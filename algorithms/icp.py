import copy
import time
import pickle
from os.path import join

import torch
import numpy as np

from algorithms.mpc_navigator_v2 import MPC
from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from gst.scripts.wrapper.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv



"""
Adapted from wafr_icp_v2.py.
- v3:
    - use plan cache to resolve infeasible planning problem.
- v2:
    - use corrected mpc with correct velocity.
    - use MPC with warm start and L2 regularization.
"""

class ICP:
    def __init__(
        self,
        args,
        config,
    ):
        self.name = "ICP"
        self.args = args
        self.config = config
        self.initialize_parameters()
        self.initialize_trajectory_predictor()
        self.initialize_model_predictive_control()
        self.initialize_calibration_environments()
        print("Algorithm initialized.")


    def initialize_parameters(self):
        # * scene parameters
        self.num_humans = self.config.sim.human_num
        self.n_agents = 1+self.num_humans # robot + other humans
        self.robot_radius = self.args.robot_radius
        self.human_radius = self.config.humans.radius
        self.robot_vel_max_magnitude = self.config.robot.v_pref
        self.robot_acc_max_magnitude = 2*self.config.robot.v_pref/self.config.env.time_step
        self.time_step = self.config.env.time_step
        self.scene_radius = self.config.sim.circle_radius
        # * trajectory predictor parameters
        self.pred_seq_len = self.args.prediction_horizon
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(join(self.args.trajectory_prediction_model_folderpath+'/checkpoint', 'args.pickle'), 'rb') as f:
            self.gst_config = pickle.load(f)
        self.gst_config.pred_seq_len = self.pred_seq_len # use pred_seq_len from argparse.
        self.obs_seq_len = self.args.observation_horizon
        self.invalid_value = self.args.trajectory_invalid_value
        # * conformal prediction parameters
        self.confidence_interval = self.args.confidence_interval
        # * calibration parameters
        self.calibration_num_processes = self.args.calibration_envs_num_processes
        self.num_calibration_episodes = self.args.calibration_size

        # self.gif_count = 0


    def initialize_trajectory_predictor(self):
        # ***** load gst *****
        self.trajectory_predictor = CrowdNavPredInterfaceMultiEnv(
            load_path=self.args.trajectory_prediction_model_folderpath,
            device=self.device,
            config=self.gst_config,
        )
        print("Trajectory prediction model loaded on device: ", self.device)


    def initialize_model_predictive_control(self):
        # ***** mpc initialization *****
        self.model_predictive_control = MPC(
            w_cte=self.args.mpc_w_cte,
            w_dv=self.args.mpc_w_dv,
            num_of_agent=self.num_humans,
            robot_r=self.robot_radius,
            human_r=self.human_radius,
            max_v=self.robot_vel_max_magnitude,
            pred_horizon=self.pred_seq_len+1,
            lookahead_step_num=self.args.mpc_lookahead_step_num,
            lookahead_step_timeinterval=self.args.mpc_lookahead_step_timeinterval,
            w_reg=self.args.mpc_w_reg,
        )
        print("Model Predictive Control initialized.")
    

    def initialize_calibration_environments(self):
        self.calibration_environments = make_vec_envs(
            self.config.env.env_name,
            self.config.env.seed,
            self.args.calibration_envs_num_processes,
            self.config.reward.gamma,
            log_dir=None,
            device='cpu',
            allow_early_resets=True,
            config=self.config,
            ax=None,
        )
        print("Calibration environments initialized.")


    def call_trajectory_predictor(self, input_traj, input_mask, measure_time=False):
        """
        Call trajectory predictor for the given input trajectory.
        - inputs
            - input_traj: np.array, input trajectory. (batch_size, n_agents, obs_seq_len, 2)
            - input_mask: np.array, input mask. (batch_size, n_agents, obs_seq_len, 1)
            - measure_time: bool, whether to measure time.
        - outputs
            - output_traj: np.array, output trajectory. (batch_size, n_agents, pred_seq_len, 2)
            - output_mask: np.array, output mask. (batch_size, n_agents, 1)
            - trajectory_predictor_inference_time: float or None, time taken for nn inference.
        """
        if measure_time:
            start_time = time.time()
        input_traj = torch.tensor(input_traj, dtype=torch.float32).to(self.device)
        input_mask = torch.tensor(input_mask, dtype=torch.float32).to(self.device)
        output_traj, output_mask = self.trajectory_predictor.forward(
            input_traj,
            input_mask,
            sampling=True,
        ) # (batch_size, n_agents, pred_seq_len, 5), (batch_size, n_agents, 1)
        output_traj = output_traj[:,:,:,:2] # only mu_x and mu_y used (batch_size, n_agents, pred_seq_len, 2)
        output_mask_traj = output_mask.unsqueeze(-1)*torch.ones_like(output_traj).to(self.device) # (batch_size, n_agents, pred_seq_len, 1)
        output_traj = output_traj*output_mask_traj+input_traj[:,:,-1:]*(1-output_mask_traj) # (batch_size, n_agents, pred_seq_len, 2)
        output_traj = output_traj.to("cpu").numpy()
        output_mask = output_mask.to("cpu").numpy()

        if measure_time:
            trajectory_predictor_inference_time = time.time()-start_time
        else:
            trajectory_predictor_inference_time = None
        # breakpoint()
        return output_traj, output_mask, trajectory_predictor_inference_time

    def mpc_robot_planning(self, scene, measure_time=False, x_init=None):
        """
        MPC robot planning for the given test scene.
        - inputs
            - scene: dict, scene for MPC planning.
                - robot_position_history: list of list, robot position history. (t, 2)
                - humans_position_history: list of list, human position history. (t, n_humans, 2)
                - humans_index_history: list of list, human index history. (t, n_humans)
                - robot_position_current: list, current robot position. (2,)
                - humans_position_current: list, current human positions. (n_humans, 2)
                - humans_velocity_current: list, current human velocities. (n_humans, 2)
                - humans_index_current: list, current human indices. (n_humans,)
                - humans_position_prediction: list of list, predicted human trajectories. (pred_seq_len, n_humans, 2)
                - humans_goal: list, human goals. (n_humans, 2)
                - humans_v_pref: list, human v_pref. (n_humans,)
                - conformal_prediction_radius: list, conformal prediction radius for humans. (pred_seq_len,)
            - measure_time: bool, whether to measure time.
            - x_init: None or list for regularization with the last iteration soluation.
        - outputs
            - sol_x: sol['x']. list.
            - robot_planned_trajectory: list of list, planned robot trajectory. (T, 2)
                - the first element is the current robot position.
            - robot_action: list, next robot action. (2,)
            - mpc_robot_planning_time: float or None, time taken for MPC planning.
        """
        # ***** MPC *****
        if measure_time:
            start_time = time.time()
        start_point = scene['robot_position_current']
        end_point = scene['robot_goal']
        conformal_pred_r = scene['conformal_prediction_radius']
        conformal_pred_r = [0.0] + conformal_pred_r
        agent_pos_pred = [scene['humans_position_current']] + scene['humans_position_prediction']
        sol, sol_status = self.model_predictive_control.Solve(start_point, end_point, agent_pos_pred, conformal_pred_r, x_init=x_init)
        print("MPC solver status:", sol_status)
        if sol_status == 'Infeasible_Problem_Detected':
            sol_x = None
            robot_planned_trajectory = None
            robot_action = None
            if measure_time:
                mpc_robot_planning_time = time.time()-start_time
            else:
                mpc_robot_planning_time = None
            return sol_x, robot_planned_trajectory, robot_action, mpc_robot_planning_time
        else:
            robot_planned_trajectory = self.model_predictive_control.get_robot_planned_traj()
            if measure_time:
                mpc_robot_planning_time = time.time()-start_time
            else:
                mpc_robot_planning_time = None
            robot_action = (np.array(robot_planned_trajectory[1])-np.array(robot_planned_trajectory[0]))/self.time_step
            robot_action = robot_action.tolist()
            sol_x = sol['x']
            return sol_x, robot_planned_trajectory, robot_action, mpc_robot_planning_time

    def predict_human_trajectories(self, scene, measure_time=False):
        """
        Predict human trajectories for the given scene.
        - inputs
            - scene: dict, scene for human trajectory prediction.
                - robot_position_history: list of list, robot position history. (t, 2)
                - humans_position_history: list of list, human position history. (t, n_humans, 2)
                - humans_index_history: list of list, human index history. (t, n_humans)
                - robot_position_current: list, current robot position. (2,)
                - humans_position_current: list, current human positions. (n_humans, 2)
                - humans_velocity_current: list, current human velocities. (n_humans, 2)
                - humans_index_current: list, current human indices. (n_humans,)
                - humans_goal: list, human goals. (n_humans, 2)
                - humans_v_pref: list, human v_pref. (n_humans,)
                - conformal_prediction_radius: list, conformal prediction radius for humans. (pred_seq_len,)
            - measure_time: bool, whether to measure time.
        - outputs
            - humans_position_prediction: list of list, predicted human trajectories. (t_future, n_humans, 2)
            - human_trajectory_prediction_time: float or None, time taken for human trajectory prediction.
        """
        if measure_time:
            start_time = time.time()
        time_steps = min(len(scene['humans_position_history']), self.obs_seq_len)
        robot_position_history = np.array(scene['robot_position_history'][-time_steps:]) # (time_steps, 2)
        humans_position_history = np.array(scene['humans_position_history'][-time_steps:]) # (time_steps, n_humans, 2)
        humans_index_history = np.array(scene['humans_index_history'][-time_steps:]) # (time_steps, n_humans)
        humans_input_mask = humans_index_history == humans_index_history[-1] # (time_steps, n_humans)
        robot_humans_input_traj_raw = np.concatenate((robot_position_history[:,np.newaxis], humans_position_history), axis=1) # (time_steps, 1+n_humans, 2)
        robot_humans_input_mask = np.concatenate((np.ones((time_steps,1)), humans_input_mask), axis=1) # (time_steps, 1+n_humans)
        if time_steps < self.obs_seq_len:
            invalid_input = np.ones((self.obs_seq_len-time_steps, self.n_agents, 2))*self.invalid_value
            input_traj = np.concatenate((invalid_input, robot_humans_input_traj_raw), axis=0)[np.newaxis] # (1, obs_seq_len, n_agents, 2)
            input_traj = input_traj.transpose((0,2,1,3))# (1, n_agents,obs_seq_len,2)
            input_mask = np.concatenate(
                (
                    np.zeros((1,self.n_agents,self.obs_seq_len-time_steps,1)),
                    robot_humans_input_mask.transpose((1,0))[:,:,np.newaxis][np.newaxis], # (1,n_agents,time_steps,1)
                ),
                axis=2,
            ) # (1,n_agents,obs_seq_len,1)
        else:
            input_traj = robot_humans_input_traj_raw[np.newaxis] # (1, obs_seq_len, n_agents, 2)
            input_traj = input_traj.transpose((0,2,1,3)) # (1, n_agents, obs_seq_len, 2)
            input_mask = robot_humans_input_mask.transpose((1,0))[:,:,np.newaxis][np.newaxis] # (1,n_agents,obs_seq_len,1)
        output_traj, _, _ = self.call_trajectory_predictor(input_traj, input_mask, measure_time=False) # np (1, n_agents, pred_seq_len, 2)
        output_traj = output_traj[0][1:] # (n_humans, pred_seq_len, 2), n_humans = n_agents - 1
        humans_position_prediction = output_traj.transpose(1,0,2).tolist() # list (pred_seq_len, n_humans, 2)
        if measure_time:
            human_trajectory_prediction_time = time.time()-start_time
        else:
            human_trajectory_prediction_time = None
        return humans_position_prediction, human_trajectory_prediction_time

    def generate_calibration_dataset(self, scene, measure_time=False):
        """
        Generate calibration dataset for the given scene.
        - inputs
            - scene: dict, scene for calibration dataset generation.
                - robot_position_history: list of list, robot position history. (t, 2)
                - humans_position_history: list of list, human position history. (t, n_humans, 2)
                - humans_index_history: list of list, human index history. (t, n_humans)
                - robot_position_current: list, current robot position. (2,)
                - humans_position_current: list, current human positions. (n_humans, 2)
                - humans_velocity_current: list, current human velocities. (n_humans, 2)
                - humans_index_current: list, current human indices. (n_humans,)
                - humans_goal: list, human goals. (n_humans, 2)
                - humans_v_pref: list, human v_pref. (n_humans,)
                - conformal_prediction_radius: list, conformal prediction radius for humans. (pred_seq_len,)
                - humans_position_prediction: list of list, predicted human trajectories. (pred_seq_len, n_humans, 2)
                - robot_planned_trajectory: list of list, planned robot trajectory. (T, 2)
            - measure_time: bool, whether to measure time.
        - outputs
            - calibration_dataset: list of dict, calibration dataset.
                - calibration_data
                    - robot_position: list of list, robot position history. (t, 2)
                    - humans_position: list of list, human position history. (t, n_humans, 2)
                    - humans_index_current: list of list, human index history. (t, n_humans)
            - calibration_dataset_generation_time: float or None, time taken for calibration dataset generation.
        """
        if measure_time:
            start_time = time.time()
        self.calibration_environments.load_snapshot(scene)
        obs = self.calibration_environments.reset(phase='test', test_case=None, from_snapshot=True, use_orca=False)
        calibration_data_processes = []
        for process_index in range(self.calibration_num_processes):
            calibration_data = {}
            calibration_data['robot_position'] = [obs['robot_position'][process_index,0].tolist()] # (t, 2)
            calibration_data['humans_position'] = [obs['human_positions'][process_index].tolist()] # (t, n_humans, 2)
            calibration_data['humans_index_current'] = [obs['humans_index_current'][process_index].tolist()] # (t, n_humans)
            calibration_data_processes.append(calibration_data)
        calibration_dataset = []
        while len(calibration_dataset) < self.num_calibration_episodes:
            obs, reward, done, info = self.calibration_environments.step(torch.zeros(self.calibration_num_processes, 2))
            if np.any(done):
                for done_process_index in np.where(done)[0]:
                    if info[done_process_index]['info'] != 'ReachGoal':
                        print(info[done_process_index]['info']+" causes early ending of iterations in Algorithm.")
                        # may be due to collision
                        # breakpoint()
                    calibration_dataset.append(calibration_data_processes[done_process_index])
                    calibration_data_processes[done_process_index] = {}
                    calibration_data_processes[done_process_index]['robot_position'] = []
                    calibration_data_processes[done_process_index]['humans_position'] = []
                    calibration_data_processes[done_process_index]['humans_index_current'] = []
            for process_index in range(self.calibration_num_processes):
                calibration_data_processes[process_index]['robot_position'].append(obs['robot_position'][process_index,0].tolist())
                calibration_data_processes[process_index]['humans_position'].append(obs['human_positions'][process_index].tolist())
                calibration_data_processes[process_index]['humans_index_current'].append(obs['humans_index_current'][process_index].tolist())
        
        if measure_time:
            calibration_dataset_generation_time = time.time()-start_time
        else:
            calibration_dataset_generation_time = None
        return calibration_dataset, calibration_dataset_generation_time
    
    def evaluate_calibration_dataset(self, calibration_dataset, measure_time=False):
        """
        Evaluate calibration dataset.
        - inputs
            - calibration_dataset: list of dict, calibration dataset.
                - calibration_data
                    - robot_position: list of list, robot position history. (t, 2)
                    - humans_position: list of list, human position history. (t, n_humans, 2)
                    - humans_index_current: list of list, human index history. (t, n_humans)
            - measure_time: bool, whether to measure time.
        - outputs
            - prediction_error_list: list of list, prediction error list. (pred_seq_len, dataset_sample_size*n_humans)
            - prediction_mask_list: list of list, prediction mask list. (pred_seq_len, dataset_sample_size*n_humans)
            - calibration_evaluation_time: float or None, time taken for calibration evaluation.
        """
        if measure_time:
            start_time = time.time()
        # ***** prediction on calibration dataset *****
        input_traj_list = []
        input_binary_mask_list = []
        human_traj_future_list = []
        human_traj_future_mask_list = []
        prediction_error_list = []
        prediction_mask_list = []
        # inference_batch_size = 3000 #128
        for _, calibration_data in enumerate(calibration_dataset):
            robot_position = np.array(calibration_data['robot_position'])
            humans_position = np.array(calibration_data['humans_position'])
            humans_index_current = np.array(calibration_data['humans_index_current']) # (t, n_humans)
            # humans_velocity = np.array(calibration_data['humans_velocity'])
            time_steps = robot_position.shape[0]
            for tt in range(0, time_steps, self.args.calibration_step):
                if tt+self.obs_seq_len+self.pred_seq_len > time_steps:
                    break
                human_traj_past = humans_position[tt:tt+self.obs_seq_len] # (obs_seq_len, n_humans, 2)
                robot_traj_past = robot_position[tt:tt+self.obs_seq_len] # (obs_seq_len, 2)     
                human_idx_in_sim_past = humans_index_current[tt:tt+self.obs_seq_len] # (obs_seq_len, n_humans)
                
                human_traj_past_mask = human_idx_in_sim_past == human_idx_in_sim_past[-1] # (obs_seq_len, n_humans)
                robot_humans_input_binary_mask = np.concatenate((np.ones((self.obs_seq_len,1)), human_traj_past_mask), axis=1) # (obs_seq_len, 1+n_humans)
                input_binary_mask = robot_humans_input_binary_mask.transpose((1,0))[:,:,np.newaxis][np.newaxis] # (1, 1+n_humans, obs_seq_len, 1)
                input_binary_mask_list.append(input_binary_mask) # (1, 1+n_humans, obs_seq_len, 1)

                human_traj_future = humans_position[tt+self.obs_seq_len:tt+self.obs_seq_len+self.pred_seq_len] # (pred_seq_len, n_humans, 2)
                human_traj_future_list.append(human_traj_future) # # (pred_seq_len, n_humans, 2)
                
                human_idx_in_sim_future = humans_index_current[tt+self.obs_seq_len:tt+self.obs_seq_len+self.pred_seq_len] # (pred_seq_len, n_humans)
                human_traj_future_mask = human_idx_in_sim_future == human_idx_in_sim_past[-1] # (pred_seq_len, n_humans)
                human_traj_future_mask_list.append(human_traj_future_mask) # (pred_seq_len, n_humans)

                input_traj = np.concatenate((robot_traj_past[:,np.newaxis], human_traj_past), axis=1) # (obs_seq_len, 1+n_humans, 2)
                input_traj = input_traj[np.newaxis] # (1, obs_seq_len, n_agents, 2)
                input_traj = input_traj.transpose((0,2,1,3)) # (1, n_agents,obs_seq_len,2)
                input_traj_list.append(input_traj)
        if len(input_traj_list) == 0:
            print("No calibration data for prediction. Ending iterations in algorithm early.")
            return None, None, None
        input_binary_mask = np.concatenate(input_binary_mask_list, axis=0) # (batch_size, 1+n_humans, obs_seq_len, 1)
        # try:
        #     input_binary_mask = np.concatenate(input_binary_mask_list, axis=0) # (batch_size, 1+n_humans, obs_seq_len, 1)
        # except:
        #     print()
        #     robot_position_last = np.array(calibration_dataset[0]['robot_position'])[-1]
        #     humans_position_last = np.array(calibration_dataset[0]['humans_position'])[-1]
        #     print(np.linalg.norm(humans_position_last-robot_position_last, axis=1))
        #     breakpoint()
        input_traj_batch = np.concatenate(input_traj_list, axis=0) # (batch_size, n_agents,obs_seq_len,2)
        input_traj_batch = torch.tensor(input_traj_batch, dtype=torch.float32).to(self.device)
        input_binary_mask = torch.tensor(input_binary_mask, dtype=torch.float32).to(self.device)
        output_traj_batch, output_binary_mask, _ = self.call_trajectory_predictor(
            input_traj_batch,
            input_binary_mask,
            measure_time=False,
        )
        output_traj_batch = output_traj_batch[:,1:] # remove robot (batch_size, n_humans, pred_seq_len, 2)
        output_traj_batch = output_traj_batch.transpose((0,2,1,3)) # (batch_size, pred_seq_len, n_humans, 2)
        human_traj_future_batch = np.stack(human_traj_future_list, axis=0) # (batch_size, pred_seq_len, n_humans, 2)
        
        output_binary_mask = output_binary_mask[:,:,0][:,np.newaxis] # (batch_size, 1, 1+n_humans)
        human_output_binary_mask = output_binary_mask[:,:,1:] # (batch_size, 1, n_humans)
        human_traj_future_mask = np.stack(human_traj_future_mask_list, axis=0) # (batch_size, pred_seq_len, n_humans)
        # breakpoint()
        human_traj_future_mask = human_traj_future_mask * human_output_binary_mask # (batch_size, pred_seq_len, n_humans)

        prediction_error = np.linalg.norm(output_traj_batch-human_traj_future_batch, axis=3) # (batch_size, pred_seq_len, n_humans)
        prediction_error = prediction_error * human_traj_future_mask # (batch_size, pred_seq_len, n_humans)

        prediction_error = prediction_error.transpose((0,2,1)) # (batch_size, n_humans, pred_seq_len)
        prediction_error = prediction_error.reshape(-1, prediction_error.shape[-1]) # (batch_size*n_humans, pred_seq_len)
        prediction_error = prediction_error.transpose((1,0)) # (pred_seq_len, batch_size*n_humans)
        prediction_error_list.append(prediction_error.tolist())

        prediction_mask = human_traj_future_mask
        prediction_mask = prediction_mask.transpose((0,2,1)) # (batch_size, n_humans, pred_seq_len)
        prediction_mask = prediction_mask.reshape(-1, prediction_mask.shape[-1]) # (batch_size*n_humans, pred_seq_len)
        prediction_mask = prediction_mask.transpose((1,0)) # (pred_seq_len, batch_size*n_humans)
        prediction_mask_list.append(prediction_mask.tolist())

        # print("number of samples in calibration dataset: ", int(np.array(prediction_error_list).shape[2]/self.num_humans))
        # breakpoint()
        print("prediction on calibration dataset finished.")
        if measure_time:
            calibration_evaluation_time = time.time()-start_time
        else:
            calibration_evaluation_time = None
        return prediction_error_list, prediction_mask_list, calibration_evaluation_time

    def conformal_prediction(self, prediction_error_list, prediction_mask_list, measure_time=False):
        """
        Conformal prediction for the given prediction error list and prediction mask list.
        - inputs
            - prediction_error_list: list of list, prediction error list. (pred_seq_len, dataset_sample_size*n_humans)
            - prediction_mask_list: list of list, prediction mask list. (pred_seq_len, dataset_sample_size*n_humans)
        - outputs
            - conformal_prediction_radius: list, conformal prediction radius for humans. (pred_seq_len,)
            - calibration_dataset_size: list, calibration dataset size for each time step. (pred_seq_len,)
            - conformal_prediction_time: float or None, time taken for conformal prediction.
        """
        if measure_time:
            start_time = time.time()
        # ***** conformal prediction *****
        error_data = np.concatenate(prediction_error_list, axis=1) # (time, all_humans)
        mask_data = np.concatenate(prediction_mask_list, axis=1) # (time, all_humans)
        conformal_prediction_radius = []
        calibration_dataset_size = []
        for pred_time_step in range(self.pred_seq_len):
            error_data_step = error_data[pred_time_step] # (all_humans)
            mask_data_step = mask_data[pred_time_step].astype(np.int64) # (all_humans)
            valid_errors_step = error_data_step[mask_data_step==1] # (n_valid_humans)
            # * we assume valid error data is large enough, so it is guranteed to find k_alpha (index of confidence interval)
            error_sorted_step = np.sort(valid_errors_step) # (n_valid_humans)
            calibration_dataset_size.append(error_sorted_step.shape[0])
            k_alpha = int(self.confidence_interval*(error_sorted_step.shape[0]+1))+1
            try:
                conformal_error_bound = error_sorted_step[k_alpha-1] # scalar
            except:
                # IndexError: index 10 is out of bounds for axis 0 with size 10
                print("warning on error_sorted_step has length of "+str(len(error_sorted_step))+", but bound index sets as "+str(k_alpha-1)+". So we use the last one.")
                conformal_error_bound = error_sorted_step[-1]
            conformal_prediction_radius.append(conformal_error_bound)
    
        if measure_time:
            conformal_prediction_time = time.time()-start_time
        else:
            conformal_prediction_time = None
        print("calibration dataset size: ", calibration_dataset_size)
        return conformal_prediction_radius, calibration_dataset_size, conformal_prediction_time

    def run_iteration(self, scene, num_iterations, measure_time=False):
        """
        Run iterations (full version of Algorithm) for the given scene.
        - inputs
            - scene: dict, scene for baseline.
                - robot_position_history: list of list, robot position history. (t, 2)
                - humans_position_history: list of list, human position history. (t, n_humans, 2)
                - humans_index_history: list of list, human index history. (t, n_humans)
                - robot_position_current: list, current robot position. (2,)
                - humans_position_current: list, current human positions. (n_humans, 2)
                - humans_velocity_current: list, current human velocities. (n_humans, 2)
                - humans_index_current: list, current human indices. (n_humans,)
                - humans_goal: list, human goals. (n_humans, 2)
                - humans_v_pref: list, human v_pref. (n_humans,)
                - conformal_prediction_radius: list, conformal prediction radius for humans. (pred_seq_len,)
            - num_iterations: int, number of iterations.
            - measure_time: bool, whether to measure time.
        - outputs
            - action: list, next robot action. (2,)
            - scene: dict, updated scene.
            - algorithm_run_record: dict, algorithm run record.
            - runtime: dict, runtime information. #! todo later
        """
        # ! may need record for every iteration.
        algorithm_run_record = {}
        if measure_time:
            algorithm_run_record['time'] = {}
        scene['conformal_prediction_radius'] = [0. for i in range(self.pred_seq_len)]
        algorithm_run_record['conformal_prediction_radius'] = [] # * do not include all-zero radius.
        algorithm_run_record['calibration_dataset_size'] = []
        humans_position_prediction, human_trajectory_prediction_time = self.predict_human_trajectories(scene, measure_time=measure_time)
        scene['humans_position_prediction'] = humans_position_prediction
        if measure_time:
            algorithm_run_record['time']['human_trajectory_prediction'] = human_trajectory_prediction_time
        algorithm_run_record['humans_position_prediction'] = humans_position_prediction
        print("human trajectory prediction finished.")
        sol_x, robot_planned_trajectory, robot_action, mpc_robot_planning_time = self.mpc_robot_planning(scene, measure_time=measure_time, x_init=None)
        if robot_action is None:
            if 'robot_plan_cache' not in scene.keys():
                raise RuntimeError("Infeasible planning problem detected at the first time step of the scene.")
            else:
                print("Infeasible planning problem detected, use plan cache to resolve.")
            scene['robot_plan_cache'] = scene['robot_plan_cache'][1:]
            scene['robot_plan_cache'][0] = scene['robot_position_current'] # ! Make the start the same as current start position, otherwise execution has slight shift.
            robot_plan_cache = np.array(scene['robot_plan_cache'])
            robot_action = (robot_plan_cache[1]-robot_plan_cache[0])/self.time_step
            robot_action = robot_action.tolist()
            algorithm_run_record['robot_plan_cache'] = copy.deepcopy(robot_plan_cache.tolist())
            runtime = None
            return robot_action, scene, algorithm_run_record, runtime
        scene['robot_planned_trajectory'] = robot_planned_trajectory
        algorithm_run_record['robot_planned_trajectory'] = []
        algorithm_run_record['robot_planned_trajectory'].append(robot_planned_trajectory)
        if measure_time:
            algorithm_run_record['time']['mpc_robot_planning'] = []
            algorithm_run_record['time']['mpc_robot_planning'].append(mpc_robot_planning_time)
            algorithm_run_record['time']['calibration_dataset_generation'] = []
            algorithm_run_record['time']['calibration_evaluation'] = []
            algorithm_run_record['time']['conformal_prediction'] = []
        print("MPC planning finished.")
        for iteration_index in range(1, num_iterations+1):
            calibration_dataset, calibration_dataset_generation_time = self.generate_calibration_dataset(scene, measure_time=measure_time)
            print("calibration dataset generated.")
            if measure_time:
                algorithm_run_record['time']['calibration_dataset_generation'].append(calibration_dataset_generation_time)
            prediction_error_list, prediction_mask_list, calibration_evaluation_time = self.evaluate_calibration_dataset(calibration_dataset, measure_time=measure_time)
            if measure_time:
                algorithm_run_record['time']['calibration_evaluation'].append(calibration_evaluation_time)
            if prediction_error_list is None:
                break
            print("calibration dataset evaluated.")
            conformal_prediction_radius, calibration_dataset_size, conformal_prediction_time = self.conformal_prediction(prediction_error_list, prediction_mask_list, measure_time=measure_time)
            if measure_time:
                algorithm_run_record['time']['conformal_prediction'].append(conformal_prediction_time)
            scene['conformal_prediction_radius'] = conformal_prediction_radius
            algorithm_run_record['conformal_prediction_radius'].append(conformal_prediction_radius)
            algorithm_run_record['calibration_dataset_size'].append(calibration_dataset_size)
            print("conformal prediction finished.")
            sol_x, robot_planned_trajectory, robot_action, mpc_robot_planning_time = self.mpc_robot_planning(scene, measure_time=measure_time, x_init=sol_x)
            if robot_action is None:
                if 'robot_plan_cache' not in scene.keys():
                    raise RuntimeError("Infeasible planning problem detected at the first time step of the scene.")
                else:
                    print("Infeasible planning problem detected, use plan cache to resolve.")
                scene['robot_plan_cache'] = scene['robot_plan_cache'][1:]
                scene['robot_plan_cache'][0] = scene['robot_position_current'] # ! Make the start the same as current start position, otherwise execution has slight shift.
                robot_plan_cache = np.array(scene['robot_plan_cache'])
                robot_action = (robot_plan_cache[1]-robot_plan_cache[0])/self.time_step
                robot_action = robot_action.tolist()
                algorithm_run_record['robot_plan_cache'] = copy.deepcopy(robot_plan_cache.tolist())
                runtime = None
                return robot_action, scene, algorithm_run_record, runtime
            
            scene['robot_planned_trajectory'] = robot_planned_trajectory
            algorithm_run_record['robot_planned_trajectory'].append(robot_planned_trajectory)
            if measure_time:
                algorithm_run_record['time']['mpc_robot_planning'].append(mpc_robot_planning_time)
            print("Iteration "+str(iteration_index)+" finished.")
        algorithm_run_record['robot_plan_cache'] = copy.deepcopy(scene['robot_planned_trajectory'])
        scene['robot_plan_cache'] = copy.deepcopy(scene['robot_planned_trajectory'])
        action = robot_action
        runtime = None # ! todo later
        return action, scene, algorithm_run_record, runtime