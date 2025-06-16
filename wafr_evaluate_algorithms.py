import os
import json
import argparse

import gym
import numpy as np

import crowd_sim
from crowd_nav.configs.config import Config

"""
- v3:
    - use plan cache to store robot plan to handle infeasible cases.
- v2:
    - use corrected mpc with correct velocity.
    - use MPC with warm start and L2 regularization.
"""
def arg_parse():
    # ***** arguments and configs *****
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, default='icp', help='icp, offcp, orca, acp.')
    parser.add_argument('-i', '--iterations', type=int, default=3, help='number of iterations of the whole CP-MPC algorithm.')
    parser.add_argument('-f', '--filename', type=str, default='240506')
    parser.add_argument('-o', '--observation_horizon', type=int, default=5, help='observation horizon for human motion. The trained gst model configuration has default observation horizon as 5.')
    parser.add_argument('-p', '--prediction_horizon', type=int, default=5, help='prediction horizon for human motion.')
    parser.add_argument('-t', '--test_scene_step', type=int, default=15, help='step in simulation which will be used as test scene.')
    parser.add_argument('-s', '--sim_random_seed', type=int, default=2081, help='random seed of crowdnav simulation.')
    parser.add_argument('-n', '--num_humans', type=int, default=5, help='number of humans in the scene.')
    parser.add_argument('-r', '--render', action='store_true', help='Render for test scene generation, mpc plan, and sth else.')
    parser.add_argument('-m', '--measure_time', action='store_true', help='Measure time for each step.')
    parser.add_argument('--fps', type=int, default=5, help='animation frame per second for mpc plan gif.')
    parser.add_argument('--calibration_size', type=int, default=100, help='number of episodes for calibration.')
    parser.add_argument('--calibration_step', type=int, default=3, help='number of time steps to jump in a calibration episode. The larger, the less calibration data.')
    parser.add_argument('--confidence_interval', type=float, default=0.95, help='confidence interval for conformal prediction.')
    parser.add_argument('--robot_radius', type=float, default=0.4, help='robot radius. default config.robot.radius=0.4.')
    parser.add_argument('--mpc_w_cte', type=float, default=1.0, help='mpc parameters.')
    parser.add_argument('--mpc_w_dv', type=float, default=5.0, help='mpc parameters.')
    parser.add_argument('--mpc_w_reg', type=float, default=0.5, help='mpc parameters.')
    parser.add_argument('--mpc_lookahead_step_num', type=int, default=50, help='mpc parameters.')
    parser.add_argument('--mpc_lookahead_step_timeinterval', type=float, default=0.25, help='mpc parameters.')
    parser.add_argument('--trajectory_prediction_model_folderpath', type=str, default='gst/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj')
    parser.add_argument('--calibration_envs_num_processes', type=int, default=8, help='number of parallel environments for calibration.')
    parser.add_argument('--trajectory_invalid_value', type=float, default=-999., help='invalid value for padding.')

    parser.add_argument('--humans_random_goal_changing', action='store_true', help='random goal changing for humans when reaching to goal.')
    parser.add_argument('--humans_random_goal_shift_max_times', type=int, default=2, help='max times of random goal shift when reaching to goal.')
    parser.add_argument('--humans_random_goal_shift_displacement_magnitude', type=float, default=2., help='max displacement xy range (unit: m) of random goal shift when reaching to goal. Default means [-2, 2].')
    parser.add_argument('--humans_end_goal_changing', action='store_true', help='end goal changing for humans when reaching to goal. Human index updated.')

    parser.add_argument('--acp_gamma', type=float, default=5e-3, help='gamma for ACP. Learning rate for ACP.')
    parser.add_argument('--acp_time_window', type=int, default=30, help='N for ACP. time window for ACP.')
    args = parser.parse_args()
    return args

args = arg_parse()
config = Config()
config.sim.human_num = args.num_humans
config.robot.radius = args.robot_radius
config.gst.obs_seq_len = args.observation_horizon

config.humans.random_goal_changing = args.humans_random_goal_changing
config.humans.random_goal_shift_max_times = args.humans_random_goal_shift_max_times
config.humans.random_goal_shift_displacement_magnitude = args.humans_random_goal_shift_displacement_magnitude
config.humans.end_goal_changing = args.humans_end_goal_changing
if args.algorithm == "icp":
    config_name = "icp_"+str(args.iterations)+"_"+str(args.calibration_size)+"_"+str(args.num_humans)+"_"+str(args.sim_random_seed)
elif args.algorithm == "offcp":
    config_name = "offcp_0_"+str(args.calibration_size)+"_"+str(args.num_humans)+"_"+str(args.sim_random_seed)
elif args.algorithm == "orca":
    config_name = "orca_0_0_"+str(args.num_humans)+"_"+str(args.sim_random_seed)
elif args.algorithm == "acp": # not added to public repo yet.
    config_name = "acp_"+str(args.acp_gamma)+"_"+str(args.acp_time_window)+"_"+str(args.num_humans)+"_"+str(args.sim_random_seed)
else:
    raise NotImplementedError
config_json_filename = "results/"+config_name+".json"
visualization_folder = "results/visualization/"+config_name
os.makedirs(visualization_folder, exist_ok=True)



# ***** crowdnav sim setup and capture test scene *****
env_name = config.env.env_name
env = gym.make(env_name)
env.configure(config)
env.thisSeed = args.sim_random_seed
env.nenv = 1

outer_case = 0 # experiments involve multiple outer cases. Change this to get to new experiments.
num_steps = 500 # 1000

# ***** algorithm *****
if args.algorithm == "icp":
    from algorithms.icp import ICP as Algorithm
    algorithm = Algorithm(args, config)
elif args.algorithm == "offcp":
    from algorithms.offcp import OffCP as Algorithm
    algorithm = Algorithm(args, config)
elif args.algorithm == "orca":
    pass
elif args.algorithm == "acp":
    from algorithms.acp import ACP as Algorithm
    algorithm = Algorithm(args, config)
else:
    raise NotImplementedError

pred_seq_len = args.prediction_horizon
obs_seq_len = args.observation_horizon
invalid_value = args.trajectory_invalid_value

# ***** test scene initialization *****
scene = {}
scene['conformal_prediction_radius'] = [0. for i in range(pred_seq_len)]

evaluation_case_record = {}

# obs = env.reset(phase='train', test_case=outer_case, from_snapshot=False, use_orca=False)
# obs = env.reset(test_case=outer_case, from_snapshot=False, use_orca=True)
if args.algorithm == "orca":
    obs = env.reset(test_case=outer_case, from_snapshot=False, use_orca=True)
else:
    obs = env.reset(test_case=outer_case, from_snapshot=False, use_orca=False)
scene['robot_goal'] = obs['robot_node'][3:5]
scene['robot_position_history'] = [] # robot_position_hist = []
scene['humans_position_history'] = [] # human_positions_hist = []
scene['humans_index_history'] = [] # humans_index_history = []

scene['robot_position_history'].append(obs['robot_position']) # list (2,)
scene['humans_position_history'].append(obs['human_positions']) # list (n_humans, 2)
scene['humans_index_history'].append(obs['humans_index_current']) # list (n_humans,)
scene['robot_position_current'] = obs['robot_position']
scene['humans_position_current'] = obs['human_positions'] # list (n_humans, 2)
scene['humans_velocity_current'] = obs['human_velocities'] # list (n_humans, 2)
scene['humans_index_current'] = obs['humans_index_current']

scene['humans_goal'] = obs['human_goals'].tolist()
scene['humans_v_pref'] = obs['human_v_pref'].tolist()

evaluation_case_record['robot_goal'] = scene['robot_goal']
evaluation_case_record['humans_goal'] = []
evaluation_case_record['humans_goal'].append(scene['humans_goal'])
evaluation_case_record['algorithm_run_record'] = []

if args.algorithm == "acp":
    evaluation_case_record['error_list_sliding_window'] = []
    evaluation_case_record['error_list_history'] = []
    evaluation_case_record['acp_delta_t_history'] = []

for step in range(num_steps):
    if args.algorithm == "icp":
        action, scene, algorithm_run_record, runtime = algorithm.run_iteration(scene, num_iterations=args.iterations, measure_time=args.measure_time)
        evaluation_case_record['algorithm_run_record'].append(algorithm_run_record)
    elif args.algorithm == "offcp":
        if step == 0:
            action, scene, algorithm_run_record, runtime = algorithm.run(scene, start=True, measure_time=False)
        else:
            action, scene, algorithm_run_record, runtime = algorithm.run(scene, start=False, measure_time=False)
        evaluation_case_record['algorithm_run_record'].append(algorithm_run_record)
    elif args.algorithm == "orca":
        action = [0, 0] # dummy action. Alerady did env.reset(use_orca=True)
    elif args.algorithm == "acp":
        action, scene, algorithm_run_record, acp_delta_t = algorithm.run(scene, evaluation_case_record)
        evaluation_case_record['algorithm_run_record'].append(algorithm_run_record)
        # evaluation_case_record['error_list_sliding_window'].append(...) # * updated within run function
        # evaluation_case_record['error_list_history'].append(...) # * updated within run function
        evaluation_case_record['acp_delta_t_history'].append(acp_delta_t)
    # breakpoint()
    obs, reward, done, info = env.step(action)
    scene['robot_position_history'].append(obs['robot_position']) # list (2,)
    scene['humans_position_history'].append(obs['human_positions']) # list (n_humans, 2)
    scene['humans_index_history'].append(obs['humans_index_current']) # list (n_humans,)
    scene['robot_position_current'] = obs['robot_position']
    scene['humans_position_current'] = obs['human_positions'] # list (n_humans, 2)
    scene['humans_velocity_current'] = obs['human_velocities'] # list (n_humans, 2)
    scene['humans_index_current'] = obs['humans_index_current']

    scene['humans_goal'] = obs['human_goals'].tolist()
    scene['humans_v_pref'] = obs['human_v_pref'].tolist()
    # humans_position_prediction: list of list, predicted human trajectories. (t_future, n_humans, 2)

    evaluation_case_record['robot_position_history'] = scene['robot_position_history']
    evaluation_case_record['humans_position_history'] = scene['humans_position_history']
    evaluation_case_record['humans_index_history'] = scene['humans_index_history']
    if args.render:
        if args.algorithm == "orca":
            env.render(
                step=step,
                traj_pred=None,
                robot_plan=None,
                conformal_prediction_radius=None,
                visualization_folder=visualization_folder,
                robot_position_history=scene['robot_position_history'],
            )
        else:
            env.render(
                step=step,
                traj_pred=np.array(scene['humans_position_prediction']).transpose(1,0,2),
                # robot_plan=np.array(scene['robot_planned_trajectory']),
                robot_plan=np.array(scene['robot_plan_cache']),
                conformal_prediction_radius=np.array(scene['conformal_prediction_radius']),
                visualization_folder=visualization_folder,
                robot_position_history=scene['robot_position_history'],
            ) # (T, 2)
    # traj_pred (num_peds, pred_seq_len, 2))
    print("rendered step: ", step)
    if done:
        break
        # breakpoint()


with open(config_json_filename, "w") as f:
    json.dump(evaluation_case_record, f)
    print(config_json_filename+" is saved.")