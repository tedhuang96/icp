import gym
import numpy as np
from numpy.linalg import norm
import copy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import CrowdSim
from collections import deque


# The class for the simulation environment used for training a DSRNN policy
# ! Copied from crowd_sim_dict_conformal_prediction.py.

class CrowdSimDict(CrowdSim):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()

        self.desiredVelocity=[0.0,0.0]
        self.human_states_window = deque() # ! to be cleaned # ! to be fixed with human indexing # ! seems unnecessary for calibration data collection. waste time.
        # ! for load snapshot reset, we load humans_index_current from test scene, but i think we will just mostly keep it as [0,1,2,3,4] for 5 humans for example.
        # ! because in real test scene, humans do not change goal after reaching goal.

    # define the observation space and the action space
    def set_robot(self, robot):
        self.robot = robot

        # set observation space and action space
        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d={}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype = np.float32)
        # only consider the robot temporal edge and spatial edges pointing from robot to each human
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,2,), dtype=np.float32)
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num, 2), dtype=np.float32)
        
        d['robot_position'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,2,), dtype = np.float32)
        d['human_positions'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num, 2), dtype = np.float32)
        d['humans_index_current'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num,), dtype = np.float32)

        # ob['humans_index_current'] = self.humans_index_current.copy() # list (n_humans) # * need copy because update is based on in-place update.

        
        # ob['robot_position'] = ob['robot_node'][:2] # list (2,)
        # ob['human_positions'] = ob['human_states'][:,:2].tolist() # list (n_humans, 2)

        # robot_position = np.array(calibration_data['robot_position'])
        # humans_position = np.array(calibration_data['humans_position'])
        # humans_index_in_sim = np.array(calibration_data['humans_index_in_sim']) # (t, n_humans)    
        self.observation_space=gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    # generate observation for each timestep
    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        ob['robot_node'] = self.robot.get_full_state_list_noV()

        self.update_last_human_states(human_visibility, reset=reset)

        # edges
        # temporal edge: robot's velocity
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])
        # spatial edges: the vector pointing from the robot position to each human's position
        ob['spatial_edges'] = np.zeros((self.human_num, 2))
        ob['human_v_pref'] = np.zeros(self.human_num)

        for i in range(self.human_num):
            relative_pos = np.array([self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
            ob['spatial_edges'][i] = relative_pos
            ob['human_v_pref'][i] = self.humans[i].v_pref
        
        ob['human_states'] = self.last_human_states # (n_humans, state_dim)
        ob['human_goals'] = self.last_human_goals # (n_humans, 2)
        ob['humans_index_current'] = self.humans_index_current.copy() # list (n_humans) # * need copy because update is based on in-place update.

        
        ob['robot_position'] = ob['robot_node'][:2] # list (2,)
        ob['human_positions'] = ob['human_states'][:,:2].tolist() # list (n_humans, 2)
        ob['human_velocities'] = ob['human_states'][:,2:4].tolist() # list (n_humans, 2)

        robot_pos = np.array(ob['robot_node'])[np.newaxis][:,:2]
        self.human_states_window.append(np.concatenate([robot_pos, self.last_human_states.copy()[:,:2]], axis=0)) # (1+n_humans, 2)
        if len(self.human_states_window) > self.obs_seq_len:
            self.human_states_window.popleft()
        # ob['human_states_seq'] = np.array(self.human_states_window) # (time_step, n_humans, state_dim)
        ob['robot_humans_state_seq'] = np.array(self.human_states_window) # (time_step, n_humans, state_dim)
        return ob


    # reset function
    # def reset(self, phase='train', test_case=None):
    def reset(self, phase='train', test_case=None, from_snapshot=False, use_orca=False):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        # * WAFR: train phase equivalent as outside evaluation
        # * WAFR: test phase equivalent as calibration
        # print("test_case: ", test_case)
        # print("from_snapshot: ", from_snapshot)
        # print("use_orca: ", use_orca)
        # if self.phase is not None:
        #    phase = self.phase
        self.phase = phase
        """
        # ! in envs.py, now we are reversed
        if envNum > 1:
            env.phase = 'test'
        else:
            env.phase = 'train'
        """
        # ! not useful
        # ! if self.test_case is not None:
        # !    test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        # ! assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0

        self.desiredVelocity = [0.0, 0.0]
        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        # ! counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
        # !                  'val': 0, 'test': self.case_capacity['val']}


        counter_offset = {'train': 0, 'test': self.case_capacity['train'], 'val': self.case_capacity['train']+self.case_capacity['test']}
        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        np_random_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        np.random.seed(np_random_seed)
        # print("env np_random_seed: ", np_random_seed)
        # np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)

        # print("phase: ", phase, " test_case: ", test_case, " from_snapshot: ", from_snapshot, " use_orca: ", use_orca, " random_seed: ", np_random_seed)
        # print("phase: {}, test_case: {}, from_snapshot: {}, use_orca: {}, random_seed: {}".format(phase, test_case, from_snapshot, use_orca, np_random_seed))
        if from_snapshot is not None:
            self.from_snapshot = from_snapshot
        self.use_orca = use_orca
        if self.from_snapshot:
            self.generate_robot_humans_from_snapshot()
        else:
            self.generate_robot_humans(phase)


        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        # ! no use ! self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]
        self.case_counter[phase] += 1
        # get robot observation
        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        random_goal_shift_max_times = 2 # ! To be added to config
        nums_goal_shifts = np.random.randint(1, random_goal_shift_max_times+1, self.human_num)
        self.goal_shift_distance_thresholds = []
        for n_goal_shift, human in zip(nums_goal_shifts, self.humans):
            human_distance_to_goal = ((human.gx-human.px)**2+(human.gy-human.py)**2)**0.5
            self.goal_shift_distance_thresholds.append(np.sort(np.random.rand(n_goal_shift))*human_distance_to_goal)


        return ob


    # step function
    def step(self, action):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        # * in our setup, action is generated by orca or snapshot data. So we don't need action input from any policy here.

        """
        action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == 'unicycle':
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0]+action.v,-self.robot.v_pref,self.robot.v_pref)
            action=ActionRot(self.desiredVelocity[0], action.r)
        """
        if self.from_snapshot:
            # snapshot robot action
            px, py = self.snapshot_data['robot_planned_trajectory'][self.robot_snapshot_time_step_count]
            try:
                px_next, py_next = self.snapshot_data['robot_planned_trajectory'][self.robot_snapshot_time_step_count+1]
            except IndexError:
                # breakpoint()
                print(self.robot_snapshot_time_step_count)
            vx, vy = (px_next-px)/self.time_step, (py_next-py)/self.time_step
            action = ActionXY(vx, vy)
            self.robot_snapshot_time_step_count += 1
        else:
            if self.use_orca:
                # orca robot action
                ob = []
                for other_human in self.humans:
                    ob.append(other_human.get_observable_state())
                action = self.robot.act(ob) # * orca action
                # robot_action = self.robot.act(ob) # * orca action
                # action = robot_action # * orca action
            action = ActionXY(*action)

        human_actions = self.get_human_actions()

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)


        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step # max episode length=time_limit/time_step


        # compute the observation

        # todo WAFR: this need to be set depending on the phase. Only do it for calibration. Not for outside.
        # todo self.check_goal_shift_threshold()


        ob = self.generate_ob(reset=False)
        

        info={'info':episode_info}


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            self.check_goal_shift_threshold()
            # if self.global_time % 5 == 0:
            #     self.update_human_goals_randomly()
        
        # Update a specific human's goal once its reached its original goal
        """
        if self.end_goal_changing:
            for human in self.humans:
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)
        """

        if self.end_goal_changing:
            for human_index, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    # * multiple humans can change index at the same time
                    self.update_human_goal(human)
                    curr_newest_human_index = max(self.humans_index_current)
                    self.humans_index_current[human_index] = curr_newest_human_index+1
                    self.reset_human_goal_shift_distance_threshold(human_index)
                    

        return ob, reward, done, info

