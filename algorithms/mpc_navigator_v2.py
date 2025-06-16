#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Path Planner
# Author: Tianchen Ji, Zhe Huang
# Email: tj12@illinois.edu, zheh4@illinois.edu
# Create Date: 2019-11-26
# Modify Date: 2024-01-11
# ---------------------------------------------------------------------------

import json
import numpy as np
import matplotlib.pyplot as plt
from casadi import nlpsol, SX, exp, vertcat # *
from matplotlib import animation
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='test scene filename')
    parser.add_argument('-a', '--animate', action='store_true', help='animate the result')
    parser.add_argument('--fps', type=int, default=5, help='animation frame per second')
    args = parser.parse_args()
    return args

# def onClick(event):
#     global pause
#     pause ^= True

class FirstStateIndex:
    def __init__(self, n):
        self.px = 0
        self.py = self.px + n
        self.vx = self.py + n
        self.vy = self.vx + n - 1

class MPC:
    def __init__(self,
                 w_cte,
                 w_dv,
                 num_of_agent,
                 robot_r,
                 human_r,
                 max_v,
                 pred_horizon,
                 lookahead_step_num,
                 lookahead_step_timeinterval,
                 w_reg=0.5,
                 ):

        # The num of MPC actions, here include vx and vy
        NUM_OF_ACTS = 2

        # The num of MPC states, here include px and py
        NUM_OF_STATES = 2

        self.obs_avoidance_horizon = min(pred_horizon, lookahead_step_num)

        self.w_cte = w_cte
        self.w_dv = w_dv
        self.w_reg = w_reg
        self.num_of_agent = num_of_agent
        self.robot_r = robot_r
        self.human_r = human_r
        self.max_v = max_v
        self.lookahead_step_num = lookahead_step_num
        self.lookahead_step_timeinterval = lookahead_step_timeinterval
        self.first_state_index_ = FirstStateIndex(self.lookahead_step_num)
        self.num_of_x_ = NUM_OF_STATES * self.lookahead_step_num + NUM_OF_ACTS * (self.lookahead_step_num - 1)
        self.num_of_g_ = NUM_OF_STATES * self.lookahead_step_num + self.num_of_agent * self.obs_avoidance_horizon

        self.pred_horizon = pred_horizon
        self.robot_planned_traj = []
        # * for animation
        self.planned_px = []
        self.planned_py = []
        self.conformal_pred_r = []
        self.agent_pos_pred = []
        self.start_point = []
        self.end_point = []

    def Solve(self, start_pos, end_pos, human_pos_pred, conformal_r, x_init=None):

        # define optimization variables
        x = SX.sym('x', self.num_of_x_)

        # define cost functions
        cost = 0.0

        if x_init is None:
            # initial variables
            x_init = [0] * self.num_of_x_
            x_init[self.first_state_index_.px:self.first_state_index_.py] = [start_pos[0]] * self.lookahead_step_num
            x_init[self.first_state_index_.py:self.first_state_index_.vx] = [start_pos[1]] * self.lookahead_step_num
        else:
            for i in range(self.first_state_index_.vx):
                diff_p = x[i] - x_init[i]
                cost += self.w_reg*(diff_p**2)

        # penalty on states
        for i in range(self.lookahead_step_num):
            cte = (x[self.first_state_index_.px + i] - end_pos[0])**2 + (x[self.first_state_index_.py + i] - end_pos[1])**2
            cost += self.w_cte * cte
        # penalty on inputs
        for i in range(self.lookahead_step_num - 2):
            dvx = x[self.first_state_index_.vx + i + 1] - x[self.first_state_index_.vx + i]
            dvy = x[self.first_state_index_.vy + i + 1] - x[self.first_state_index_.vy + i]
            cost += self.w_dv*(dvx**2) + self.w_dv*(dvy**2)

        # define lowerbound and upperbound of x
        x_lowerbound_ = [-exp(10)] * self.num_of_x_
        x_upperbound_ = [exp(10)] * self.num_of_x_
        for i in range(self.first_state_index_.vx, self.num_of_x_):
            x_lowerbound_[i] = -self.max_v
            x_upperbound_[i] = self.max_v

        # define lowerbound and upperbound of g constraints
        g_lowerbound_ = [0] * self.num_of_g_
        g_upperbound_ = [0] * self.num_of_g_

        g_lowerbound_[self.first_state_index_.px] = start_pos[0]
        g_lowerbound_[self.first_state_index_.py] = start_pos[1]

        g_upperbound_[self.first_state_index_.px] = start_pos[0]
        g_upperbound_[self.first_state_index_.py] = start_pos[1]

        for i in range(self.obs_avoidance_horizon - 1):
            safety_r = self.robot_r + self.human_r + conformal_r[1 + i]
            for j in range(self.num_of_agent):
                g_lowerbound_[(1 + self.first_state_index_.py + self.lookahead_step_num 
                               + i + j * self.obs_avoidance_horizon)] = safety_r**2
                g_upperbound_[(1 + self.first_state_index_.py + self.lookahead_step_num 
                               + i + j * self.obs_avoidance_horizon)] = exp(10)

        # define g constraints
        g = [None] * self.num_of_g_
        g[self.first_state_index_.px] = x[self.first_state_index_.px]
        g[self.first_state_index_.py] = x[self.first_state_index_.py]
        for i in range(self.num_of_agent):
            g[self.first_state_index_.py + self.lookahead_step_num + i * self.obs_avoidance_horizon] = 0

        for i in range(self.lookahead_step_num - 1):
            curr_px_index = i + self.first_state_index_.px
            curr_py_index = i + self.first_state_index_.py
            curr_vx_index = i + self.first_state_index_.vx
            curr_vy_index = i + self.first_state_index_.vy

            curr_px = x[curr_px_index]
            curr_py = x[curr_py_index]
            curr_vx = x[curr_vx_index]
            curr_vy = x[curr_vy_index]

            next_px = x[1 + curr_px_index]
            next_py = x[1 + curr_py_index]

            next_m_px = curr_px + curr_vx * self.lookahead_step_timeinterval
            next_m_py = curr_py + curr_vy * self.lookahead_step_timeinterval

            # equality constraints
            g[1 + curr_px_index] = next_px - next_m_px
            g[1 + curr_py_index] = next_py - next_m_py

            # inequality constraints
            if i < self.obs_avoidance_horizon - 1:
                for j in range(self.num_of_agent):
                    g[1 + curr_py_index + self.lookahead_step_num + j * self.obs_avoidance_horizon] = (
                            next_px - human_pos_pred[1 + i][j][0])**2 + (next_py - human_pos_pred[1 + i][j][1])**2

        # create the NLP
        nlp = {'x':x, 'f':cost, 'g':vertcat(*g)}

        # solver options
        opts = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"] = 0
        opts["ipopt.tol"] = 0.01
        opts["ipopt.compl_inf_tol"] = 0.001
        opts["ipopt.constr_viol_tol"] = 0.01

        solver = nlpsol('solver', 'ipopt', nlp, opts)

        # solve the NLP
        res = solver(x0=x_init, lbx=x_lowerbound_, ubx=x_upperbound_, lbg=g_lowerbound_, ubg=g_upperbound_)
        
        self.planned_px = res['x'][0:1 * self.lookahead_step_num]
        self.planned_py = res['x'][1 * self.lookahead_step_num:2 * self.lookahead_step_num]
        self.planned_px = np.array(self.planned_px).squeeze()
        self.planned_py = np.array(self.planned_py).squeeze()
        self.robot_planned_traj = np.array([self.planned_px, self.planned_py]).T.tolist()
        # * for animation
        self.conformal_pred_r = conformal_r
        self.agent_pos_pred = human_pos_pred
        self.start_point = start_pos
        self.end_point = end_pos

        return res, solver.stats()['return_status']

    def get_robot_planned_traj(self):
        return self.robot_planned_traj
    
    def generate_animation(self, filename, fps):
         # create animation
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.plot(self.planned_px, self.planned_py, 'o-', color='g', label='planned trajectory')
        self.ax.plot(self.start_point[0], self.start_point[1], 'o', color='b', label='start point')
        self.ax.plot(self.end_point[0], self.end_point[1], 'o', color='orange', label='target point')
        # for i in range(num_of_agent):
        #     plt.plot(agent_goal[i][0], agent_goal[i][1], 'o', color='m')
        self.current_pos, = self.ax.plot([],[], ls='None', color='k', marker='o', label='current position')
        self.current_zone, = self.ax.plot([],[], 'k--')
        self.agent_pos_list = []
        self.agent_zone_list = []
        self.agent_cs_list = []
        for i in range(self.num_of_agent):
            agent_pos, = self.ax.plot([],[], ls='None', color='r', marker='o')
            agent_zone, = self.ax.plot([],[], 'r--')
            agent_cs, = self.ax.plot([],[], 'b--')
            self.agent_pos_list.append(agent_pos)
            self.agent_zone_list.append(agent_zone)
            self.agent_cs_list.append(agent_cs)
        # plt.legend(loc='upper left')
        # self.ax.axis('equal')
        self.ax.axis([-6, 6, -6, 6])
        # self.ax.grid()
        ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.planned_px), interval=1)
        output_filename = filename+'.gif'
        ani.save(output_filename, writer='pillow', fps=fps)
        print("animation saved to", output_filename)



    def animate(self, time_stamp):
        """
        This is only to visualize how the robot will move along the planned trajectory.
        No receding horizon control is applied here.
        """
        theta = np.arange(0, 2*np.pi, 0.01)
        self.current_pos.set_data([self.planned_px[time_stamp]], [self.planned_py[time_stamp]])
        current_zone_x = self.planned_px[time_stamp] + self.robot_r * np.cos(theta)
        current_zone_y = self.planned_py[time_stamp] + self.robot_r * np.sin(theta)
        self.current_zone.set_data(current_zone_x, current_zone_y)
        # plot agent positions
        if time_stamp < self.pred_horizon:
            for i in range(self.num_of_agent):
                # breakpoint()
                self.agent_pos_list[i].set_data([self.agent_pos_pred[time_stamp][i][0]], [self.agent_pos_pred[time_stamp][i][1]])
                agent_zone_x = self.agent_pos_pred[time_stamp][i][0] + self.human_r * np.cos(theta)
                agent_zone_y = self.agent_pos_pred[time_stamp][i][1] + self.human_r * np.sin(theta)
                self.agent_zone_list[i].set_data(agent_zone_x, agent_zone_y)
                human_safety_r = self.human_r + self.conformal_pred_r[time_stamp]
                agent_cs_x = self.agent_pos_pred[time_stamp][i][0] + human_safety_r * np.cos(theta)
                agent_cs_y = self.agent_pos_pred[time_stamp][i][1] + human_safety_r * np.sin(theta)
                self.agent_cs_list[i].set_data(agent_cs_x, agent_cs_y)
        else:
            for i in range(self.num_of_agent):
                self.agent_pos_list[i].set_data([], [])
                self.agent_zone_list[i].set_data([], [])
                self.agent_cs_list[i].set_data([], [])


if __name__ == '__main__':
    args = arg_parse()

    # MPC parameters
    w_cte = 1.0
    w_dv = 5.0
    lookahead_step_num = 50
    lookahead_step_timeinterval = 0.25

    # read data
    sample_path = args.filename+'.json'
    with open(sample_path) as f:
        sample_data = json.load(f)
        print(sample_path+" loaded.")
    
    # agent trajectory prediction parameters
    pred_horizon = sample_data["human_prediction_steps"] + 1

    # start point and end point of ego robot
    start_point = sample_data["robot_position_current"]
    end_point = sample_data["robot_goal"]

    # threshold of safety
    robot_r = sample_data["robot_radius"]
    human_r = sample_data["human_radius"]
    conformal_pred_r = sample_data["conformal_prediction_radius"]
    conformal_pred_r = [0.0] + conformal_pred_r
    print("conformal prediction radius:", conformal_pred_r)

    # max vx, vy
    max_v = sample_data["robot_vel_max_magnitude"]

    # current and predicted human positions
    agent_pos_pred = sample_data["humans_position_future"]
    agent_pos_pred = [sample_data["humans_position_current"]] + agent_pos_pred
    num_of_agent = len(agent_pos_pred[1])
    agent_goal = sample_data["humans_goal"]

    mpc_ = MPC(w_cte=w_cte,
               w_dv=w_dv,
               num_of_agent=num_of_agent,
               robot_r=robot_r,
               human_r=human_r,
               max_v=max_v,
               pred_horizon=pred_horizon,
               lookahead_step_num=lookahead_step_num,
               lookahead_step_timeinterval=lookahead_step_timeinterval)

    # solve for optimal control actions
    sol, sol_status = mpc_.Solve(start_point, end_point, agent_pos_pred, conformal_pred_r)
    print("solver status:", sol_status)
    sample_data["robot_planned_trajectory"] = mpc_.get_robot_planned_traj()
    output_sample_path = 'mpc_'+args.filename+'.json'
    with open(output_sample_path, 'w') as outfile:
        json.dump(sample_data, outfile)
        print("planned trajectory saved to", output_sample_path)
    if args.animate:
        mpc_.generate_animation(args.filename, args.fps)