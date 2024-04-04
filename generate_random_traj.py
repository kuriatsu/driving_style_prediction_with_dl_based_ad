#!/usr/bin/python3
# -*-coding:utf-8-*-

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import pickle


class Config():
    # simulation parameters

    def __init__(self):
        # robot parameter
        self.max_speed = 3.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.5  # [m/ss]
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_reso = 0.005  # [m/s]
        self.yawrate_reso = 0.5 * math.pi / 180.0  # [rad/s]
        self.v_num = 5 
        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]

        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 10.0 
        self.obstacle_cost_gain = 0.5

        self.robot_radius = 0.5  # [m]
        self.obs_radius = 1.0  # [m]
        self.goal_radius = 1.0

        self.yawrate_num = 100 # rollout candidate number
        self.trajectory_mu = 50.0 # mean when randomly select trajectory 
        self.trajectory_sd = 2.0 # deviation when randomly select trajectory
        self.trajectory_index = 50 # selected trajectory (used when fix_trajectory=True)

def calc_dynamic_window(x, config):
    """ calc dynamic window (boundary of rollout)
    """
    ## dynamic window from robot specification (capability)
    Vs = [config.min_speed, config.max_speed, -config.max_yawrate, config.max_yawrate]

    ## dynamic window from motion model (consider current state)
    Vd = [
        x[3] - config.max_accel * config.dt, # min_speed from motion model
        x[3] + config.max_accel * config.dt, # max_speed from motion model
        x[4] - config.max_dyawrate * config.dt, # min_yaw from motion model
        x[4] + config.max_dyawrate * config.dt # max_yaw from motion model
      ]
    ## get possible control window
    dw = [
        max(Vs[0], Vd[0]),
        min(Vs[1], Vd[1]),
        max(Vs[2], Vd[2]),
        min(Vs[3], Vd[3])
      ]

    return dw

def motion(x, u, dt):
    """ calculate vehicle next step
    """
    x[2] += u[1] * dt # yaw
    x[0] += u[0] * math.cos(x[2]) * dt # x
    x[1] += u[0] * math.sin(x[2]) * dt # y
    x[3] = u[0] # speed
    x[4] = u[1] # omega

    return x

def calc_trajectory(xinit, v, y, config):
    """calculate trajectory
    ~args~
    xinit: initial ego position
    v: input speed
    y: input yaw
    config: config
    ~return~
    trajectory
    """
    x = np.array(xinit)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)

        traj = np.vstack((traj, x))
        time += config.dt

    return traj


def calc_final_input(x, u, dw, config, goal, ob):
    """select trajectory from candidate generated by dynamic window
    ~args~
    x: ego position
    u: speed and yaw
    dw: candidate trajectory 
    config: config
    goal: goal position
    ob: obstacle position
    ~return~
    u: selected speed and yaw
    best_traj: selected trajectoy
    """
    xinit = x[:]
    min_cost = 10000.0
    min_u = u
    min_u[0] = 0.0
    best_traj = np.array([x])

    v = dw[1] 
    trajs = []
    us = []
    for y in np.linspace(dw[2], dw[3], config.yawrate_num):
        traj = calc_trajectory(xinit, v, y, config)
        # plt.plot(traj[:, 0], traj[:, 1], "-k", alpha=0.2)
        trajs.append(traj)
        us.append([v, y])

    # mu = config.trajectory_mu
    # sd = config.trajectory_sd
    # index = int(truncnorm.rvs((0 - mu)/sd, (config.yawrate_num-mu)/sd, loc=mu, scale=sd))
    index = config.trajectory_index
    min_u = us[index]
    best_traj = trajs[index]

    return min_u, best_traj


def dwa_control(x, u, config, goal, ob):
    """output dynamic window
    """
    dw = calc_dynamic_window(x, config)
    u, traj = calc_final_input(x, u, dw, config, goal, ob)
    return u, traj

def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    """draw evo vehicle heading pose
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), head_length = width, head_width = width)
    plt.plot(x, y)

def check_collision(traj, ob, config):
    """check collision
    ~args~
    traj: trajectory history
    ob: obstacle position [[x, y],,, ]
    config: config
    ~return~
    collision=True
    """
    ## calc obstacle cost: collision = inf, free = 0
    skip_n = 2
    minr = float("inf")
    
    for j in range(0, len(traj[:, 1]), skip_n):
        for i in range(len(ob[:, 0])):
            ox = ob[i, 0]
            oy = ob[i, 1]
            dx = traj[j, 0] - ox
            dy = traj[j, 1] - oy

            ## distance to obstacle
            r = math.sqrt(dx**2 + dy**2)

            if r <= config.obs_radius + config.robot_radius:
                print("collide")
                return True
            
    ## collision free
    return False 
    
def run(config, goal, ob, fix_trajectory=True, show_anim=True):
    """run each iteration
    ~args~
    data: config, result saving
    goal: goal position
    ob: obstacle position
    fix_trajectory: fix trajectory selection at every time step
    show_anim: show result with matplotlib 
    ~return~

    """
    ## initial state [x, y, yaw, v, omega]
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u = np.array([0.0, 0.0])
    traj = np.array(x)

    ## set trajectory index before start
    if fix_trajectory:
        mu = config.trajectory_mu
        sd = config.trajectory_sd
        ## truncated norm
        config.trajectory_index = int(truncnorm.rvs((0 - mu)/sd, (config.yawrate_num-mu)/sd, loc=mu, scale=sd))

    for i in range(80):
        u, ltraj = dwa_control(x, u, config, goal, ob)
        x = motion(x, u, config.dt)
        traj = np.vstack((traj, x))

        ## randomly selelct trajectory at each time step 
        if not fix_trajectory:
            mu = config.trajectory_mu
            sd = config.trajectory_sd
            ## truncated norm
            config.trajectory_index = int(truncnorm.rvs((0 - mu)/sd, (config.yawrate_num-mu)/sd, loc=mu, scale=sd))

        if show_anim:
            ## trajectory plot
            plt.plot(ltraj[:, 0], ltraj[:, 1], "-g")
            ## vehicle plot
            plt.plot(x[0], x[1], "or", ms=config.robot_radius*60)
            plot_arrow(x[0], x[1], x[2])
            ## goal plot
            plt.plot(goal[0], goal[1], "xb")
            ## obstacle plot
            plt.plot(ob[:, 0], ob[:, 1] , "ok", ms=config.obs_radius*60)

            plt.xlim(0.0, 14.0)
            plt.ylim(-4.0, 4.0)
            plt.grid(True)
            plt.pause(0.0001)
            plt.cla()

        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.goal_radius:
            print("goal")
            break

    return traj

def main():
    datas = []
    iteration = 100 # data collection in each parameter 
    goal = np.array([14, 0])
    ob = np.matrix([[8.0, 0.0]])
    fix_trajectory = False # True: select same id of trajectory in each time step, False: sample id from distribution in each time step  
    show_anim = False

    ## parameter of truncated gaussian distribution
    mu_list = [50, 45, 55, 40, 60]
    sd_list = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]

    ## data collection with each parameters
    for mu in mu_list:
        for sd in sd_list:
            config = Config()
            config.trajectory_mu = mu
            config.trajectory_sd = sd 
            for _ in range(iteration):
                traj = run(config, goal, ob, fix_trajectory, show_anim)
                is_collision = check_collision(traj, ob, config)

                data = {"traj_mu": mu,
                        "traj_sd": sd,
                        "traj": traj,
                        "collision": is_collision
                        }
                datas.append(data)

    ## save data
    with open("data.pickle", "wb") as f:
        pickle.dump(datas, f)
        print("saved")

    ## show result
    plt.cla()
    plt.plot(goal[0], goal[1], "xb")
    plt.plot(ob[:, 0], ob[:, 1] , "ok")
    for i, data in enumerate(datas):
        traj_arr = np.array(data["traj"])
        plt.xlim(0.0, 12.0)
        plt.ylim(-4.0, 4.0)
        plt.plot(traj_arr[:, 0], traj_arr[:, 1], 
                 label=f"u:{data['traj_mu']} sd:{data['traj_sd']}", 
                     color=plt.cm.RdYlBu(mu_list.index(data["traj_mu"])/(2*len(mu_list))+sd_list.index(data["traj_sd"])/(2*len(sd_list))))
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
