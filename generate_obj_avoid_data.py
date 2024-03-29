#!/usr/bin/python3
# -*-coding:utf-8-*-

import math
import numpy as np
import matplotlib.pyplot as plt

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
        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 10.0 
        self.robot_radius = 0.2  # [m]
        self.goal_radius = 1.0

def calc_dynamic_window(x, config):
    ## dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed, -config.max_yawrate, config.max_yawrate]
    ## dynamic window from motion model
    Vd = [
        x[3] - config.max_accel * config.dt,
        x[3] + config.max_accel * config.dt,
        x[4] - config.max_dyawrate * config.dt,
        x[4] + config.max_dyawrate * config.dt
      ]
    dw = [
        max(Vs[0], Vd[0]),
        min(Vs[1], Vd[1]),
        max(Vs[2], Vd[2]),
        min(Vs[3], Vd[3])
      ]

    return dw

def motion(x, u, dt):
    x[2] += u[1] * dt # yaw
    x[0] += u[0] * math.cos(x[2]) * dt # x
    x[1] += u[0] * math.sin(x[2]) * dt # y
    x[3] = u[0] # speed
    x[4] = u[1] # omega

    return x

def calc_trajectory(xinit, v, y, config):
    x = np.array(xinit)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)

        traj = np.vstack((traj, x))
        time += config.dt

    return traj

def calc_to_goal_cost(traj, goal, config):
    ## calc goal cost: 2D norm
    goal_magnitude = math.sqrt(goal[0]**2 + goal[1]**2)
    traj_magnitude = math.sqrt(traj[-1, 0]**2 + traj[-1, 1]**2)
    dot_product = (goal[0]*traj[-1, 0]) + (goal[1]*traj[-1, 1])
    error = dot_product / (goal_magnitude*traj_magnitude)
    error_angle = math.acos(error)
    cost = config.to_goal_cost_gain * error_angle

    return cost

def calc_obstacle_cost(traj, ob, config):
    ## calc obstacle cost: collision = inf, free = 0
    skip_n = 2
    minr = float("inf")
    for j in range(0, len(traj[:, 1]), skip_n):
        for i in range(len(ob[:, 0])):
            ox = ob[i, 0]
            oy = ob[i, 1]
            dx = traj[j, 0] - ox
            dy = traj[j, 1] - oy

            r = math.sqrt(dx**2 + dy**2)
            if r <= config.robot_radius:
                print(f"collision r:{r}")
                return float("Inf")
            
            if minr >= r:
                minr = r

    ## collision free
    return 0.5 / minr 

def calc_final_input(x, u, dw, config, goal, ob):
    xinit = x[:]
    min_cost = 10000.0
    min_u = u
    min_u[0] = 0.0
    best_traj = np.array([x])

    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):
            traj = calc_trajectory(xinit, v, y, config)
            
            plt.plot(traj[:, 0], traj[:, 1], "-k", alpha=0.2)
            ## cost
            to_goal_cost = calc_to_goal_cost(traj, goal, config)
            speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3])
            ob_cost = calc_obstacle_cost(traj, ob, config)
            final_cost = to_goal_cost + speed_cost + ob_cost

            if min_cost >= final_cost:
                min_cost = final_cost
                min_u = [v, y]
                best_traj = traj

    return min_u, best_traj


def dwa_control(x, u, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    u, traj = calc_final_input(x, u, dw, config, goal, ob)
    return u, traj

def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), head_length = width, head_width = width)
    plt.plot(x, y)

    
def main():
    ## initial state [x, y, yaw, v, omega]
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([12, 0])
    u = np.array([0.0, 0.0])
    ob = np.matrix([
                    [6.0, 0.0],
                    ])
    traj = np.array(x)
    config = Config()
    show_anim = True

    for i in range(1000):
        u, ltraj = dwa_control(x, u, config, goal, ob)
        x = motion(x, u, config.dt)
        traj = np.vstack((traj, x))

        if show_anim:
            plt.plot(ltraj[:, 0], ltraj[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1] , "ok")
            plot_arrow(x[0], x[1], x[2])
            plt.xlim(0.0, 12.0)
            plt.ylim(-4.0, 4.0)
            plt.grid(True)
            plt.pause(0.0001)
            plt.cla()

        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.goal_radius:
            print("goal")
            break

    if show_anim:
        plt.plot(x[0], x[1], "xr")
        plt.plot(goal[0], goal[1], "xb")
        plt.plot(ob[:, 0], ob[:, 1] , "ok")
        plt.plot(traj[:, 0], traj[:, 1], "-r")
        plt.xlim(0.0, 12.0)
        plt.ylim(-4.0, 4.0)
        plt.show()

if __name__ == "__main__":
    main()
