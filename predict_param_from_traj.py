#!/usr/bin/python3
# -*- cofing:utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import math
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import random

def main():

    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f)

    iteration = 10 
    data_per_param = 100
    result = pd.DataFrame(columns=["gt_mu", "gt_sd", "batch_num", "err_mu", "err_sd"])

    param_mu = []
    param_sd = []
    for d in data:
        param_mu.append(d["traj_mu"])
        param_sd.append(d["traj_sd"])

    param_mu = set(param_mu)
    param_sd = set(param_sd)

    for mu in param_mu:
        for sd in param_sd:
            target_data = []
            for d in data:
                if d["traj_mu"] == mu and d["traj_sd"] == sd:
                    target_data.append(d) 

            mu_gt = 0.1 * (mu-50.0)/50.0 * (40.0 * math.pi / 180.0)
            sd_gt = 0.1 * sd/50 * (40 * math.pi / 180.0)
            for batch_num in range(1, data_per_param+1):
                error_mu = []
                error_sd = []
                for _ in range(iteration):
                    train_data = [] 
                    for d in random.sample(target_data, batch_num):
                        for i, omega in enumerate(d["traj"][:, 4]):
                            if i==0: continue
                            train_data.append(omega-d["traj"][i-1, 4])

                    train_data = np.array(train_data)
                    mu_est = np.mean(train_data) 
                    sd_est = np.std(train_data)
                    error_mu = abs(mu_est-mu_gt) 
                    error_sd = abs(sd_est-sd_gt) 
                    # error_mu = mu_est 
                    # error_sd = sd_est 
                    print(f"sd_gt: {sd_gt}, sd: {sd}, mu_est: {mu_est}, sd_est: {sd_est}")


                    buf = pd.DataFrame([[mu, sd, batch_num, error_mu, error_sd]], columns=result.columns)
                    result = pd.concat([result, buf], ignore_index=True)

    print(result)
    sns.color_palette("tab10")
    sns.lineplot(result, x="batch_num", y="err_mu", hue="gt_sd", style="gt_mu")
    plt.show()
    sns.lineplot(result, x="batch_num", y="err_sd", hue="gt_sd", style="gt_mu")
    plt.show()




if __name__ == "__main__":
    main()

