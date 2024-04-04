#!/usr/bin/python3
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

def main():
    file = sys.argv[1]
    with open(file, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(columns=["mu", "sd", "collision", "trial", "collision_prob"])
    for d in data:

        mu = int(d["traj_mu"])
        sd = int(d["traj_sd"])
        name = f"{mu}-{sd}"
        if name in df.index:
            df.loc[name, "trial"] += 1
            df.loc[name, "collision"] += int(d["collision"])
        else:
            buf = pd.DataFrame(index=[name], data=[[d["traj_mu"], d["traj_sd"], int(d["collision"]), 1, 0.0]], columns=df.columns)
            df = pd.concat([df, buf])

    for i, row in df.iterrows():
        df.loc[i, "collision_prob"] = row.collision / row.trial


    fig, axes = plt.subplots()

    col = []
    for i in df.sort_values(by=["mu"]).mu.drop_duplicates():
        row = []
        for j in df.sort_values(by=["sd"]).sd.drop_duplicates():
            print(df[(df.mu==i)&(df.sd==j)])
            if len(df[(df.mu==i)&(df.sd==j)]) > 0:
                row.append(df[(df.mu==i)&(df.sd==j)].iloc[-1]["collision_prob"])
            else:
                row.append(0)
        col.append(row)

    mat = np.array(col)
    for (i, j), z in np.ndenumerate(mat):
        axes.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    print(mat)
    axes.matshow(mat)
    axes.set_yticklabels(df.sort_values(by=["sd"]).sd.drop_duplicates())
    axes.set_xticklabels(df.sort_values(by=["mu"]).mu.drop_duplicates())
    plt.show()


    total_trial = df["trial"].sum()
    total_collision = df["collision"].sum()
    total_collision_prob = df["collision"].sum() / df["trial"].sum()
    mu = df["mu"].sum() / len(df)
    sd = df["sd"].sum() / len(df)
    buf = pd.DataFrame(index=["general"], data=[[mu, sd, total_collision, total_trial, total_collision_prob]], columns=df.columns)
    df = pd.concat([df, buf])
    print(df)

if __name__ == "__main__":
    main()
