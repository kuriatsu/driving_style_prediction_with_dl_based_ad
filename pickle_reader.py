import pandas as pd
import math
import random
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def read_pickle_file(file):
    pickle_data = pd.read_pickle(file)
    return pickle_data

def pickup_data(data, mu, sd, sample_num):
    target_data = []
    for d in data:
        if d["traj_mu"] == mu and d["traj_sd"] == sd:
            target_data.append(d) 

    train_data = [] 
    for d in random.sample(target_data, sample_num):
        # for i, omega in enumerate(d["traj"][:, 4]):
        #     if i==0: continue
            ## get difference
        #     train_data.append(omega-d["traj"][i-1, 4])

        for i, index in enumerate(d["traj"][:, 5]):
            train_data.append(index)

    return train_data

def save_data(data):

    with open("prediction.pickle", "wb") as f:
        pickle.dump(data, f)

def show_data(filename):

    with open(filename, "rb") as f:
        data = pickle.load(f)

    result = pd.DataFrame(columns=["mu", "sd", "gt_mu", "gt_sd", "pred_mu", "pred_sd", "batch_num", "err_mu", "err_sd"])
    for row in data:
        mu = row[0]
        sd = row[1]
        mu_gt = 0.1 * (mu-50.0)/50.0 * (40.0 * math.pi / 180.0)
        sd_gt = 0.1 * sd/50 * (40 * math.pi / 180.0)
        mu_error = abs(row[2] - mu_gt)
        sd_error = abs(row[3] - sd_gt)

        buf = pd.DataFrame([[mu, sd, mu_gt, sd_gt, row[2], row[3], row[5], mu_error, sd_error]], columns=result.columns)
        result = pd.concat([result, buf], ignore_index=True)

    print(result)
    fig, ax = plt.subplots()
    sns.color_palette("tab10")
    sns.scatterplot(result, x="batch_num", y="err_mu", hue="gt_sd", style="gt_mu")
    plt.show()
    sns.lineplot(result, x="batch_num", y="err_sd", hue="gt_sd", style="gt_mu")
    plt.show()


# if __name__ == "__main__":
#     show_data(sys.argv[1])
