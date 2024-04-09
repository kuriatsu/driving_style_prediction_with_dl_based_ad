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

    result = pd.DataFrame(columns=["mu", "sd", "mu_gt", "sd_mu", "mu_pred", "sd_pred", "batch_num", "mu_err", "sd_err"])
    for row in data:
        mu = row[2]
        sd = row[3]
        mu_gt = 0.1 * (mu-50.0)/50.0 * (40.0 * math.pi / 180.0)
        sd_gt = 0.1 * sd/50 * (40 * math.pi / 180.0)
        mu_error = abs(row[0] - mu)
        sd_error = abs(row[1] - sd)

        buf = pd.DataFrame([[int(row[0]), int(row[1]), mu_gt, sd_gt, mu, sd, row[5], mu_error, sd_error]], columns=result.columns)
        result = pd.concat([result, buf], ignore_index=True)

    print(result)
    with open("prediction_summary.pickle", "wb") as f:
        pickle.dump(result, f)

    fig, ax = plt.subplots()
    sns.color_palette("tab10")
    sns.lineplot(result, x="batch_num", y="mu_err", hue="sd", style="mu")
    plt.show()
    sns.lineplot(result, x="batch_num", y="sd_err", hue="sd", style="mu")
    plt.show()


# if __name__ == "__main__":
#     show_data(sys.argv[1])
