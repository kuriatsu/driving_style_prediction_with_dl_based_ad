#!/usr/bin/python3
# -*-coding:utf-8-*-

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import sys


class MLP(nn.Module):
    def __init__(self):
        self.layers == nn.Sequential(
                nn.Linear(20, 16),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 2)
                )
        def forward(self, x):
            return self.layers(x)


def main():
    with open(sys.argv[1], "rb") as f:
        datas = pickle.load(f)

    four_length_data = []
    for i in range(4, len(datas)):
        print([datas[j]["traj"] for j in range(i-3, i)])
        four_length_data.append([datas[j]["traj"] for j in range(i-3, i)])
        print(i)


    train_data = torch.utils.data.DataLoader(four_length_data, batch_size=10, shuffle=True, num_workers=1)
    test_data = torch.utils.data.DataLoader(four_length_data, batch_size=10, shuffle=True, num_workers=1)
    print(train_data)


if __name__ == "__main__":
    main()
    
