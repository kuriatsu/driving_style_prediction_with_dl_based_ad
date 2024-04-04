#!/usr/bin/python3
# -*-cofing:utf-8-*-

import torch
import torch.nn as nn
import pickle
import sys


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
            )

    def forward(self, x):
        return self.layers(x)

def main():
    with open(sys.argv[1], "rb") as f:
        datas = pickle.load(f)

    train_data = torch.utils.data.DataLoader(datas, batch_size=10, shuffle=True, num_workers=1)
    test_data = torch.utils.data.DataLoader(datas, batch_size=10, shuffle=True, num_workers=1)

    mlp = MLP()
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adagrad(mlp.parameters(), lr=1e-4)
    device = torch.device("cuda")
    epoch = 0

    while epoch < 5:
        for i, data in enumerate(datas):
            input_tensor = torch.as_tensor(train_data[:][0:1], dtype=torch.float32, device=device)
            print([[train_data["trajectory_mu"][j], train_data["trajectory_sd"][j]] for j in range(len(data))])
            output_tensor = torch.as_tensor([[train_data["trajectory_mu"][j], train_data["trajectory_sd"][j]] for j in range(len(data))], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = mlp(input_tensor)
            loss = loss_function(pred, output_tensor)
            loss.backward()
            optimier.step()
            epoch += 1
    

    input_tensor = torch.as_tensor(test_data[:][0:1], dtype=torch.float32, device=device)
    output_tensor = torch.as_tensor([[test_data["trajectory_mu"][j], test_data["trajectory_sd"][j]] for j in range(len(data))], dtype=torch.float32, device=device)
    mlp.eval()

    with torch.no_grad():
        outputs = mlp(input_tensor)
        predicted_labels = outputs.squeeze().tolist()

    predicted_labels = np.array(predicted_labels)
    test_targets = np.array(output_tensor)
    mse = mean_squared_error(test_targets, predicted_labels)
    r2 = r2_score(test_targets, predicted_labels)
    print(f"MSE: {mse}, R2: {r2}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
