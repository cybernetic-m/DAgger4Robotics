import torch
import torch.nn as nn

class SimplePolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.fc1 = nn.Linear(input_dim, 256)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, output_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            #nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(256, 128),
            #nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(128, output_dim),
            #nn.BatchNorm1d(512),
            #nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)