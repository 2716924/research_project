"""Neural network shared by every RCT arm (same topology for all variants)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinAtarQNetwork(nn.Module):
    """Small conv network suited to MinAtar's 10x10 boolean grids."""

    def __init__(self, in_channels, num_actions, num_layers=2, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        flat_features = 16 * 8 * 8

        layers = []
        current_dim = flat_features
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, num_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = self.hidden_layers(x)
        return self.output_layer(x)
