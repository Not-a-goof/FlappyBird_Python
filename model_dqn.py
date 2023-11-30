import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define network layers
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, output_dim)  # Directly outputting the Q-values for each action

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # No activation function here as we need raw scores for Q-values
        return x
