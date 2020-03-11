import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.input   = nn.Linear(8,   64)
        self.hidden1 = nn.Linear(64, 64)
        self.output  = nn.Linear(64, 4)

        self.relu = nn.ReLU()


    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)

        return x

