import torch
import torch.nn as nn

class Q_network(torch.nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=128):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_states, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_actions)


    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

