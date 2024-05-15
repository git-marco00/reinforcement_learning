import torch
import torch.nn.functional as F

class Critic(torch.nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=64):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_states, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

