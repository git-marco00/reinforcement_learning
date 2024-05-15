import torch
import torch.nn.functional as F

class Actor(torch.nn.Module):
    def __init__(self, n_states, n_actions, hidden=64):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_states, hidden)
        self.layer2 = torch.nn.Linear(hidden, hidden)
        self.layer3 = torch.nn.Linear(hidden, n_actions)
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return self.softmax(x)

