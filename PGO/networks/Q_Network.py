import torch

class Q_network(torch.nn.Module):
    def __init__(self, n_states, n_actions, n_hidden = 16):
        super().__init__()

        self.layer1 = torch.nn.Linear(n_states, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x_l1 = self.layer1(x)
        x_l2 = self.layer2(x_l1)
        return x_l2

