import torch
import torch.nn.functional as F

class Actor(torch.nn.Module):
    def __init__(self, n_states, n_actions, hidden=256):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_states, hidden)
        self.layer2 = torch.nn.Linear(hidden, hidden)
        self.layer3 = torch.nn.Linear(hidden, n_actions*2)  # 4 in total: main_engine_mean, lateral_engine_mean, main_engine_std, lateral_engines_std
        self.n_actions = n_actions

        self.scale_factor = torch.tensor([1.5, 2], requires_grad=False)
        self.bias_factor = torch.tensor([-0.5, -1], requires_grad=False)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))      # => [0,x]
        x = F.leaky_relu(self.layer2(x))      # => [0,x]
        x = F.sigmoid(self.layer3(x))   # => [0,1]
        
        # [main_engine_mean, lateral_engine_mean, main_engine_std, lateral_engine_std]
        #   [-0.5, 1]           [-1, 1]

        return x[:, :self.n_actions] * self.scale_factor + self.bias_factor, x[:, self.n_actions:] + 0.01

class Critic(torch.nn.Module):
    def __init__(self, input_dim, n_hidden=256):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, 1)


    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        return x
    