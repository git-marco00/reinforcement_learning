import torch
import torch.nn.functional as F

class Deterministic_Policy(torch.nn.Module):
    def __init__(self, n_states, n_actions, hidden=256):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_states, hidden)
        self.layer2 = torch.nn.Linear(hidden, hidden)
        self.layer3 = torch.nn.Linear(hidden, n_actions)
        self.n_actions = n_actions
        
        self.scale_factor = torch.tensor([1.5, 2], requires_grad=False)
        self.bias_factor = torch.tensor([-0.5, -1], requires_grad=False)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.sigmoid(self.layer3(x))
        
        # [main_engine, lateral_engine]
        #   [-0.5, 1]       [-1,1]
        return x[:, :self.n_actions] * self.scale_factor + self.bias_factor
    
class Compatible_Deterministic_Q(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.layer1(x)
