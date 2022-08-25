import torch
import torch.nn as nn

class Critic(nn.Module):

    def __init__(self, layers):
        
        super().__init__()
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):

        return self.network(state)