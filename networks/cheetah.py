import torch
import torch.nn as nn

from .network import Network

class Cheetah(Network):

    def actor_network(self):

        return [
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 2*self.action_dim),
        ]

    def critic_network(self):

        return [
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ]