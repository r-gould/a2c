import torch
import torch.nn as nn

class Actor(nn.Module):

    def __init__(self, layers, low, high):
        
        super().__init__()

        self.network = nn.Sequential(*layers)
        self.low = low
        self.high = high
        self.scale = max(abs(low), abs(high))

    def mean_logstd(self, state):

        mean_logstd = self.network(state)
        mean, logstd = torch.chunk(mean_logstd, chunks=2, dim=-1)
        mean = self.scale * torch.tanh(mean)
        return mean, logstd

    def forward(self, state):

        state = torch.Tensor(state)
        mean, logstd = self.mean_logstd(state)
        std = torch.exp(logstd)
        
        dist = torch.distributions.normal.Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, self.low, self.high)
        
        return action.detach().numpy(), log_prob