import torch
import torch.nn as nn
import numpy as np

from .actor import Actor
from .critic import Critic

class A2C(nn.Module):

    def __init__(self, env, network, gamma, entropy=1e-3):
        
        super().__init__()
        
        self.env = env
        self.low = self.env.action_space.low.min()
        self.high = self.env.action_space.high.max()
    
        self.actor = Actor(network.actor_network(), self.low, self.high)
        self.critic = Critic(network.critic_network())
        self.gamma = gamma
        self.entropy = entropy
        self.compiled = False

    def compile(self, actor_optim, critic_optim):
        
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.compiled = True

    def episode_update(self, horizon, render):

        assert self.compiled

        score = 0
        state = self.env.reset()
        actor_losses, critic_losses = [], []

        for _ in range(horizon):

            if render:
                self.env.render()

            action, log_prob = self.actor(state)

            next_state, reward, done, _ = self.env.step(action)

            state, next_state = torch.from_numpy(state).float(), torch.from_numpy(next_state).float()
            value, next_value = self.critic(state), self.critic(next_state)
            target = reward + (1 - done) * self.gamma * next_value
            
            critic_loss = (target.detach() - value)**2

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            advantage = (target - value).detach()

            actor_loss = -log_prob * advantage
            actor_loss += -log_prob * self.entropy

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            score += reward
            state = next_state.numpy()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            if done:
                break

        return score, np.mean(actor_losses), np.mean(critic_losses)