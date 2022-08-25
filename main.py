import gym
import torch

import networks

from a2c import A2C
from trainer import Trainer
from utils import state_action_dims, plot_data

NETWORKS = {
    "HalfCheetah-v4" : networks.Cheetah,
    "InvertedPendulum-v4" : networks.Pendulum,
}

def main(env_str, actor_lr, critic_lr, episodes, horizon, gamma):

    assert env_str in NETWORKS.keys()

    env = gym.make(env_str)
    network_cls = NETWORKS.get(env_str)
    state_dim, action_dim = state_action_dims(env)

    network = network_cls(state_dim, action_dim)
    agent = A2C(env, network, gamma)

    actor_optim = torch.optim.Adam(agent.actor.parameters(), actor_lr)
    critic_optim = torch.optim.Adam(agent.critic.parameters(), critic_lr)
    agent.compile(actor_optim, critic_optim)

    scores, actor_losses, critic_losses = Trainer.train(agent, episodes, horizon, display_rate=50, render_rate=250)
    plot_data(scores, actor_losses, critic_losses, name=env_str)


if __name__ == "__main__":

    main(env_str="InvertedPendulum-v4", 
         actor_lr=1e-4, critic_lr=1e-3,
         episodes=750, horizon=99999,
         gamma=0.9)