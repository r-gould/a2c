import numpy as np
import matplotlib.pyplot as plt

def state_action_dims(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return state_dim, action_dim

def plot_data(scores, actor_losses, critic_losses, name="Scores", window=10):

    smooth_scores = [np.mean(scores[i:i+window]) for i in range(len(scores))]
    
    plt.title(name)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.plot(scores, label="Raw")
    plt.plot(smooth_scores, label="Smoothed")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/{name}.png")
    plt.show()

    plt.title("Actor loss")
    plt.plot(actor_losses)
    plt.show()

    plt.title("Critic loss")
    plt.plot(critic_losses)
    plt.show()
