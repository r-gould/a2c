import torch

class Trainer:

    @staticmethod
    def train(agent, episodes, horizon, display_rate=50, render_rate=100,
              save_model=True):
        
        assert agent.compiled

        scores = []
        actor_losses, critic_losses = [], []

        for ep in range(1, episodes+1):

            render = (render_rate is not None 
                     and ep % render_rate == 0)

            score, actor_loss, critic_loss = agent.episode_update(horizon, render)

            display = (display_rate is not None 
                      and ep % display_rate == 0)
            
            if display:
                print("Episode:", ep)
                print("Score:", score)
                print("Actor loss:", actor_loss)
                print("Critic loss:", critic_loss)
                print()
                
            scores.append(score)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if save_model:
            torch.save(agent.state_dict(), "a2c/saved/agent.pt")
            
        return scores, actor_losses, critic_losses