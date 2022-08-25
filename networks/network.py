class Network:
    
    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

    def actor_network(self):

        raise NotImplementedError()

    def critic_network(self):
        
        raise NotImplementedError()