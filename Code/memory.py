MEM_SIZE = 1000 # taille de la simulation
class Memory:
    def __init__(self):
        self.theta = []
        self.states = []
        self.actions = []
        self.rewards = []
    def append(self,theta,state,action,reward):
        self.theta.append(theta)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    def get(self,i):
        self.theta[i],self.states[i],self.actions[i],self.rewards[i]