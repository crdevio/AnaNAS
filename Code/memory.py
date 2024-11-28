import numpy as np

MEM_SIZE = 1000 # taille de la simulation
class Memory:
    def __init__(self):
        self.theta = []
        self.states = []
        self.actions = []
        self.rewards = []
    def append(self,theta,state,state_b,action,reward):
        self.theta.append(theta)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    def get(self,i):
        self.theta[i],self.states[i],self.states_before[i],self.actions[i],self.rewards[i]

    def sample(self,N):
        idx = np.random.choice(len(self.actions) - 1, N)
        states = np.array(self.states)[idx]
        next_states = np.array(self.states)[idx+1]
        actions = np.array(self.actions)[idx]
        rewards = np.array(self.rewards)[idx]
        theta = np.array(self.theta)[idx]
        return states,next_states,actions,rewards,theta
