from collections import namedtuple, deque
import numpy as np
import torch

MEM_SIZE = 10000 # taille de la simulation

class Memory:
    def __init__(self):
        self.theta = deque([], maxlen = MEM_SIZE)
        self.states = deque([], maxlen = MEM_SIZE)
        self.actions = deque([], maxlen = MEM_SIZE)
        self.rewards = deque([], maxlen = MEM_SIZE)
        self.terminals = deque([], maxlen = MEM_SIZE)
    def append(self,theta,state,state_b,action,reward,terminal):
        self.theta.append(theta)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
    def get(self,i):
        self.theta[i],self.states[i],self.states_before[i],self.actions[i],self.rewards[i]

    def sample(self,N):
        idx = np.random.choice(len(self.actions) - 1, N)
        cones = np.array([i[0] for i in self.states])[idx]
        speeds = np.array([i[1] for i in self.states])[idx]
        goals = np.array([i[2] for i in self.states])[idx]
        next_cones = np.array([i[0] for i in self.states])[idx+1]
        next_speeds = np.array([i[1] for i in self.states])[idx+1]
        next_goals = np.array([i[2] for i in self.states])[idx+1]
        actions = np.array(self.actions)[idx]
        rewards = np.array(self.rewards)[idx]
        terminals = np.array(self.terminals)[idx]
        return torch.tensor(cones, dtype=torch.float32),torch.tensor(speeds, dtype=torch.float32),torch.tensor(goals, dtype=torch.float32),torch.tensor(next_cones, dtype=torch.float32),torch.tensor(next_speeds, dtype=torch.float32),torch.tensor(next_goals, dtype=torch.float32),torch.tensor(actions),torch.tensor(rewards,dtype=torch.float32),terminals
