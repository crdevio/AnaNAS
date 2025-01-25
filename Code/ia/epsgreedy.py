import numpy as np
from ia.dqn import DQN
from ia.constants import *

class EpsilonGreedy:
    def __init__(self, policy:DQN, epsilon):
        self.policy = policy
        self.eps = epsilon

    def __call__(self, state):
        (cone, vitesse, goal) = state
        if np.random.random() > self.eps:
            return self.policy(cone.to(DEVICE), vitesse.to(DEVICE), goal.to(DEVICE))
        else:
            return torch.tensor([[np.random.random() for i in range(4)] for i in range(cone.shape[0])], device=DEVICE)