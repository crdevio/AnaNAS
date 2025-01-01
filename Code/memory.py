from collections import namedtuple, deque
import numpy as np
import torch

MEM_SIZE = 10000 # taille de la simulation

class Memory:
    def __init__(self):

        self.mem_index = 0
        self.size = 0

        #self.theta = deque([], maxlen = MEM_SIZE)
        #States contient cone, vitesse, goal sous forme de tuple. Il faut couper en trois pour optimiser le calcul
        #On verra plus tard mais peut-être que c'est encore mieux de faire des torch.tensor
        self.cone = torch.zeros((MEM_SIZE, 101, 51, 3), dtype=torch.float32)
        self.vitesse = torch.zeros((MEM_SIZE,), dtype=torch.float32)
        self.goal = torch.zeros((MEM_SIZE,2), dtype=torch.float32)
        self.actions = torch.zeros((MEM_SIZE,), dtype=torch.int32)
        self.rewards = torch.zeros((MEM_SIZE,), dtype=torch.float32)
        self.terminals = np.zeros((MEM_SIZE,), dtype=bool)

        '''
        self.states = deque([], maxlen = MEM_SIZE)
        self.actions = deque([], maxlen = MEM_SIZE)
        self.rewards = deque([], maxlen = MEM_SIZE)
        self.terminals = deque([], maxlen = MEM_SIZE)
        '''

    def append(self,cone,vitesse,goal,action,reward,terminal):
        #self.theta.append(theta)
        self.cone[self.mem_index] = torch.from_numpy(cone)
        self.vitesse[self.mem_index] = vitesse
        self.goal[self.mem_index] = torch.tensor([goal[0], goal[1]])
        self.actions[self.mem_index] = action
        self.rewards[self.mem_index] = reward
        self.terminals[self.mem_index] = terminal

        self.mem_index = (self.mem_index + 1) % MEM_SIZE
        self.size = min(self.size + 1, MEM_SIZE)

        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        '''

    def get(self,i):
        #self.theta[i],self.states[i],self.states_before[i],self.actions[i],self.rewards[i]
        return self.cone[i],self.vitesse[i],self.goal[i],self.actions[i],self.rewards[i]

    def sample(self,N):
        idx = np.random.choice(self.size, N, replace=False)

        '''
        cones = np.array([i[0] for i in self.states])[idx]
        speeds = np.array([i[1] for i in self.states])[idx]
        goals = np.array([i[2] for i in self.states])[idx]
        next_cones = np.array([i[0] for i in self.states])[idx+1]
        next_speeds = np.array([i[1] for i in self.states])[idx+1]
        next_goals = np.array([i[2] for i in self.states])[idx+1]
        actions = np.array(self.actions)[idx]
        rewards = np.array(self.rewards)[idx]
        terminals = np.array(self.terminals)[idx]
        '''
        
        #On doit retourner sous la forme cones, vitesse, goal, next_cones, next_vitesse, next_goal, actions, rewards, terminals
        return self.cone[idx], self.vitesse[idx], self.goal[idx], self.cone[(idx + 1) % MEM_SIZE], self.vitesse[(idx + 1) % MEM_SIZE], self.goal[(idx + 1) % MEM_SIZE], self.actions[idx], self.rewards[idx], self.terminals[idx]