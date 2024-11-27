import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import Memory
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 1000


def state_reward(car):
    score = 1/(np.linalg.norm(np.array([car.x_position,car.y_position]) - np.array(car.goal))+0.05)
    if car.collision: 
        score = - 10000000
    return score

    
class DQN(nn.Module):
    def __init__(self, input_samples, output_features):
        super(DQN, self).__init__()
        self.memory = Memory()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Pooling after conv1
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Pooling after conv2
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * (input_samples // 4), 128)  # Adjust based on pooling output
        self.fc2 = nn.Linear(128, output_features)
        
    def forward(self, x):
            x = x.transpose(1, 2)
            
            # Convolutional layers + activation + pooling
            x = F.relu(self.conv1(x))  # Apply ReLU after conv1
            x = self.pool1(x)          # Apply max pooling after conv1
            
            x = F.relu(self.conv2(x))  # Apply ReLU after conv2
            x = self.pool2(x)          # Apply max pooling after conv2
            
            # Flatten the tensor before passing it to fully connected layers
            x = x.view(x.size(0), -1)  # Flatten
            
            # Fully connected layers + activation
            x = F.relu(self.fc1(x))    # Apply ReLU after fc1
            x = self.fc2(x)            # Output layer (no activation for regression or classification)
            return x
            #j = torch.argmax(x)
            #return torch.tensor([1 if i==j else 0 for i in range(x.shape[0])])
class EpsilonGreedy:
    def __init__(self,policy,epsilon):
        self.policy = policy
        self.eps = epsilon
    def __call__(self,state):
        (cone) = state
        if np.random.random() > self.eps:
            return self.policy(cone)
        else:
            return torch.tensor([[np.random.random() for i in range(4)] for i in range(state.shape[0])])


def decide(cone,speed,car):
    cone = torch.tensor(np.array(cone),dtype=torch.float32)
    cone = cone.view(1,cone.shape[0],cone.shape[1])
    rep = greedy(cone)
    rep = [x>0.5 for x in rep[0]]
    print("REWARD: ",state_reward(car))
    return rep # up down left right

def reward(voiture,pos_avant):
    d_before = np.linalg.norm(voiture.goal - pos_avant)
    d_after = np.linalg.norm(voiture.goal - np.array([voiture.x_position,voiture.y_position]))
    return - (d_after - d_before)

model = DQN(2011,4)
greedy = EpsilonGreedy(model,EPS_START)

