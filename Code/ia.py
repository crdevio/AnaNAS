import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import Memory
EPS_START = 1.
EPS_DECAY = 0.95
EPS_MIN = 0.1


def state_reward(car):
    # ATTENTION y  a peut-être une couille entre l'appel à state_reward et la valeur de collision.
    score = 1000/(np.linalg.norm(np.array([car.x_position,car.y_position]) - np.array(car.goal))+0.05)
    if car.collision: 
        score = - 100
    return score


class DQN(nn.Module):
    def __init__(self, input_samples, output_features):
        super(DQN, self).__init__()
        self.memory = Memory()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)  # Output: (4, 64, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(1 * (input_samples // 1), 16)
        self.fc2 = nn.Linear(16 + 3, output_features)

    def forward(self, x, vitesse, goal):
        # Passage des données convolutives
        x = x.transpose(1, 3)
        x = self.conv1(x)
        x = self.pool(x)
        #x = F.relu(self.conv2(x))
        #x = self.pool2(x)

        # Mise en forme pour les couches linéaires
        x = x.view(x.size(0), -1)
        x = F.relu(x)
        x = F.relu(self.fc1(x))

        # Ajout de la vitesse comme seconde entrée
        vitesse = vitesse.view(-1, 1)  # S'assurer que vitesse a la bonne forme
        goal = goal.view(-1,2)
        x = torch.cat((x, vitesse, goal), dim=1).to(torch.float32)

        # Dernière couche linéaire
        x = self.fc2(x)

        return x

class EpsilonGreedy:
    def __init__(self,policy,epsilon):
        self.policy = policy
        self.eps = epsilon
    def __call__(self,state):
        (cone,vitesse,goal) = state
        if np.random.random() > self.eps:
            return self.policy(cone,vitesse,goal)
        else:
            return torch.tensor([[np.random.random() for i in range(4)] for i in range(cone.shape[0])])
        


def decide(cone,speed,car,greedy,model,do_opti):
    cone = torch.tensor(np.array(cone),dtype=torch.float32)
    cone = cone.view(1,cone.shape[0],cone.shape[1],cone.shape[2])
    if do_opti:
        rep = greedy((cone,torch.tensor(car.vitesse),torch.tensor(car.get_relative_goal_position())))
    else: rep = model(cone,torch.tensor(car.vitesse),torch.tensor(car.get_relative_goal_position()))
    j = torch.argmax(rep,dim=1)
    rep = torch.tensor(
        [
        [1 if i==j[k] else 0 for i in range(rep.shape[1])]
        for k in range(rep.shape[0])]
        )[0]
    return rep # up down left right

def reward(voiture,pos_avant):
    d_before = np.linalg.norm(voiture.goal - pos_avant)
    d_after = np.linalg.norm(voiture.goal - np.array([voiture.x_position,voiture.y_position]))
    return - (d_after - d_before)
#model = DQN(3923,4)
#greedy = EpsilonGreedy(model,EPS_START)
#t = 0



