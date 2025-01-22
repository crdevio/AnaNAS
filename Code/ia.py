import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import Memory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {DEVICE} device")

EPS_START = 1.
EPS_DECAY = 1e-3        #dans le TP 1e-5
EPS_MIN = 0.1


def state_reward(car):
    # ATTENTION y a peut-être une couille entre l'appel à state_reward et la valeur de collision.
    score = 40*np.exp(-np.linalg.norm(np.array([car.x_position, car.y_position]) - np.array(car.goal))/100.)
    vector_to_goal = np.array([car.x_position, car.y_position]) - np.array(car.goal)
    theta_g = np.atan2(vector_to_goal[1],vector_to_goal[0])
    score -= 4*np.abs(theta_g)
    if car.collision:
        score = - 100
    return score



class DQN(nn.Module):
    def __init__(self, input_samples, output_features):
        super(DQN, self).__init__()
        self.memory = Memory()

        #self.time_emb_layer = nn.Linear(1,1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)  # Output: (4, 64, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(1 * (input_samples // 1), 16)
        self.fc2 = nn.Linear(16 + 3, 16)
        self.fc3 = nn.Linear(16,output_features)

        # Move the model to the specified device
        self.to(DEVICE)

    def forward(self, x, vitesse, goal):
        vitesse = vitesse.view(-1, 1).to(DEVICE)
        goal = goal.view(-1, 2).to(DEVICE)
        #time_emb = self.get_time_embedding(torch.arctan(goal[:,1]/goal[:,0]))
        
        # Add time embedding as extra channels to the image
        #time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Make it (batch_size, 2, 1, 1)
        #time_emb = self.time_emb_layer(time_emb)
        
        x = x.transpose(1, 3).to(DEVICE)
        #x += time_emb
        x = self.conv1(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(x)
        x = F.relu(self.fc1(x))

        x = torch.cat((x, vitesse, goal), dim=1).to(torch.float32)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def get_time_embedding(self, time_step):
        # Convert time_step to tensor and get the time embedding
        embedding = torch.sin(time_step)

        return embedding


class EpsilonGreedy:
    def __init__(self, policy, epsilon):
        self.policy = policy
        self.eps = epsilon

    def __call__(self, state):
        (cone, vitesse, goal) = state
        if np.random.random() > self.eps:
            return self.policy(cone.to(DEVICE), vitesse.to(DEVICE), goal.to(DEVICE))
        else:
            return torch.tensor([[np.random.random() for i in range(4)] for i in range(cone.shape[0])], device=DEVICE)


def decide(cone, speed, car, greedy, model, do_opti):
    cone = torch.tensor(np.array(cone), dtype=torch.float32, device=DEVICE)
    cone = cone.view(1, cone.shape[0], cone.shape[1], cone.shape[2])
    if do_opti:
        rep = greedy((cone, torch.tensor(car.vitesse, device=DEVICE), torch.tensor(car.get_relative_goal_position(), device=DEVICE,dtype=torch.float32)))
    else:
        rep = model(cone, torch.tensor(car.vitesse, device=DEVICE), torch.tensor(car.get_relative_goal_position(), device=DEVICE))
    j = torch.argmax(rep, dim=1)
    rep = torch.tensor(
        [
            [1 if i == j[k] else 0 for i in range(rep.shape[1])]
            for k in range(rep.shape[0])]
    )[0]
    return rep  # up down left right


def reward(voiture, pos_avant):
    print("attention je suis utilisée")
    d_before = np.linalg.norm(voiture.goal - pos_avant)
    d_after = np.linalg.norm(voiture.goal - np.array([voiture.x_position, voiture.y_position]))
    return -(d_after - d_before)