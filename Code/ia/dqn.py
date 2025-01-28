import torch.nn as nn
from ia.memory import Memory
from ia.constants import *
import torch.nn.functional as F
""""
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

        self.to(DEVICE)

    def forward(self, x, vitesse, goal):
        vitesse = vitesse.view(-1, 1).to(DEVICE)
        goal = goal.view(-1, 2).to(DEVICE)

        #time_emb = self.get_time_embedding(torch.arctan(goal[:,1]/goal[:,0]))
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
        embedding = torch.sin(time_step)

        return embedding
"""
class DQN(nn.Module):
    def __init__(self, input_size, output_features):
        super(DQN, self).__init__()
        
        # Convolutions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)  # Réduit H/2 et W/2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)  # Réduit encore H/4 et W/4
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Force la sortie des convolutions à une taille fixe (4x4)
        
        # Fully connected layers
        conv_output_size = 32 * 4 * 4  # Après le pooling : (32 channels) x (4x4)
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64 + 3, 32)  # Inclut vitesse (1) + goal (2)
        self.fc3 = nn.Linear(32, output_features)

        # Device
        self.to(DEVICE)

    def forward(self, x, vitesse, goal):
        vitesse = vitesse.view(-1, 1).to(DEVICE)
        goal = goal.view(-1, 2).to(DEVICE)

        # Convolutions
        x = x.transpose(1, 3).to(DEVICE)  # (batch_size, H, W, C) -> (batch_size, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Fixe la taille de sortie à (batch_size, 32, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 32 * 4 * 4)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.cat((x, vitesse, goal), dim=1).to(torch.float32)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
