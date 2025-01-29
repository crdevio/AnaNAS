import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ia.constants import *

import numpy as np

def state_reward(car):
    # Calcul de la distance au but
    distance_to_goal = np.linalg.norm(np.array([car.x_position, car.y_position]) - np.array(car.goal))
    
    # Calcul de l'orientation vers le but
    vector_to_goal = np.array(car.goal) - np.array([car.x_position, car.y_position])
    theta_g = np.arctan2(vector_to_goal[1], vector_to_goal[0]) 
    theta_diff = np.abs(theta_g - car.orientation)
    
    # Récompense basée sur la proximité du but (moins c'est loin, mieux c'est)
    distance_reward = 100 / (1 + (distance_to_goal / 10)**2)
    
    # Récompense basée sur l'alignement avec l'objectif (plus l'orientation est correcte, mieux c'est)
    orientation_reward = np.cos(theta_diff)
    
    # Pénalité pour la collision
    if car.collision:
        return -100  # Punir fortement en cas de collision
    
    # Récompense pour atteindre le but
    goal_reached_reward = 100 if distance_to_goal < 1 else 0
    
    # Pénalité pour une grande vitesse (si la vitesse est trop élevée, la voiture peut être hors de contrôle)
    speed_penalty = 0 # -1 * max(0, car.vitesse - 10)  # Pénalise si la vitesse dépasse 10 unités (à ajuster selon le cas)
    
    # Pénalité pour des changements d'orientation brusques (pour éviter une conduite erratique)
    orientation_penalty = -1 * np.abs(car.orientation - np.arctan2(vector_to_goal[1], vector_to_goal[0]))  # Encourager une conduite plus fluide
    
    # Récompense pour la conduite fluide (basée sur la vitesse et la distance)
    efficiency_reward = max(0, 5 - (distance_to_goal / (car.vitesse + 1)))  # Encourager à atteindre l'objectif en moins de temps
    
    # Total de la récompense
    reward = distance_reward + orientation_reward + goal_reached_reward + speed_penalty + orientation_penalty + efficiency_reward
    
    return reward


def decide(cone, speed, car, greedy, model, do_opti):
    cone = torch.tensor(np.array(cone), dtype=torch.float32, device=DEVICE)
    cone = cone.view(1, cone.shape[0], cone.shape[1], cone.shape[2])
    rep = greedy((cone, torch.tensor(car.vitesse, device=DEVICE), torch.tensor(car.get_relative_goal_position(), device=DEVICE,dtype=torch.float32)))
    j = torch.argmax(rep, dim=1)
    rep = torch.tensor(
        [
            [1 if i == j[k] else 0 for i in range(rep.shape[1])]
            for k in range(rep.shape[0])]
    )[0]
    return rep  # up down left right