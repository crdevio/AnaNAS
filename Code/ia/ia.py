import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ia.constants import *


def state_reward(car):
    # ATTENTION y a peut-être une couille entre l'appel à state_reward et la valeur de collision.
    return 100/(0.01 + np.linalg.norm(np.array([car.x_position, car.y_position]) - np.array(car.goal))/10)
    score = 40*np.exp(-np.linalg.norm(np.array([car.x_position, car.y_position]) - np.array(car.goal))/100.)
    vector_to_goal = np.array(car.goal) - np.array([car.x_position, car.y_position])
    theta_g = np.atan2(vector_to_goal[1],vector_to_goal[0])
    print(f'Angle theta : {np.abs(theta_g)}')
    score -= 4*np.abs(theta_g)
    if car.collision:
        score = -100
    return score

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