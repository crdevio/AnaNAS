import numpy as np

def decide(cone,speed):
    print(cone.shape)
    return [1,0,0,0] # up down left right

def reward(voiture,pos_avant):
    d_before = np.linalg.norm(voiture.goal - pos_avant)
    d_after = np.linalg.norm(voiture.goal - np.array([voiture.x_position,voiture.y_position]))
    return - (d_after - d_before)
