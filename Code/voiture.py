#Pour tester la voiture, lancer test_voiture.py

import pygame
from dynamic import *
from math import cos, sin, pi, atan
import numpy as np

VITESSE_MAX = 100
VITESSE_MIN = -20
FROTTEMENT = 0.5
VITESSE_ROT = 1.2 #C'est des gradient par seconde ou par appuis. A long terme, il faudra vérifier qu'on ne tourne pas trop 
FPS = 60
LONGUEUR = 10
LARGEUR = 5
VITESSE_ROT_NECESS = 20

RAYON_CONE = 64
ANGLE_CONE = 2*pi / 3
LARGEUR_CONE = 32


class Voiture(Dynamic):
    def __init__(self, position = (0,0), ia = True, goal = (200,200)):

        self.x_position = position[0]
        self.y_position = position[1]
        self.orientation = 0 #C'est en radient, son vecteur d'orientation sera donc (cos(orientation), sin(orientation))
        self.vitesse = 0
        self.ia = ia
        self.collision = False
        self.goal = goal
        self.up,self.down,self.left,self.right = 0,0,0,0

        #Pour l'instant le cone va en fait se transformer en rectangle. C'est les coordonées si la voiture est centrée en 0
        self.cone = []
        for i in range(0 + LONGUEUR // 2, 101 + LONGUEUR // 2):
            self.cone.append([])
            for j in range(-25, 26):
                self.cone[-1].append([i, j])
        self.cone = np.array(self.cone, int)
        print(self.cone.shape)
                    

    def update(self, dt, events):

        self.x_position += cos(self.orientation) * self.vitesse * dt
        self.y_position += sin(self.orientation) * self.vitesse * dt

        accelere = False
        descelere = False

        if not self.ia:
            self.up, self.down, self.left, self.right = events[pygame.K_z], events[pygame.K_s], events[pygame.K_d], events[pygame.K_q]
        if self.up:
            self.avance(dt)
            accelere = True
        if self.down:
            self.recule(dt)
            descelere = True
        if self.left:
            self.tourne_droite(dt)
        if self.right:
            self.tourne_gauche(dt)
        if (not accelere) and (not descelere):
            self.ralenti(dt)

    def avance(self, dt):
        self.vitesse += dt * (VITESSE_MAX - self.vitesse * FROTTEMENT)
    def recule(self, dt): 
        self.vitesse += dt * (VITESSE_MIN - self.vitesse * FROTTEMENT)
    def ralenti(self, dt):
        self.vitesse += dt * (0 - self.vitesse * FROTTEMENT)
    def tourne_droite(self, dt):
        self.orientation += VITESSE_ROT * dt * min(self.vitesse/VITESSE_ROT_NECESS,1)
    def tourne_gauche(self, dt):
        self.orientation -= VITESSE_ROT * dt * min(self.vitesse/VITESSE_ROT_NECESS,1)
    
    def get_shape(self):
        return ["rect", "blue", (self.x_position, self.y_position, LARGEUR, LONGUEUR),self.orientation]
    
    def get_relative_goal_position(self):
        dx = self.goal[0] - self.x_position
        dy = self.goal[1] - self.y_position

        # Rotation inverse pour obtenir les coordonnées relatives
        x_rel = np.cos(-self.orientation) * dx - np.sin(-self.orientation) * dy
        y_rel = np.sin(-self.orientation) * dx + np.cos(-self.orientation) * dy

        return x_rel, y_rel

    def get_cone(self):

        '''
        
        cone_actuel = []

        for i in range(RAYON_CONE):
            cone_actuel.append([])
            for j in range(LARGEUR_CONE):

                x, y = self.x_position,self.y_position

                x += cos(self.orientation-ANGLE_CONE/2+j/LARGEUR_CONE*ANGLE_CONE)*i
                y += sin(self.orientation-ANGLE_CONE/2+j/LARGEUR_CONE*ANGLE_CONE)*i

                cone_actuel[-1].append([int(x),int(y)])

        return cone_actuel

        '''

        #Il faut calculer le centre de la voiture, faire la rotation et ajouter le centre

        x_avant_gauche = self.x_position + LONGUEUR * cos(self.orientation)
        y_avant_gauche = self.y_position + LONGUEUR * sin(self.orientation)

        x_avant_droite = x_avant_gauche - LARGEUR * sin(self.orientation)
        y_avant_droite = y_avant_gauche + LARGEUR * cos(self.orientation)

        x_milieu = (x_avant_droite + self.x_position) / 2
        y_milieu = (y_avant_droite + self.y_position) / 2

        matrice_rotation = np.array([[cos(self.orientation), sin(self.orientation)], [-sin(self.orientation), cos(self.orientation)]])

        return self.cone @ matrice_rotation + np.array([x_milieu, y_milieu])
        
    
    def get_inputs(self,inputs):
        if self.ia:
            self.up,self.down,self.left,self.right = inputs[0],inputs[1],inputs[2],inputs[3]

