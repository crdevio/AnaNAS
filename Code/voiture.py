#Pour tester la voiture, lancer test_voiture.py

import pygame
from dynamic import *
from math import cos, sin

VITESSE_MAX = 20
VITESSE_MIN = -7
FROTTEMENT = 0.5
VITESSE_ROT = 0.2 #C'est des gradient par seconde ou par appuis. A long terme, il faudra vérifier qu'on ne tourne pas trop 
FPS = 60



class Voiture(Dynamic):
    def __init__(self):
        self.x_position = 0
        self.y_position = 0
        self.orientation = 0 #C'est en radient, son vecteur d'orientation sera donc (cos(orientation), sin(orientation))
        self.vitesse = 0

    def update(self, dt, events):
        self.x_position += cos(self.orientation) * self.vitesse * dt
        self.y_position += sin(self.orientation) * self.vitesse * dt
        accelere = False
        descelere = False
        if events[pygame.K_z]:
            self.avance(dt)
            accelere = True
        if events[pygame.K_s]:
            self.recule(dt)
            descelere = True
        if events[pygame.K_d]:
            self.tourne_droite(dt)
        if events[pygame.K_q]:
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
        self.orientation += VITESSE_ROT * dt
    def tourne_gauche(self, dt):
        self.orientation -= VITESSE_ROT * dt
    
    def get_shape(self):
        return [["rect", "blue", (self.x_position, self.y_position, 30, 50),self.orientation]]
