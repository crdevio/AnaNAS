#Pour tester la voiture, lancer test_voiture.py

import pygame
from math import cos, sin

VITESSE_MAX = 3
VITESSE_MIN = -3
FROTTEMENT = 1
VITESSE_ROT = 0.05 #C'est des gradient par seconde ou par appuis. A long terme, il faudra vérifier qu'on ne tourne pas trop 
FPS = 60


class Voiture():
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
        for event in events:
            if event == pygame.K_UP:
                self.avance(dt)
                accelere = True
            elif event == pygame.K_DOWN:
                self.recule(dt)
                descelere = True
            elif event == pygame.K_RIGHT:
                self.tourne_droite(dt)
            elif event == pygame.K_LEFT:
                self.tourne_gauche(dt)
        if (not accelere) and (not descelere):
            self.ralenti(dt)
    def avance(self, dt):
        self.vitesse += dt * (VITESSE_MAX - self.vitesse)
    def recule(self, dt): 
        self.vitesse += dt * (VITESSE_MIN - self.vitesse)
    def ralenti(self, dt):
        self.vitesse += dt * (0 - self.vitesse)
    def tourne_droite(self, dt):
        self.orientation += VITESSE_ROT * dt
    def tourne_gauche(self, dt):
        self.orientation -= VITESSE_ROT * dt