#Pour la voiture, on a besoin du centre de la  voiture (juste une paire de points), de son orientation (un vecteur normalisé qui va donner dans quel sens elle se dirige) et sa vitesse
#(Pour ça on utilise une fonction déjà toute faite comme on a dit précédemen
#Augmente la vitesse on utilise la fonciton 
#Pour l'instant on fait seulement un truc binaire pour la vitesse, soit la vitesse est max, soit elle est min
#Pour que ça marche mieux, on utilise une equation diffénrentielle de la forme dv/dt = c - fv, ça va conger vers une vitesse max égale à c / f

import pygame
from math import cos, sin

VITESSE_MAX = 3
VITESSE_MIN = -3
FROTTEMENT = 1
VITESSE_ROT = 0.05 #C'est des gradient par seconde ou par appuis. A long terme, il faudra vérifier qu'on ne tourne pas trop 

class Voiture():
    def __init__(self):
        self.x_position = 0
        self.y_position = 0
        self.orientation = 0 #C'est en radient, son vecteur d'orientation sera donc (cos(orientation), sin(orientation))
        self.vitesse = 0
    def update(self, events):
        self.x_position += cos(self.orientation) * self.vitesse
        self.y_position += sin(self.orientation) * self.vitesse
        if self.x_position < 0:
            self.x_position = 0
        if self.y_position == 0:
            self.y_position = 0
        for event in events:
            if event == pygame.K_UP:
                self.augmente_vitesse()
            elif event == pygame.K_DOWN:
                self.diminue_vitesse()
            elif event == pygame.K_RIGHT:
                self.tourne_droite()
            elif event == pygame.K_LEFT:
                self.tourne_gauche()
    def augmente_vitesse(self):
        self.vitesse = VITESSE_MAX
    def diminue_vitesse(self): 
        self.vitesse = VITESSE_MIN
    def tourne_droite(self):
        self.orientation -= VITESSE_ROT
    def tourne_gauche(self):
        self.orientation += VITESSE_ROT