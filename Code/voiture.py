#Pour tester la voiture, lancer test_voiture.py

import pygame
from dynamic import *
from math import cos, sin, pi, atan

VITESSE_MAX = 100
VITESSE_MIN = -20
FROTTEMENT = 0.5
VITESSE_ROT = 1.2 #C'est des gradient par seconde ou par appuis. A long terme, il faudra vérifier qu'on ne tourne pas trop 
FPS = 60
LONGUEUR = 10
LARGEUR = 5
VITESSE_ROT_NECESS = 20

RAYON_CONE = 100
ANGLE_CONE = pi / 8



class Voiture(Dynamic):
    def __init__(self):
        self.x_position = 0
        self.y_position = 0
        self.orientation = 0 #C'est en radient, son vecteur d'orientation sera donc (cos(orientation), sin(orientation))
        self.vitesse = 0

        self.cone = []
        for x_cone in range(1, RAYON_CONE):
            for y_cone in range(RAYON_CONE):
                if x_cone**2 + y_cone**2 > RAYON_CONE**2:
                    break
                angle = atan(y_cone / x_cone)
                if angle > -ANGLE_CONE and angle < ANGLE_CONE:
                    self.cone.append((x_cone, y_cone))
                    

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
        self.orientation += VITESSE_ROT * dt * min(self.vitesse/VITESSE_ROT_NECESS,1)
    def tourne_gauche(self, dt):
        self.orientation -= VITESSE_ROT * dt * min(self.vitesse/VITESSE_ROT_NECESS,1)
    
    def get_shape(self):
        return [["rect", "blue", (self.x_position, self.y_position, LARGEUR, LONGUEUR),self.orientation]]

    def get_cone(self):

        x_avant_gauche = self.x_position + LONGUEUR * cos(self.orientation)
        y_avant_gauche = self.y_position + LONGUEUR * sin(self.orientation)

        x_avant_droite = x_avant_gauche + LARGEUR * sin(self.orientation)
        y_avant_droite = y_avant_gauche + LARGEUR * cos(self.orientation)

        x_milieu_devant = (x_avant_gauche + x_avant_droite) / 2
        y_milieu_devant = (y_avant_gauche + y_avant_droite) / 2

        x_milieu = (x_avant_gauche + self.x_position) / 2
        y_milieu = (y_avant_gauche + self.y_position) / 2

        cone_actuel = []

        for i in range(len(self.cone)):

            x, y = cone[i][0], cone[i][1]

            x += x_milieu_devant
            y += y_milieu_devant

            x_p = x - x_milieu
            y_p = y - y_milieu

            x_pp = x_p * cos(self.orientation) - y_p * sin(self.orientation)
            y_pp = x_p * sin(self.orientation) + y_p * cos(self.orientation)

            x_rot = x_pp + x_milieu
            y_rot = y_pp + y_milieu

            cone_actuel.append([x_rot, y_rot])

        return cone_actuel
    

