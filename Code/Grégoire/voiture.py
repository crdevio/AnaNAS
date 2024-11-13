



#Pour la voiture, on a besoin du centre de la  voiture (juste une paire de points), de son orientation (un vecteur normalisé qui va donner dans quel sens elle se dirige) et sa vitesse
#(Pour ça on utilise une fonction déjà toute faite comme on a dit précédemen
#Augmente la vitesse on utilise la fonciton 
#Pour l'instant on fait seulement un truc binaire pour la vitesse, soit la vitesse est max, soit elle est min

from pygame import event
from math import cos, sin

VITESSE_MAX = 5
VITESSE_MIN = -2
VITESSE_ROT = 0.05 #C'est des gradient par seconde ou par appuis. A long terme, il faudra vérifier qu'on ne tourne pas trop 

class Voiture(Dynamic):
    def __init__(self):
        self.centre = (0, 0)
        self.orientation = 0 #C'est en radient, son vecteur d'orientation sera donc (cos(orientation), sin(orientation))
        self.vitesse = 0
    def update(self, events):
        self.centre += (cos(self.orientation), sin(self.orientation)) * self.vitesse
        for event in events:
            if event == K_PAGEUP:
                self.augmente_vitesse()
            elif event == K_PAGEDOWN:
                self.diminue_vitesse()
            elif event == K_RSHIFT:
                self.tourne_droite()
            elif event == K_LSHIFT:
                self.tourne_gauche()
    def augmente_vitesse(self):
        self.vitesse = VITESSE_MAX
    def diminue_vitesse(self): 
        self.vitesse = VITESSE_MIN
    def tourne_droite(self):
        self.orientation -= VITESSE_ROT
    def tourne_gauche_self(self):
        self.orientation += VITESSE_ROT