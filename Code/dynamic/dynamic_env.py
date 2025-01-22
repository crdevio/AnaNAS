from dynamic.dynamic import Dynamic
from dynamic.voiture import Voiture 
from dynamic.redlight import RedLightGreenLight
import numpy as np
import torch

class DynamicEnvironnement:
    def __init__(self, decide = None) -> None:
        self.dynamic_objects = []
        self.cars = []
        self.decide = decide
    def add(self,dynamic):
        self.dynamic_objects.append(dynamic)
    def add_car(self,car):
        self.cars.append(car)
    def __getitem__(self, index):
        if index < len(self.dynamic_objects):
            return self.dynamic_objects[index]
        else:
            return self.cars[index - len(self.dynamic_objects)]
                
    def decisions(self,img,mem,t):
        img_np = np.array(img)
        """
        Appelé QUE SI ia est activé
        """
        for car in self.cars:
            cone = car.get_cone()
            #cone = np.int32(np.min(np.concatenate((cone.reshape(101, 51, 1, 2), to_compare), axis=2), axis=2))
            cone = np.int32(cone)
            cone = img_np[cone[:, :, 0], cone[:, :, 1]]
            inputs = self.decide(cone,car.vitesse,car)
            car.get_inputs(inputs)
            return (cone, car.vitesse, car.get_relative_goal_position(), torch.argmax(inputs))
    def update_env(self,delta_t, keys = None):
        for i in self:
            i.update(delta_t, keys)
    def get_shape_env(self):
        rep = []
        for i in self:
                rep += [[i.get_shape(),i]]
        return rep