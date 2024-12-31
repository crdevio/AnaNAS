import ia
import torch
import numpy as np


class Dynamic:
    def __init__(self,pos) -> None:
        self.pos = pos
    def update(self,delta_t,keys=None):
        pass
    def get_shape(self):
        return []

class RedLightGreenLight(Dynamic):
    def __init__(self, pos, delay,radius) -> None:
        super().__init__(pos)
        self.delay = delay
        self.current_delay = delay
        self.color = 0
        self.radius = radius

    def get_shape(self):
        if self.color==0:return ["circle","green",self.pos,self.radius]
        else:return ["circle","red",self.pos,self.radius]

    def update(self, delta_t, keys = None,decide = None):
        self.current_delay -= delta_t
        if self.current_delay<=0:
            self.current_delay = self.delay
            self.color = 1-self.color


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
        return (self.dynamic_objects[index] if index < len(self.dynamic_objects) else self.cars[index - len(self.dynamic_objects)])
    
    def decisions(self,img,mem,t):
        img_np = np.array(img)
        to_compare = np.zeros((51, 25, 1, 2))
        to_compare[:, :, :, 0] = img_np.shape[0] - 1
        to_compare[:, :, :, 1] = img_np.shape[1] - 1
        """
        Appelé QUE SI ia est activé
        """
        for car in self.cars:
            cone = car.get_cone()
            print(cone.shape)
            cone = np.int32(np.max(np.concatenate((cone.reshape(51, 25, 1, 2), to_compare), axis=2), axis=2))
            cone = img_np[cone[:, :, 0], cone[:, :, 1]]
            inputs = self.decide(cone,car.vitesse,car)
            car.get_inputs(inputs)
            return (cone, car.vitesse, car.get_relative_goal_position(), torch.argmax(inputs), ia.state_reward(car))
    def update_env(self,delta_t, keys = None):
        for i in self:
            i.update(delta_t, keys)
    def get_shape_env(self):
        rep = []
        for i in self:
            rep += [[i.get_shape(),i]]
        return rep
    


