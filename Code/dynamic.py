import ia
import torch


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
        """
        Appelé QUE SI ia est activé
        """
        for e in self.cars:
            cone = e.get_cone()
            cone = [img[int(c[0])][int(c[1])] for c in cone]
            mem.states.append((cone,e.vitesse,e.get_relative_goal_position()))
            inputs = self.decide(cone,e.vitesse,e)
            mem.actions.append(torch.argmax(inputs))
            mem.rewards.append(ia.state_reward(e))
            e.get_inputs(inputs)
    def update_env(self,delta_t, keys = None):
        for i in self:
            i.update(delta_t, keys)
    def get_shape_env(self):
        rep = []
        for i in self:
            rep += [[i.get_shape(),i]]
        return rep
    


