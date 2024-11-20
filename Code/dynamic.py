class Dynamic:
    def __init__(self,pos) -> None:
        self.pos = pos
    def update(self,delta_t):
        pass
    def get_shape(self):
        return [[]]

class RedLightGreenLight(Dynamic):
    def __init__(self, pos, delay,radius) -> None:
        super().__init__(pos)
        self.delay = delay
        self.current_delay = delay
        self.color = 0
        self.radius = radius

    def get_shape(self):
        if self.color==0:return [["circle","green",self.pos,self.radius]]
        else:return [["circle","red",self.pos,self.radius]]

    def update(self, delta_t):
        self.current_delay -= delta_t
        if self.current_delay<=0:
            self.current_delay = self.delay
            self.color = 1-self.color


class DynamicEnvironnement:
    def __init__(self) -> None:
        self.dynamic_objects = []
    def add(self,static):
        self.dynamic_objects.append(static)
    def __getitem__(self, index):
        return self.dynamic_objects[index]
    def update_env(self,delta_t):
        for i in self:
            i.update(delta_t)
    def get_shape_env(self):
        rep = []
        for i in self:
            rep += i.get_shape()
        return rep
    

env = DynamicEnvironnement()
env.add(RedLightGreenLight((0,0),1.,2))
env.add(RedLightGreenLight((20,20),2.,3))

for i in range(10):
    print(f"New step {i+1}")
    env.update_env(1.)
    print(env.get_shape_env())

