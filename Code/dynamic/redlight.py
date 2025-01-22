from dynamic.dynamic import Dynamic

class RedLightGreenLight(Dynamic):
    def __init__(self, pos, delay,radius) -> None:
        super().__init__(pos)
        self.delay = delay
        self.current_delay = delay
        self.color = 0
        self.radius = radius

    def get_shape(self):
        if self.color==0:
            return ["circle","green",self.pos,self.radius]
        else:
            return ["circle","red",self.pos,self.radius]

    def update(self, delta_t, keys = None,decide = None):
        self.current_delay -= delta_t
        if self.current_delay<=0:
            self.current_delay = self.delay
            self.color = 1-self.color