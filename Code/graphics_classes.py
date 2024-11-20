class Camera:
    def __init__(self,weight,height,x=0,y=0):
        self.w = weight
        self.h = height
        self.x = x
        self.y = y
    def move_left(self, step = 1):  self.x -= step
    def move_right(self, step = 1): self.x += step
    def move_up(self,step = 1):     self.y -= step
    def move_down(self, step = 1):  self.y += step