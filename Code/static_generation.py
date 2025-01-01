from PIL import Image, ImageDraw
import numpy as np

LARGEUR_ROUTE = 15

class Static:
    def __init__(self,pos) -> None:
        self.pos = pos
    def draw_self(self,draw):
        pass

class StraightRoad(Static):
    def __init__(self,pos,width,length,angle) -> None:
        super().__init__(pos)
        angle-=np.pi/2
        corners = [
        (-width, 0),
        (width, 0),
        (width, length),
        (-width, length)
    ]
        x,y = pos
        rotated_corners = [
        (
            x + cx * np.cos(angle) - cy * np.sin(angle),
            y + cx * np.sin(angle) + cy * np.cos(angle)
        )
        for cx, cy in corners
    ]
        self.vertices = rotated_corners
    def draw_self(self,draw):
        draw.polygon(self.vertices, fill="white")


class StaticEnvironnement:
    def __init__(self) -> None:
        self.static_objects = []
    def add(self,static):
        self.static_objects.append(static)
    def __getitem__(self, index):
        return self.static_objects[index]


def generate_images(env,width,height,file='output/image.png'):
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    for obj in env:
        obj.draw_self(draw)
    image.save(file)

static_env = StaticEnvironnement()
"""
static_env.add(StraightRoad((40,40),LARGEUR_ROUTE,80,0))
static_env.add(StraightRoad((135,25),LARGEUR_ROUTE,80,np.pi/2))
static_env.add(StraightRoad((40,25),LARGEUR_ROUTE,80,np.pi/2))
static_env.add(StraightRoad((25,120),LARGEUR_ROUTE,125,0))
"""
static_env.add(StraightRoad((40, 40), LARGEUR_ROUTE,150,0))
generate_images(static_env,500,500,file='output/short.png')
