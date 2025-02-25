from PIL import Image, ImageDraw
import numpy as np

LARGEUR_ROUTE = 15
FILE_URL = 'output/circuit_4straight.png'
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

class CurvedRoad(Static):
    def __init__(self, pos, radius, width, start_angle, end_angle):
        super().__init__(pos)
        self.radius = radius
        self.width = width
        self.start_angle = start_angle
        self.end_angle = end_angle

        # Generate inner and outer arcs
        inner_arc = [
            (
                pos[0] + (radius * np.cos(angle)),
                pos[1] + (radius * np.sin(angle))
            )
            for angle in np.linspace(start_angle, end_angle, num=100)
        ]
        outer_arc = [
            (
                pos[0] + ((radius + width) * np.cos(angle)),
                pos[1] + ((radius + width) * np.sin(angle))
            )
            for angle in np.linspace(end_angle, start_angle, num=100)
        ]

        # Combine arcs to create a closed polygon
        self.vertices = inner_arc + outer_arc

    def draw_self(self, draw):
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
static_env.add(StraightRoad((50,50),(20),300,0))
static_env.add(StraightRoad((350,100),(300),30,0))

#static_env.add(CurvedRoad((50,260),150,60,np.pi/2,np.pi/4))
#static_env.add(CurvedRoad((303,513),150,60,-3*np.pi/4,-np.pi/4))
print(f"Génère dans {FILE_URL}")
generate_images(static_env,1000,1000,file=FILE_URL )
