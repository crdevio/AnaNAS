# Example file showing a basic pygame "game loop"
import pygame
from dynamic import *
from voiture import *
from graphics_classes import Camera,Polygon
import numpy as np
import pygame.surfarray


pygame.init()
font = pygame.font.Font(None, 36)
RES_AFFICHAGE = (600,600)
FPS = 60 
CAMERA_SPEED = 100
class Simulation:
    def __init__(self, dyn_env=DynamicEnvironnement(),res = RES_AFFICHAGE, static_url = "", drawing = True, dt = 0.01):
        self.running = True
        self.clock = pygame.time.Clock()
        self.clock.tick(FPS)
        self.screen = pygame.display.set_mode(res)
        if static_url == "":
            self.static_img = None
        else:
            self.static_img = pygame.image.load(static_url).convert()
            self.static_arr = pygame.surfarray.array3d(self.static_img)
        self.drawing = drawing
        if drawing:
            self.time_manager = lambda: self.clock.get_time()/1000.0 # in sec
            self.camera = Camera(res[0],res[1])
        else:
            self.time_manager = lambda: dt

        self.dyn_env = dyn_env

    def update(self):
        self.clock.tick(FPS)
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                self.running = False
        keys = pygame.key.get_pressed()
        dt = self.time_manager()
        if self.drawing:
            # Je mets ça la pour pas avoir à recalculer keys dans draw.
            if keys[pygame.K_LEFT]:
                self.camera.move_left(CAMERA_SPEED * dt)
            if keys[pygame.K_RIGHT]:
                self.camera.move_right(CAMERA_SPEED * dt)
            if keys[pygame.K_UP]:
                self.camera.move_up(CAMERA_SPEED * dt)
            if keys[pygame.K_DOWN]:
                self.camera.move_down(CAMERA_SPEED * dt)
        self.dyn_env.update_env(dt,keys)


    def draw(self):
        if not self.drawing: return
        if self.static_img == None:
            print("Nathan t'as pas set static_img")
            print("Clement cette phrase ne veut rien dire")
            print("De toute facon faudra qu'on reorganise les fichiers entre eux et ya d'autres trus qui vont pas")
            return
        image_array = pygame.surfarray.array3d(pygame.display.get_surface())
        self.dyn_env.decisions(image_array)
        self.screen.fill("black")
        self.screen.blit(self.static_img,(- self.camera.x, - self.camera.y))
        shapes = self.dyn_env.get_shape_env()
        for shape in shapes:
            if shape[0] == "circle":
                # shape = "circle",(r,g,b),(center_x,center_y),radius
                x = shape[2][0] - self.camera.x
                y = shape[2][1] - self.camera.y
                pygame.draw.circle(self.screen, shape[1],(x,y), shape[3])
            if shape[0] == "line":
                # shape = "line",(r,g,b),(start_x,start_y),(end_x,end_y),width
                x = shape[2][0] - self.camera.x
                y = shape[2][1] - self.camera.y
                pygame.draw.line(self.screen,shape[1],(x,y), shape[3] - (self.camera.x,self.camera.y),shape[4])
            if shape[0] == "rect":
                # shape = "rect", (r,g,b), (x,y,lenx,leny), angle
                pos = shape[2][0]-self.camera.x,shape[2][1]-self.camera.y
                width,length = shape[2][2],shape[2][3]
                angle = shape[3]
                angle+=np.pi/2
                corners = [
                (-width, -length),
                (width, -length),
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
                #polygone = Polygon(rotated_corners)
                c = False
                for e in rotated_corners:
                    x,y = int(e[0]),int(e[1])
                    x+=int(self.camera.x)
                    y += int(self.camera.y)
                    if sum(self.static_arr[x][y]) == 0:
                        c= True 
                    

                pygame.draw.polygon(self.screen,shape[1],rotated_corners)
        if self.dyn_env.cars[0].ia:
            text = "Decision: "
            if self.dyn_env.cars[0].up:
                text+="UP "
            if self.dyn_env.cars[0].down:
                text+="DOWN "
            if self.dyn_env.cars[0].left:
                text+="LEFT "
            if self.dyn_env.cars[0].right:
                text+="RIGHT "
            text_surface = font.render(text, True, (255, 0,0))  # True for anti-aliasing
            self.screen.blit(text_surface, (200, 200))
        pygame.display.flip()

       
dyn_env = DynamicEnvironnement()

dyn_env.add(RedLightGreenLight((100,100),2,5))
dyn_env.add_car(Voiture(position=(40,40),ia=True))
            
jeu = Simulation(static_url="output/image.png",dyn_env = dyn_env)
while jeu.running:
    jeu.update()
    jeu.draw()

pygame.quit()