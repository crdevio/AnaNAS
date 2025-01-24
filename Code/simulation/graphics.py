import pygame
from dynamic.voiture import *
from dynamic.dynamic import *
from dynamic.dynamic_env import DynamicEnvironnement
from graphics_classes import Camera,Polygon
import numpy as np
import pygame.surfarray
import torch
import random
from ia.ia import DEVICE
from ia.memory import Memory
from ia.dqn import DQN
from ia.epsgreedy import EpsilonGreedy,EPS_START,EPS_DECAY,EPS_MIN
import sys
import ia

RES_AFFICHAGE = (600,600)
FPS = 600 
CAMERA_SPEED = 100
GOAL = (400,40) #(140,122)

#mettre dedans les urls des fichiers et leur goal
"""
STATIC_URLS = {"output/decaler.png" : [(230,160),[(80,140,0)]],
               "output/short.png" : (170,140),
               "output/straight.png":[(400,140),[(80,140,0),(150,140,0),(210,140,0)]]
               "output/curved.png" : [(170,220),[(80,140,0),(130,160,np.pi/4)]],
               "output/squared.png": [(220,220),[(80,250,0)]],
               "output/squared2.png": [(220,280),[(80,250,0)]]}
"""

STATIC_URLS = {"output/squared.png": [(220,220),[(80,250,0)]],
               "output/squared2.png": [(220,280),[(80,250,0)]]}

STATIC_URLS_LIST = list(STATIC_URLS.keys())

epsilon_dict = {key: 1. for key in STATIC_URLS_LIST}

pygame.init()
font = pygame.font.Font(None, 36)

class Simulation:

    def __init__(self, dyn_env=DynamicEnvironnement(),res = RES_AFFICHAGE, drawing = True, dt = 0.017):
        static_url = STATIC_URLS_LIST[random.randint(0,len(STATIC_URLS_LIST)-1)]
        self.static_url = static_url
        self.GOAL = STATIC_URLS[static_url][0]
        pygame.init()
        font = pygame.font.Font(None, 36)
        self.running = True
        self.clock = pygame.time.Clock()
        self.clock.tick(FPS)
        self.screen = pygame.display.set_mode(res)
        self.font = pygame.font.Font(None, 50)  # None pour utiliser la police par défaut, 50 pour la taille
        if static_url == "":
            self.static_img = None
        else:
            self.static_img = pygame.image.load(static_url).convert()
            self.static_arr = pygame.surfarray.array3d(self.static_img)
            for i in range(-2,3):
                for j in range(-2,3):
                    self.static_arr[self.GOAL[0]+i][self.GOAL[1]+j] = np.array([255,0,0])
            """
            self.static_arr[(self.static_arr == [0, 0, 0]).all(axis=-1)] = [1, 0, 0]
            self.static_arr[(self.static_arr == [255, 255, 255]).all(axis=-1)] = [0, 1, 0]
            self.static_arr[(self.static_arr == [255, 0, 0]).all(axis=-1)] = [0, 0, 1]
            """
        self.drawing = drawing
        if False:
            self.time_manager = lambda: self.clock.get_time()/1000.0 # in sec
        else:
            self.time_manager = lambda: dt


        self.camera = Camera(res[0],res[1])

        self.dyn_env = dyn_env

    def update(self,mem,t,sim):
        self.clock.tick(FPS)
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                pygame.quit()
        keys = pygame.key.get_pressed()
        dt = self.time_manager()
        if self.drawing:
            # Je mets ça la pour pas avoir à recalculer keys dans draw.
            if keys[pygame.K_SPACE]:
                sim.do_opti = not sim.do_opti
            if keys[pygame.K_LEFT]:
                self.camera.move_left(CAMERA_SPEED * dt)
            if keys[pygame.K_RIGHT]:
                self.camera.move_right(CAMERA_SPEED * dt)
            if keys[pygame.K_UP]:
                self.camera.move_up(CAMERA_SPEED * dt)
            if keys[pygame.K_DOWN]:
                self.camera.move_down(CAMERA_SPEED * dt)
        self.dyn_env.update_env(dt,keys)
        #image_array = pygame.surfarray.array3d(pygame.display.get_surface())
        return self.dyn_env.decisions(self.static_arr,mem,t)
    def draw(self,score):
        if not self.drawing: 
            return
        if self.static_img == None:
            print("Nathan t'as pas set static_img")
            print("Clement cette phrase ne veut rien dire")
            print("De toute facon faudra qu'on reorganise les fichiers entre eux et ya d'autres trus qui vont pas")
            return
        

        self.screen.fill("black")
        self.screen.blit(self.static_img,(- self.camera.x, - self.camera.y))
        shapes = self.dyn_env.get_shape_env()
        for bidule in shapes:
            shape,car = bidule[0],bidule[1]
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
                car.collision = False
                cone = car.get_cone()
                pygame.draw.polygon(self.screen, (27, 255, 89), [cone[0][0],cone[0][-1],cone[-1][-1],cone[-1][0]], width=5)
                for e in rotated_corners:
                    x,y = int(e[0]),int(e[1])
                    x+=int(self.camera.x)
                    y += int(self.camera.y)
                    if (self.static_arr[x][y] == [0,0,0]).all():
                        c= True
                        car.collision = True
                pygame.draw.polygon(self.screen,shape[1],rotated_corners)
        pygame.draw.circle(self.screen, "red", (self.GOAL[0] - self.camera.x, self.GOAL[1] - self.camera.y), 6)
        text_surface = self.font.render(f"Score: {score}", True, (255, 0,0))  # True pour l'anti-aliasing
        self.screen.blit(text_surface, (300,500))
        pygame.display.flip()

'''
On dirait que ces lignes servent à rien mais pour l'instant je les laisse le temps de vérifier       
dyn_env = DynamicEnvironnement()

dyn_env.add_car(Voiture(position=(40,140),ia=True))
'''