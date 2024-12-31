import pygame
from dynamic import *
from voiture import *
from graphics_classes import Camera,Polygon
import numpy as np
import pygame.surfarray
import torch
import torch.optim as optim
import torch.nn as nn
from memory import Memory
from ia import DQN,EpsilonGreedy,EPS_START,EPS_DECAY,EPS_MIN,DEVICE
import sys

RES_AFFICHAGE = (600,600)
FPS = 600 
CAMERA_SPEED = 100
GOAL = (400,40) #(140,122)
SAVE_EVERY = 10
INPUT_SAMPLE = 1200
NB_EPOCH = 1000
BATCH_SIZE = 2


pygame.init()
font = pygame.font.Font(None, 36)

class Simulation:

    def __init__(self, dyn_env=DynamicEnvironnement(),res = RES_AFFICHAGE, static_url = "", drawing = True, dt = 0.017):
        pygame.init()
        font = pygame.font.Font(None, 36)
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
        image_array = pygame.surfarray.array3d(pygame.display.get_surface())
        return self.dyn_env.decisions(image_array,mem,t)
    def draw(self):
        if not self.drawing: return
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
                for e in rotated_corners:
                    x,y = int(e[0]),int(e[1])
                    x+=int(self.camera.x)
                    y += int(self.camera.y)
                    somm = 0
                    try: somm = sum(self.static_arr[x][y])
                    except:pass
                    if somm== 0:
                        c= True
                        car.collision = True
                pygame.draw.polygon(self.screen,shape[1],rotated_corners)
        pygame.draw.circle(self.screen,"red",(GOAL[0]-self.camera.x,GOAL[1]-self.camera.y),2)
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

class DeepQAgent:

    def __init__(self, T=100,game_per_epoch = 10, gamma=0.5, lr = 0.01, weight_path = None, do_opti = True, target_update_freq = 50):

        self.memory = Memory()
        self.t = 0
        self.num_sim=0
        self.game_per_epoch = game_per_epoch
        self.iter = 0
        self.T = T
        self.policy_model = DQN(INPUT_SAMPLE,4).to(DEVICE)
        self.target_model = DQN(INPUT_SAMPLE,4).to(DEVICE)

        if weight_path != None: 
            self.policy_model.load_state_dict(torch.load(weight_path, weights_only=True))
            self.target_model.load_state_dict(torch.load(weight_path,weights_only=True))
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(),lr = lr)
        self.policy_epsgreedy = EpsilonGreedy(self.policy_model,EPS_START)

        self.jeu = None
        self.gamma = gamma
        self.criterion = nn.HuberLoss()
        self.do_opti = do_opti
        self.update_freq_delay = 0
        self.target_update_freq = target_update_freq

    def etape1(self):

        self.jeu = Simulation(static_url="output/straight.png",dyn_env = None)
        for _ in range(self.game_per_epoch):
            dyn_env = DynamicEnvironnement(
                lambda cone,speed,car: ia.decide(cone,speed,car,self.policy_epsgreedy,self.policy_model,self.do_opti)
            )
            dyn_env.add(RedLightGreenLight((100,100),2,5))
            dyn_env.add_car(Voiture(position=(80,40),ia=True, goal=GOAL))
            self.jeu.dyn_env = dyn_env
            self.t = 0
            while self.t < self.T:
                self.t += 1
                cone, vitesse, goal, actions, rewards = self.jeu.update(self.memory,self.t,self)
                is_terminal = False
                if self.t == (self.T):
                    is_terminal = True
                    print("Final Reward: ",self.memory.rewards[-1])
                self.jeu.draw()
                if self.memory.size >= BATCH_SIZE and self.do_opti:
                    self.optimize_model()
                for car in dyn_env.cars:
                    if car.collision:return
                self.update_freq_delay+=1
                if self.update_freq_delay >= self.target_update_freq:
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    self.update_freq_delay=0
                self.memory.append(cone, vitesse, goal, actions, rewards, is_terminal)

    def optimize_model(self):

        self.policy_optimizer.zero_grad()
        cones,speeds,goals,next_cones,next_speeds,next_goals,actions,rewards,terminals = self.memory.sample(BATCH_SIZE)
        mask = torch.tensor((1. - terminals.astype(float)))
        cones = cones.to(DEVICE)
        speeds = speeds.to(DEVICE)
        goals = goals.to(DEVICE)
        mask = torch.tensor((1. - terminals.astype(float))).to(DEVICE)
        target_value = torch.max(self.target_model(cones, speeds,goals), dim=1)[0].to(DEVICE)
        y = rewards.to(DEVICE) + mask * self.gamma * target_value
        y_predicted = self.policy_model(cones,speeds,goals)
        rewards_predicted = y_predicted[torch.arange(BATCH_SIZE),actions.to(DEVICE)].type(torch.float64).to(DEVICE)
        loss = self.criterion(y,rewards_predicted)
        loss.backward()

        print(f"Loop {self.iter}: {loss.item()}, epsilon : {self.policy_epsgreedy.eps}")
        self.policy_optimizer.step()
        if self.iter % SAVE_EVERY==0:
            torch.save(self.policy_model.state_dict(), "./weights")
        
    def loop(self, nb_epoch):

        for _ in range(nb_epoch):
            self.iter+=1
            #self.memoire = Memory()
            self.etape1()
            self.policy_epsgreedy.eps=max(EPS_DECAY*self.policy_epsgreedy.eps, EPS_MIN)
        pass
        pygame.quit()