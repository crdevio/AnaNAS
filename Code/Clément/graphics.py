# Example file showing a basic pygame "game loop"
import pygame
from graphics_classes import Camera

pygame.init()
class Simulation:
    def __init__(self,res = (600,600), static_url = "",dynamic_list = [], drawing = True, dt = 0.01):
        self.running = True
        self.clock = pygame.time.Clock()
        self.clock.tick(60)
        self.screen = pygame.display.set_mode(res)
        if static_url == "":
            self.static_img = None
        else:
            self.static_img = pygame.image.load(static_url).convert()
        self.dyn_list = dynamic_list
        self.drawing = drawing
        if drawing:
            self.time_manager = lambda: self.clock.get_time()/1000.0 # in sec
            self.camera = Camera(res[0],res[1])
        else:
            self.time_manager = lambda: dt
    def update(self):
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                self.running = False
        keys = pygame.key.get_pressed()
        dt = self.time_manager()
        for e in self.dyn_list:
            # Ici on devra réfléchir car la fonction update a besoin de leur donner à chacun leur cône de vision si nécessaire. C'est un placeholder pour débug
            e.update(dt,keys) # dans les codes de update, faire keys[pygame.K_LEFT] / des trucs similaires.
        if self.drawing:
            # Je mets ça la pour pas avoir à recalculer keys dans draw.
            if keys[pygame.K_LEFT]:
                self.camera.move_left(100 * dt)
            if keys[pygame.K_RIGHT]:
                self.camera.move_right(100 * dt)
            if keys[pygame.K_UP]:
                self.camera.move_up(100 * dt)
            if keys[pygame.K_DOWN]:
                self.camera.move_down(100 * dt)
    def draw(self):
        if not self.drawing: return
        if self.static_img == None:
            print("Nathan t'as pas set static_img")
            print("Clement cette phrase ne veut rien dire")
            print("De toute facon faudra qu'on reorganise les fichiers entre eux et ya d'autres trus qui vont pas")
            return
        self.screen.fill("black")
        self.screen.blit(self.static_img,(- self.camera.x, - self.camera.y))
        for e in self.dyn_list:
            for shape in e.get_shapes():
                if shape[0] == "circle":
                    # shape = "circle",(r,g,b),(center_x,center_y),radius
                    pygame.draw.circle(self.screen, shape[1],shape[2] - (self.camera.x, self.camera.y), shape[3])
                if shape[0] == "line":
                    # shape = "line",(r,g,b),(start_x,start_y),(end_x,end_y),width
                    pygame.draw.line(self.screen,shape[1],shape[2] - (self.camera.x,self.camera.y), shape[3] - (self.camera.x,self.camera.y),shape[4])
                if shape[0] == "rect":
                    # shape = "rect", (r,g,b), (x,y,lenx,leny)
                    pygame.draw.rect(self.screen,shape[1],(shape[2][0] - self.camera.x,shape[2][1] - self.camera.y, shape[2][2],shape[2][3]))
        pygame.display.flip()

       
            
jeu = Simulation(static_url="static.png")
while jeu.running:
    jeu.update()
    jeu.draw()

pygame.quit()