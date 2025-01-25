from ia.memory import Memory
from ia.dqn import DQN
from ia.constants import *
from ia.epsgreedy import *
import ia.ia as ia
import torch
import torch.optim as optim
import torch.nn as nn
from simulation.graphics import Simulation, epsilon_dict, STATIC_URLS
from dynamic.dynamic_env import DynamicEnvironnement
import random
from dynamic.voiture import Voiture

def choose_rd_from_list(l):
    return l[random.randint(0, len(l) - 1)]

class DeepQAgent:
    #dans le TP, lr = 1e-4
    def __init__(self, T=100, game_per_epoch = 10, gamma=0.5, lr = 1e-3, weight_path = None, do_opti = True, target_update_freq = 1000, eps = None):

        self.memory = Memory()
        self.t = 0
        self.num_sim = 0
        self.game_per_epoch = game_per_epoch
        self.iter = 0
        self.T = T
        self.eps_decay = 0.
        self.policy_model = DQN(INPUT_SAMPLE, 4).to(DEVICE)
        self.target_model = DQN(INPUT_SAMPLE, 4).to(DEVICE)
        eps_start = EPS_START
        if eps != None:
            eps_start = eps
        if not do_opti: print("The model is in test mode: it will drive 100% using the model, no randomness.")
        else: print("The model is in training mode: it will start from EPS_START and decrease to EPS_MIN.")
        if weight_path != None: 
            print(f"Loading weights from {weight_path}")
            self.policy_model.load_state_dict(torch.load(weight_path, weights_only=True))
            self.target_model.load_state_dict(torch.load(weight_path, weights_only=True))
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr = lr)
        self.policy_epsgreedy = EpsilonGreedy(self.policy_model,eps_start)

        self.jeu = None
        self.global_t = 0
        self.gamma = gamma
        self.criterion = nn.HuberLoss()
        self.do_opti = do_opti
        self.update_freq_delay = 0
        self.target_update_freq = target_update_freq

    def etape1(self):
        mode = "train"

        
        """ Nathan c'est quoi ça ? Tu pourras le bouger au bon endroit stp
        if self.iter % TEST_EVRY <= len(STATIC_URLS_LIST) - 1 and self.global_t >= WARMUP_PHASE:
            mode = "test"
            static_url = STATIC_URLS_LIST[self.iter % TEST_EVRY]
            self.policy_epsgreedy.eps = 0
            print("TEST situation number", self.iter % TEST_EVRY)
        """
        
        self.jeu = Simulation(dyn_env=None)
        self.policy_epsgreedy.eps = epsilon_dict[self.jeu.static_url]
        is_terminal = False
        for _ in range(self.game_per_epoch):
            dyn_env = DynamicEnvironnement(
                lambda cone,speed,car: ia.decide(cone,speed,car,self.policy_epsgreedy,self.policy_model,self.do_opti)
            )
            starting_pos = choose_rd_from_list(STATIC_URLS[self.jeu.static_url][1])
            dyn_env.add_car(Voiture(position=starting_pos[:2], ia=True, goal=self.jeu.GOAL, orientation=starting_pos[2]))
            self.jeu.dyn_env = dyn_env
            self.t = 0
            while self.t < self.T and not is_terminal:
                self.t += 1
                self.global_t += 1
                cone, vitesse, goal, actions= self.jeu.update(self.memory, self.t, self)
                if self.t == (self.T):
                    is_terminal = True
                    print("Final Reward: ", self.memory.rewards[self.memory.mem_index-1])
                    self.eps_decay = EPS_DECAY
                rewards = ia.state_reward(dyn_env.cars[0])
                self.jeu.draw(rewards)
                if self.global_t >= WARMUP_PHASE and self.do_opti and self.global_t % MODEL_UPDATE_EVERY == 0:
                    self.optimize_model()
                for car in dyn_env.cars:
                    if car.collision:
                        print("CRASH")
                        is_terminal = True
                        self.eps_decay = EPS_DECAY
                for car in dyn_env.cars:
                    if (car.x_position-self.jeu.GOAL[0])**2 + (car.y_position-self.jeu.GOAL[1])**2 <= GOAL_RADIUS:
                        print("GOAL")
                        is_terminal = True
                        rewards = 100
                        print("Final Reward: ", self.memory.rewards[self.memory.mem_index-1])
                        self.eps_decay = 5*EPS_DECAY
                self.update_freq_delay += 1
                if self.update_freq_delay >= self.target_update_freq:
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    self.update_freq_delay = 0
                self.memory.append(cone, vitesse, goal, actions, rewards, is_terminal)

    def optimize_model(self):

        self.policy_optimizer.zero_grad()
        cones,speeds,goals,next_cones,next_speeds,next_goals,actions,rewards,terminals = self.memory.sample(BATCH_SIZE)
        mask = torch.tensor((1. - terminals.astype(float)))
        mask = torch.tensor((1. - terminals.astype(float))).to(DEVICE)
        target_value = torch.max(self.target_model(next_cones, next_speeds,next_goals), dim=1)[0].to(DEVICE)
        y = rewards.to(DEVICE) + mask * self.gamma * target_value
        y_predicted = self.policy_model(cones,speeds,goals)
        rewards_predicted = y_predicted[torch.arange(BATCH_SIZE),actions.to(DEVICE)].type(torch.float64).to(DEVICE)
        loss = self.criterion(y,rewards_predicted)
        loss.backward()

        if self.global_t % SHOW_INFO_EVERY == 0:
            print(f"Loop {self.iter}: {loss.item()}, epsilon : {self.policy_epsgreedy.eps}")
        self.policy_optimizer.step()
        
    def loop(self, nb_epoch):

        for _ in range(nb_epoch):
            self.iter += 1
            #self.memoire = Memory()
            self.etape1()
            if self.global_t >= WARMUP_PHASE:
                epsilon_dict[self.jeu.static_url] = max(epsilon_dict[self.jeu.static_url] - self.eps_decay, EPS_MIN)
            if self.iter % SAVE_EVERY == 0:
                torch.save(self.policy_model.state_dict(), "./weights")
                print("saved")
        pass
        pygame.quit()