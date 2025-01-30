import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {DEVICE} device")

SAVE_EVERY = 10
MODEL_UPDATE_EVERY = 4
NB_EPOCH = 100000
BATCH_SIZE = 32
SHOW_INFO_EVERY = 500
WARMUP_PHASE = 2000  #20 000 dans le TP
TEST_EVRY = 100
GOAL_RADIUS = 200
LARGEUR_CONE =50
LONGUEUR_CONE = 150
INPUT_SAMPLE = 2 * LARGEUR_CONE * LONGUEUR_CONE
MEM_SIZE = 10000 # 100 000 dans le TP.
T_VALUE = 1000
EPS_START = 1.
EPS_DECAY = 1e-3       #dans le TP 1e-5
EPS_MIN = 0.1
EPS_TEST = 0.4