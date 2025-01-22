import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {DEVICE} device")

SAVE_EVERY = 10
MODEL_UPDATE_EVERY = 4
INPUT_SAMPLE = 5000
NB_EPOCH = 10000
BATCH_SIZE = 32
SHOW_INFO_EVERY = 500
WARMUP_PHASE = 2000  #20 000 dans le TP
TEST_EVRY = 100
GOAL_RADIUS = 200