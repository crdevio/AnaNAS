# Format
## `nom_fichier`
Number Epochs: X
Last Epsilon: X
Cone Height: X
Cone Width: X
Mem Size: X
Model Type: [DEFAULT]

# Saved models list

## eps0p1_little_cone
Number Epochs: 4160
Last Epsilon: 0.1
Cone Height: 75
Cone Width: 200
Mem Size: 30 000
Model Type: [DEFAULT]

## longtrain_circuit1
Number Epochs: 1470
Last Epsilon: 0.1
Cone Heights: 75
COne Width: 300
Mem Size: 30000
EPS Decay: 5e-4
Map: le premier circuit avec 2 goals.
Résultat: il a bien appris le deuxième mais pas du tout le premier. je pense que c'est dû à la différence de distance des objectifs.

## 2conv2map
Number Epochs: 2945
Last Epsilon: 0.1
Cone Heights: 75
COne Width: 300
Mem Size: 30000
EPS Decay: 5e-4
MODELE: DQN AVEC 2 CONV !
Map: le premier circuit avec 2 goals.
Résultat: il a bien appris le deuxième mais pas du tout le premier. je pense que c'est dû à la différence de distance des objectifs.


## circuit_4goal
Nb epochs: 530
Résultat: Il a bien appris certains côtés, et surtout il peut revenir en arrière quand il se rate !
SAVE_EVERY = 10
MODEL_UPDATE_EVERY = 4
NB_EPOCH = 100000
BATCH_SIZE = 32
SHOW_INFO_EVERY = 500
WARMUP_PHASE = 2000  #20 000 dans le TP
TEST_EVRY = 100
GOAL_RADIUS = 200
LARGEUR_CONE =75
LONGUEUR_CONE = 200
INPUT_SAMPLE = 2 * LARGEUR_CONE * LONGUEUR_CONE
MEM_SIZE = 10000 # 100 000 dans le TP.
T_VALUE = 1000
EPS_START = 1.
EPS_DECAY = 1e-3       #dans le TP 1e-5
EPS_MIN = 0.1
EPS_TEST = 0.4

