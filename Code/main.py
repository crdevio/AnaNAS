from ia.deepq_agent import DeepQAgent
import argparse
from ia.constants import *
parser = argparse.ArgumentParser(
                    prog='anaNAS',
                    description='A simple car simulation that learn how to drive with RL',
                    epilog='Text at the bottom of help')
parser.add_argument('-l', '--load', type=str, default=None)
parser.add_argument('-s', '--save', type=str, default=None)
parser.add_argument('-e', '--epsilon', type=float, default=None)
parser.add_argument('-t','--test', type=bool, default=False)
args = parser.parse_args()

d = DeepQAgent(game_per_epoch=1, T=T_VALUE, gamma=0.99, weight_path=args.load, save_path=args.save, do_opti=(not args.test), eps=args.epsilon)
d.loop(NB_EPOCH)