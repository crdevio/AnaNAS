from graphics import DeepQAgent,NB_EPOCH
import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('-f', '--filename')
args = parser.parse_args()

d = DeepQAgent(game_per_epoch=1, T=300, gamma=0.99, weight_path=args.filename,do_opti=True)
d.loop(NB_EPOCH)