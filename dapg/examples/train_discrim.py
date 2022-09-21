import pickle
import numpy as np
import time as timer
import os
import argparse
from mjrl.algos.discriminator import Discriminator
from mjrl.utils.wandb import init_wandb


parser = argparse.ArgumentParser(description='Train discriminator.')
parser.add_argument('--itr', type=int, default=10000, help='training epochs')
parser.add_argument('--wandb_activate', type=bool, default=False, help='activate wandb for logging')
parser.add_argument('--wandb_entity', type=str, default='', help='wandb entity')
parser.add_argument('--wandb_project', type=str, default='', help='wandb project')
parser.add_argument('--wandb_group', type=str, default='', help='wandb group')
parser.add_argument('--wandb_name', type=str, default='', help='wandb name')
parser.add_argument('--save_id', type=str, default='0', help='identification number for each run')

args = parser.parse_args()
if args.wandb_activate:
    if len(args.wandb_project) == 0:
        args.wandb_project = 'hand_dapg'
    if len(args.wandb_group) == 0:
        args.wandb_group = ''
    if len(args.wandb_name) == 0:
        args.wandb_name = 'discriminator_'+args.save_id
    init_wandb(args)
os.makedirs('./model', exist_ok=True) # data saving dir

ts = timer.time()

cwd = os.getcwd()
print(cwd)
# load data
envs = ['pen-v0', 'door-v0', 'hammer-v0']
model = Discriminator(itr=int(1e5))

for env in envs:
    try:
        # load rl collected paths
        rl_data_dir = f"collect_data/data/{env}"
        with open(rl_data_dir+'.pkl', 'rb') as f:
            rl_paths = pickle.load(f)

        # load demo paths
        demo_data_dir = f'../demonstrations/{env}_demos'
        with open(demo_data_dir+'.pickle', 'rb') as f:
            demo_paths = pickle.load(f)
        print(f"Load {len(rl_paths)} RL paths and {len(demo_paths)} demo paths for env {env}.")
        model.process_data(env, true_paths=demo_paths, fake_paths=rl_paths)
    except:
        print(f"Fail to load data for env: {env}")

print("========================================")
print("Starting discriminator training phase")
print("========================================")

model.train()
print("time taken = %f" % (timer.time()-ts))
