import pickle
import numpy as np
import time as timer
import os
import argparse
from mjrl.algos.discriminator import Discriminator
from mjrl.utils.wandb import init_wandb
from common import Envs

parser = argparse.ArgumentParser(description='Train discriminator.')
parser.add_argument('--itr', type=int, default=10000, help='training epochs')
parser.add_argument('--wandb_activate', action='store_true', help='activate wandb for logging')
parser.add_argument('--wandb_entity', type=str, default='', help='wandb entity')
parser.add_argument('--wandb_project', type=str, default='', help='wandb project')
parser.add_argument('--wandb_group', type=str, default='', help='wandb group')
parser.add_argument('--wandb_name', type=str, default='', help='wandb name')
parser.add_argument('--save_id', type=str, default='0', help='identification number for each run')
parser.add_argument('--leave_one_out', type=str, default=None, help='leave one environment for test, the rest for training')

args = parser.parse_args()
if args.wandb_activate:
    if len(args.wandb_project) == 0:
        args.wandb_project = 'hand_dapg'
    if len(args.wandb_group) == 0:
        args.wandb_group = ''
    if len(args.wandb_name) == 0:
        if args.leave_one_out is not None:
            args.wandb_name = f'discriminator_no_{args.leave_one_out}_'+args.save_id
        else:
            args.wandb_name = 'discriminator_'+args.save_id
    init_wandb(args)
os.makedirs('./model', exist_ok=True) # data saving dir

ts = timer.time()

cwd = os.getcwd()
print(cwd)
# load data
if args.leave_one_out is not None:
    envs = Envs
    envs.remove(args.leave_one_out)
model = Discriminator(itr=args.itr, save_logs=True, log_dir=f'./discriminator_no_{args.leave_one_out}')

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

model.train(model_path=f'./model/no_{str(args.leave_one_out)}/')
print("time taken = %f" % (timer.time()-ts))
