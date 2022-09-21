import pickle
import numpy as np
import time as timer
import os
from mjrl.algos.discriminator import Discriminator

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
