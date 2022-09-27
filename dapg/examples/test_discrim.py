"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.discriminator import Discriminator

import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
import gym
from collections import deque
import torch
import numpy as np
import pygame


# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--render', type=bool, default=False, help='render the scene')
parser.add_argument('--discriminator_reward', type=bool, default=False, help='with discriminator as additional reward')
parser.add_argument('--record_video', type=bool, default=False, help='whether recording the video')
parser.add_argument('--record_video_interval', type=int, default=1000, help='record video interval (episode)')
parser.add_argument('--record_video_length', type=int, default=100, help='record video length')
parser.add_argument('--wandb_activate', type=bool, default=False, help='activate wandb for logging')
parser.add_argument('--wandb_entity', type=str, default='', help='wandb entity')
parser.add_argument('--wandb_project', type=str, default='', help='wandb project')
parser.add_argument('--wandb_group', type=str, default='', help='wandb group')
parser.add_argument('--wandb_name', type=str, default='', help='wandb name')
parser.add_argument('--save_id', type=str, default='0', help='identification number for each run')

args = parser.parse_args()
# if not specified
# if args.record_video_interval is None:
#     args['record_video_interval'] = 2
# if args.record_video_length is None:
#     args['record_video_length'] = 100
print("If render, do 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so'.")
print("If record video, undo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so'.")

print(args)

JOB_DIR = args.output
if not os.path.exists(JOB_DIR):
    os.mkdir(JOB_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG', 'NPGDiscriminator']])

EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)
log_dir = str('_'.join([job_data['env'], job_data['algorithm'], args.save_id]))

# ===============================================================================
# Train Loop
# ===============================================================================

e = GymEnv(job_data['env'])
policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

if not job_data['algorithm'] in ['DAPG', 'NPGDiscriminator']:
    # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
    demo_paths = None

# ===============================================================================
# RL Agent
# ===============================================================================
rl_agent = NPG(e, policy, baseline, normalized_step_size=job_data['rl_step_size'],
        seed=job_data['seed'], save_logs=True, log_dir=log_dir, discriminator_reward=args.discriminator_reward)

if args.record_video and not args.render:
    e.on_screen = False
    record_video_interval = args.record_video_interval
    record_video_length = args.record_video_length
    # env.is_vector_env = True
    video_path = "data/videos"
    current_dir = os.getcwd()
    e = gym.wrappers.RecordVideo(e, video_path,\
            episode_trigger=lambda episode: episode % record_video_interval == 0, # record the videos every * steps
            # video_length=record_video_length) # record full episode if uncomment
            )
    print(f'Save video to: {current_dir}/{video_path}')
else:
    e.on_screen = True

env_name = ['relocate-v0', 'pen-v0', 'door-v0', 'hammer-v0'][0]
# load rl collected paths
rl_data_dir = f"collect_data/data/{env_name}"
with open(rl_data_dir+'.pkl', 'rb') as f:
    rl_paths = pickle.load(f)
# load demo paths
demo_data_dir = f'../demonstrations/{env_name}_demos'
with open(demo_data_dir+'.pickle', 'rb') as f:
    demo_paths = pickle.load(f)

hand_dim = 24

model = Discriminator()
model.load_model(path='./model/model')
feature = model.feature
discriminator = model.discriminator

obs_buffer = deque(maxlen=model.frame_num)
act_buffer = deque(maxlen=model.frame_num)

fps = 10 # render frequency
background_colour = (255, 255, 255)

def init_screen(background_colour):
    # set pygame window position
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (600,900)  # (x, y)
    pygame.init()
    
    clock = pygame.time.Clock()
    SIZE = WIDTH, HEIGHT = (400, 200)
    # screen = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
    screen = pygame.display.set_mode(SIZE)
    screen.fill(background_colour)
    myFont1 = pygame.font.SysFont("arial", 30)
    myFont2 = pygame.font.SysFont("arial", 100)
    myFont = [myFont1, myFont2]
    return screen, myFont, clock

def rollout(env, policy, num_traj=3, eval_mode=False, env_kwargs=None):
    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv) or isinstance(env.env, GymEnv): # env might have a wrapper, e.g. RecordVideo
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError


    horizon = env.horizon
    paths = []

    for ep in range(num_traj):
        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []

        o = env.reset()
    
        screen, myFont, clock = init_screen(background_colour)
        done = False
        t = 0

        path = demo_paths[ep]  # rl_paths or demo_paths
        action_seq = path['actions']
        while t < horizon and t<len(action_seq) and done != True:
            # a, agent_info = policy.get_action(o)
            # if eval_mode:
            #     a = agent_info['evaluation']
            a = action_seq[t][-hand_dim:]  # only last dims are for hand
            obs_buffer.append(o)
            act_buffer.append(a)
            env_info_base = env.get_env_infos()
            # noise = np.random.uniform(-1, 1, a.shape[0])
            # a += noise
            next_o, r, done, env_info_step = env.step(a)

            if env.on_screen:
                env.render()

            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            # agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1
            if len(obs_buffer) == model.frame_num and len(act_buffer) == model.frame_num:
                sample, _ = model.sample_processing(env.env_id, np.array(list(obs_buffer)), np.array(list(act_buffer)))
                sample = torch.FloatTensor(sample)
                x = feature(sample)
                p = discriminator(x).squeeze().detach().numpy()
                str_p = "{:.4f}". format(p)  # remove e form
                print(f"Step: {t}, discriminator output: {str_p}")

                # render discriminator score
                dt = clock.tick(fps) / 1000
                screen.fill(background_colour)  # clear screen
                fixed_text = myFont[0].render('Discriminator Output:', True, (0, 0, 0))  # (r,g,b)
                text = myFont[1].render(str_p[:4], True, (0, 0, 0))  # (r,g,b)  
                screen.blit(fixed_text, (0, 0)) # (x, y)
                screen.blit(text, (50, 50)) # (x, y)
                coef = 100
                radius = 2
                pygame.draw.rect(screen, [0, 0, 0], [340-radius, 50-radius, 40+radius, coef+radius], 2)  # borders of bar
                pygame.draw.rect(screen, [0, 0, 255], [340, coef*(1-p)+50, 40, coef*p], 0)  # (r,g,b) (left, top, width, height), line width
                pygame.display.update()


rollout(e, rl_agent.policy)