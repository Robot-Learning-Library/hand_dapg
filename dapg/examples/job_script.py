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
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
from mjrl.utils.wandb import init_wandb
from mjrl.utils.fc_network import FCNetwork
from mjrl.algos.npg_discrim import NPGDiscriminator

import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
import gym

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--render', action='store_true', help='render the scene')
parser.add_argument('--discriminator_reward', action='store_true', help='with discriminator as additional reward')
parser.add_argument('--adaptive_scale', action='store_true', help='whether using adaptive scale of discriminator reward')
parser.add_argument('--warm_up', type=int, default=0, help='warm up steps without discriminator reward')
parser.add_argument('--record_video', action='store_true', help='whether recording the video')
parser.add_argument('--record_video_interval', type=int, default=1000, help='record video interval (episode)')
parser.add_argument('--record_video_length', type=int, default=100, help='record video length')
parser.add_argument('--wandb_activate', action='store_true', help='activate wandb for logging')
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
job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']
EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)
print(f'Configurations: \n ------------------------------------------------\n{job_data}')
print(args, args.discriminator_reward)
if args.discriminator_reward:
    log_dir = str('_'.join([job_data['env'], job_data['algorithm'], args.save_id, 'reg_reward']))
else:
    log_dir = str('_'.join([job_data['env'], job_data['algorithm'], args.save_id]))
if args.wandb_activate:
    if len(args.wandb_project) == 0:
        args.wandb_project = 'hand_dapg'
    if len(args.wandb_group) == 0:
        args.wandb_group = ''
    if len(args.wandb_name) == 0:
        args.wandb_name = log_dir
    init_wandb(args)

# ===============================================================================
# Train Loop
# ===============================================================================

e = GymEnv(job_data['env'])
policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

# Get demonstration data if necessary and behavior clone
if job_data['algorithm'] != 'NPG':
    print("========================================")
    print("Collecting expert demonstrations")
    print("========================================")
    demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))

    bc_agent = BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                  lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
    in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
    bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
    bc_agent.set_variance_with_data(out_scale)

    ts = timer.time()
    print("========================================")
    print("Running BC with expert demonstrations")
    print("========================================")
    bc_agent.train()
    print("========================================")
    print("BC training complete !!!")
    print("time taken = %f" % (timer.time() - ts))
    print("========================================")

    if job_data['eval_rollouts'] >= 1:
        score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
        print("Score with behavior cloning = %f" % score[0][0])

if not job_data['algorithm'] in ['DAPG', 'NPGDiscriminator']:
    # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
    demo_paths = None

# ===============================================================================
# RL Loop
# ===============================================================================
if job_data['algorithm'] in ['NPG', 'BCRL']:
    rl_agent = NPG(e, policy, baseline, normalized_step_size=job_data['rl_step_size'],
            seed=job_data['seed'], save_logs=True, log_dir=log_dir, discriminator_reward=args.discriminator_reward, adaptive_scale=args.adaptive_scale)
elif job_data['algorithm'] == 'DAPG':
    rl_agent = DAPG(e, policy, baseline, demo_paths,
                    normalized_step_size=job_data['rl_step_size'],
                    lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                    seed=job_data['seed'], save_logs=True, log_dir=log_dir,
                    discriminator_reward=args.discriminator_reward
                    )
elif job_data['algorithm'] == 'NPGDiscriminator':
    discriminator = FCNetwork(e.spec.observation_dim+e.spec.action_dim, 1, hidden_sizes=job_data['policy_size'], output_nonlinearity='sigmoid')
    rl_agent = NPGDiscriminator(e, policy, baseline, discriminator, demo_paths, normalized_step_size=job_data['rl_step_size'],
                seed=job_data['seed'], save_logs=True, log_dir=log_dir)
else:
    raise NotImplementedError

print("========================================")
print("Starting reinforcement learning phase")
print("========================================")

ts = timer.time()
train_agent(job_name=JOB_DIR,
            agent=rl_agent,
            parser_args=args,
            seed=job_data['seed'],
            niter=job_data['rl_num_iter'],
            gamma=job_data['rl_gamma'],
            gae_lambda=job_data['rl_gae'],
            num_cpu=job_data['num_cpu'],
            sample_mode='trajectories',
            num_traj=job_data['rl_num_traj'],
            save_freq=job_data['save_freq'],
            evaluation_rollouts=job_data['eval_rollouts'])
print("time taken = %f" % (timer.time()-ts))
