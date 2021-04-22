#from stable_baselines3.common.env_checker import check_env
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from gym_duckietown.simulator import Simulator
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper, \
    DiscreteWrapper, PositiveVelocityActionWrapper, Map1EvalRewardWrapper
from aido_code.reward_wrappers import DtRewardPosAngle, DtRewardVelocity, DtRewardWrapperDistanceTravelled
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, DQN, A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
import os.path as osp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from sb_code.custom_callbacks import SaveNormalization, CustomBestModelCallback
from sb_code.Logger import Logger
from sys import platform
import torch
import random
import shutil

# Below 2 lines is for Windows 10 Environment. Comment if running on other OS
if platform == 'win32':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Below is training on linux and if GPU is available
if platform == 'linux' and torch.cuda.is_available():
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(640, 480))
    display.start()

PROJECT_PATH = osp.abspath(osp.dirname(osp.dirname(__file__)))


# Step 1. Initialize some parameters
custom_params = {
    # ENVIORNMENT Set up
    'map': 'map2',
    'ep_len': 1500,
    'seed': 2,
    # WRAPPERS
    'USING_VAE' : True, # whether to use VAE
    'VAE_LATENT_DIM': 512,
    'FRAME_STACK' : 20,
    'USING_NORMALIZATION' : True,
    'discrete': True,
    # TRAINING
    'eval_freq': 5000, # All are steps running
    'save_freq': 30000,
    'eval_episodes': 5,
    'restore': False,
    'load_path': osp.join(PROJECT_PATH, "results", "dqn", "2021-04-19_dqn", "2021-04-19_14-52-12_dqn", "phong_best", "phong_best"),
    # ALGORITHMS PARAMETERS
    'algo' : 'dqn',
    'sac_parameters': {
        'buffer_size': int(1e5),
        'gradient_steps': 64,
        'train_freq': 64,
        'optimize_memory_usage': True,
        'learning_starts': 1000
    },
    'dqn_parameters': {
        'optimize_memory_usage': True,
        'buffer_size': int(2e5),
        'exploration_fraction': 0.25,
        'gamma': .98,
        'train_freq': (1, 'episode'),
        'gradient_steps': 10,
        'target_update_interval': 50000
    },
    'a2c_parameters': {
        'use_sde': True,
        'normalize_advantage': True,
    },
    'ppo_parameters': {
        'batch_size': 256,
        'n_epochs': 20,
        'gamma': 0.99,
        'gae_lambda': 0.9,
        'sde_sample_freq': 4,
        'learning_rate': 3e-5,
        'use_sde': True,
        'clip_range': 0.4
        #'policy_kwargs': dict(log_std_init=-2,ortho_init=False)
    }
}
custom_params['policy'] = 'MlpPolicy' if custom_params['USING_VAE'] else 'CnnPolicy'

# Step 2. Initialize the environment

np.random.seed(custom_params['seed'])
torch.manual_seed(custom_params['seed'])
random.seed(custom_params['seed'])

def setup_env():
    env = DuckietownEnv(
        map_name=custom_params['map'],
        domain_rand=False,
        draw_bbox=False,
        max_steps=custom_params['ep_len'],
        seed=custom_params['seed']
    )
    train_env = env
    eval_env = env

    # My Wrappers
    #train_env = DtRewardWrapperDistanceTravelled(train_env)
    #train_env = DtRewardVelocity(train_env)
    train_env = RewardWrapper(train_env)
    train_env = ResizeWrapper(train_env, shape=(60, 80, 3))

    if custom_params['map'] == 'map1':
        eval_env = Map1EvalRewardWrapper(eval_env)
    eval_env = ResizeWrapper(eval_env, shape=(60, 80, 3))

    if custom_params['discrete']:
        train_env = DiscreteWrapper(train_env)
        eval_env = DiscreteWrapper(eval_env)

    if custom_params['USING_VAE']:
        train_env = NormalizeWrapper(train_env)  # No need to use normalization if image
        train_env = FinalLayerObservationWrapper(train_env, latent_dim=custom_params['VAE_LATENT_DIM'], map=custom_params['map'])

        eval_env = NormalizeWrapper(eval_env)  # No need to use normalization if image
        eval_env = FinalLayerObservationWrapper(eval_env, latent_dim=custom_params['VAE_LATENT_DIM'],
                                                 map=custom_params['map'])

    return train_env, eval_env

# Step 3. Check the custom environment. Must do it before any wrappers
#check_env(env)

# Step 3.a Our Wrapper
train_env, eval_env = setup_env()

# Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
eval_env = make_vec_env(lambda: eval_env, n_envs=1)
if custom_params['algo'] == 'dqn':
    env = make_vec_env(lambda: train_env, n_envs=1)
else:
    env = make_vec_env(lambda: train_env, n_envs=1)

# Step 3.b Passing through Normalization and stack frame (Optional)

env = VecFrameStack(env, n_stack=custom_params['FRAME_STACK']) # Use 1 for now because we use image
eval_env = VecFrameStack(eval_env, n_stack=custom_params['FRAME_STACK']) # Use 1 for now because we use image

if not custom_params['USING_VAE']:
    env = VecTransposeImage(env) # Uncomment if using 3d obs
    eval_env = VecTransposeImage(eval_env)

if custom_params['USING_NORMALIZATION']:
    env = VecNormalize(env, norm_obs=True, norm_reward=False) # If using normalize, must save
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

# Step 4. Make Logger corrsponding to the name of algorithm
logger = Logger(custom_params['algo'])
shutil.copy(osp.join(PROJECT_PATH, "sb_code", "wrapper.py"), logger.output_dir)

# Step 5. Creating callbacks

checkpoint_callback = CheckpointCallback(save_freq=custom_params['save_freq'], save_path=logger.output_dir,
                                         name_prefix='rl_model')

custom_bestmodel_callback = CustomBestModelCallback(eval_env=eval_env, eval_freq=custom_params['eval_freq'], logger=logger, custom_params=custom_params)

savestats_callback = SaveNormalization(save_path=osp.join(logger.output_dir, "vec_normalization.pkl")) # If using normalize, must create this callback

eval_callback = EvalCallback(eval_env = eval_env, n_eval_episodes=custom_params['eval_episodes'], callback_on_new_best=savestats_callback,
                             eval_freq=custom_params['eval_freq'],
                             best_model_save_path=osp.join(logger.output_dir, "best_model"),
                             log_path=osp.join(logger.output_dir, "results"))

callback = CallbackList([checkpoint_callback, eval_callback, custom_bestmodel_callback])

if custom_params['algo'] == 'sac':
    model = SAC(policy = custom_params['policy'], env=env, verbose=1, **custom_params['sac_parameters'], tensorboard_log=logger.output_dir)
elif custom_params['algo'] == 'dqn':
    model = DQN(policy = custom_params['policy'], env=env, verbose=1, **custom_params['dqn_parameters'], tensorboard_log=logger.output_dir)
    if custom_params['restore'] == True:
        print(f'Loading retrained model at {custom_params["load_path"]}')
        model = DQN.load(custom_params['load_path'], env=env, exploration_initial_eps=0.7)
elif custom_params['algo'] == 'a2c':
    model = A2C(policy = custom_params['policy'], env=env, verbose=1, **custom_params['a2c_parameters'], tensorboard_log=logger.output_dir)
elif custom_params['algo'] == 'ppo':
    model = PPO(policy = custom_params['policy'], env=env, verbose=1, **custom_params['ppo_parameters'], tensorboard_log=logger.output_dir)
else:
    raise ValueError("Invalid algo")
logger.save_config(custom_params)

if custom_params['algo'] == 'dqn':
    model.learn(total_timesteps=10000000, log_interval=1000, callback=callback) # Log_interval = number of episodes
else:
    model.learn(total_timesteps=10000000, log_interval=100, callback=callback) # Log_interval = number of episodes

