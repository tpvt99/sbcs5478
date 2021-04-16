from stable_baselines3 import SAC, DQN, A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
import os.path as osp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from sb_code.custom_callbacks import SaveNormalization, CustomBestModelCallback
from sb_code.Logger import Logger
from sys import platform
import torch
from aido_code.env_setup import setup
import numpy as np
import random
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
    'env_config': {
        'crop_image_top': False,
        'resized_image': (64, 64),
        'action_type': 'leftright_braking',
        'reward_function': 'posangle'
    },
    # WRAPPERS
    'USING_VAE' : False, # whether to use VAE
    'VAE_LATENT_DIM': 1028,
    'FRAME_STACK' : 3,
    'USING_NORMALIZATION' : True,
    # TRAINING
    'eval_freq': 1000, # All are steps running
    'save_freq': 50000,
    'eval_episodes': 5,
    # ALGORITHMS PARAMETERS
    'algo' : 'ppo',
    'sac_parameters': {
        'buffer_size': int(1e5),
        'gradient_steps': 64,
        'train_freq': 64,
        'optimize_memory_usage': True,
        'learning_starts': 1000
    },
    'dqn_parameters': {
        'optimize_memory_usage': True,
        'buffer_size': int(5e5)
    },
    'a2c_parameters': {
        'use_sde': True,
        'normalize_advantage': True,
    },
    'ppo_parameters': {
        'batch_size': 256,
        'n_epochs': 20,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        #'sde_sample_freq': 4,
        'clip_range_vf': 0.2,
        'learning_rate': 3e-5,
        'use_sde': True,
        'clip_range': 0.4,
    }
}
custom_params['policy'] = 'MlpPolicy' if custom_params['USING_VAE'] else 'CnnPolicy'

# Step 2. Initialize the environment

env = setup(configs=custom_params)
np.random.seed(custom_params['seed'])
torch.manual_seed(custom_params['seed'])
random.seed(custom_params['seed'])

# Step 2. Check the custom environment. Must do it before any wrappers
#check_env(env)


# Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
eval_env = make_vec_env(lambda: env, n_envs=1)
env = make_vec_env(lambda: env, n_envs=2)

# Step 3.b Passing through Normalization and stack frame (Optional)

env = VecFrameStack(env, n_stack=custom_params['FRAME_STACK']) # Use 1 for now because we use image
eval_env = VecFrameStack(eval_env, n_stack=custom_params['FRAME_STACK']) # Use 1 for now because we use image

if not custom_params['USING_VAE']:
    env = VecTransposeImage(env) # Uncomment if using 3d obs
    eval_env = VecTransposeImage(eval_env)

if custom_params['USING_NORMALIZATION']:
    env = VecNormalize(env, norm_obs=True, norm_reward=True) # If using normalize, must save
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

# Step 4. Make Logger corrsponding to the name of algorithm
logger = Logger(custom_params['algo'] + '_aido')

# Step 5. Creating callbacks

checkpoint_callback = CheckpointCallback(save_freq=custom_params['save_freq'], save_path=logger.output_dir,
                                         name_prefix='rl_model')

custom_bestmodel_callback = CustomBestModelCallback(eval_env=eval_env, eval_freq=custom_params['eval_freq'], logger=logger)

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
elif custom_params['algo'] == 'a2c':
    model = A2C(policy = custom_params['policy'], env=env, verbose=1, **custom_params['a2c_parameters'], tensorboard_log=logger.output_dir)
elif custom_params['algo'] == 'ppo':
    model = PPO(policy = custom_params['policy'], env=env, verbose=1, **custom_params['ppo_parameters'], tensorboard_log=logger.output_dir)
else:
    raise ValueError("Invalid algo")
logger.save_config(custom_params)

model.learn(total_timesteps=10000000, log_interval=100, callback=callback) # Log_interval = number of episodes
