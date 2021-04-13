from stable_baselines3.common.env_checker import check_env
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, PPO, DDPG, TD3
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
import os.path as osp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EventCallback, BaseCallback
from sb_code.Logger import Logger
from sys import platform
import torch
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

# Step 1. Initialize the environment
env = DuckietownEnv(
            map_name='map3',
            domain_rand=False,
            draw_bbox=False,
            max_steps=1500,
            seed=2
        )

# Step 2. Check the custom environment. Must do it before any wrappers
check_env(env)

# Step 3.a Our Wrapper
env = RewardWrapper(env)
env = ResizeWrapper(env, shape=(64, 80, 3))
env = NormalizeWrapper(env)
env = FinalLayerObservationWrapper(env, latent_dim=1028, map="map3")

#eval_env = env # Make a seperate evaluation environment without vectorized version as in train env

# Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
env = make_vec_env(lambda: env, n_envs=1)
# Step 3.b Passing through Normalization and stack frame (Optional)

env = VecFrameStack(env, n_stack=4) # Use 1 for now because we use image
#env = VecTransposeImage(env) # Uncomment if using 3d obs
#env = VecNormalize(env, norm_obs=True, norm_reward=True) # If using normalize, must save

# Step 4. Make Logger corrsponding to the name of algorithm
logger = Logger("ppo")

# Step 5. Creating callbacks
class SaveNormalization(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: (Optional[BaseCallback]) Callback that will be called
        when an event is triggered.
    :param verbose: (int)
    """
    def __init__(self, env = None, save_path=None):
        super(SaveNormalization, self).__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.model.get_vec_normalize_env() is not None:
            self.model.get_vec_normalize_env().save(self.save_path)
            if self.verbose > 1:
                print(f"Saving VecNormalize to {self.save_path}")
        return True

checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=logger.output_dir,
                                         name_prefix='rl_model')

savestats_callback = SaveNormalization(save_path=osp.join(logger.output_dir, "vec_normalization.pkl")) # If using normalize, must create this callback

eval_callback = EvalCallback(eval_env = env, n_eval_episodes=5, callback_on_new_best=savestats_callback,
                             eval_freq=2000,
                             best_model_save_path=osp.join(logger.output_dir, "best_model"),
                             log_path=osp.join(logger.output_dir, "results"))

callback = CallbackList([checkpoint_callback, eval_callback])

model = PPO(policy = "MlpPolicy", env=env, verbose=1, tensorboard_log=logger.output_dir)
model.learn(total_timesteps=100000, log_interval=10, callback=callback) # Log_interval = number of episodes
