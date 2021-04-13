from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
import os.path as osp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EventCallback, BaseCallback
from sb_code.Logger import Logger
from sys import platform
import torch
import gym
from stable_baselines3.common.evaluation import evaluate_policy
# Below is training on linux and if GPU is available
from stable_baselines3.sac.policies import MlpPolicy

PROJECT_PATH = osp.abspath(osp.dirname(osp.dirname(__file__)))

# Step 1. Initialize the environment
env = gym.make('CartPole-v1')

# Step 2. Check the custom environment. Must do it before any wrappers
check_env(env)


#eval_env = env # Make a seperate evaluation environment without vectorized version as in train env

# Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
env = make_vec_env(lambda: env, n_envs=1)
# Step 3.b Passing through Normalization and stack frame (Optional)

env = VecFrameStack(env, n_stack=1) # Use 1 for now because we use image
#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.) # If using normalize, must save

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
                             eval_freq=1000,
                             best_model_save_path=osp.join(logger.output_dir, "best_model"),
                             log_path=osp.join(logger.output_dir, "results"))

callback = CallbackList([checkpoint_callback, eval_callback])

model = PPO('MlpPolicy', env=env, verbose=1)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Before mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


model.learn(total_timesteps=150000, log_interval=5, callback=callback) # Log_interval = number of episodes
