from stable_baselines3.common.env_checker import check_env
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
import os.path as osp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EventCallback, BaseCallback
from sb_code.Logger import Logger

# Below 2 lines is for Windows 10 Environment. Comment if running on other OS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PROJECT_PATH = osp.abspath(osp.dirname(osp.dirname(__file__)))

# Step 1. Initialize the environment
env = DuckietownEnv(
            map_name='map1',
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
env = FinalLayerObservationWrapper(env)

#eval_env = env # Make a seperate evaluation environment without vectorized version as in train env

# Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
env = make_vec_env(lambda: env, n_envs=1)
# Step 3.b Passing through Normalization and stack frame (Optional)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=100.) # If using normalize, must save
env = VecFrameStack(env, n_stack=1) # Use 1 for now because we use image
#env = VecTransposeImage(env) # Uncomment if using 3d obs

# Step 4. Make Logger corrsponding to the name of algorithm
logger = Logger("sac")

# Step 5. Creating callbacks
class SaveStaticsits(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: (Optional[BaseCallback]) Callback that will be called
        when an event is triggered.
    :param verbose: (int)
    """
    def __init__(self, env = None, save_path=None):
        super(SaveStaticsits, self).__init__()
        self.save_path = save_path
        self.env = env

    def _on_step(self) -> bool:
        self.env.save(self.save_path)
        return True

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=logger.output_dir,
                                         name_prefix='rl_model')
savestats_callback = SaveStaticsits(env=env, save_path=osp.join(logger.output_dir, "stats")) # If using normalize, must create this callback
eval_callback = EvalCallback(eval_env = env, n_eval_episodes=10, callback_on_new_best=savestats_callback,
                             eval_freq=5000,
                             best_model_save_path=osp.join(logger.output_dir, "best_model"),
                             log_path=osp.join(logger.output_dir, "results"))


callback = CallbackList([checkpoint_callback, eval_callback])

model = SAC(policy = "MlpPolicy", env=env, verbose=1, buffer_size=int(1e6))
model.learn(total_timesteps=1000000, log_interval=4, callback=callback)
