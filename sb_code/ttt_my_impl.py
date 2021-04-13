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

# Below is training on linux and if GPU is available


PROJECT_PATH = osp.abspath(osp.dirname(osp.dirname(__file__)))

# Step 1. Initialize the environment
env = gym.make('MountainCarContinuous-v0')

# Step 2. Check the custom environment. Must do it before any wrappers
check_env(env)

results_dir = osp.join(PROJECT_PATH, "results", "ppo", "2021-04-12_ppo", "2021-04-12_23-12-27_ppo")

#env = make_vec_env(lambda: env, n_envs=1)
# Step 3.b Passing through Normalization and stack frame (Optional)
#env = VecFrameStack(env, n_stack=2) # Use 1 for now because we use image
#env = VecNormalize.load(osp.join(results_dir, "vec_normalization.pkl"), env)

# Step 5. Creating callbacks
model = PPO.load(osp.join(results_dir, "best_model", "best_model.zip"))

# Load the saved statistics
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

obs = env.reset()
steps = 0
rewards = 0
done, state = False, None
while True:
    # Get action
    env.render()
    action, state = model.predict(obs, state=state, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f'Step {steps} Reward {reward} with info {info}')

    steps += 1
    rewards += reward

    if done:
        break

print(steps)
print(rewards)