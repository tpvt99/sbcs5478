from stable_baselines3.common.env_checker import check_env
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper, DiscreteWrapper
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, A2C, DQN, PPO
from stable_baselines3.common.vec_env import VecNormalize
import os.path as osp
from gym_duckietown.simulator import Simulator

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
from global_configuration import PROJECT_PATH
import json
# Below 2 lines is for Windows 10 Environment. Comment if running on other OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from gym.wrappers.resize_observation import ResizeObservation

results_dir = osp.join(PROJECT_PATH, "results", "ppo_racing", "2021-04-13_ppo_racing", "2021-04-13_16-17-49_ppo_racing")

with open(osp.join(results_dir, "config.json"), 'r') as f:
    custom_params = json.load(f)

env = gym.make('CarRacing-v0')
env = ResizeObservation(env, 64)


if custom_params['algo'] == 'dqn':
    env = DiscreteWrapper(env)

# Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
env = make_vec_env(lambda: env, n_envs=1)
# Step 3.b Passing through Normalization and stack frame (Optional)

env = VecFrameStack(env, n_stack=custom_params['FRAME_STACK']) # Use 1 for now because we use image
if not custom_params['USING_VAE']:
    env = VecTransposeImage(env) # Uncomment if using 3d obs
if custom_params['USING_NORMALIZATION']:
    env = VecNormalize.load(osp.join(results_dir, "vec_normalization.pkl"), env)

# Load the agent
if custom_params['algo'] == 'sac':
    model = SAC.load(osp.join(results_dir, "best_model", "best_model.zip"))
elif custom_params['algo'] == 'a2c':
    model = A2C.load(osp.join(results_dir, "best_model", "best_model.zip"))
elif custom_params['algo'] == 'dqn':
    model = DQN.load(osp.join(results_dir, "best_model", "best_model.zip"))
elif custom_params['algo'] == 'ppo':
    model = PPO.load(osp.join(results_dir, "best_model", "best_model.zip"))

else:
    raise ValueError("Error model")

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
    action, state = model.predict(obs, state=state, deterministic=False)
    obs, reward, done, info = env.step(action)
    print(f'Step {steps} Action {action} with Reward {reward} with info {info}')

    steps += 1
    rewards += reward

    if done:
        break

print(steps)
print(rewards)