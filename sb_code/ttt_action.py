from stable_baselines3.common.env_checker import check_env
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecNormalize
import os.path as osp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
from global_configuration import PROJECT_PATH

# Below 2 lines is for Windows 10 Environment. Comment if running on other OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


env = DuckietownEnv(
            map_name='map3',
            domain_rand=False,
            draw_bbox=False,
            max_steps=1500,
            seed=2
        )

env = RewardWrapper(env)
env = ResizeWrapper(env, shape=(64, 80, 3))
env = NormalizeWrapper(env)
env = FinalLayerObservationWrapper(env, latent_dim=1028, map="map3")

results_dir = osp.join(PROJECT_PATH, "results", "ddpg", "2021-04-13_ddpg", "2021-04-13_09-18-56_ddpg")

# Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
env = make_vec_env(lambda: env, n_envs=1)
# Step 3.b Passing through Normalization and stack frame (Optional)
env = VecFrameStack(env, n_stack=4) # Use 1 for now because we use image
#env = VecNormalize.load(osp.join(results_dir, "vec_normalization.pkl"), env)

#env = VecTransposeImage(env) # Uncomment if using 3d obs
# Load the agent
model = PPO.load(osp.join(results_dir, "best_model", "best_model.zip"))
#model = PPO.load(osp.join(results_dir, "best_model.zip"))

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

    print(f'Step {steps}  reward {reward}with action {action} and its shape: {action.shape}')

    steps += 1
    rewards += reward

    if done:
        break

print(steps)
print(rewards)