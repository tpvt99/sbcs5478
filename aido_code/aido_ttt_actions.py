
from stable_baselines3 import SAC, A2C, DQN, PPO
import os.path as osp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
from global_configuration import PROJECT_PATH
import json
from aido_code.env_setup import setup_for_test
# Below 2 lines is for Windows 10 Environment. Comment if running on other OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

results_dir = osp.join(PROJECT_PATH, "results", "ppo_aido", "2021-04-16_ppo_aido", "2021-04-16_10-39-38_ppo_aido")

with open(osp.join(results_dir, "config.json"), 'r') as f:
    custom_params = json.load(f)

env = setup_for_test(custom_params)

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
    model = PPO.load(osp.join(results_dir, "best_model", "best_model"))
    #model = PPO.load(osp.join(results_dir, "phong_best", "phong_best"))
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
    action, state = model.predict(obs, state=state, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f'Step {steps} Action {action} with Reward {reward} with info {info}')

    steps += 1
    rewards += reward

    if done:
        break

print(steps)
print(rewards)