from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper, DiscreteWrapper, PositiveVelocityActionWrapper
from aido_code.reward_wrappers import DtRewardPosAngle, DtRewardVelocity
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, A2C, DQN, PPO
import os.path as osp

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecTransposeImage
from global_configuration import PROJECT_PATH
import json
# Below 2 lines is for Windows 10 Environment. Comment if running on other OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#results_dir = osp.join(PROJECT_PATH, "results", "dqn", "2021-04-19_dqn", "2021-04-19_14-52-12_dqn") # original map1
results_dir = osp.join(PROJECT_PATH, "results", "dqn", "2021-04-19_dqn", "2021-04-19_18-04-39_dqn") # finetune map1
#results_dir = osp.join(PROJECT_PATH, "results", "dqn", "2021-04-19_dqn", "2021-04-19_20-01-28_dqn") # finetune map1 with original reward
#results_dir = osp.join(PROJECT_PATH, "results", "ppo", "2021-04-19_ppo", "2021-04-19_16-59-46_ppo")




with open(osp.join(results_dir, "config.json"), 'r') as f:
    custom_params = json.load(f)

env = DuckietownEnv(
            map_name=custom_params['map'],
            domain_rand=False,
            draw_bbox=False,
            max_steps=1500,
            #seed=custom_params['seed']
            seed=2
            )

#env = DtRewardPosAngle(env)
#env = DtRewardVelocity (env)
#env = RewardWrapper(env)
env = ResizeWrapper(env, shape=(60, 80, 3))


if custom_params['discrete']:
    env = DiscreteWrapper(env)

if custom_params['USING_VAE']:
    env = NormalizeWrapper(env) # No need to use normalization if image
    env = FinalLayerObservationWrapper(env, latent_dim=1028, map=custom_params['map'])


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
    #model = DQN.load(osp.join(results_dir, "best_model", "best_model.zip"), env=env)
    #model = DQN.load(osp.join(results_dir, "phong_best", "phong_best.zip"), env=env)
    model = DQN.load(osp.join(results_dir, "rl_model_150000_steps.zip"), env=env)
elif custom_params['algo'] == 'ppo':
    #model = PPO.load(osp.join(results_dir, "best_model", "best_model"), env=env, seed=custom_params['seed'])
    model = PPO.load(osp.join(results_dir, "phong_best", "phong_best"), env=env)
else:
    raise ValueError("Error model")

# Load the saved statistics
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

seed_lists = [2, 3, 5, 9, 12]

for episode in range(len(seed_lists)):

    #env.seed(9)
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

        if done:
            break

        steps += 1
        rewards += reward

    print(f'Seed {seed_lists[episode]} steps {steps} and rewards {rewards}')
    print('-------')
    break
