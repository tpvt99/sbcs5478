from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper, \
    DiscreteWrapper, PositiveVelocityActionWrapper, InfoWrapperEval
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



#results_dir = osp.join(PROJECT_PATH, "results", "dqn", "2021-04-20_dqn", "2021-04-20_23-38-53_dqn") # original map1 train again
results_dir = osp.join(PROJECT_PATH, "results", "dqn", "2021-04-22_dqn", "2021-04-22_17-41-00_dqn") # original map2 train again

SEED = 2

with open(osp.join(results_dir, "config.json"), 'r') as f:
    custom_params = json.load(f)

env = DuckietownEnv(
            map_name=custom_params['map'],
            domain_rand=False,
            draw_bbox=False,
            max_steps=1500,
            seed=SEED
            )

#env = DtRewardPosAngle(env)
#env = DtRewardVelocity (env)
#env = RewardWrapper(env)
env = InfoWrapperEval(env)
env = ResizeWrapper(env, shape=(60, 80, 3))


if custom_params['discrete']:
    env = DiscreteWrapper(env)

if custom_params['USING_VAE']:
    env = NormalizeWrapper(env) # No need to use normalization if image
    env = FinalLayerObservationWrapper(env, latent_dim=custom_params['VAE_LATENT_DIM'], map=custom_params['map'])


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
    #model = DQN.load(osp.join(results_dir, "phong_best", "phong_best_221000.zip"), env=env)
    model = DQN.load(osp.join(results_dir, "rl_model_510000_steps.zip"), env=env)
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


txt_file = open(f'{custom_params["map"]}_seed{SEED}.txt', 'w')

for episode in range(1):

    obs = env.reset()

    steps = 0
    rewards = 0
    done, state = False, None
    while True:
        # Get action
        env.render()
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        discrete_action_tobe_wrote = env.unwrapped.envs[0].env.action_list[int(action)]
        print(f'Step {steps} Action {discrete_action_tobe_wrote} with Reward {reward}')

        steps += 1
        rewards += reward

        txt_file.write(f"{discrete_action_tobe_wrote[0]},{discrete_action_tobe_wrote[1]}\n")
        if done:
            break
        print('-------')

    print(f'Seed {SEED} steps {steps} and rewards {rewards}')
    print('-------')
    break
txt_file.close()