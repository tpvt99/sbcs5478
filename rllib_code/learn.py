from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, RewardWrapper, FinalLayerObservationWrapper, DiscreteWrapper
import os.path as osp
from sb_code.Logger import Logger
from sys import platform
import torch
from global_configuration import PROJECT_PATH
import ray
from gym_duckietown.simulator import Simulator

if platform == 'win32':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Below is training on linux and if GPU is available
if platform == 'linux' and torch.cuda.is_available():
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(640, 480))
    display.start()

# Step 1. Initialize the environment
env = Simulator(
            map_name='map1',
            domain_rand=False,
            draw_bbox=False,
            max_steps=1500,
            seed=2
        )

# Step 3.a Our Wrapper
#env = RewardWrapper(env)
#env = ResizeWrapper(env, shape=(64, 80, 3))
#env = NormalizeWrapper(env)

ray.init(local_mode=True)
tune.run(PPOTrainer, config={"env": env, "framework": 'torch', "log_level": "INFO"})


