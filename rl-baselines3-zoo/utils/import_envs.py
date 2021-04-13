from gym.envs.registration import register
from gym.envs.registration import registry

try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

# My Custome environments
if 'DuckietownEnv-v1' not in registry.env_specs:
    register(
        id='DuckietownEnv-v1',
        entry_point='gym_duckietown.envs.duckietown_env:DuckietownEnv',
        kwargs={
            'map_name' : 'map1',
            'domain_rand' : False,
            'draw_bbox' : False,
            'max_steps' : 1500,
            'seed' : 2
        })