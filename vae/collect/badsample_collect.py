
import matplotlib.pyplot as plt
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper
import numpy as np

def get_observations(num_samples, seed):
    env = DuckietownEnv(
        map_name = 'map3',
        domain_rand = False,
        draw_bbox = False,
        max_steps = 2000,
        seed = seed
    )
    env = ResizeWrapper(env, shape=(64, 80, 3))
    env = NormalizeWrapper(env)

    samples = num_samples
    time_steps = 0
    obs_array = np.zeros(shape=(samples, 64, 80, 3))
    while time_steps < samples:
        # collect trajectories
        env.seed(seed)
        obs = env.reset()
        obs_array[time_steps] = obs
        time_steps += 1
        while True:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            rollout_done = done or time_steps >= samples
            if rollout_done:
                break

            obs_array[time_steps] = obs
            time_steps += 1

        seed+=1
        print(time_steps)

    return obs_array

samples = 12000
observation = get_observations(samples, 0)
observation = np.stack(observation, axis=0)
np.save('badsamples.npy', observation)