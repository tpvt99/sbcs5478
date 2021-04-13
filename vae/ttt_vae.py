from vae.model_torch import AutoEncoder
import math
import matplotlib.pyplot as plt
import gym
from gym import spaces
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from sb_code.wrapper import NormalizeWrapper, ResizeWrapper
import numpy as np
import torch
from torchvision import transforms
import os.path as osp
from global_configuration import PROJECT_PATH
from sys import platform

if platform == 'win32':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_observations(num_samples, seed):
    env = DuckietownEnv(
        map_name = 'map1',
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

samples = 10


train_data1 = np.load('./data/map3/dataset.npy')[0:3000]
train_data2 = np.load('./data/map3/badsamples.npy')[0:3000]
indexs = np.random.choice(6000, samples)
data = np.concatenate([train_data2, train_data1], axis=0)[indexs].astype(np.float32)

# data = get_observations(samples, 3)
# data = np.stack(data, axis=0).astype(np.float32)

latent_dim = 1028
vae = AutoEncoder(input_shape=(64, 80, 3), latent_dim=latent_dim)
vae.load_state_dict(torch.load(osp.join(PROJECT_PATH, "vae", "checkpoints", "map3", "latent_1028", "best_model.pt"),
                               map_location=torch.device("cpu")))

plt.figure(figsize=(20, 4))

for i in range(samples):
      # display original
      ax = plt.subplot(2, samples, i + 1)
      plt.imshow(data[i])
      plt.title("original")
      #plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # display reconstruction
      ax = plt.subplot(2, samples, i + 1 + samples)
      reconstruction = vae(torch.from_numpy(data[i][None].transpose(0, 3, 1, 2)))
      plt.imshow(reconstruction[0].detach().numpy().transpose(1, 2, 0))
      plt.title("reconstructed")
      #plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
plt.show()