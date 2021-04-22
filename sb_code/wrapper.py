import gym
import cv2
from gym import spaces
import numpy as np
import os.path as osp
from gym_duckietown.simulator import NotInLane, REWARD_INVALID_POSE
from vae.model_torch import AutoEncoder
from global_configuration import PROJECT_PATH
import torch
import numpy as np
from torchvision import transforms
from itertools import product
from PIL import Image
import matplotlib.pyplot as plt
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape
        self.count = 0

    def observation(self, observation):
        resized_img = cv2.resize(observation, dsize=(self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        # if self.env.unwrapped.step_count in [0, 1]:
        #     print(f'Count {self.env.unwrapped.step_count}, Position {self.env.unwrapped.cur_pos} '
        #           f'Angle {self.env.unwrapped.cur_angle} and speed {self.env.unwrapped.speed}')
        #     #plt.imshow()
        #     plt.imsave(f'f{self.count}_step{self.env.unwrapped.step_count}.jpeg', resized_img)
        #     self.count+=1
        return resized_img

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env=None):
        super(RewardWrapper, self).__init__(env)

    def reward(self, reward):
        #if reward == REWARD_INVALID_POSE:
        #    return -100.0

        pos = self.env.cur_pos
        angle = self.env.cur_angle
        speed = self.env.speed
        col_penalty = self.env._proximity_penalty2(pos, angle)

        dist_to_stop = 1000.0
        dist_thresh = 0.30

        for obj in self.objects:
            if obj.kind == "sign_stop":
                dist_to_stop = min(dist_to_stop, ((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)

        # Get the position relative to the right lane tangent
        try:
            lp = self.env.get_lane_pos2(pos, angle)
        except NotInLane:
            reward = 40 * col_penalty
        else:

            # Compute the reward
            if dist_to_stop < dist_thresh:
                reward = (
                    # + 1.0  * lp.dot_dir
                        - 10.0 * np.abs(lp.dist)
                        + 40.0 * col_penalty
                        + 1.0 * ((2.2 / (1 + self.speed)) - 1) * lp.dot_dir
                )
            else:
                # Compute the reward
                reward = (
                        + 1.0 * self.speed * lp.dot_dir
                        - 10.0 * np.abs(lp.dist)
                        + 40.0 * col_penalty
                )

        #
        if self.speed > 0.15 and dist_to_stop < 0.3:
            reward -= 1.5

        if math.fabs(self.speed - 0) <= 1e-10:
            reward -= 0.8

        return reward

class Map1EvalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env=None):
        super(Map1EvalRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == REWARD_INVALID_POSE:
            return 0.0

        return reward


class InfoWrapperEval(gym.RewardWrapper):
    def __init__(self, env=None):
        super(InfoWrapperEval, self).__init__(env)

    def get_dist_to_stop(self):
        dist_to_stop = 1000.0

        for obj in self.env.unwrapped.objects:
            if obj.kind == "sign_stop":
                dist_to_stop = min(dist_to_stop, ((self.env.unwrapped.cur_pos[0] - obj.pos[0]) ** 2 + (
                            self.env.unwrapped.cur_pos[2] - obj.pos[2]) ** 2) ** 0.5)

        return dist_to_stop

    def reward(self, reward):
        print(f'Speed: {self.env.unwrapped.speed} '
              f'DistanceToStop: {self.get_dist_to_stop()}')
              #f'Collistion {self.env.unwrapped._proximity_penalty2(self.env.unwrapped.cur_pos, self.env.unwrapped.cur_angle)}'
              #f'LP: {self.env.unwrapped.get_lane_pos2(self.env.unwrapped.cur_pos, self.env.unwrapped.cur_angle)}')
        return reward


class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        discrete_v = [0.1, 0.35, 1.0]

        denominators = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 4, 8, 12, 16, 20]
        discrete_w = [-np.pi / x for x in denominators] + [-1.0, 0.0, 1.0] + [np.pi / x for x in denominators]
        for x in np.arange(0.01, 0.11, 0.01):
            discrete_w.append(np.round(x, 2))
            discrete_w.append(-np.round(x, 2))

        for x in np.arange(0.001, 0.01, 0.002):
            discrete_w.append(np.round(x, 3))
            discrete_w.append(-np.round(x, 3))

        for w in range(len(discrete_w)):
            if discrete_w[w] == 0:
                discrete_w[w] = 0

        self.action_list = list(product(discrete_v, discrete_w))
        self.action_list.append((0.0, 1.0))
        self.action_list.append((0.0, -np.pi / 2))
        self.action_list.append((0.0, np.pi / 2))

        self.action_dim = len(self.action_list)

        #print("Action space:", self.action_dim)
        #print(self.action_list)


        self.action_space = spaces.Discrete(self.action_dim)

    def action(self, action):
        assert np.isscalar(action) or isinstance(action, int), "Action is not an integer"
        assert action >= 0 and action < self.action_dim
        return np.array(self.action_list[action])


class FinalLayerObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, latent_dim: int, map: str):
        super(FinalLayerObservationWrapper, self).__init__(env)
        self.latent_dim = latent_dim
        self.vae = AutoEncoder(input_shape = self.observation_space.shape, latent_dim=self.latent_dim)
        self.vae.load_state_dict(torch.load(
            osp.join(PROJECT_PATH, "vae", "checkpoints", map, f"latent_{self.latent_dim}", "best_model.pt"),
            map_location=torch.device(device)))

        self.encoder = self.vae.encoder
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.cur_angle_dim = 1 # HARD CODE
        self.speed_dim = 1 # HARD CODE
        self.cur_pos_dim = 3 # HARD CODE
        self.dist_to_stop_dim = 1

        self.obs_dim = self.latent_dim + self.cur_pos_dim + self.speed_dim + self.cur_angle_dim + self.dist_to_stop_dim
        self.observation_space.shape = (self.obs_dim,)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)

    def get_dist_to_stop(self):
        dist_to_stop = 1000.0

        for obj in self.env.unwrapped.objects:
            if obj.kind == "sign_stop":
                dist_to_stop = min(dist_to_stop, ((self.env.unwrapped.cur_pos[0] - obj.pos[0]) ** 2 + (
                            self.env.unwrapped.cur_pos[2] - obj.pos[2]) ** 2) ** 0.5)

        return dist_to_stop

    def observation(self, observation):
        #1. Check dimension
        assert observation.ndim == 3, "Dimension of image must be 3"
        observation=observation.astype(np.float32)
        # 1. Extract first latent to go into Encoder
        encoded_observation = self.encoder(self.transforms(observation)[None])
        encoded_observation = encoded_observation.detach().numpy()[0]

        output_observation = np.concatenate([encoded_observation,
                                             np.array([self.get_dist_to_stop()]),
                                             self.env.unwrapped.cur_pos,
                                             np.array([self.env.unwrapped.speed]),
                                             np.array([self.env.unwrapped.cur_angle])])
        return output_observation

class PositiveVelocityActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(PositiveVelocityActionWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=np.array([-1, -1.]), high=np.array([1., 1.]), shape=(2,))

    def action(self, action):
        assert action.ndim == 1
        #action[0] = np.clip(action[0], 0.0, 1.0)
        return action