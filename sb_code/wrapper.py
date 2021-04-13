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

    def observation(self, observation):
        resized_img = cv2.resize(observation, dsize=(self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
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
        if reward == REWARD_INVALID_POSE:
            return -50.0

        # pos = self.env.cur_pos
        # angle = self.env.cur_angle
        # speed = self.env.speed
        # col_penalty = self.env._proximity_penalty2(pos, angle)
        #
        # # Get the position relative to the right lane tangent
        # try:
        #     lp = self.env.get_lane_pos2(pos, angle)
        # except NotInLane:
        #     reward = 40 * col_penalty
        # else:
        #
        #     # Compute the reward
        #     reward = (
        #             +1.0 * speed * lp.dot_dir +
        #             -10 * np.abs(lp.dist) +
        #             +40 * col_penalty
        #     )
        #
        # dist_to_stop = 1000.0
        # # print("number of objects = ", len(self.objects))
        # for obj in self.objects:
        #     if obj.kind == "sign_stop":
        #         dist_to_stop = min(dist_to_stop, ((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)
        #
        # if self.speed > 0.15 and dist_to_stop < 0.3:
        #     reward = -100.0

        if reward > 0:
            reward += 10
        else:
            reward += 2
        return reward

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        velocity_list = [0.1, 0.25, 0.35, 0.4, 0.5, 0.7, 0.9, 1.0]
        steering_list = [-np.pi, -np.pi/8, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi/8, np.pi]
        action_list = list(product(velocity_list, steering_list))
        self.action_dim = len(action_list)
        self.index_to_actions = {key:val for key,val in enumerate(action_list)}

        self.action_space = spaces.Discrete(self.action_dim)

    def action(self, action):
        assert np.isscalar(action) or isinstance(action, int), "Action is not an integer"
        assert action >= 0 and action < self.action_dim
        return np.array(self.index_to_actions[action])


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

        self.obs_dim = self.latent_dim + self.cur_pos_dim + self.speed_dim + self.cur_angle_dim
        self.observation_space.shape = (self.obs_dim,)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)

    def observation(self, observation):
        #1. Check dimension
        assert observation.ndim == 3, "Dimension of image must be 3"
        observation=observation.astype(np.float32)
        # 1. Extract first latent to go into Encoder
        encoded_observation = self.encoder(self.transforms(observation)[None])
        encoded_observation = encoded_observation.detach().numpy()[0]

        output_observation = np.concatenate([encoded_observation,
                                             self.env.unwrapped.cur_pos,
                                             np.array([self.env.unwrapped.speed]),
                                             np.array([self.env.unwrapped.cur_angle])])
        return output_observation