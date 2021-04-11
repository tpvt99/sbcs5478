import gym
import cv2
from gym import spaces
import numpy as np
# from src_mb.vae.model import VariationalAutoEncoder
import os.path as osp
from gym_duckietown.simulator import NotInLane, REWARD_INVALID_POSE
PROJECT_PATH = osp.abspath(osp.dirname(osp.dirname(__file__)))

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
            return -100

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
            reward += 4

        return reward

# class FinalLayerObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env=None, vae_loader_path = None, latent_dim: int=128):
#         super(FinalLayerObservationWrapper, self).__init__(env)
#         self.latent_dim = latent_dim
#         self.vae = VariationalAutoEncoder(input_shape = self.observation_space.shape, latent_dim=self.latent_dim)
#         if vae_loader_path is not None:
#             self.vae.load_weights(vae_loader_path)
#         else:
#             self.vae.load_weights(osp.join(PROJECT_PATH, "src_mb", "vae", "vae_checkpoints", "latent_128", "vae_tr199"))
#         self.encoder = self.vae.encoder
#         self.encoder.trainable = False # set to not train
#
#         self.cur_angle_dim = 1 # HARD CODE
#         self.speed_dim = 1 # HARD CODE
#         self.cur_pos_dim = 3 # HARD CODE
#
#         self.observation_space.shape = (self.latent_dim,)
#         self.observation_space = spaces.Box(0.0, 1.0, (self.latent_dim,), dtype=np.float32)
#
#     def observation(self, observation):
#         #1. Check dimension
#         if len(observation.shape) == 1 or len(observation.shape)==3:
#             observation = observation[None]
#         assert observation.shape[0] == 1, "There should be only 1 observation passing through this every time called"
#         # 1. Extract first latent to go into Encoder
#         encoded_observation = self.encoder(observation)
#         encoded_observation = encoded_observation[0].numpy()
#         # output_observation = np.concatenate([encoded_observation,
#         #                                      self.env.unwrapped.cur_pos,
#         #                                      np.array([self.env.unwrapped.speed]),
#         #                                      np.array([self.env.unwrapped.cur_angle])])
#         return encoded_observation