from stable_baselines3.common.callbacks import BaseCallback
import os.path as osp
import os
from sb_code.Logger import Logger
import numpy as np
import math
from stable_baselines3.common.vec_env import sync_envs_normalization
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from sb_code.wrapper import NormalizeWrapper, ResizeWrapper, FinalLayerObservationWrapper, DiscreteWrapper

class SaveNormalization(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: (Optional[BaseCallback]) Callback that will be called
        when an event is triggered.
    :param verbose: (int)
    """
    def __init__(self, save_path=None):
        super(SaveNormalization, self).__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.model.get_vec_normalize_env() is not None:
            self.model.get_vec_normalize_env().save(self.save_path)
            if self.verbose > 1:
                print(f"Saving VecNormalize to {self.save_path}")
        return True

class CustomBestModelCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param eval_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, eval_env, eval_freq: int, logger: Logger, custom_params:dict, name_prefix: str = "phong_best"):
        super(CustomBestModelCallback, self).__init__()
        self.eval_freq = eval_freq
        self.name_prefix = name_prefix
        self.eval_env = eval_env
        self.save_path = osp.join(logger.output_dir, name_prefix)
        self.custom_logger = logger
        self.custom_params = custom_params

        # For evaluation
        self.reward_list, self.length_list = [], []
        self.best_reward_all = -math.inf
        self.best_reward_solo = -math.inf

        self.seeds_dict = {
                "map1": [2, 3, 5, 9, 12],
                "map2": [1, 2, 3, 5, 7, 8, 13, 16],
                "map3": [1, 2, 4, 8, 9, 10, 15, 21],
                "map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
                "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
            }

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        self.training_env = self.model.get_env()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls >= 50000: # 50000 is learning_start

            episode_lengths = self.eval_env.unwrapped.envs[0].get_episode_lengths()
            episode_rewards = self.eval_env.unwrapped.envs[0].get_episode_rewards()

            solo_reward_list = episode_rewards[-self.custom_params['eval_episodes']:]
            solo_length_list = episode_lengths[-self.custom_params['eval_episodes']:]

            mean_reward = np.mean(episode_rewards[-500:])
            std_reward = np.std(episode_rewards[-500:])
            mean_length = np.mean(episode_lengths[-500:])
            std_length = np.std(episode_lengths[-500:])

            if mean_reward > self.best_reward_all:
                path = os.path.join(self.save_path, f"{self.name_prefix}")
                self.best_reward_all = mean_reward
                self.model.save(path)
                print(f"Saving model checkpoint to {path}")
                self.model.save(osp.join(self.save_path, f"{self.name_prefix}_{str(int(mean_reward)).replace('.', '_')}"))

            if np.mean(solo_reward_list) > self.best_reward_solo:
                self.best_reward_solo = np.mean(solo_reward_list)
                path = osp.join(self.save_path, f"{self.name_prefix}_solo_{str(int(self.best_reward_solo)).replace('.', '_')}")
                self.model.save(path)
                print(f"Saving solo model checkpoint to {path}")

            print(f'Custom Eval timesteps {self.n_calls} meanReward: {mean_reward:.2f} stdReward: {std_reward:.2f} '
                  f'meanLength {mean_length:.2f} stdLength: {std_length:.2f} bestReward: {self.best_reward_all:.2f} '
                  f'rewardSolo {np.mean(solo_reward_list):.2f} lengthSolo {np.mean(solo_length_list):.2f} '
                  f'bestRewardSolo: {self.best_reward_solo:.2f}')

            self.custom_logger.log_tabular('n_calls', self.n_calls)
            self.custom_logger.log_tabular('mean_reward', mean_reward)
            self.custom_logger.log_tabular('std_reward', std_reward)
            self.custom_logger.log_tabular('mean_length', mean_length)
            self.custom_logger.log_tabular('std_length', std_length)
            self.custom_logger.log_tabular('best_reward', self.best_reward_all)
            self.custom_logger.log_tabular('mean_reward_solo', np.mean(solo_reward_list))
            self.custom_logger.log_tabular('best_reward_solo', self.best_reward_solo)
            self.custom_logger.log_tabular('explr_rate', self.model.exploration_rate)

            self.custom_logger.dump_tabular()
            if self.custom_params['algo'] == 'dqn':
                print('Exploration rate: ', self.model.exploration_rate)

            print('-------------------------')
        return True