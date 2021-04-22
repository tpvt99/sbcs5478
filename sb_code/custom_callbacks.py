from stable_baselines3.common.callbacks import BaseCallback
import os.path as osp
import os
from sb_code.Logger import Logger
import numpy as np
import math
from stable_baselines3.common.vec_env import sync_envs_normalization
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv, VecEnv
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

    def setup_env(self, custom_params, seed):

        env = DuckietownEnv(
            map_name=custom_params['map'],
            domain_rand=False,
            draw_bbox=False,
            max_steps=1500,
            seed=seed
        )

        env = ResizeWrapper(env, shape=(60, 80, 3))

        if custom_params['discrete']:
            env = DiscreteWrapper(env)

        if custom_params['USING_VAE']:
            env = NormalizeWrapper(env)  # No need to use normalization if image
            env = FinalLayerObservationWrapper(env, latent_dim=custom_params['VAE_LATENT_DIM'],
                                               map=custom_params['map'])

        # Step 3.b. To make Vectorized Environment to be able to use Normalize or FramStack (Optional)
        env = make_vec_env(lambda: env, n_envs=1)
        # Step 3.b Passing through Normalization and stack frame (Optional)

        env = VecFrameStack(env, n_stack=custom_params['FRAME_STACK'])  # Use 1 for now because we use image
        if not custom_params['USING_VAE']:
            env = VecTransposeImage(env)  # Uncomment if using 3d obs

        return env

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls >= 50000: # 50000 is learning_start
            seed_lists = self.seeds_dict[self.custom_params['map']][0:5]
            solo_reward_list, solo_length_list = [], []

            for i in range(len(seed_lists)):
                eval_env = self.setup_env(self.custom_params, seed=seed_lists[i])

                if not isinstance(eval_env, VecEnv):
                    eval_env = DummyVecEnv([lambda: eval_env])

                if isinstance(eval_env, VecEnv):
                    assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

                if not isinstance(self.training_env, type(self.eval_env)):
                    raise ValueError(
                        "Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

                sync_envs_normalization(self.training_env, eval_env)

                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    eval_env,
                    n_eval_episodes=1,
                    render=False,
                    deterministic=True,
                    return_episode_rewards=False,
                    warn=False
                )
                self.reward_list.extend(episode_rewards)
                self.length_list.extend(episode_lengths)
                solo_reward_list.extend(episode_rewards)
                solo_length_list.extend(episode_lengths)

            mean_reward = np.mean(self.reward_list[-500:])
            std_reward = np.std(self.reward_list[-500:])
            mean_length = np.mean(self.length_list[-500:])
            std_length = np.std(self.length_list[-500:])

            if mean_reward > self.best_reward_all:
                path = os.path.join(self.save_path, f"{self.name_prefix}")
                self.best_reward = mean_reward
                self.model.save(path)
                print(f"Saving model checkpoint to {path}")

            if np.mean(solo_reward_list) > self.best_reward_solo:
                self.best_reward_solo = np.mean(solo_reward_list)
                path = osp.join(self.save_path, f"{self.name_prefix}_{self.n_calls}")
                self.model.save(path)
                print(f"Saving solo model checkpoint to {path}")

            print(f'Custom Eval at timesteps {self.n_calls} with meanReward: {mean_reward:.3f},  stdReward: {std_reward:.3f} '
                  f'meanLength {mean_length:.3f}  stdLength: {std_length:.3f} bestReward: {self.best_reward:.3f}'
                  f' rewardSolo {np.mean(solo_reward_list):.3f} lengthSolo {np.mean(solo_length_list):3.f} bestRewardSolo: {self.best_reward_solo}')

            self.custom_logger.log_tabular('n_calls', self.n_calls)
            self.custom_logger.log_tabular('mean_reward', mean_reward)
            self.custom_logger.log_tabular('std_reward', std_reward)
            self.custom_logger.log_tabular('mean_length', mean_length)
            self.custom_logger.log_tabular('std_length', std_length)
            self.custom_logger.log_tabular('best_reward', self.best_reward)
            self.custom_logger.log_tabular('mean_reward_solo', np.mean(solo_reward_list))
            self.custom_logger.log_tabular('best_reward_solo', self.best_reward_solo)

            self.custom_logger.dump_tabular()
        if self.custom_params['algo'] == 'dqn':
            print('Exploration rate: ', self.model.exploration_rate)
        print('-------------------------')
        return True