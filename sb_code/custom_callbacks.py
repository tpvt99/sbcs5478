from stable_baselines3.common.callbacks import BaseCallback
import os.path as osp
import os
from sb_code.Logger import Logger
import numpy as np

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

    def __init__(self, eval_env, eval_freq: int, logger: Logger, algo:str, name_prefix: str = "phong_best"):
        super(CustomBestModelCallback, self).__init__()
        self.eval_freq = eval_freq
        self.name_prefix = name_prefix
        self.eval_env = eval_env
        self.save_path = osp.join(logger.output_dir, name_prefix)
        self.custom_logger = logger
        self.algo_type = algo

        # For evaluation
        self.best_reward = None

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_lengths = self.eval_env.unwrapped.envs[0].get_episode_lengths()
            episode_rewards = self.eval_env.unwrapped.envs[0].get_episode_rewards()

            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards[-1000:])
                std_reward = np.std(episode_rewards[-1000:])
                mean_length = np.mean(episode_lengths[-1000:])
                std_length = np.std(episode_lengths[-1000:])

                if self.best_reward == None and len(episode_lengths) > 50:
                    self.best_reward = mean_reward

                if self.best_reward != None:
                    print(f'Custom Eval at num_timesteps {self.n_calls} with mean reward: {mean_reward:.5f},  std reward: {std_reward:.5f} '
                      f'mean length {mean_length:.5f}  std length: {std_length:.5f} best reward: {self.best_reward:.5f}')
                else:
                    print(
                        f'Custom Eval at num_timesteps {self.n_calls} with mean reward: {mean_reward:.5f},  std reward: {std_reward:.5f} '
                        f'mean length {mean_length:.5f}  std length: {std_length:.5f}')

                if self.best_reward != None and mean_reward > self.best_reward:
                    path = os.path.join(self.save_path, f"{self.name_prefix}")
                    self.best_reward = mean_reward
                    self.model.save(path)
                    print(f"Saving model checkpoint to {path}")

                self.custom_logger.log_tabular('n_calls', self.n_calls)
                self.custom_logger.log_tabular('mean_reward', mean_reward)
                self.custom_logger.log_tabular('std_reward', std_reward)
                self.custom_logger.log_tabular('mean_length', mean_length)
                self.custom_logger.log_tabular('std_length', std_length)
                self.custom_logger.log_tabular('best_reward', self.best_reward if self.best_reward != None else '')
                self.custom_logger.dump_tabular()
            if self.algo_type == 'dqn':
                print('Exploration rate: ', self.model.exploration_rate)
            print('-------------------------')
        return True