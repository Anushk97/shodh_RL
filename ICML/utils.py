import wandb
import json
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class RLHFLogger:
    def __init__(self):
        self.reward_history = []
        self.kl_div_history = []

    def log_step(self, reward, kl_div, gp_answer, eq_answer):
        self.reward_history.append(reward)
        self.kl_div_history.append(kl_div)

        wandb.log(
            {
                "reward": reward,
                "kl_divergence": kl_div,
                "gp_eq_match_rate": int(gp_answer == eq_answer),
            }
        )


def load_questions(file_path="./list_of_questions_new.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def log_training_results(epoch, q_id, gp_answer, eq_solution):
    with open(f"results_epoch_{epoch}.txt", "a") as f:
        f.write(
            f"Question {q_id}:\nGP Answer: {gp_answer}\nEQ Solution: {eq_solution}\n\n"
        )

    wandb.log(
        {
            "epoch": epoch,
            "question_id": q_id,
            "gp_answer": gp_answer,
            "eq_solution": eq_solution,
        }
    )


class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            reward = self.locals.get("rewards")[0]
            self.episode_rewards.append(reward)

            wandb.log(
                {
                    "reward": reward,
                    "episode_length": self.n_calls,
                    "answer": info.get("answer", ""),
                    "solution": info.get("solution", ""),
                    "mean_reward": np.mean(
                        self.episode_rewards[-100:]
                    ),  # moving average
                    "total_steps": self.n_calls,
                }
            )
        return True

    def _on_training_start(self):
        wandb.watch(self.model.policy)

    def _on_rollout_end(self):
        wandb.log(
            {
                "policy_loss": self.model.logger.name_to_value["train/policy_loss"],
                "value_loss": self.model.logger.name_to_value["train/value_loss"],
                "explained_variance": self.model.logger.name_to_value[
                    "train/explained_variance"
                ],
            }
        )


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 1  # log every step
        self.writer = SummaryWriter(self.logger.dir)

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:

            self.writer.add_scalar(
                "rewards/mean_reward", self.locals["rewards"][0], self.n_calls
            )

            if len(self.model.logger.name_to_value) > 0:
                self.writer.add_scalar(
                    "losses/policy_loss",
                    self.model.logger.name_to_value["train/policy_loss"],
                    self.n_calls,
                )
                self.writer.add_scalar(
                    "losses/value_loss",
                    self.model.logger.name_to_value["train/value_loss"],
                    self.n_calls,
                )
                self.writer.add_scalar(
                    "losses/explained_variance",
                    self.model.logger.name_to_value["train/explained_variance"],
                    self.n_calls,
                )
        return True

    def _on_training_end(self):
        self.writer.close()
