from rlhf_config import RLHFConfig
from utils import RLHFLogger, load_questions, log_training_results
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import torch.nn.functional as F
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from policy import LanguagePPOPolicy
import wandb
from typing import Optional
import numpy as np
import torch
from env import MathRLHFEnv
from utils import WandbLoggingCallback, TensorboardCallback


class PPORLHFTrainer:
    def __init__(
        self,
        env,
        model_name: str,
        total_timesteps: int,
        learning_rate: float = 3e-4,
        n_steps: int = 256,
        batch_size: int = 32,
        n_epochs: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = "./ppo_rlhf_tensorboard/",
        policy_kwargs: Optional[dict] = None,
    ):
        self.env = DummyVecEnv([lambda: env])
        self.model_name = model_name
        self.total_timesteps = total_timesteps

        # if policy_kwargs is None:
        #     policy_kwargs = {
        #         "net_arch": [dict(pi=[256, 128], vf=[256, 128])]
        #     }

        self.model = PPO(
            policy=LanguagePPOPolicy,
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

        # self.callback = WandbLoggingCallback()

    def train(self):
        try:
            wandb.init(
                project="rlhf-ppo",
                name=self.model_name,
                config={
                    "total_timesteps": self.total_timesteps,
                    "learning_rate": self.model.learning_rate,
                    "n_steps": self.model.n_steps,
                    "batch_size": self.model.batch_size,
                    "n_epochs": self.model.n_epochs,
                },
            )
            callback = [WandbLoggingCallback(), TensorboardCallback()]

            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback,
                progress_bar=True,
            )
            self.model.save(f"ppo_rlhf_{self.model_name}_final")

        except Exception as e:
            print(f"Training failed with error: {e}")
            raise e

        finally:
            wandb.finish()
            self.env.close()


### TRAINER 01 (Using custom PPO) ###
def main():
    config = RLHFConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "microsoft/phi-1.5"

    print(f"Loading {model_name}...")
    gp_model = AutoModelForCausalLM.from_pretrained(
        config.gp_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    eq_model = AutoModelForCausalLM.from_pretrained(
        config.eq_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    # optimize models
    gp_model.config.use_cache = True
    eq_model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    # gp_model = gp_model.to(device)
    # eq_model =eq_model.to(device)

    print(f"GP Model dtype: {next(gp_model.parameters()).dtype}")
    print(f"EQ Model dtype: {next(eq_model.parameters()).dtype}")

    gp_model.gradient_checkpointing_enable()
    eq_model.gradient_checkpointing_enable()

    questions = load_questions()

    env = MathRLHFEnv(gp_model, eq_model, tokenizer, questions)

    trainer = PPORLHFTrainer(
        env=env,
        model_name="phi-math-reasoning-stable",
        total_timesteps=len(questions) * config.max_epochs,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.max_epochs,
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
    )
    trainer.train()


### TRAINER 02 (using PPO from SB library) ###
def train_gp_with_rlhf():
    wandb.init(project="math-rlhf", name="phi-math-reasoning-ppo")

    config = RLHFConfig()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    print("Loading models...")
    model_name = "microsoft/phi-1.5"

    gp_model = AutoModelForCausalLM.from_pretrained(
        config.gp_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    eq_model = AutoModelForCausalLM.from_pretrained(
        config.eq_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gp_model.config.pad_token_id = tokenizer.pad_token_id
    eq_model.config.pad_token_id = tokenizer.pad_token_id
    gp_model.config.use_cache = True
    eq_model.config.use_cache = True

    print("Loading questions...")
    questions = load_questions()

    print("Creating environment...")
    env = MathRLHFEnv(gp_model, eq_model, tokenizer, questions)
    env = DummyVecEnv([lambda: env])

    print("Setting up PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        tensorboard_log="./ppo_math_tensorboard/",
        clip_range=0.2,
        ent_coef=0.01,  # encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 128],  # smaller policy network
                vf=[256, 128],  # smaller value network
            ),
            activation_fn=torch.nn.ReLU,
        ),
    )

    try:
        print("Starting training...")
        total_timesteps = len(questions) * 5

        checkpoint_callback = CheckpointCallback(
            save_freq=100, save_path="./logs", name_prefix="rlhf_checkpoint"
        )

        wandb_callback = WandbLoggingCallback(verbose=1)
        tensorboard_callback = TensorboardCallback()

        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, wandb_callback, tensorboard_callback],
            progress_bar=True,
        )

        print("Saving final models...")
        model.save("./models/ppo_math_model_final")
        gp_model.save_pretrained("./models/improved_phi_model_final")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    finally:
        env.close()
        wandb.finish()


### RUN TRAINER 01 ###
if __name__ == "__main__":
    main()

### RUN TRAINER 02 ###
# if __name__ == "__main__":
#     train_gp_with_rlhf()
