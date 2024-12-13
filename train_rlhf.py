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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from models import LanguagePPOPolicy
import wandb
from typing import Optional
import numpy as np
from torch.distributions import Categorical
import gc
import torch
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


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
            verbose=1
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
                }
            )
            callback = [WandbLoggingCallback(), TensorboardCallback()]
            
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback,
                progress_bar=True
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
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # model_config = AutoConfig.from_pretrained("microsoft/phi-2")
    # model_config.use_cache = False  # More stable

    model_name = "microsoft/phi-1_5"
    
    print(f"Loading {model_name}...")
    gp_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    eq_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    # optimize models
    gp_model.config.use_cache = True
    eq_model.config.use_cache = True 

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
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
        policy_kwargs={
            "net_arch": dict(
                pi=[64, 64],
                vf=[64, 64]
            )
        }
    )
    trainer.train()

### ENVIRONMENT ###
class MathRLHFEnv(gym.Env):
    def __init__(self, gp_model, eq_model, tokenizer, questions):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gp_model = gp_model
        self.eq_model = eq_model
        self.tokenizer = tokenizer
        self.questions = questions
        self.current_question_idx = 0
        self.training = True

        self.top_k = 1000  # reduce from full vocab size
        # self.action_space = spaces.Box(
        #     low=-10.0,
        #     high=10.0,
        #     shape=(self.tokenizer.vocab_size,),
        #     dtype=np.float32
        # )

        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        
        self.max_length = 128 
        self.observation_space = spaces.Box(
            low=0,
            high=self.tokenizer.vocab_size,
            shape=(self.max_length,),
            dtype=np.int64
        )

        self.current_solution = None
        
        # self.gp_model = self.gp_model.to(self.device)
        # self.eq_model = self.eq_model.to(self.device)
    
    def generate_answer(self, model, prompt, device):
        try:
            if model == self.eq_model:
                formatted_prompt = (
                    f"Problem: {prompt}\n\n"
                    "Solve this step by step:\n"
                    "1. Understand what is given\n"
                    "2. Write the equations\n"
                    "3. Solve step by step\n"
                    "4. State the final answer\n\n"
                    "Solution:\n"
                )
            else:
                formatted_prompt = (
                    f"Problem: {prompt}\n"
                    "Let me solve this step by step:\n"
                )
                
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            ).to(next(model.parameters()).device)
            
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    min_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    no_repeat_ngram_size=3
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                return generated_text.replace(prompt, "").strip()
                
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Error: Could not generate answer"
        finally:
            model.train()

    def set_training(self, training: bool):
        self.training = training
    
    def calculate_reward(self, gp_answer, correct_solution):
        if not gp_answer or not correct_solution:
            return 0.0
        
        try:
            gp_answer = gp_answer.strip().lower()
            correct_solution = correct_solution.strip().lower()
            
            if gp_answer.startswith("error:") or correct_solution.startswith("error:"):
                return 0.0

            reasoning_score = 0.0
            steps_score = 0.0
            answer_match_score = 0.0
            
            # raw text
            print("\nText Analysis:")
            print(f"Raw GP Answer: {gp_answer}")
            print('--------------------------------')
            print(f"Raw Correct Solution (EQ): {correct_solution}")
            print('--------------------------------')
            
            math_terms = ['add', 'subtract', 'multiply', 'divide', 'equals', 
                        'sum', 'total', 'difference', 'product', 'quotient',
                        '+', '-', '*', '/', '=', '$']
            
            # score reasoning steps
            gp_steps = set(word for word in gp_answer.split() 
                        if word in math_terms or any(c.isdigit() for c in word))
            correct_steps = set(word for word in correct_solution.split() 
                            if word in math_terms or any(c.isdigit() for c in word))
            
            if correct_steps:
                steps_score = len(gp_steps.intersection(correct_steps)) / len(correct_steps)
            
            # score answer matching
            if "final answer" in gp_answer and "final answer" in correct_solution:
                gp_final = gp_answer[gp_answer.find("final answer"):].split('\n')[0]
                correct_final = correct_solution[correct_solution.find("final answer"):].split('\n')[0]
                answer_match_score = 1.0 if gp_final == correct_final else 0.0
            
            # score reasoning
            reasoning_indicators = ['because', 'therefore', 'since', 'so', 'as']
            reasoning_score = any(indicator in gp_answer for indicator in reasoning_indicators)
            
            # combine scores (weighted average)
            final_score = (
                0.4 * answer_match_score +  
                0.4 * steps_score +         
                0.2 * float(reasoning_score) # explanation is good but less critical
            )
            
            print(f"\nScoring Breakdown:")
            print(f"Answer Match Score: {answer_match_score:.2f}")
            print(f"Steps Score: {steps_score:.2f}")
            print(f"Reasoning Score: {reasoning_score:.2f}")
            print(f"Final Score: {final_score:.2f}\n")
            
            return float(final_score)
            
        except Exception as e:
            print(f"Reward calculation error: {e}")
            return 0.0
    
    def reset(self):
        self.current_question_idx = (self.current_question_idx + 1) % len(self.questions)
        question = self.questions[self.current_question_idx]
        
        eq_prompt = (
            "You are a math equation generator. Convert this problem into "
            "mathematical equations, solve them step by step, and provide "
            f"the final numerical answer:\n{question}\nSolution:"
        )
        self.current_solution = self.generate_answer(
            self.eq_model, 
            question, 
            next(self.eq_model.parameters()).device
        )
        
        encoded = self.tokenizer(
            question,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np"
        )
        obs = encoded['input_ids'][0]
        
        return obs
    
    def step(self, action):
        try:            
            question = self.questions[self.current_question_idx]
            action_tensor = torch.FloatTensor(action).to(next(self.gp_model.parameters()).device)
            action_probs = F.softmax(action_tensor, dim=-1)
            
            # answer with gp_model
            try:
                formatted_prompt = (
                    f"Problem: {question}\n"
                    "Solution: Let me solve this step by step.\n"
                    "Final answer: $"
                )
                
                inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=True,
                return_attention_mask=True
            ).to(next(self.gp_model.parameters()).device)
                
                with torch.no_grad():
                    outputs = self.gp_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=100,
                        min_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        num_beams=1,
                        early_stopping=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0
                    )
                    
                    answer = self.tokenizer.decode(
                        outputs[0], 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
            except Exception as e:
                print(f"Generation error: {e}")
                answer = "Error: Generation failed"

            reward = self.calculate_reward(answer, self.current_solution)
            self.current_question_idx = (self.current_question_idx + 1) % len(self.questions)
            
            next_obs = self.reset()
            done = True
            
            info = {
                "answer": answer,
                "solution": self.current_solution,
                "reward": reward
            }

            # print('INFO...', info)
            
            return next_obs, reward, done, info
            
        except Exception as e:
            print(f"Step error: {str(e)}")
            return self.reset(), 0.0, True, {}

### LOGGING ###
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
            
            wandb.log({
                "reward": reward,
                "episode_length": self.n_calls,
                "answer": info.get("answer", ""),
                "solution": info.get("solution", ""),
                "mean_reward": np.mean(self.episode_rewards[-100:]),  # moving average
                "total_steps": self.n_calls
            })
        return True
    
    def _on_training_start(self):
        wandb.watch(self.model.policy)
    
    def _on_rollout_end(self):
        wandb.log({
            "policy_loss": self.model.logger.name_to_value["train/policy_loss"],
            "value_loss": self.model.logger.name_to_value["train/value_loss"],
            "explained_variance": self.model.logger.name_to_value["train/explained_variance"]
        })

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_training_start(self):
        self._log_freq = 1  # log every step
        self.writer = SummaryWriter(self.logger.dir)
        
    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            
            self.writer.add_scalar('rewards/mean_reward', 
                                 self.locals['rewards'][0], 
                                 self.n_calls)
            
            if len(self.model.logger.name_to_value) > 0:
                self.writer.add_scalar('losses/policy_loss',
                                     self.model.logger.name_to_value['train/policy_loss'],
                                     self.n_calls)
                self.writer.add_scalar('losses/value_loss',
                                     self.model.logger.name_to_value['train/value_loss'],
                                     self.n_calls)
                self.writer.add_scalar('losses/explained_variance',
                                     self.model.logger.name_to_value['train/explained_variance'],
                                     self.n_calls)
        return True
    
    def _on_training_end(self):
        self.writer.close()


### TRAINER 02 (using PPO from SB library) ###
def train_gp_with_rlhf():
    wandb.init(project="math-rlhf", name="phi-math-reasoning-ppo")
    
    config = RLHFConfig()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading models...")
    model_name = "microsoft/phi-1_5"
    
    gp_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    eq_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
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
        ent_coef=0.01,      # encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 128],    # smaller policy network
                vf=[256, 128]     # smaller value network
            ),
            activation_fn=torch.nn.ReLU
        )
    )
    
    try:
        print("Starting training...")
        total_timesteps = len(questions) * 5
        
        checkpoint_callback = CheckpointCallback(
            save_freq=100, 
            save_path='./logs',
            name_prefix='rlhf_checkpoint'
        )
        
        wandb_callback = WandbLoggingCallback(verbose=1)
        tensorboard_callback = TensorboardCallback()

        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, wandb_callback, tensorboard_callback],
            progress_bar=True
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
# if __name__ == "__main__":
#     main()

### RUN TRAINER 02 ###
if __name__ == "__main__":
    train_gp_with_rlhf()


