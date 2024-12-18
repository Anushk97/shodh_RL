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
        max_grad_norm: float = 1.0,
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

    model_name = "microsoft/phi-1.5"
    
    print(f"Loading {model_name}...")
    gp_model = AutoModelForCausalLM.from_pretrained(
        config.gp_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    eq_model = AutoModelForCausalLM.from_pretrained(
        config.eq_model_path,
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
        gamma=0.99,  
        gae_lambda=0.95, 
        clip_range=0.1,  
        ent_coef=0.01, 
        policy_kwargs={
            "net_arch": dict(
                pi=[128, 128],
                vf=[128, 128]
            )
        }
    )
    trainer.train()

### ENVIRONMENT ###
class MathRLHFEnv(gym.Env):
    def __init__(self, gp_model, eq_model, tokenizer, questions):
        super().__init__()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Environment using device: {self.device}")
        
        self.gp_model = gp_model.to(self.device)
        self.eq_model = eq_model.to(self.device)
        self.tokenizer = tokenizer
        self.questions = questions
        self.current_question_idx = 0
        self.gp_model.train()
        self.eq_model.eval()
        self.optimizer = torch.optim.AdamW(
            self.gp_model.parameters(), 
            lr=1e-5,
            weight_decay=0.01,
            eps=1e-7,
            foreach=True 
        )
        self.optimizer_state = None
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        self.max_length = 128
        self.observation_space = spaces.Box(
            low=0,
            high=self.tokenizer.vocab_size,
            shape=(self.max_length,),
            dtype=np.int64
        )
        
        self.current_sequence = None
        self.current_solution = None

    def _get_observation(self):
        seq = self.current_sequence[0].detach().cpu().numpy()
        padded = np.zeros(self.max_length, dtype=np.int64)
        seq_len = min(len(seq), self.max_length)
        padded[:seq_len] = seq[:seq_len]
        return padded

    def step(self, action):
        try:
            if self.optimizer_state is None:
                self.optimizer_state = {
                    'state': self.optimizer.state_dict()['state'],
                    'param_groups': self.optimizer.state_dict()['param_groups']
                }
            
            outputs = self.gp_model(self.current_sequence)
            logits = outputs.logits[:, -1, :]
            target = torch.tensor([action], device=self.device)
            
            # restore optimizer state
            self.optimizer.load_state_dict({
                'state': self.optimizer_state['state'],
                'param_groups': self.optimizer_state['param_groups']
            })
            
            # calculate loss and update
            loss = F.cross_entropy(logits, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gp_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer_state = {
                'state': self.optimizer.state_dict()['state'],
                'param_groups': self.optimizer.state_dict()['param_groups']
            }
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                next_token = torch.tensor([[action]], device=self.device)
                self.current_sequence = torch.cat([
                    self.current_sequence, 
                    next_token
                ], dim=1)
            
            current_text = self.tokenizer.decode(self.current_sequence[0])
            step_reward = self.calculate_step_reward(current_text)
            
            done = (action == self.tokenizer.eos_token_id or 
                   len(self.current_sequence[0]) >= self.max_length)
            
            obs = self._get_observation()
            
            info = {
                "text": current_text,
                "loss": loss.item(),
                "step_reward": step_reward
            }
            
            if done:
                final_reward = self.calculate_reward(current_text, self.current_solution)
                total_reward = step_reward + final_reward
                info["final_reward"] = final_reward
                return obs, total_reward, done, info
            
            return obs, step_reward, done, info
            
        except Exception as e:
            print(f"Step error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc() 
            return self.reset(), 0.0, True, {}

    def calculate_step_reward(self, current_text):
        reward = 0.0
        
        # reward for mathematical terms and symbols
        math_terms = [
            'add', 'subtract', 'multiply', 'divide', 'equals',
            '+', '-', '*', '/', '=', 
            'sum', 'difference', 'product', 'quotient',
            'equation', 'solution', 'step'
        ]
        
        for term in math_terms:
            if term in current_text.lower():
                reward += 0.1
        
        # reward for numerical values
        if any(char.isdigit() for char in current_text):
            reward += 0.2
            
        # reward for step-by-step structure
        if "step" in current_text.lower():
            reward += 0.3
            
        # reward for complete sentences
        sentences = current_text.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                reward += 0.1
                
        # penalize very short or very long generations
        if len(current_text) < 10:
            reward -= 0.2
        elif len(current_text) > 500:
            reward -= 0.3
            
        # penalize repetition
        words = current_text.lower().split()
        if len(words) > len(set(words)): 
            reward -= 0.1
            
        return float(reward)

    def calculate_reward(self, gp_answer, correct_solution):
        if not gp_answer or not correct_solution:
            return 0.0
        
        try:
            gp_answer = gp_answer.strip().lower()
            correct_solution = correct_solution.strip().lower()
            
            reward = 0.0
            math_terms = [
                'add', 'subtract', 'multiply', 'divide', 'equals',
                '+', '-', '*', '/', '=', 
                'sum', 'difference', 'product', 'quotient'
            ]
            
            for term in math_terms:
                if term in gp_answer and term in correct_solution:
                    reward += 1.0
                    
            gp_numbers = set(''.join(filter(str.isdigit, gp_answer)))
            correct_numbers = set(''.join(filter(str.isdigit, correct_solution)))
            
            if gp_numbers and correct_numbers:
                reward += 2.0 * len(gp_numbers.intersection(correct_numbers))
                
            if "step" in gp_answer and "step" in correct_solution:
                reward += 2.0
                
            if "answer" in gp_answer.lower() and "answer" in correct_solution.lower():
                gp_last_line = gp_answer.split('\n')[-1]
                correct_last_line = correct_solution.split('\n')[-1]
                if any(n in correct_last_line for n in gp_numbers):
                    reward += 3.0
            max_possible_reward = 10.0
            normalized_reward = min(reward, max_possible_reward)
            
            return float(normalized_reward)
            
        except Exception as e:
            print(f"Reward calculation error: {e}")
            return 0.0

    def reset(self):
        question = self.questions[self.current_question_idx]
        self.current_question_idx = (self.current_question_idx + 1) % len(self.questions)
        
        prompt = f"Problem: {question}\nSolution: Let me solve this step by step.\n"
        self.current_solution = self.generate_answer(
            self.eq_model, 
            question,
            self.device
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        self.current_sequence = encoded['input_ids']
        return self._get_observation()
    
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
                "mean_reward": np.mean(self.episode_rewards[-100:]),
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
        self._log_freq = 1 
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
    import os
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
    
    # Use mixed precision for efficiency
    # from torch.cuda.amp import autocast, GradScaler
    # scaler = GradScaler()
    
    wandb.init(project="math-rlhf", name="phi-math-reasoning-ppo")
    
    config = RLHFConfig()
    
    #mps or gpu or cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Training using device: {device}")
    
    print("Loading models...")
    gp_model = AutoModelForCausalLM.from_pretrained(
        config.gp_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).to(device)
    
    eq_model = AutoModelForCausalLM.from_pretrained(
        config.eq_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).to(device)
    
    #disable caching
    gp_model.config.use_cache = False
    eq_model.config.use_cache = False
    
    gp_model.gradient_checkpointing_enable()
    eq_model.gradient_checkpointing_enable()
    
    model_name = "microsoft/phi-1.5"
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
        learning_rate=1e-4,
        n_steps=64,      
        batch_size=16,   
        n_epochs=3,      
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],
                vf=[64, 64]
            )
        ),
        verbose=1
    )
    
    try:
        print("Starting training...")
        total_timesteps = len(questions) * 50
        model.learn(
            total_timesteps=total_timesteps,
            callback=[WandbLoggingCallback()],
            progress_bar=True
        )
        
        print("Saving final models...")
        gp_model.save_pretrained("./models/finetuned_phi_math")
        
    except Exception as e:
        print(f"Training failed: {e}")
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


