from dataclasses import dataclass

@dataclass
class RLHFConfig:
    gp_model_path: str = "microsoft/phi-1.5" 
    eq_model_path: str = "microsoft/phi-1.5" 
    # hyperparameters
    learning_rate: float = 1e-4 
    max_epochs: int = 10 
    batch_size: int = 64  
    max_grad_norm: float = 1.0
    ppo_epsilon: float = 0.2
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    kl_target: float = 0.1
    n_steps: int = 512