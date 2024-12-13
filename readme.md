# RLHF Training for Mathematical Reasoning

This project implements Reinforcement Learning from Human Feedback (RLHF) to improve mathematical reasoning capabilities of language models.

## Overview

The system uses PPO (Proximal Policy Optimization) to fine-tune language models for mathematical problem solving. It consists of:

- A base language model (Microsoft's Phi-1.5)
- A reward model that evaluates mathematical reasoning
- A PPO training loop for policy optimization
- Evaluation metrics and logging via WandB and TensorBoard

## Installation

1. Clone the repository: ```git clone https://github.com/Anushk97/shodh_RL.git```

2. Install dependencies: ```pip install -r requirements.txt```

3. Run the training script (choose trainer 01 or trainer 02): ```python train_rlhf.py```

4. View training results in WandB and TensorBoard.

