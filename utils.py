import wandb
import json

class RLHFLogger:
    def __init__(self):
        self.reward_history = []
        self.kl_div_history = []
        
    def log_step(self, reward, kl_div, gp_answer, eq_answer):
        self.reward_history.append(reward)
        self.kl_div_history.append(kl_div)
        
        wandb.log({
            'reward': reward,
            'kl_divergence': kl_div,
            'gp_eq_match_rate': int(gp_answer == eq_answer)
        })

def load_questions(file_path="./list_of_questions_new.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def log_training_results(epoch, q_id, gp_answer, eq_solution):
    with open(f"results_epoch_{epoch}.txt", "a") as f:
        f.write(f"Question {q_id}:\nGP Answer: {gp_answer}\nEQ Solution: {eq_solution}\n\n")
    
    wandb.log({
        'epoch': epoch,
        'question_id': q_id,
        'gp_answer': gp_answer,
        'eq_solution': eq_solution
    })