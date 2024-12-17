import torch
import gym
from gym import spaces
import numpy as np
import torch.nn.functional as F


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
            dtype=np.int64,
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
                    f"Problem: {prompt}\n" "Let me solve this step by step:\n"
                )

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True,
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
                    no_repeat_ngram_size=3,
                )

                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
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
            print("--------------------------------")
            print(f"Raw Correct Solution (EQ): {correct_solution}")
            print("--------------------------------")

            math_terms = [
                "add",
                "subtract",
                "multiply",
                "divide",
                "equals",
                "sum",
                "total",
                "difference",
                "product",
                "quotient",
                "+",
                "-",
                "*",
                "/",
                "=",
                "$",
            ]

            # score reasoning steps
            gp_steps = set(
                word
                for word in gp_answer.split()
                if word in math_terms or any(c.isdigit() for c in word)
            )
            correct_steps = set(
                word
                for word in correct_solution.split()
                if word in math_terms or any(c.isdigit() for c in word)
            )

            if correct_steps:
                steps_score = len(gp_steps.intersection(correct_steps)) / len(
                    correct_steps
                )

            # score answer matching
            if "final answer" in gp_answer and "final answer" in correct_solution:
                gp_final = gp_answer[gp_answer.find("final answer") :].split("\n")[0]
                correct_final = correct_solution[
                    correct_solution.find("final answer") :
                ].split("\n")[0]
                answer_match_score = 1.0 if gp_final == correct_final else 0.0

            # score reasoning
            reasoning_indicators = ["because", "therefore", "since", "so", "as"]
            reasoning_score = any(
                indicator in gp_answer for indicator in reasoning_indicators
            )

            # combine scores (weighted average)
            final_score = (
                0.4 * answer_match_score
                + 0.4 * steps_score
                + 0.2 * float(reasoning_score)  # explanation is good but less critical
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
        self.current_question_idx = (self.current_question_idx + 1) % len(
            self.questions
        )
        question = self.questions[self.current_question_idx]

        eq_prompt = (
            "You are a math equation generator. Convert this problem into "
            "mathematical equations, solve them step by step, and provide "
            f"the final numerical answer:\n{question}\nSolution:"
        )
        self.current_solution = self.generate_answer(
            self.eq_model, question, next(self.eq_model.parameters()).device
        )

        encoded = self.tokenizer(
            question,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
        )
        obs = encoded["input_ids"][0]

        return obs

    def step(self, action):
        try:
            question = self.questions[self.current_question_idx]
            action_tensor = torch.FloatTensor(action).to(
                next(self.gp_model.parameters()).device
            )
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
                    return_attention_mask=True,
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
                        length_penalty=1.0,
                    )

                    answer = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
            except Exception as e:
                print(f"Generation error: {e}")
                answer = "Error: Generation failed"

            reward = self.calculate_reward(answer, self.current_solution)
            self.current_question_idx = (self.current_question_idx + 1) % len(
                self.questions
            )

            next_obs = self.reset()
            done = True

            info = {
                "answer": answer,
                "solution": self.current_solution,
                "reward": reward,
            }

            # print('INFO...', info)

            return next_obs, reward, done, info

        except Exception as e:
            print(f"Step error: {str(e)}")
            return self.reset(), 0.0, True, {}
