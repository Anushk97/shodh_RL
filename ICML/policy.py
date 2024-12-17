import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn.functional as F
from typing import Tuple
from torch.distributions import Categorical

class LanguagePPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # decide which token to generate (actor)
        self.action_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        # estimate current state (critic)
        self.value_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_float = obs.float()
        features = self.feature_extractor(obs_float)
        
        
        action_logits = self.action_net(features)
        action_dist = Categorical(logits=action_logits)
        
        if deterministic:
            actions = torch.argmax(action_logits, dim=1)
        else:
            actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)
        
        values = self.value_net(features)
        return actions, values, log_probs

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_float = obs.float()
        features = self.feature_extractor(obs_float)
        
        action_logits = self.action_net(features)
        action_dist = Categorical(logits=action_logits)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        values = self.value_net(features)
        
        return values, log_probs, entropy

    def predict_values(self, obs):
        """
        Get the estimated values for given observations.
        """
        obs_float = obs.float()
        features = self.feature_extractor(obs_float)
        values = self.value_net(features)
        return values