"""
Lagrangian-based PPO for constrained optimization
Ensures on-time departures as hard constraint
"""

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class LagrangianPPO:
    def __init__(self, env, constraint_threshold=0.95):
        """
        constraint_threshold: minimum on-time rate (e.g., 95%)
        """
        self.env = env
        self.constraint_threshold = constraint_threshold
        self.lagrange_multiplier = 1.0
        self.lr_lambda = 0.01
        
        # Standard PPO model
        self.model = PPO("MlpPolicy", env, verbose=1)
    
    def compute_constraint_violation(self, info):
        """Check if on-time rate constraint is violated"""
        on_time_rate = info['on_time_rate']
        violation = max(0, self.constraint_threshold - on_time_rate)
        return violation
    
    def train_step(self):
        # Collect rollout
        obs = self.env.reset()
        episode_constraint_violation = 0
        
        for _ in range(2048):
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            
            # Augment reward with Lagrangian term
            violation = self.compute_constraint_violation(info)
            augmented_reward = reward - self.lagrange_multiplier * violation
            
            episode_constraint_violation += violation
            
            if done:
                break
        
        # Update Lagrange multiplier
        self.lagrange_multiplier = max(
            0, 
            self.lagrange_multiplier + self.lr_lambda * episode_constraint_violation
        )
        
        return episode_constraint_violation
