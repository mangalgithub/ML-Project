"""
Proximal Policy Optimization (PPO) for EV Charging
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch

class PPOAgent:
    def __init__(self, env, learning_rate=3e-4):
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=[256, 256, 128]
            ),
            verbose=1,
            tensorboard_log="./results/tensorboard/"
        )
    
    def train(self, total_timesteps=500000):
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=EvalCallback(
                eval_env,
                eval_freq=10000,
                best_model_save_path='./results/models/ppo_best'
            )
        )
    
    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
