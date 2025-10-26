"""
Soft Actor-Critic (SAC) for continuous control
"""

from stable_baselines3 import SAC

class SACAgent:
    def __init__(self, env, learning_rate=3e-4):
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256]
            ),
            verbose=1,
            tensorboard_log="./results/tensorboard/"
        )
    
    def train(self, total_timesteps=500000):
        self.model.learn(total_timesteps=total_timesteps)
