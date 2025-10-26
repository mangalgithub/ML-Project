"""
Training Script for EV Fleet V2G RL Agents
Trains PPO, SAC, and Lagrangian-PPO agents with hyperparameter optimization

Usage:
    python train_rl_agents.py --agent ppo --timesteps 500000
    python train_rl_agents.py --agent sac --timesteps 500000
    python train_rl_agents.py --agent all --optimize

Author: EV Fleet RL Project
Date: October 2025
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import torch

# Stable Baselines3 imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Custom imports (you'll create these)
# from ev_depot_env import EVDepotEnv  # Your RL environment
# from data_generator import EVFleetDataGenerator


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'ppo': {
        'total_timesteps': 500000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {
            'net_arch': [256, 256, 128]
        }
    },
    'sac': {
        'total_timesteps': 500000,
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'learning_starts': 1000,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'policy_kwargs': {
            'net_arch': [256, 256]
        }
    },
    'lagrangian_ppo': {
        'total_timesteps': 500000,
        'constraint_threshold': 0.95,  # Min on-time rate
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'lagrange_lr': 0.01,
        'policy_kwargs': {
            'net_arch': [256, 256, 128]
        }
    }
}


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, env, config: Dict = None):
        """
        Initialize PPO agent
        
        Args:
            env: Gymnasium environment
            config: Configuration dictionary (uses default if None)
        """
        self.env = env
        self.config = config or TRAINING_CONFIG['ppo']
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            policy_kwargs=self.config['policy_kwargs'],
            verbose=1,
            tensorboard_log="./results/tensorboard/ppo/"
        )
        
        print("✅ PPO Agent initialized")
    
    def train(self, total_timesteps: int = None, callbacks: list = None):
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Number of timesteps to train
            callbacks: List of callbacks
        """
        timesteps = total_timesteps or self.config['total_timesteps']
        
        print(f"\n{'='*70}")
        print(f"🚀 Training PPO Agent for {timesteps:,} timesteps")
        print(f"{'='*70}\n")
        
        self.model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n✅ PPO training complete!")
    
    def save(self, path: str):
        """Save trained model"""
        self.model.save(path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        self.model = PPO.load(path, env=self.env)
        print(f"📂 Model loaded from {path}")
    
    def predict(self, observation, deterministic: bool = True):
        """Get action from policy"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action


# ============================================================================
# SAC AGENT
# ============================================================================

class SACAgent:
    """Soft Actor-Critic Agent"""
    
    def __init__(self, env, config: Dict = None):
        """
        Initialize SAC agent
        
        Args:
            env: Gymnasium environment
            config: Configuration dictionary (uses default if None)
        """
        self.env = env
        self.config = config or TRAINING_CONFIG['sac']
        
        # Create SAC model
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.config['learning_rate'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            tau=self.config['tau'],
            gamma=self.config['gamma'],
            learning_starts=self.config['learning_starts'],
            train_freq=self.config['train_freq'],
            gradient_steps=self.config['gradient_steps'],
            ent_coef=self.config['ent_coef'],
            policy_kwargs=self.config['policy_kwargs'],
            verbose=1,
            tensorboard_log="./results/tensorboard/sac/"
        )
        
        print("✅ SAC Agent initialized")
    
    def train(self, total_timesteps: int = None, callbacks: list = None):
        """
        Train the SAC agent
        
        Args:
            total_timesteps: Number of timesteps to train
            callbacks: List of callbacks
        """
        timesteps = total_timesteps or self.config['total_timesteps']
        
        print(f"\n{'='*70}")
        print(f"🚀 Training SAC Agent for {timesteps:,} timesteps")
        print(f"{'='*70}\n")
        
        self.model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n✅ SAC training complete!")
    
    def save(self, path: str):
        """Save trained model"""
        self.model.save(path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        self.model = SAC.load(path, env=self.env)
        print(f"📂 Model loaded from {path}")
    
    def predict(self, observation, deterministic: bool = True):
        """Get action from policy"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action


# ============================================================================
# LAGRANGIAN PPO AGENT
# ============================================================================

class LagrangianPPOAgent:
    """
    Lagrangian-based PPO for constrained optimization
    Ensures on-time departure constraint while minimizing cost
    """
    
    def __init__(self, env, config: Dict = None):
        """
        Initialize Lagrangian PPO agent
        
        Args:
            env: Gymnasium environment
            config: Configuration dictionary
        """
        self.env = env
        self.config = config or TRAINING_CONFIG['lagrangian_ppo']
        
        # Constraint parameters
        self.constraint_threshold = self.config['constraint_threshold']
        self.lagrange_multiplier = 1.0
        self.lagrange_lr = self.config.get('lagrange_lr', 0.01)
        
        # Create base PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            policy_kwargs=self.config['policy_kwargs'],
            verbose=1,
            tensorboard_log="./results/tensorboard/lagrangian_ppo/"
        )
        
        # Track constraint violations
        self.constraint_violations = []
        self.lagrange_history = []
        
        print("✅ Lagrangian-PPO Agent initialized")
        print(f"   Constraint threshold: {self.constraint_threshold:.1%} on-time rate")
    
    def compute_constraint_violation(self, info: Dict) -> float:
        """
        Compute constraint violation (negative = satisfied, positive = violated)
        
        Args:
            info: Episode info dictionary containing 'on_time_rate'
            
        Returns:
            Violation amount (0 if satisfied)
        """
        on_time_rate = info.get('on_time_rate', 1.0)
        violation = max(0, self.constraint_threshold - on_time_rate)
        return violation
    
    def train(self, total_timesteps: int = None, callbacks: list = None):
        """
        Train with Lagrangian constraint
        
        Args:
            total_timesteps: Number of timesteps to train
            callbacks: List of callbacks
        """
        timesteps = total_timesteps or self.config['total_timesteps']
        
        print(f"\n{'='*70}")
        print(f"🚀 Training Lagrangian-PPO Agent for {timesteps:,} timesteps")
        print(f"{'='*70}\n")
        
        # Custom training loop with Lagrangian updates
        n_episodes = 0
        timestep = 0
        
        while timestep < timesteps:
            # Collect episode
            obs, info = self.env.reset()
            episode_reward = 0
            episode_violation = 0
            done = False
            
            while not done and timestep < timesteps:
                # Get action from policy
                action, _ = self.model.predict(obs, deterministic=False)
                
                # Take step
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Compute constraint violation
                violation = self.compute_constraint_violation(info)
                episode_violation += violation
                
                # Augment reward with Lagrangian penalty
                augmented_reward = reward - self.lagrange_multiplier * violation
                
                episode_reward += reward
                timestep += 1
            
            # Update Lagrange multiplier (gradient ascent on dual)
            avg_violation = episode_violation / max(1, timestep)
            self.lagrange_multiplier = max(
                0, 
                self.lagrange_multiplier + self.lagrange_lr * avg_violation
            )
            
            # Track metrics
            self.constraint_violations.append(episode_violation)
            self.lagrange_history.append(self.lagrange_multiplier)
            
            n_episodes += 1
            
            if n_episodes % 10 == 0:
                print(f"Episode {n_episodes} | Timestep {timestep}/{timesteps} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Lambda: {self.lagrange_multiplier:.3f} | "
                      f"On-time: {info.get('on_time_rate', 0):.1%}")
        
        print("\n✅ Lagrangian-PPO training complete!")
    
    def save(self, path: str):
        """Save trained model and Lagrangian parameters"""
        self.model.save(path)
        
        # Save Lagrangian parameters
        lagrange_data = {
            'lagrange_multiplier': self.lagrange_multiplier,
            'constraint_threshold': self.constraint_threshold,
            'violations_history': self.constraint_violations,
            'lagrange_history': self.lagrange_history
        }
        
        with open(f"{path}_lagrange.json", 'w') as f:
            json.dump(lagrange_data, f, indent=2)
        
        print(f"💾 Model and Lagrangian parameters saved to {path}")
    
    def load(self, path: str):
        """Load trained model and Lagrangian parameters"""
        self.model = PPO.load(path, env=self.env)
        
        # Load Lagrangian parameters
        try:
            with open(f"{path}_lagrange.json", 'r') as f:
                lagrange_data = json.load(f)
            
            self.lagrange_multiplier = lagrange_data['lagrange_multiplier']
            self.constraint_violations = lagrange_data['violations_history']
            self.lagrange_history = lagrange_data['lagrange_history']
        except FileNotFoundError:
            print("⚠️  Lagrangian parameters not found, using defaults")
        
        print(f"📂 Model loaded from {path}")
    
    def predict(self, observation, deterministic: bool = True):
        """Get action from policy"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

def optimize_hyperparameters(env, agent_type: str = 'ppo', n_trials: int = 50):
    """
    Optimize hyperparameters using Optuna
    
    Args:
        env: Training environment
        agent_type: 'ppo' or 'sac'
        n_trials: Number of optimization trials
        
    Returns:
        Best hyperparameters dictionary
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 Starting Hyperparameter Optimization for {agent_type.upper()}")
    print(f"   Trials: {n_trials}")
    print(f"{'='*70}\n")
    
    def objective(trial):
        """Optuna objective function"""
        
        if agent_type == 'ppo':
            # Suggest hyperparameters
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
            n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
            ent_coef = trial.suggest_loguniform('ent_coef', 0.001, 0.1)
            
            # Create config
            config = {
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'gamma': gamma,
                'ent_coef': ent_coef,
                'total_timesteps': 100000,  # Shorter for optimization
                'n_epochs': 10,
                'policy_kwargs': {'net_arch': [256, 256]}
            }
            
            # Train agent
            agent = PPOAgent(env, config)
            agent.train(total_timesteps=100000)
            
        elif agent_type == 'sac':
            # Suggest hyperparameters
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
            gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
            tau = trial.suggest_uniform('tau', 0.001, 0.01)
            
            # Create config
            config = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'gamma': gamma,
                'tau': tau,
                'total_timesteps': 100000,
                'buffer_size': 100000,
                'policy_kwargs': {'net_arch': [256, 256]}
            }
            
            # Train agent
            agent = SACAgent(env, config)
            agent.train(total_timesteps=100000)
        
        # Evaluate
        eval_reward = evaluate_agent(agent, env, n_episodes=10)
        
        return eval_reward
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10000)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✅ Optimization complete!")
    print(f"   Best value: {study.best_value:.2f}")
    print(f"   Best params: {study.best_params}")
    
    return study.best_params


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_agent(agent, env, n_episodes: int = 10) -> float:
    """
    Evaluate agent performance
    
    Args:
        agent: Trained agent
        env: Environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Average episode reward
    """
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"   Evaluation: {avg_reward:.2f} ± {std_reward:.2f} (n={n_episodes})")
    
    return avg_reward


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='Train EV Fleet RL Agents')
    parser.add_argument('--agent', type=str, default='ppo', 
                       choices=['ppo', 'sac', 'lagrangian_ppo', 'all'],
                       help='Agent type to train')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Training timesteps')
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Optuna trials for optimization')
    parser.add_argument('--output-dir', type=str, default='results/models/',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
     # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/tensorboard/', exist_ok=True)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     EV Fleet V2G RL Training - Week 3 Implementation             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Load environment
    from src.environments.ev_depot_env import EVDepotEnv
    from data_generator import EVFleetDataGenerator
    
    print("📂 Loading dataset...")
    generator = EVFleetDataGenerator()
    dataset = generator.load_dataset('data/processed/')
    
    print("🏗️  Creating environment...")
    env = EVDepotEnv(dataset)
    env = Monitor(env)  # Wrap for logging
    
    print("✅ Environment loaded successfully!\n")
    
    # Hyperparameter optimization
    if args.optimize:
        if args.agent in ['ppo', 'all']:
            best_params = optimize_hyperparameters(env, 'ppo', args.n_trials)
            print(f"\nBest PPO params: {best_params}")
        
        if args.agent in ['sac', 'all']:
            best_params = optimize_hyperparameters(env, 'sac', args.n_trials)
            print(f"\nBest SAC params: {best_params}")
        
        return
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env,
        eval_freq=10000,
        n_eval_episodes=5,
        best_model_save_path=args.output_dir,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=args.output_dir,
        name_prefix='checkpoint'
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train agents
    agents_to_train = ['ppo', 'sac', 'lagrangian_ppo'] if args.agent == 'all' else [args.agent]
    
    for agent_name in agents_to_train:
        print(f"\n{'='*70}")
        print(f"Training {agent_name.upper()}")
        print(f"{'='*70}")
        
        if agent_name == 'ppo':
            agent = PPOAgent(env)
            agent.train(args.timesteps, callbacks)
            agent.save(f"{args.output_dir}/ppo_final")
            
        elif agent_name == 'sac':
            agent = SACAgent(env)
            agent.train(args.timesteps, callbacks)
            agent.save(f"{args.output_dir}/sac_final")
            
        elif agent_name == 'lagrangian_ppo':
            agent = LagrangianPPOAgent(env)
            agent.train(args.timesteps)
            agent.save(f"{args.output_dir}/lagrangian_ppo_final")
        
        # Evaluate
        print(f"\n📊 Evaluating {agent_name.upper()}...")
        evaluate_agent(agent, env, n_episodes=20)
    
    print(f"\n{'='*70}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nModels saved to: {args.output_dir}")
    print("View training progress:")
    print(f"  tensorboard --logdir results/tensorboard/")


if __name__ == "__main__":
    main()