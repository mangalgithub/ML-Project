import sys
import os

# Add project root directory (two levels up) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from data_generator import EVFleetDataGenerator
from src.environments.ev_depot_env import EVDepotEnv
import numpy as np

class GreedyCharging:
    def get_action(self, observation, env):
        actions = []
        for charger_id in range(env.n_chargers):
            vehicle_id = np.where(env.charger_assignment == charger_id)[0]
            if len(vehicle_id) == 0:
                actions.append(0.0)
                continue
            vid = vehicle_id[0]
            current_soc = env.vehicle_soc[vid]
            target_soc = env.vehicles.iloc[vid]['soc_target']
            current_price = env.prices.iloc[env.current_step]['price_per_kwh']

            if current_soc < target_soc:
                if current_price < 0.20:
                    actions.append(1.0)  # Max charge
                else:
                    actions.append(0.3)  # Slow charge
            elif current_soc > target_soc + 0.05:
                if current_price > 0.30:
                    actions.append(-0.5)  # Discharge for V2G
                else:
                    actions.append(0.0)   # Idle
            else:
                actions.append(0.0)  # Maintain
        return np.array(actions)

# 1. Load dataset
generator = EVFleetDataGenerator()
dataset = generator.load_dataset('data/processed/')

# 2. Create environment
env = EVDepotEnv(dataset)

# 3. Instantiate greedy policy agent
greedy_agent = GreedyCharging()

# 4. Run one full episode
observation, info = env.reset()
total_reward = 0.0

for step in range(env.total_steps):
    action = greedy_agent.get_action(observation, env)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

# 5. Get and print metrics
metrics = env.get_episode_metrics()
print("\n=== Greedy Policy Results ===")
print(f"Total Reward: {total_reward:.2f}")
print(f"Total Cost: €{-total_reward:.2f}")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Optional: Break out key results
print(f"\nSummary:\nCost (€): {metrics['total_cost']:.2f}")
print(f"Peak (kW): {metrics['peak_demand']:.2f}")
print(f"On-time (%): {metrics['on_time_rate']*100:.2f}")
