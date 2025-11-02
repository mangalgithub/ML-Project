import numpy as np
from data_generator import EVFleetDataGenerator
from src.environments.ev_depot_env import EVDepotEnv

# Import your RL libraries
from stable_baselines3 import PPO, SAC

# ---- Greedy Baseline ----
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
                    actions.append(1.0)
                else:
                    actions.append(0.3)
            elif current_soc > target_soc + 0.05:
                if current_price > 0.30:
                    actions.append(-0.5)
                else:
                    actions.append(0.0)
            else:
                actions.append(0.0)
        return np.array(actions)

# ---- MILP Baseline ----
import cvxpy as cp
class MILPOptimization:
    def optimize(self, env, horizon=96):
        n_steps = min(horizon, env.total_steps - env.current_step)
        charge_power = cp.Variable((env.n_chargers, n_steps))
        discharge_power = cp.Variable((env.n_chargers, n_steps))
        soc = cp.Variable((env.n_vehicles, n_steps + 1))
        peak_power = cp.Variable(nonneg=True)
        energy_costs = []
        for t in range(n_steps):
            price = env.prices.iloc[env.current_step + t]['price_per_kwh']
            v2g_price = env.prices.iloc[env.current_step + t]['v2g_sell_price']
            charge_cost = cp.sum(charge_power[:, t]) * price * env.timestep_hours
            discharge_revenue = cp.sum(discharge_power[:, t]) * v2g_price * env.timestep_hours
            energy_costs.append(charge_cost - discharge_revenue)
        total_cost = cp.sum(energy_costs)
        peak_penalty = env.peak_penalty_coef * peak_power
        objective = cp.Minimize(total_cost + peak_penalty)
        constraints = []
        constraints.append(peak_power >= 0)
        # Initial SoC
        for vid in range(env.n_vehicles):
            constraints.append(soc[vid, 0] == env.vehicle_soc[vid])
        for vid in range(env.n_vehicles):
            v_spec = env.vehicles.iloc[vid]
            capacity = v_spec['battery_capacity']
            for t in range(n_steps):
                cid = env.charger_assignment[vid]
                if cid >= 0:
                    charge_energy = charge_power[cid, t] * env.timestep_hours * v_spec['charging_efficiency']
                    discharge_energy = discharge_power[cid, t] * env.timestep_hours / v_spec['discharging_efficiency']
                    constraints.append(
                        soc[vid, t+1] == soc[vid, t] + charge_energy/capacity - discharge_energy/capacity
                    )
                    constraints.append(charge_power[cid, t] >= 0)
                    constraints.append(charge_power[cid, t] <= v_spec['max_charging_power'])
                    constraints.append(discharge_power[cid, t] >= 0)
                    constraints.append(discharge_power[cid, t] <= v_spec['max_discharging_power'])
                else:
                    constraints.append(soc[vid, t+1] == soc[vid, t])
                constraints.append(soc[vid, t] >= v_spec['soc_min'] * 0.95)
                constraints.append(soc[vid, t] <= v_spec['soc_max'] * 1.05)
        for t in range(n_steps):
            base_load = env.base_load.iloc[env.current_step + t]['base_load_kw']
            total_power = base_load + cp.sum(charge_power[:, t]) - cp.sum(discharge_power[:, t])
            constraints.append(peak_power >= total_power)
        for _, schedule in env.schedules.iterrows():
            if env.current_step <= schedule['departure_step'] < env.current_step + n_steps:
                vid = int(schedule['vehicle_id'])
                t_dep = int(schedule['departure_step'] - env.current_step)
                target_soc = env.vehicles.iloc[vid]['soc_target']
                relaxed_soc_max = env.vehicles.iloc[vid]['soc_max'] * 1.05
                adjusted_target_soc = min(target_soc, relaxed_soc_max)
                constraints.append(soc[vid, t_dep] >= adjusted_target_soc)
        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver=cp.ECOS, verbose=False)
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Solver failed with status {problem.status}")
            return None, None
        return charge_power.value, discharge_power.value

def eval_greedy(env):
    agent = GreedyCharging()
    obs, info = env.reset()
    for _ in range(env.total_steps):
        action = agent.get_action(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return env.get_episode_metrics()

def eval_milp(env):
    milp = MILPOptimization()
    env.reset()
    total_energy_charged = 0.0
    rolling_horizon = 96
    while env.current_step < env.total_steps:
        charge_power, discharge_power = milp.optimize(env, horizon=rolling_horizon)
        if charge_power is None or discharge_power is None:
            env.current_step += rolling_horizon
            continue
        n_steps = charge_power.shape[1]
        for t in range(n_steps):
            for vid in range(env.n_vehicles):
                cid = env.charger_assignment[vid]
                v_spec = env.vehicles.iloc[vid]
                capacity = v_spec['battery_capacity']
                if cid >= 0:
                    charge_energy = charge_power[cid, t] * env.timestep_hours * v_spec['charging_efficiency']
                    discharge_energy = discharge_power[cid, t] * env.timestep_hours / v_spec['discharging_efficiency']
                    soc_change = charge_energy / capacity - discharge_energy / capacity
                    env.vehicle_soc[vid] = min(v_spec['soc_max'], max(v_spec['soc_min'], env.vehicle_soc[vid] + soc_change))
                    total_energy_charged += max(charge_energy, 0)
            env.current_step += 1
    m = env.get_episode_metrics()
    m['energy_charged'] = total_energy_charged
    return m

def eval_rl(model_path, env, RLClass):
    model = RLClass.load(model_path, env=env)
    obs, info = env.reset()
    for _ in range(env.total_steps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return env.get_episode_metrics()

def format_metrics(metrics):
    return f"{metrics['total_cost']:.0f} | {metrics['peak_demand']:.0f} | {metrics['on_time_rate']*100:.1f} | {metrics.get('energy_charged', metrics.get('total_energy_charged', 0)):.0f}"

if __name__ == "__main__":
    generator = EVFleetDataGenerator()
    dataset = generator.load_dataset('data/processed/')
    env = EVDepotEnv(dataset)
    rows = []

    # Greedy
    metrics_greedy = eval_greedy(env)
    rows.append(("Greedy", metrics_greedy))
    print("Greedy: " + format_metrics(metrics_greedy))

    # MILP
    metrics_milp = eval_milp(env)
    rows.append(("MILP", metrics_milp))
    print("MILP: " + format_metrics(metrics_milp))

    # PPO
    metrics_ppo = eval_rl("results/models/ppo_final.zip", env, PPO)
    rows.append(("PPO", metrics_ppo))
    print("PPO: " + format_metrics(metrics_ppo))

    # SAC
    metrics_sac = eval_rl("results/models/sac_final.zip", env, SAC)
    rows.append(("SAC", metrics_sac))
    print("SAC: " + format_metrics(metrics_sac))

    # Lagrangian-PPO (usually PPO-based)
    metrics_lppo = eval_rl("results/models/lagrangian_ppo_final.zip", env, PPO)
    rows.append(("Lagrangian-PPO", metrics_lppo))
    print("Lagrangian-PPO: " + format_metrics(metrics_lppo))

    # Print as table
    print("\nMethod | Total Cost | Peak Demand | On-Time Departure (%) | Energy Charged (kWh)")
    print("-"*65)
    for label, m in rows:
        print(f"{label:15s} | {m['total_cost']:10.0f} | {m['peak_demand']:11.0f} | {m['on_time_rate']*100:19.1f} | {m.get('energy_charged', m.get('total_energy_charged', 0)):19.0f}")
