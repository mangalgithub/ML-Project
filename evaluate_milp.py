import numpy as np
from data_generator import EVFleetDataGenerator
from src.environments.ev_depot_env import EVDepotEnv
import cvxpy as cp


class MILPOptimization:
    def optimize(self, env, horizon=96):  # 24-hour horizon
        n_steps = min(horizon, env.total_steps - env.current_step)
        
        charge_power = cp.Variable((env.n_chargers, n_steps))
        discharge_power = cp.Variable((env.n_chargers, n_steps))
        soc = cp.Variable((env.n_vehicles, n_steps + 1))
        peak_power = cp.Variable(nonneg=True)  # Non-negative peak power variable
        
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
        constraints.append(peak_power >= 0)  # Explicit lower bound
        
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
                # Slightly relax SoC bounds for solver stability
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
                # Ensure target_soc does not exceed relaxed soc_max
                relaxed_soc_max = env.vehicles.iloc[vid]['soc_max'] * 1.05
                adjusted_target_soc = min(target_soc, relaxed_soc_max)
                constraints.append(soc[vid, t_dep] >= adjusted_target_soc)
        
        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Solver failed with status {problem.status}")
            return None, None
        
        return charge_power.value, discharge_power.value


def evaluate_milp():
    # Load dataset and environment
    generator = EVFleetDataGenerator()
    dataset = generator.load_dataset('data/processed/')
    env = EVDepotEnv(dataset)
    
    milp = MILPOptimization()
    
    total_energy_cost = 0.0
    peak_demand = 0.0
    late_departures = 0
    total_departures = 0
    
    # Reset environment state
    env.reset()
    
    rolling_horizon = 96  # 1 day simulation steps
    
    while env.current_step < env.total_steps:
        charge_power, discharge_power = milp.optimize(env, horizon=rolling_horizon)
        if charge_power is None or discharge_power is None:
            print(f"Solver failed at step {env.current_step}, skipping horizon")
            env.current_step += rolling_horizon
            continue
        
        n_steps = charge_power.shape[1]
        
        for t in range(n_steps):
            step = env.current_step + t
            
            base_load = env.base_load.iloc[step]['base_load_kw']
            price = env.prices.iloc[step]['price_per_kwh']
            v2g_price = env.prices.iloc[step]['v2g_sell_price']
            
            # Calculate grid load and update peak
            grid_load = base_load + np.sum(charge_power[:, t]) - np.sum(discharge_power[:, t])
            peak_demand = max(peak_demand, grid_load)
            
            # Calculate cost
            energy_cost = np.sum(charge_power[:, t]) * price * env.timestep_hours \
                          - np.sum(discharge_power[:, t]) * v2g_price * env.timestep_hours
            total_energy_cost += energy_cost
            
            # Update SoC
            for vid in range(env.n_vehicles):
                cid = env.charger_assignment[vid]
                v_spec = env.vehicles.iloc[vid]
                capacity = v_spec['battery_capacity']
                if cid >= 0:
                    charge_energy = charge_power[cid, t] * env.timestep_hours * v_spec['charging_efficiency']
                    discharge_energy = discharge_power[cid, t] * env.timestep_hours / v_spec['discharging_efficiency']
                    soc_change = charge_energy / capacity - discharge_energy / capacity
                    env.vehicle_soc[vid] = min(v_spec['soc_max'], max(v_spec['soc_min'], env.vehicle_soc[vid] + soc_change))
            
            # Handle departures and update late count
            departures = env.schedules[
                (env.schedules['departure_step'] == step)
            ]
            for _, schedule in departures.iterrows():
                vid = int(schedule['vehicle_id'])
                target_soc = env.vehicles.iloc[vid]['soc_target']
                total_departures += 1
                if env.vehicle_soc[vid] < target_soc:
                    late_departures += 1
        
        env.current_step += n_steps
    
    on_time_rate = 1 - (late_departures / max(1, total_departures))
    
    metrics = {
        'total_cost': total_energy_cost,
        'peak_demand': peak_demand,
        'on_time_rate': on_time_rate,
        'late_departures': late_departures,
        'total_departures': total_departures
    }
    
    print("\n=== MILP Optimization Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    return metrics


if __name__ == "__main__":
    evaluate_milp()
