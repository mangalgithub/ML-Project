"""
MILP Optimization (Small-scale perfect information)
Minimizes: energy_cost + peak_penalty
Subject to: SoC constraints, charger limits, departure requirements
"""

import cvxpy as cp

class MILPOptimization:
    def optimize(self, env, horizon=96):  # 24-hour horizon
        n_steps = min(horizon, env.total_steps - env.current_step)
        
        # Decision variables
        charge_power = cp.Variable((env.n_chargers, n_steps))
        discharge_power = cp.Variable((env.n_chargers, n_steps))
        soc = cp.Variable((env.n_vehicles, n_steps + 1))
        peak_power = cp.Variable()
        
        # Objective: minimize cost + peak penalty
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
        
        # Constraints
        constraints = []
        
        # Initial SoC
        for vid in range(env.n_vehicles):
            constraints.append(soc[vid, 0] == env.vehicle_soc[vid])
        
        # SoC dynamics and limits
        for vid in range(env.n_vehicles):
            v_spec = env.vehicles.iloc[vid]
            capacity = v_spec['battery_capacity']
            
            for t in range(n_steps):
                # Find charger for this vehicle
                cid = env.charger_assignment[vid]
                if cid >= 0:
                    # SoC update
                    charge_energy = charge_power[cid, t] * env.timestep_hours * v_spec['charging_efficiency']
                    discharge_energy = discharge_power[cid, t] * env.timestep_hours / v_spec['discharging_efficiency']
                    
                    constraints.append(
                        soc[vid, t+1] == soc[vid, t] + charge_energy/capacity - discharge_energy/capacity
                    )
                    
                    # Power limits
                    constraints.append(charge_power[cid, t] >= 0)
                    constraints.append(charge_power[cid, t] <= v_spec['max_charging_power'])
                    constraints.append(discharge_power[cid, t] >= 0)
                    constraints.append(discharge_power[cid, t] <= v_spec['max_discharging_power'])
                else:
                    constraints.append(soc[vid, t+1] == soc[vid, t])
                
                # SoC bounds
                constraints.append(soc[vid, t] >= v_spec['soc_min'])
                constraints.append(soc[vid, t] <= v_spec['soc_max'])
        
        # Peak power constraint
        for t in range(n_steps):
            base_load = env.base_load.iloc[env.current_step + t]['base_load_kw']
            total_power = base_load + cp.sum(charge_power[:, t]) - cp.sum(discharge_power[:, t])
            constraints.append(peak_power >= total_power)
        
        # Departure constraints
        for _, schedule in env.schedules.iterrows():
            if env.current_step <= schedule['departure_step'] < env.current_step + n_steps:
                vid = schedule['vehicle_id']
                t_dep = schedule['departure_step'] - env.current_step
                target_soc = env.vehicles.iloc[vid]['soc_target']
                constraints.append(soc[vid, t_dep] >= target_soc)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        return charge_power.value, discharge_power.value
