"""
Greedy Baseline: Charge immediately when price is low, 
discharge when price is high (if above target SoC)
"""

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
            
            # Simple logic:
            # - If SoC < target and price < 0.20: charge at max
            # - If SoC > target and price > 0.30: discharge
            # - Otherwise: idle
            
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
