"""
EV Fleet Data Generator for V2G Peak Shaving RL Project
Generates realistic synthetic data for depot charging optimization

Author: EV Fleet RL Project
Date: October 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os


class EVFleetDataGenerator:
    """
    Generate synthetic but realistic EV fleet operational data for RL training
    
    Features:
    - Heterogeneous fleet with varying battery capacities (40-80 kWh)
    - Realistic daily operation patterns (morning departure, evening return)
    - Time-of-use electricity tariff with peak/off-peak pricing
    - Vehicle-to-Grid (V2G) capability for all chargers
    - Depot base load simulation
    - 15-minute time resolution (96 timesteps per day)
    
    Example:
        >>> generator = EVFleetDataGenerator(n_vehicles=20, n_chargers=15)
        >>> dataset = generator.generate_complete_dataset()
        >>> generator.save_dataset(dataset, 'data/processed/')
    """
    
    def __init__(self, 
                 n_vehicles: int = 20,
                 n_chargers: int = 15,
                 simulation_days: int = 7,
                 timestep_minutes: int = 15,
                 seed: int = 42):
        """
        Initialize the EV fleet data generator
        
        Args:
            n_vehicles: Number of EVs in the fleet (default: 20)
            n_chargers: Number of charging stations at depot (default: 15)
            simulation_days: Number of days to simulate (default: 7)
            timestep_minutes: Time resolution in minutes (default: 15)
            seed: Random seed for reproducibility (default: 42)
        """
        self.n_vehicles = n_vehicles
        self.n_chargers = n_chargers
        self.simulation_days = simulation_days
        self.timestep_minutes = timestep_minutes
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Calculate time parameters
        self.steps_per_day = (24 * 60) // timestep_minutes  # 96 steps for 15-min resolution
        self.total_steps = self.steps_per_day * simulation_days
        
        # Vehicle parameters (based on commercial EVs: Nissan eNV200, Ford E-Transit)
        self.battery_capacity_min = 40.0   # kWh (small commercial EV)
        self.battery_capacity_max = 80.0   # kWh (large commercial EV)
        self.charging_power_max = 50.0     # kW (Level 2 DC fast charging)
        self.discharging_power_max = 30.0  # kW (V2G discharge capability)
        self.charging_efficiency = 0.95    # 95% charging efficiency
        self.discharging_efficiency = 0.90 # 90% discharging efficiency (inverter losses)
        
        # SoC constraints for battery health
        self.soc_min_safe = 0.20  # Don't discharge below 20%
        self.soc_max_safe = 0.90  # Don't charge above 90%
        
        print(f"✅ EVFleetDataGenerator initialized")
        print(f"   Fleet size: {n_vehicles} vehicles, {n_chargers} chargers")
        print(f"   Simulation: {simulation_days} days ({self.total_steps} timesteps)")
        print(f"   Resolution: {timestep_minutes} minutes per timestep")
    
    def generate_complete_dataset(self) -> Dict:
        """
        Generate all required data components for the simulation
        
        Returns:
            Dictionary containing:
            - vehicles: DataFrame with vehicle specifications
            - schedules: DataFrame with route schedules (departure/arrival times)
            - prices: DataFrame with electricity prices (time-of-use tariff)
            - base_load: DataFrame with depot base load profile
            - chargers: DataFrame with charger specifications
            - metadata: Dictionary with simulation parameters
        """
        
        print("\n" + "="*70)
        print("🔄 GENERATING EV FLEET DATASET")
        print("="*70)
        
        # 1. Generate vehicle specifications
        print("\n[1/5] Generating vehicle specifications...")
        vehicles = self._generate_vehicle_specs()
        print(f"      ✓ Generated specs for {len(vehicles)} vehicles")
        
        # 2. Generate route schedules
        print("[2/5] Generating route schedules...")
        schedules = self._generate_route_schedules()
        print(f"      ✓ Generated {len(schedules)} trips")
        
        # 3. Generate electricity prices
        print("[3/5] Generating electricity tariff...")
        prices = self._generate_electricity_prices()
        print(f"      ✓ Generated {len(prices)} price points")
        
        # 4. Generate base load
        print("[4/5] Generating depot base load...")
        base_load = self._generate_base_load()
        print(f"      ✓ Generated {len(base_load)} load points")
        
        # 5. Generate charger specifications
        print("[5/5] Generating charger specifications...")
        chargers = self._generate_charger_specs()
        print(f"      ✓ Generated specs for {len(chargers)} chargers")
        
        # Package everything into a dictionary
        dataset = {
            'vehicles': vehicles,
            'schedules': schedules,
            'prices': prices,
            'base_load': base_load,
            'chargers': chargers,
            'metadata': {
                'n_vehicles': self.n_vehicles,
                'n_chargers': self.n_chargers,
                'simulation_days': self.simulation_days,
                'timestep_minutes': self.timestep_minutes,
                'total_steps': self.total_steps,
                'steps_per_day': self.steps_per_day,
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'seed': self.seed
            }
        }
        
        print("\n✅ Dataset generation complete!")
        self._print_dataset_summary(dataset)
        
        return dataset
    
    def _generate_vehicle_specs(self) -> pd.DataFrame:
        """
        Generate heterogeneous vehicle specifications
        
        Returns:
            DataFrame with columns:
            - vehicle_id: Unique identifier (0 to n_vehicles-1)
            - battery_capacity: Total battery capacity in kWh
            - max_charging_power: Maximum charging power in kW
            - max_discharging_power: Maximum V2G discharge power in kW
            - charging_efficiency: Charging efficiency (0-1)
            - discharging_efficiency: Discharging efficiency (0-1)
            - soc_min: Minimum safe State of Charge (0-1)
            - soc_max: Maximum safe State of Charge (0-1)
            - soc_target: Target SoC at departure (0-1)
        """
        
        vehicles = []
        
        for vid in range(self.n_vehicles):
            # Random battery capacity (uniform distribution across range)
            capacity = np.random.uniform(
                self.battery_capacity_min, 
                self.battery_capacity_max
            )
            
            # Slight variations in efficiency (Gaussian distribution around mean)
            charge_eff = np.clip(
                np.random.normal(self.charging_efficiency, 0.02),
                0.90, 0.98
            )
            discharge_eff = np.clip(
                np.random.normal(self.discharging_efficiency, 0.02),
                0.85, 0.95
            )
            
            # Target SoC varies slightly by vehicle (some drivers more conservative)
            target_soc = np.clip(
                np.random.normal(0.80, 0.05),
                0.70, 0.85
            )
            
            vehicles.append({
                'vehicle_id': vid,
                'battery_capacity': round(capacity, 2),
                'max_charging_power': self.charging_power_max,
                'max_discharging_power': self.discharging_power_max,
                'charging_efficiency': round(charge_eff, 3),
                'discharging_efficiency': round(discharge_eff, 3),
                'soc_min': self.soc_min_safe,
                'soc_max': self.soc_max_safe,
                'soc_target': round(target_soc, 2)
            })
        
        return pd.DataFrame(vehicles)
    
    def _generate_route_schedules(self) -> pd.DataFrame:
        """
        Generate realistic route schedules for fleet vehicles
        
        Patterns simulated:
        - 80% of vehicles: Standard day shift (depart 6-10 AM, return 4-10 PM)
        - 20% of vehicles: Alternative patterns (night shift, multiple trips, long routes)
        
        Returns:
            DataFrame with columns:
            - vehicle_id: Which vehicle (0 to n_vehicles-1)
            - day: Day of simulation (0-indexed)
            - departure_step: Timestep when vehicle leaves depot
            - arrival_step: Timestep when vehicle returns to depot
            - departure_hour: Hour of departure (e.g., 7.5 = 7:30 AM)
            - arrival_hour: Hour of arrival (e.g., 18.0 = 6:00 PM)
            - energy_consumed_kwh: Energy used during trip (kWh)
            - soc_at_arrival: State of Charge when vehicle returns (0-1)
        """
        
        schedules = []
        
        # Get vehicle capacities for SoC calculation
        vehicle_capacities = {}
        for vid in range(self.n_vehicles):
            capacity = np.random.uniform(
                self.battery_capacity_min,
                self.battery_capacity_max
            )
            vehicle_capacities[vid] = capacity
        
        for day in range(self.simulation_days):
            for vid in range(self.n_vehicles):
                
                # Decide pattern for this vehicle-day
                is_standard_pattern = np.random.random() < 0.80
                
                if is_standard_pattern:
                    # Standard daily pattern: morning departure, evening return
                    
                    # Departure time: 6-10 AM (Gaussian around 7:30 AM)
                    departure_hour = np.random.normal(7.5, 1.0)
                    departure_hour = np.clip(departure_hour, 6.0, 10.0)
                    
                    # Return time: 4-10 PM (Gaussian around 6:00 PM)
                    arrival_hour = np.random.normal(18.0, 1.5)
                    arrival_hour = np.clip(arrival_hour, 16.0, 22.0)
                    
                    # Ensure arrival after departure
                    if arrival_hour <= departure_hour:
                        arrival_hour = departure_hour + np.random.uniform(6.0, 10.0)
                        arrival_hour = min(arrival_hour, 23.5)
                    
                    # Time away (hours)
                    time_away = arrival_hour - departure_hour
                    
                    # Energy consumption: 2-3.5 kWh per hour (typical for delivery routes)
                    hourly_consumption = np.random.uniform(2.0, 3.5)
                    energy_consumed = hourly_consumption * time_away
                    
                else:
                    # Alternative pattern: night shift, long routes, or early start
                    
                    pattern_type = np.random.choice(['night_shift', 'long_route', 'early_start'])
                    
                    if pattern_type == 'night_shift':
                        # Depart evening, return early morning next day
                        departure_hour = np.random.uniform(20.0, 23.0)
                        arrival_hour = np.random.uniform(4.0, 8.0)
                        time_away = (24.0 - departure_hour) + arrival_hour
                        
                    elif pattern_type == 'long_route':
                        # Full day route
                        departure_hour = np.random.uniform(5.0, 7.0)
                        arrival_hour = np.random.uniform(20.0, 23.0)
                        time_away = arrival_hour - departure_hour
                        
                    else:  # early_start
                        # Very early morning departure
                        departure_hour = np.random.uniform(4.0, 6.0)
                        arrival_hour = np.random.uniform(14.0, 18.0)
                        time_away = arrival_hour - departure_hour
                    
                    hourly_consumption = np.random.uniform(2.0, 3.5)
                    energy_consumed = hourly_consumption * time_away
                
                # Convert hours to timesteps
                departure_step = int(
                    day * self.steps_per_day + 
                    departure_hour * (60 / self.timestep_minutes)
                )
                
                arrival_step = int(
                    day * self.steps_per_day + 
                    arrival_hour * (60 / self.timestep_minutes)
                )
                
                # Handle next-day arrivals (night shift)
                if arrival_hour < departure_hour:
                    arrival_step += self.steps_per_day
                
                # Cap energy consumption at reasonable maximum
                energy_consumed = min(energy_consumed, 65.0)
                
                # Calculate SoC at arrival
                # Assume vehicle starts at target SoC (0.8), subtract energy used
                capacity = vehicle_capacities[vid]
                soc_depletion = energy_consumed / capacity
                soc_at_arrival = max(0.20, 0.80 - soc_depletion)
                
                schedules.append({
                    'vehicle_id': vid,
                    'day': day,
                    'departure_step': departure_step,
                    'arrival_step': arrival_step,
                    'departure_hour': round(departure_hour, 2),
                    'arrival_hour': round(arrival_hour, 2),
                    'energy_consumed_kwh': round(energy_consumed, 2),
                    'soc_at_arrival': round(soc_at_arrival, 3)
                })
        
        df = pd.DataFrame(schedules)
        
        # Sort by departure time for easier processing
        df = df.sort_values(['departure_step', 'vehicle_id']).reset_index(drop=True)
        
        return df
    
    def _generate_electricity_prices(self) -> pd.DataFrame:
        """
        Generate time-of-use electricity tariff
        
        Price structure (€/kWh):
        - Peak hours (5-9 PM): €0.35 base
        - Mid-peak (7-11 AM, 1-5 PM): €0.20 base
        - Off-peak (11 PM - 7 AM): €0.10 base
        
        Additional factors:
        - Daily variation (weekly pattern)
        - Random noise (market volatility)
        - Weekend discount (10% reduction)
        
        Returns:
            DataFrame with columns:
            - timestep: Time index (0 to total_steps-1)
            - hour: Hour of day (0-24, float)
            - day: Day number
            - day_of_week: Day of week (0=Monday, 6=Sunday)
            - price_per_kwh: Grid purchase price (€/kWh)
            - v2g_sell_price: V2G feed-in price (€/kWh)
            - price_category: 'peak', 'mid-peak', or 'off-peak'
        """
        
        prices = []
        
        for step in range(self.total_steps):
            # Calculate time information
            hour = (step % self.steps_per_day) * self.timestep_minutes / 60.0
            day = step // self.steps_per_day
            day_of_week = day % 7  # 0=Monday, 6=Sunday
            
            # Base price by time of day
            if 17.0 <= hour < 21.0:  # Peak: 5-9 PM
                base_price = 0.35
                category = 'peak'
            elif (7.0 <= hour < 11.0) or (13.0 <= hour < 17.0):  # Mid-peak
                base_price = 0.20
                category = 'mid-peak'
            else:  # Off-peak
                base_price = 0.10
                category = 'off-peak'
            
            # Weekend discount (Saturday=5, Sunday=6)
            if day_of_week >= 5:
                base_price *= 0.90
            
            # Weekly pattern (wholesale market fluctuation)
            weekly_factor = 1.0 + 0.10 * np.sin(2 * np.pi * day / 7.0)
            
            # Random noise (market volatility)
            noise = np.random.normal(0, 0.015)
            
            # Final price
            price = base_price * weekly_factor + noise
            price = max(0.05, price)  # Price floor at €0.05/kWh
            
            # V2G sell price (typically 70-80% of purchase price)
            v2g_price = price * 0.75
            
            prices.append({
                'timestep': step,
                'hour': round(hour, 2),
                'day': day,
                'day_of_week': day_of_week,
                'price_per_kwh': round(price, 4),
                'v2g_sell_price': round(v2g_price, 4),
                'price_category': category
            })
        
        return pd.DataFrame(prices)
    
    def _generate_base_load(self) -> pd.DataFrame:
        """
        Generate depot base load (non-EV electricity consumption)
        
        Includes:
        - Lighting (varies by time of day)
        - HVAC (heating/cooling)
        - Office equipment
        - Maintenance tools and equipment
        
        Returns:
            DataFrame with columns:
            - timestep: Time index
            - hour: Hour of day
            - base_load_kw: Power consumption in kW
        """
        
        base_loads = []
        
        for step in range(self.total_steps):
            hour = (step % self.steps_per_day) * self.timestep_minutes / 60.0
            day = step // self.steps_per_day
            day_of_week = day % 7
            
            # Base load pattern by time of day
            if 6.0 <= hour < 18.0:  # Daytime operations
                base = 100.0  # kW
            elif 18.0 <= hour < 22.0:  # Evening (reduced)
                base = 60.0
            else:  # Night (minimal)
                base = 30.0
            
            # Weekend reduction
            if day_of_week >= 5:
                base *= 0.70
            
            # Seasonal variation (heating/cooling needs)
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day / 365.0)
            
            # Random variation
            noise = np.random.normal(0, 5.0)
            
            load = base * seasonal_factor + noise
            load = max(10.0, load)  # Minimum baseline
            
            base_loads.append({
                'timestep': step,
                'hour': round(hour, 2),
                'base_load_kw': round(load, 2)
            })
        
        return pd.DataFrame(base_loads)
    
    def _generate_charger_specs(self) -> pd.DataFrame:
        """
        Generate charger specifications
        
        Returns:
            DataFrame with columns:
            - charger_id: Unique identifier
            - max_power: Maximum power in kW
            - v2g_capable: V2G capability (True/False)
            - efficiency: Charger efficiency (0-1)
            - charger_type: 'Level2' or 'DC_Fast'
        """
        
        chargers = []
        
        for cid in range(self.n_chargers):
            # Mix of charger types (half DC Fast, half Level 2)
            if cid < self.n_chargers // 2:
                charger_type = 'DC_Fast'
                max_power = 50.0
            else:
                charger_type = 'Level2'
                max_power = 22.0
            
            chargers.append({
                'charger_id': cid,
                'max_power': max_power,
                'v2g_capable': True,  # All chargers V2G capable
                'efficiency': 0.95,
                'charger_type': charger_type
            })
        
        return pd.DataFrame(chargers)
    
    def _print_dataset_summary(self, dataset: Dict):
        """Print comprehensive dataset statistics"""
        
        print("\n" + "="*70)
        print("📊 DATASET SUMMARY")
        print("="*70)
        
        # General configuration
        print("\n🔧 CONFIGURATION")
        print(f"   Vehicles:          {self.n_vehicles}")
        print(f"   Chargers:          {self.n_chargers}")
        print(f"   Simulation period: {self.simulation_days} days ({self.total_steps} timesteps)")
        print(f"   Time resolution:   {self.timestep_minutes} minutes/step")
        
        # Vehicle statistics
        vehicles_df = dataset['vehicles']
        print("\n🚗 VEHICLE FLEET")
        print(f"   Battery capacity:  {vehicles_df['battery_capacity'].min():.1f} - {vehicles_df['battery_capacity'].max():.1f} kWh")
        print(f"   Average capacity:  {vehicles_df['battery_capacity'].mean():.1f} kWh")
        print(f"   Max charge power:  {vehicles_df['max_charging_power'].max():.0f} kW")
        print(f"   Max V2G power:     {vehicles_df['max_discharging_power'].max():.0f} kW")
        
        # Schedule statistics
        schedules_df = dataset['schedules']
        print("\n📅 ROUTE SCHEDULES")
        print(f"   Total trips:       {len(schedules_df)}")
        print(f"   Trips per vehicle: {len(schedules_df) / self.n_vehicles:.1f}")
        print(f"   Avg energy/trip:   {schedules_df['energy_consumed_kwh'].mean():.1f} kWh")
        print(f"   Avg SoC arrival:   {schedules_df['soc_at_arrival'].mean():.1%}")
        
        # Price statistics
        prices_df = dataset['prices']
        print("\n💰 ELECTRICITY TARIFF")
        print(f"   Price range:       €{prices_df['price_per_kwh'].min():.3f} - €{prices_df['price_per_kwh'].max():.3f}/kWh")
        print(f"   Average price:     €{prices_df['price_per_kwh'].mean():.3f}/kWh")
        print(f"   V2G sell price:    €{prices_df['v2g_sell_price'].mean():.3f}/kWh (avg)")
        
        # Peak/off-peak breakdown
        peak_count = len(prices_df[prices_df['price_category'] == 'peak'])
        print(f"   Peak hours:        {peak_count/len(prices_df)*100:.1f}% of time")
        
        # Base load statistics
        base_load_df = dataset['base_load']
        print("\n⚡ DEPOT BASE LOAD")
        print(f"   Load range:        {base_load_df['base_load_kw'].min():.1f} - {base_load_df['base_load_kw'].max():.1f} kW")
        print(f"   Average load:      {base_load_df['base_load_kw'].mean():.1f} kW")
        
        # Charger statistics
        chargers_df = dataset['chargers']
        print("\n🔌 CHARGING INFRASTRUCTURE")
        print(f"   DC Fast chargers:  {len(chargers_df[chargers_df['charger_type']=='DC_Fast'])}")
        print(f"   Level 2 chargers:  {len(chargers_df[chargers_df['charger_type']=='Level2'])}")
        print(f"   V2G capable:       {chargers_df['v2g_capable'].sum()}/{len(chargers_df)}")
        
        print("\n" + "="*70)
    
    def save_dataset(self, dataset: Dict, output_dir: str = 'data/processed/'):
        """
        Save generated dataset to CSV files
        
        Args:
            dataset: Generated dataset dictionary
            output_dir: Output directory path (default: 'data/processed/')
        """
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving dataset to {output_dir}...")
        
        # Save each component as CSV
        dataset['vehicles'].to_csv(f'{output_dir}/vehicles.csv', index=False)
        print(f"   ✓ vehicles.csv")
        
        dataset['schedules'].to_csv(f'{output_dir}/schedules.csv', index=False)
        print(f"   ✓ schedules.csv")
        
        dataset['prices'].to_csv(f'{output_dir}/prices.csv', index=False)
        print(f"   ✓ prices.csv")
        
        dataset['base_load'].to_csv(f'{output_dir}/base_load.csv', index=False)
        print(f"   ✓ base_load.csv")
        
        dataset['chargers'].to_csv(f'{output_dir}/chargers.csv', index=False)
        print(f"   ✓ chargers.csv")
        
        # Save metadata as JSON
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(dataset['metadata'], f, indent=2)
        print(f"   ✓ metadata.json")
        
        print(f"\n✅ Dataset saved successfully!")
        print(f"   Location: {os.path.abspath(output_dir)}")
    
    def load_dataset(self, input_dir: str = 'data/processed/') -> Dict:
        """
        Load previously generated dataset from files
        
        Args:
            input_dir: Directory containing saved dataset
            
        Returns:
            Dataset dictionary with all components
        """
        
        print(f"📂 Loading dataset from {input_dir}...")
        
        dataset = {
            'vehicles': pd.read_csv(f'{input_dir}/vehicles.csv'),
            'schedules': pd.read_csv(f'{input_dir}/schedules.csv'),
            'prices': pd.read_csv(f'{input_dir}/prices.csv'),
            'base_load': pd.read_csv(f'{input_dir}/base_load.csv'),
            'chargers': pd.read_csv(f'{input_dir}/chargers.csv'),
        }
        
        # Load metadata
        with open(f'{input_dir}/metadata.json', 'r') as f:
            dataset['metadata'] = json.load(f)
        
        print("✅ Dataset loaded successfully!")
        return dataset


# ============================================================================
# EXAMPLE USAGE AND MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  EV Fleet Data Generator for V2G Peak Shaving RL Project         ║
    ║  Generates realistic synthetic data for reinforcement learning   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize generator with default parameters
    generator = EVFleetDataGenerator(
        n_vehicles=20,          # 20 EVs in fleet
        n_chargers=15,          # 15 charging stations
        simulation_days=7,      # 1 week simulation
        timestep_minutes=15,    # 15-minute resolution
        seed=42                 # Reproducible results
    )
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset()
    
    # Save to files
    generator.save_dataset(dataset, output_dir='data/processed/')
    
    print("\n" + "="*70)
    print("🎉 DATA GENERATION COMPLETE!")
    print("="*70)
    print("\nYou can now use this dataset for:")
    print("  1. Training baseline algorithms (Greedy, MILP)")
    print("  2. Training RL agents (PPO, SAC, Lagrangian-PPO)")
    print("  3. Evaluating performance metrics")
    print("\nNext steps:")
    print("  - Run baseline algorithms: python src/baselines/greedy.py")
    print("  - Train RL agents: python src/training/train.py")
    print("  - Evaluate results: python src/evaluation/evaluate_all.py")
    print("="*70)