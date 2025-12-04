def config_setting():
    config = {
        "controlled_vehicles": 1,
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy", "heading"],
            "normalize":False
        },
        "action": {
            "type": "ContinuousAction",  
            "acceleration_range": [-3.0, 3.0], 
            "steering_range": [-0.2, 0.2],  
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 30,  # [s]
        "initial_spacing": 2,
        "collision_reward": -1,  # The reward received when colliding with a vehicle.
        "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
        "simulation_frequency": 10,  # [Hz]
        "policy_frequency": 10,  # [Hz]5
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": False,
        "offscreen_rendering": False,
        "offroad_terminal": True
    }
    return config


def default_config_setting():
    config = {
        "observation": {
            "type": "Kinematics"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,  # [s]
        "initial_spacing": 2,
        "collision_reward": -1,  # The reward received when colliding with a vehicle.
        "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        
    }
    return config

