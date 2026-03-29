import os
import sys
import gymnasium as gym
import numpy as np
from sumo_rl import SumoEnvironment

# 1. FIXED IMPORT: Match the exact class name (q_learning_agent)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qLearningAgent import qLearningAgent

# 2. FIXED HELPER FUNCTION: Handle the dictionary from sumo-rl
def discretize_state(obs):
    """
    Extracts the array from the sumo-rl dictionary and rounds 
    the continuous numbers so the Q-Table doesn't explode.
    """
    if isinstance(obs, dict):
        # Grab the array for the first (and only) traffic light
        state_array = list(obs.values())[0]
    else:
        state_array = obs
        
    # Round to the nearest whole number and convert to tuple
    return tuple(np.round(state_array).astype(int))

# 1. Setup Paths & Environment
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
net_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection.net.xml')
route_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection-vhvh.rou.xml')

env = SumoEnvironment(
    net_file=net_file,
    route_file=route_file,
    use_gui=False,
    num_seconds=3600, 
    delta_time=5
)

# 2. Initialize Agent
valid_actions = list(range(env.action_space.n))

# FIXED INITIALIZATION: Use the exact class name
agent = qLearningAgent(
    action_space=valid_actions,
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=1.0, # Start 100% random
    exploration_decay=0.95 # Slowly become less random each episode
)

# 3. The Training Loop (Episodes)
EPISODES = 10 # Let's train for 10 full runs to start
max_steps = 720 # 3600 seconds / 5 delta_time

print("Starting Training...")

for episode in range(EPISODES):
    # Reset the environment
    res = env.reset()
    raw_obs = res[0] if isinstance(res, tuple) else res
    
    total_wait = 0
    
    # Inner loop
    for step in range(max_steps):
        # Discretize state
        current_state = discretize_state(raw_obs) 
        
        # Agent chooses action
        action = agent.choose_action(current_state)
        
        # Step the environment
        sumo_actions = {ts: action for ts in env.ts_ids}
        step_res = env.step(sumo_actions)
        
# Unpack safe step
        if len(step_res) == 5:
            next_raw_obs, reward_dict, terminated, truncated, info = step_res
            is_done = terminated or truncated
        else:
            next_raw_obs, reward_dict, is_done, info = step_res
        
        # FIXED REWARD: Extract the scalar number if it's a dictionary
        if isinstance(reward_dict, dict):
            reward = list(reward_dict.values())[0]
        else:
            reward = reward_dict

        # Discretize the new state
        next_state = discretize_state(next_raw_obs)
        
        # AGENT LEARNS (Now it's passing a clean float!)
        agent.learn(current_state, action, reward, next_state)
        
        # Track metrics
        # Some versions of sumo-rl nest the info dictionary
        if 'system_total_waiting_time' in info:
            total_wait += info['system_total_waiting_time']
        elif isinstance(info, dict) and env.ts_ids[0] in info:
             total_wait += info[env.ts_ids[0]].get('system_total_waiting_time', 0)

        # Move forward
        raw_obs = next_raw_obs
        
        if is_done and step > 20:
            break
            
    # Decay exploration
    agent.decay_exploration()
    
    print(f"Episode {episode + 1}/{EPISODES} | Total Wait Time: {total_wait:.2f} | Epsilon: {agent.exploration_rate:.2f}")

env.close()

# Show what the agent learned!
print("\nTraining Complete! Here is the Q-Table brain:")
agent.plot_q_values()