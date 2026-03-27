import os
import gymnasium as gym
from sumo_rl import SumoEnvironment

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

net_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection.net.xml')
# If vhvh continues to fail, try 'single-intersection-gen.rou.xml'
route_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection-vhvh.rou.xml')

env = SumoEnvironment(
    net_file=net_file,
    route_file=route_file,
    use_gui=False,
    num_seconds=3600, 
    delta_time=5,
    begin_time=0
)

print(f"Starting simulation logic...")
res = env.reset()
obs = res[0] if isinstance(res, tuple) else res

total_wait = 0
step_count = 0
# We want to simulate 1 hour (3600s) / 5s per step = 720 steps
max_steps = 720 

print("Running Baseline Simulation (vhvh flow)...")

# Change to a FOR loop so we aren't at the mercy of the 'done' variable initially
for i in range(max_steps):
    # Action dictionary for all traffic lights
    actions = {ts: env.action_space.sample() for ts in env.ts_ids}
    
    step_res = env.step(actions)
    
    # Handle both 4-value and 5-value returns from Gym
    if len(step_res) == 5:
        next_obs, reward, terminated, truncated, info = step_res
        is_done = terminated or truncated
    else:
        next_obs, reward, is_done, info = step_res
    
    # Accumulate wait time
    # SUMO-RL sometimes nests metrics under the agent ID
    current_step_wait = info.get('system_total_waiting_time', 0)
    total_wait += current_step_wait
    step_count += 1

    if step_count % 50 == 0:
        print(f"Step {step_count} | Cumulative Wait: {total_wait:.1f}s")

    # Only stop if is_done is true AND we've actually progressed past the start
    if is_done and step_count > 20:
        print(f"Simulation finished naturally at step {step_count}")
        break

avg_wait = total_wait / step_count if step_count > 0 else 0

print(f"\n--- FINAL BASELINE RESULTS ---")
print(f"Total Steps Run: {step_count}")
print(f"Total Cumulative Waiting Time: {total_wait:.2f} seconds")
print(f"Average Waiting Time per Step: {avg_wait:.2f} seconds")

env.close()