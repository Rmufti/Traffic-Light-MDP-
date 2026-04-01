"""
Train a PPO agent on the single-intersection traffic light environment
using a CUSTOM reward: -(waiting_time + alpha * lane_queue_variance).

Usage: python src/train_ppo_custom.py
"""
import os
import sys
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment

# ---------------------------------------------------------------------------
# Custom reward function
# ---------------------------------------------------------------------------
def balanced_reward(traffic_signal):
    lane_halting = [
        traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
        for lane in traffic_signal.lanes
    ]
    total_queued = sum(lane_halting)
    lane_variance = float(np.var(lane_halting)) if len(lane_halting) > 0 else 0.0

    # Scaled down to match DQN
    alpha = 0.05
    return -(total_queued + alpha * lane_variance)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

net_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection.net.xml')
route_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection-vhvh.rou.xml')

model_save_path = os.path.join(base_dir, 'models', 'ppo_custom')
csv_save_path = os.path.join(base_dir, 'outputs', 'ppo_custom_metrics.csv')

os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
# INCREASED EPISODES
NUM_EPISODES = 200
NUM_SECONDS = 3600
DELTA_TIME = 5
MAX_STEPS_PER_EP = NUM_SECONDS // DELTA_TIME

# ---------------------------------------------------------------------------
# Callback 
# ---------------------------------------------------------------------------
class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_waits = []
        self.episode_queues = []
        self.episode_speeds = []
        self.step_in_ep = 0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        self.episode_waits.append(info.get("system_total_waiting_time", 0))
        self.episode_queues.append(info.get("agents_total_stopped", 0))
        self.episode_speeds.append(info.get("system_mean_speed", 0))
        self.step_in_ep += 1
        return True

    def reset_episode(self):
        self.episode_waits = []
        self.episode_queues = []
        self.episode_speeds = []
        self.step_in_ep = 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 60)
print("  PPO Training — Custom Reward (queue + lane‑balance)")
print("=" * 60)

all_metrics = []

for episode in range(1, NUM_EPISODES + 1):
    print(f"\n--- Episode {episode}/{NUM_EPISODES} ---")

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        single_agent=True,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        reward_fn=balanced_reward,
        sumo_warnings=False,
    )

    if episode == 1:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4, 
            n_steps=MAX_STEPS_PER_EP, # 720 steps
            batch_size=120,           # 720 divides perfectly by 120
            ent_coef=0.01, 
            verbose=0,
        )
    else:
        model.set_env(env)

    cb = EpisodeMetricsCallback()
    model.learn(total_timesteps=MAX_STEPS_PER_EP, callback=cb, reset_num_timesteps=False)

    avg_wait = np.mean(cb.episode_waits) if cb.episode_waits else 0
    avg_queue = np.mean(cb.episode_queues) if cb.episode_queues else 0
    avg_speed = np.mean(cb.episode_speeds) if cb.episode_speeds else 0
    final_wait = cb.episode_waits[-1] if cb.episode_waits else 0

    all_metrics.append({
        "episode": episode,
        "avg_waiting_time": round(avg_wait, 2),
        "final_waiting_time": round(final_wait, 2),
        "avg_queue_length": round(avg_queue, 2),
        "avg_speed": round(avg_speed, 4),
        "steps": cb.step_in_ep,
    })

    print(f"  Steps: {cb.step_in_ep}")
    print(f"  Avg Waiting Time: {avg_wait:.2f}s")
    print(f"  Final Waiting Time: {final_wait:.2f}s")
    print(f"  Avg Queue Length: {avg_queue:.2f}")
    print(f"  Avg Speed: {avg_speed:.4f} m/s")

    env.close()

model.save(model_save_path)
print(f"\nModel saved to {model_save_path}.zip")

with open(csv_save_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
    writer.writeheader()
    writer.writerows(all_metrics)
print(f"Metrics saved to {csv_save_path}")

print("\n" + "=" * 60)
print("  PPO Custom Reward Training Complete!")
print("=" * 60)