"""
Train a DQN agent on the single-intersection traffic light environment
using the default sumo-rl reward (diff-waiting-time).

Usage: python src/train_dqn.py
"""
import os
import sys
import csv
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

net_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection.net.xml')
route_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection-vhvh.rou.xml')
model_save_path = os.path.join(base_dir, 'models', 'dqn_standard')
csv_save_path = os.path.join(base_dir, 'outputs', 'dqn_standard_metrics.csv')

os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
NUM_EPISODES = 5
NUM_SECONDS = 3600
DELTA_TIME = 5
MAX_STEPS_PER_EP = NUM_SECONDS // DELTA_TIME  # 720

# ---------------------------------------------------------------------------
# Episode metrics collector callback
# ---------------------------------------------------------------------------
class EpisodeMetricsCallback(BaseCallback):
    """Collects per-step info from the environment during training."""

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
# Main training loop – one episode at a time so we can log per-episode
# ---------------------------------------------------------------------------
print("=" * 60)
print("  DQN Training — Standard Reward (diff-waiting-time)")
print("=" * 60)

all_metrics = []

for episode in range(1, NUM_EPISODES + 1):
    print(f"\n--- Episode {episode}/{NUM_EPISODES} ---")

    # Create a fresh environment each episode
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        single_agent=True,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        sumo_warnings=False,
    )

    if episode == 1:
        # First episode: create the model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=100,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_final_eps=0.02,
            batch_size=64,
            verbose=0,
        )
    else:
        # Subsequent episodes: swap in the new env
        model.set_env(env)

    cb = EpisodeMetricsCallback()
    model.learn(total_timesteps=MAX_STEPS_PER_EP, callback=cb, reset_num_timesteps=False)

    # Summarise episode
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

# Save model
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}.zip")

# Save metrics CSV
with open(csv_save_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
    writer.writeheader()
    writer.writerows(all_metrics)
print(f"Metrics saved to {csv_save_path}")

print("\n" + "=" * 60)
print("  DQN Standard Training Complete!")
print("=" * 60)
