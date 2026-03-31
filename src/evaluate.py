"""
Evaluate all models side-by-side:
  1. Random baseline
  2. Q-Learning (trained on the fly for a few episodes, then evaluated)
  3. DQN — standard reward
  4. DQN — custom reward

Saves per-step metrics for every evaluation episode to
  outputs/evaluation_results.csv

Usage: python src/evaluate.py
"""
import os
import sys
import csv
import numpy as np
from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment

# So we can import models/qLearningAgent.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.qLearningAgent import qLearningAgent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

net_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection.net.xml')
route_file = os.path.join(base_dir, 'networks', 'simple', 'single-intersection-vhvh.rou.xml')
dqn_std_path = os.path.join(base_dir, 'models', 'dqn_standard.zip')
dqn_cust_path = os.path.join(base_dir, 'models', 'dqn_custom.zip')
output_csv = os.path.join(base_dir, 'outputs', 'evaluation_results.csv')
os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)

NUM_EVAL_EPISODES = 3
NUM_SECONDS = 3600
DELTA_TIME = 5
MAX_STEPS = NUM_SECONDS // DELTA_TIME


# ---------------------------------------------------------------------------
# Helper: discretize state for Q-learning (same as src/train.py)
# ---------------------------------------------------------------------------
def discretize_state(obs):
    if isinstance(obs, dict):
        state_array = list(obs.values())[0]
    else:
        state_array = obs
    return tuple(np.round(state_array).astype(int))


# ---------------------------------------------------------------------------
# Runner functions for each model type
# ---------------------------------------------------------------------------
def run_random(env):
    """Random action baseline."""
    obs, info = env.reset()
    rows = []
    for step in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "step": step,
            "waiting_time": info.get("system_total_waiting_time", 0),
            "queue_length": info.get("agents_total_stopped", 0),
            "mean_speed": info.get("system_mean_speed", 0),
        })
        if terminated or truncated:
            break
    env.close()
    return rows


def run_qlearning(env, num_train_episodes=5):
    """
    Train a Q-learning agent for a few episodes, then evaluate once.
    Returns evaluation-episode rows.
    """
    action_list = list(range(env.action_space.n))
    agent = qLearningAgent(
        action_space=action_list,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.90,
    )

    # Quick training
    for ep in range(num_train_episodes):
        obs, info = env.reset()
        state = discretize_state(obs)
        for _ in range(MAX_STEPS):
            action = agent.choose_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(obs)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if terminated or truncated:
                break
        agent.decay_exploration()
    env.close()

    # Evaluation episode (greedy)
    eval_env = SumoEnvironment(
        net_file=net_file, route_file=route_file,
        use_gui=False, single_agent=True,
        num_seconds=NUM_SECONDS, delta_time=DELTA_TIME,
        sumo_warnings=False,
    )
    agent.exploration_rate = 0.0  # greedy
    obs, info = eval_env.reset()
    state = discretize_state(obs)
    rows = []
    for step in range(MAX_STEPS):
        action = agent.choose_action(state)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        state = discretize_state(obs)
        rows.append({
            "step": step,
            "waiting_time": info.get("system_total_waiting_time", 0),
            "queue_length": info.get("agents_total_stopped", 0),
            "mean_speed": info.get("system_mean_speed", 0),
        })
        if terminated or truncated:
            break
    eval_env.close()
    return rows


def run_dqn(env, model_path):
    """Load a trained DQN and evaluate deterministically."""
    model = DQN.load(model_path, env=env)
    obs, info = env.reset()
    rows = []
    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "step": step,
            "waiting_time": info.get("system_total_waiting_time", 0),
            "queue_length": info.get("agents_total_stopped", 0),
            "mean_speed": info.get("system_mean_speed", 0),
        })
        if terminated or truncated:
            break
    env.close()
    return rows


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def make_env():
    return SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        single_agent=True,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        sumo_warnings=False,
    )


print("=" * 60)
print("  Evaluation — All Models")
print("=" * 60)

all_rows = []

models_to_run = [
    ("Random", lambda: run_random(make_env())),
    ("Q-Learning", lambda: run_qlearning(make_env(), num_train_episodes=5)),
]

if os.path.exists(dqn_std_path):
    models_to_run.append(("DQN_Standard", lambda: run_dqn(make_env(), dqn_std_path)))
else:
    print(f"WARNING: {dqn_std_path} not found — skipping DQN Standard evaluation")

if os.path.exists(dqn_cust_path):
    models_to_run.append(("DQN_Custom", lambda: run_dqn(make_env(), dqn_cust_path)))
else:
    print(f"WARNING: {dqn_cust_path} not found — skipping DQN Custom evaluation")

for model_name, runner_fn in models_to_run:
    for ep in range(1, NUM_EVAL_EPISODES + 1):
        print(f"\n  Running {model_name} — Episode {ep}/{NUM_EVAL_EPISODES} ...")
        rows = runner_fn()
        for r in rows:
            r["model"] = model_name
            r["episode"] = ep
        all_rows.extend(rows)

        # Quick summary
        avg_w = np.mean([r["waiting_time"] for r in rows])
        avg_q = np.mean([r["queue_length"] for r in rows])
        avg_s = np.mean([r["mean_speed"] for r in rows])
        print(f"    Avg Wait: {avg_w:.1f}s | Avg Queue: {avg_q:.1f} | Avg Speed: {avg_s:.3f} m/s")

# Save CSV
fieldnames = ["model", "episode", "step", "waiting_time", "queue_length", "mean_speed"]
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nResults saved to {output_csv}")
print("=" * 60)
print("  Evaluation Complete!")
print("=" * 60)
