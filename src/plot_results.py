"""
Generate comparison graphs from evaluation results.

Reads:
  outputs/evaluation_results.csv
  outputs/dqn_standard_metrics.csv
  outputs/dqn_custom_metrics.csv

Produces (all saved to outputs/):
  1. waiting_time_comparison.png
  2. queue_length_comparison.png
  3. mean_speed_comparison.png
  4. summary_table.png
  5. training_curve.png

Usage: python src/plot_results.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
out_dir = os.path.join(base_dir, 'outputs')

eval_csv = os.path.join(out_dir, 'evaluation_results.csv')
dqn_std_csv = os.path.join(out_dir, 'dqn_standard_metrics.csv')
dqn_cust_csv = os.path.join(out_dir, 'dqn_custom_metrics.csv')

# Style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'Random': '#e74c3c',
    'Q-Learning': '#f39c12',
    'DQN_Standard': '#3498db',
    'DQN_Custom': '#2ecc71',
}

# ---------------------------------------------------------------------------
# Load evaluation data
# ---------------------------------------------------------------------------
if not os.path.exists(eval_csv):
    print(f"ERROR: {eval_csv} not found. Run evaluate.py first.")
    sys.exit(1)

df = pd.read_csv(eval_csv)
models = df['model'].unique()
print(f"Models found: {list(models)}")

# Compute mean across episodes for each (model, step)
grouped = df.groupby(['model', 'step']).agg(
    waiting_time=('waiting_time', 'mean'),
    queue_length=('queue_length', 'mean'),
    mean_speed=('mean_speed', 'mean'),
).reset_index()


# ---------------------------------------------------------------------------
# 1. Waiting Time Comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for model in models:
    sub = grouped[grouped['model'] == model]
    ax.plot(sub['step'], sub['waiting_time'],
            label=model, color=COLORS.get(model, 'gray'), alpha=0.9)
ax.set_xlabel('Simulation Step', fontsize=13)
ax.set_ylabel('Total Waiting Time (s)', fontsize=13)
ax.set_title('Total Waiting Time Comparison', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'waiting_time_comparison.png'), dpi=150)
print("Saved waiting_time_comparison.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Queue Length Comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for model in models:
    sub = grouped[grouped['model'] == model]
    ax.plot(sub['step'], sub['queue_length'],
            label=model, color=COLORS.get(model, 'gray'), alpha=0.9)
ax.set_xlabel('Simulation Step', fontsize=13)
ax.set_ylabel('Queue Length (vehicles halting)', fontsize=13)
ax.set_title('Queue Length Comparison', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'queue_length_comparison.png'), dpi=150)
print("Saved queue_length_comparison.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Mean Speed Comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for model in models:
    sub = grouped[grouped['model'] == model]
    ax.plot(sub['step'], sub['mean_speed'],
            label=model, color=COLORS.get(model, 'gray'), alpha=0.9)
ax.set_xlabel('Simulation Step', fontsize=13)
ax.set_ylabel('Mean Speed (m/s)', fontsize=13)
ax.set_title('Mean Speed Comparison', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'mean_speed_comparison.png'), dpi=150)
print("Saved mean_speed_comparison.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Summary Table
# ---------------------------------------------------------------------------
summary_rows = []
for model in models:
    sub = df[df['model'] == model]
    summary_rows.append({
        'Model': model,
        'Avg Wait Time (s)': f"{sub['waiting_time'].mean():.1f}",
        'Avg Queue Length': f"{sub['queue_length'].mean():.1f}",
        'Avg Speed (m/s)': f"{sub['mean_speed'].mean():.3f}",
    })

summary_df = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(models)))
ax.axis('off')
ax.set_title('Model Performance Summary', fontsize=15, fontweight='bold', pad=20)

col_labels = summary_df.columns.tolist()
cell_text = summary_df.values.tolist()

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colours
for i in range(1, len(cell_text) + 1):
    color = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(color)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
print("Saved summary_table.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Training Curve (DQN Standard vs Custom over episodes)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
has_curve = False

if os.path.exists(dqn_std_csv):
    std = pd.read_csv(dqn_std_csv)
    ax.plot(std['episode'], std['avg_waiting_time'], 'o-',
            label='DQN Standard', color=COLORS['DQN_Standard'], linewidth=2, markersize=8)
    has_curve = True

if os.path.exists(dqn_cust_csv):
    cust = pd.read_csv(dqn_cust_csv)
    ax.plot(cust['episode'], cust['avg_waiting_time'], 's-',
            label='DQN Custom', color=COLORS['DQN_Custom'], linewidth=2, markersize=8)
    has_curve = True

if has_curve:
    ax.set_xlabel('Training Episode', fontsize=13)
    ax.set_ylabel('Avg Waiting Time (s)', fontsize=13)
    ax.set_title('DQN Training Curve — Standard vs Custom Reward', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_curve.png'), dpi=150)
    print("Saved training_curve.png")
else:
    print("No training CSVs found — skipping training_curve.png")

plt.close(fig)

print("\n" + "=" * 60)
print("  All plots saved to outputs/")
print("=" * 60)
