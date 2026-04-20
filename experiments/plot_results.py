# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:17:20 2026

@author: mikun
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data you just extracted
file_path = "parsed_results_rssa_attack_perception_noise_high.csv"
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Create a time step column (0 to 5000)
steps = range(len(df))

# 2. Set up the figure with two stacked graphs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Graph 1: Efficiency (Distance to Goal) ---
ax1.plot(steps, df['dist_goal_right_arm'], label='Relaxed SSA (RSSA) with following attack: high perception noise', color='blue', linewidth=2)
ax1.set_ylabel("Distance to Goal (m)", fontsize=12, fontweight='bold')
ax1.set_title("Task Efficiency: Reaching the Target", fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- Graph 2: Safety (Distance to Obstacle) ---
ax2.plot(steps, df['min_dist_to_env'], label='Relaxed SSA (RSSA) with following attack: high perception noise', color='red', linewidth=2)
# Add a horizontal line at 0 to show the collision boundary
ax2.axhline(0, color='black', linestyle='--', label='Collision Boundary')
ax2.set_ylabel("Distance to Obstacle (m)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Simulation Steps", fontsize=12, fontweight='bold')
ax2.set_title("Safety: Avoiding Dynamic Obstacles", fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# 3. Clean up the layout and save the image
plt.tight_layout()
save_name = "tradeoff_curve_rssa_attack_perception_noise_high.png"
plt.savefig(save_name, dpi=300)
print(f"Success! Graph saved as {save_name}")