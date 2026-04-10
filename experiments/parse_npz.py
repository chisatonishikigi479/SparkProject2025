import numpy as np
import pandas as pd

file_path = "results_rssa.json/data.npz"
print(f"Loading compressed data from: {file_path}")
data = np.load(file_path, allow_pickle=True)

csv_data = {}
for key in data.files:
    arr = data[key]
    
    # 1. Grab standard 1D arrays (like seed, done, dist_goal_base)
    if arr.ndim == 1:
        csv_data[key] = arr
    elif arr.ndim == 0:
        csv_data[key] = [arr.item()]
        
    # 2. Unpack the 2D array (Left/Right Arms)
    elif arr.ndim == 2:
        if key == "dist_goal_arm":
            csv_data["dist_goal_right_arm"] = arr[:, 0]
            csv_data["dist_goal_left_arm"] = arr[:, 1]
            
    # 3. Flatten the 3D array (25 robot parts vs 10 obstacles)
    elif arr.ndim == 3:
        if key == "dist_robot_to_env":
            # Find the absolute minimum distance across all body parts (axis 1) and obstacles (axis 2)
            csv_data["min_dist_to_env"] = np.min(arr, axis=(1, 2))

# 4. Save the cleanly formatted data!
try:
    df = pd.DataFrame(csv_data)
    save_name = "parsed_rssa_results_final.csv"
    df.to_csv(save_name, index=False)
    print(f"\nSuccess! Your complete tradeoff data is saved to: {save_name}")
except ValueError as e:
    print(f"\nError grouping data: {e}")