import numpy as np
import pandas as pd
from pathlib import Path

def parse_spark_npz(file_path: str, output_dir=None):
    """Robust SPARK parser with Infeasibility Rate & Slack Usage"""
    
    file_path = Path(file_path)
    data = np.load(file_path, allow_pickle=True)
    
    if output_dir is None:
        output_dir = Path("parsed_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {file_path.name}")

    csv_data = {}
    episode = {}   

    for key in data.files:
        arr = data[key]
        if arr.size == 0:
            continue

        if arr.ndim == 1:
            csv_data[key] = arr

            if key in ["infeasible", "infeasibility", "qp_infeasible"]:
                infeas_count = np.sum(arr)
                total_steps = len(arr)
                episode["infeasibility_rate"] = [infeas_count / total_steps * 100]
                episode["infeas_count"] = [infeas_count]
                episode["total_steps"] = [total_steps]

            if key in ["slack", "slack_value", "slack_var"]:
                episode["mean_slack"] = [np.mean(arr)]
                episode["max_slack"] = [np.max(arr)]
                episode["total_slack"] = [np.sum(arr)]
                episode["slack_active_rate"] = [np.mean(arr > 0) * 100]   # % of timesteps with slack > 0

        elif arr.ndim == 2 and key == "dist_goal_arm":
            csv_data["dist_goal_right_arm"] = arr[:, 0]
            csv_data["dist_goal_left_arm"] = arr[:, 1]

        elif arr.ndim == 3 and key == "dist_robot_to_env":
            min_per_step = np.min(arr, axis=(1, 2))
            csv_data["min_dist_to_env"] = min_per_step
            episode["min_dist_to_env"] = [np.min(min_per_step)]
            episode["mean_dist_to_env"] = [np.mean(min_per_step)]

    df_timestep = pd.DataFrame(csv_data)

    base_episode = {
        "file": [file_path.parent.name.replace(".json", "")],
        "algo": [file_path.parent.name.split("_")[2] if len(file_path.parent.name.split("_")) > 2 else "unknown"],
        "constraint_level": ["D2" if "D2" in file_path.parent.name else "D1"],
        "attack_type": [next((x for x in ["perception_noise", "latency", "nominal"] if x in file_path.parent.name), "nominal")],
        "attack_level": [next((x for x in ["low", "medium", "high"] if x in file_path.parent.name), "none")],
    }
    
    df_episode = pd.DataFrame({**base_episode, **episode})

    base_name = file_path.parent.name.replace(".json", "")
    
    df_timestep.to_csv(output_dir / f"{base_name}_timestep.csv", index=False)
    df_episode.to_csv(output_dir / f"{base_name}_episode.csv", index=False)

    print(f"Saved in 'parsed_results/' → {base_name}_episode.csv")
    return df_timestep, df_episode


if __name__ == "__main__":
    results_dir = Path("..")   
    
    all_episodes = []
    
    for npz_file in results_dir.rglob("data.npz"):
        if "ExtB" in str(npz_file):
            try:
                _, episode_df = parse_spark_npz(npz_file)
                all_episodes.append(episode_df)
            except Exception as e:
                print(f"Failed {npz_file.name}: {e}")

    if all_episodes:
        final_summary = pd.concat(all_episodes, ignore_index=True)
        
        # Nice ordering of columns
        cols = ["file", "algo", "constraint_level", "attack_type", "attack_level",
                "infeasibility_rate", "mean_slack", "max_slack", "slack_active_rate",
                "min_dist_to_env", "mean_dist_to_env", "total_steps"]
        final_summary = final_summary[[c for c in cols if c in final_summary.columns]]
        
        final_summary.to_csv("ExtensionB_Full_Summary.csv", index=False)
        print("\nFINAL SUMMARY SAVED: ExtensionB_Full_Summary.csv")
        print(final_summary.round(4))
    else:
        print("No Extension B files found.")