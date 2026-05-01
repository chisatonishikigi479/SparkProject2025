import numpy as np
import pandas as pd
from pathlib import Path
import sys

def load_and_summarize(folder_path: Path):
    """Load data.npz from inside each result folder"""
    npz_file = folder_path / "data.npz"
    
    if not npz_file.exists():
        print(f"No data.npz found in {folder_path.name}")
        return None
    
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        summary = {
            "folder": folder_path.name,
            "algo": "unknown",
            "constraint_level": "unknown",
            "attack_type": "nominal",
            "attack_level": "none",
            "min_dist": np.nan,
            "mean_dist": np.nan,
            "collision_rate": np.nan,
            "infeasibility_rate": np.nan,
            "mean_slack": np.nan,
            "max_slack": np.nan,
            "slack_active_rate": np.nan,
        }
        
        name = folder_path.name.lower()
        if "ssa" in name: summary["algo"] = "ssa"
        elif "rssa" in name: summary["algo"] = "rssa"
        elif "cbf" in name: summary["algo"] = "cbf"
        elif "rcbf" in name: summary["algo"] = "rcbf"
        
        if "d1" in name: summary["constraint_level"] = "D1"
        elif "d2" in name: summary["constraint_level"] = "D2"
        
        if "perception_noise" in name: summary["attack_type"] = "perception_noise"
        elif "latency" in name: summary["attack_type"] = "latency"
        
        for lvl in ["low", "medium", "high", "nominal"]:
            if lvl in name:
                summary["attack_level"] = lvl
                break

        for key in data.files:
            arr = data[key]
            if arr.size == 0:
                continue
            flat = arr.flatten()
            
            if any(x in key.lower() for x in ["dist", "distance"]):
                summary["min_dist"] = float(np.min(flat))
                summary["mean_dist"] = float(np.mean(flat))
                summary["collision_rate"] = float(np.mean(flat < 0.01))
            
            if any(x in key.lower() for x in ["infeas", "infeasible"]):
                summary["infeasibility_rate"] = float(np.mean(flat) * 100)
            
            if "slack" in key.lower():
                summary["mean_slack"] = float(np.mean(flat))
                summary["max_slack"] = float(np.max(flat))
                summary["slack_active_rate"] = float(np.mean(flat > 0) * 100)

        return pd.DataFrame([summary])
    
    except Exception as e:
        print(f"Failed {folder_path.name}: {e}")
        return None


if __name__ == "__main__":
    results_dir = Path("results_attacks")
    
    if not results_dir.exists():
        print("Folder 'results_attacks' not found!")
        sys.exit(1)
    
    experiment_folders = [f for f in results_dir.iterdir() if f.is_dir()]
    print(f"Found {len(experiment_folders)} experiment folders.\n")
    
    all_summaries = []
    for folder in experiment_folders:
        row = load_and_summarize(folder)
        if row is not None:
            all_summaries.append(row)
    
    if not all_summaries:
        print("No valid results could be loaded.")
        sys.exit(1)
    
    final_df = pd.concat(all_summaries, ignore_index=True)
    
    cols = ["folder", "algo", "constraint_level", "attack_type", "attack_level",
            "min_dist", "mean_dist", "collision_rate", "infeasibility_rate",
            "mean_slack", "max_slack", "slack_active_rate"]
    
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    
    print("=== SUMMARY OF ALL EXPERIMENTS ===")
    print(final_df.round(4))
    
    print("\n=== GROUPED MEAN ===")
    grouped = final_df.groupby(["algo", "constraint_level", "attack_type", "attack_level"]).mean(numeric_only=True)
    print(grouped.round(4))
    
    final_df.to_csv("constraint_conflict_summary.csv", index=False)
    grouped.to_csv("constraint_conflict_grouped.csv")
    
    print("\nAnalysis completed successfully!")
    print("Files saved:")
    print("   → constraint_conflict_summary.csv")
    print("   → constraint_conflict_grouped.csv")