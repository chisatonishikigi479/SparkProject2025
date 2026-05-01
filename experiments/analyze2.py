import numpy as np
import pandas as pd
from pathlib import Path
import sys

def analyze_results(results_folder="results_attacks_v2_G1MobileBase"):
    results_dir = Path(results_folder)
    
    if not results_dir.exists():
        print(f"Folder '{results_folder}' not found!")
        print("Current folders found:")
        for p in Path(".").iterdir():
            if p.is_dir() and ("result" in p.name.lower() or "attack" in p.name.lower()):
                print("   ", p.name)
        sys.exit(1)

    experiment_folders = [f for f in results_dir.iterdir() if f.is_dir()]
    print(f"Found {len(experiment_folders)} experiment folders in '{results_folder}'.\n")

    records = []

    for folder in experiment_folders:
        npz_file = folder / "data.npz"
        if not npz_file.exists():
            print(f"No data.npz in {folder.name}")
            continue

        try:
            data = np.load(npz_file, allow_pickle=True)
            name = folder.name.lower()

            record = {
                "folder": folder.name,
                "algo": next((a for a in ["ssa", "rssa", "cbf", "rcbf"] if a in name), "unknown"),
                "constraint_level": "D2" if "d2" in name else "D1",
                "attack_type": "perception_noise" if "perception" in name else 
                               "latency" if "latency" in name else "nominal",
                "attack_level": next((l for l in ["low", "medium", "high"] if l in name), "nominal"),
            }

            for key in data.files:
                arr = data[key]
                if arr.size == 0:
                    continue
                flat = arr.flatten()

                if any(d in key.lower() for d in ["dist", "distance"]):
                    record["min_dist"] = float(np.min(flat))
                    record["mean_dist"] = float(np.mean(flat))
                    record["collision_rate"] = float(np.mean(flat < 0.01))

                if any(i in key.lower() for i in ["infeas", "infeasible"]):
                    record["infeasibility_rate"] = float(np.mean(flat) * 100)

                if "slack" in key.lower():
                    record["mean_slack"] = float(np.mean(flat))
                    record["max_slack"] = float(np.max(flat))
                    record["slack_active_%"] = float(np.mean(flat > 0) * 100)

            records.append(record)

        except Exception as e:
            print(f"Failed {folder.name}: {e}")

    if not records:
        print("No valid data loaded.")
        return

    df = pd.DataFrame(records)

    print("=== OVERALL SUMMARY ===")
    print(df.round(4))

    print("\n=== GROUPED MEAN COMPARISON ===")
    grouped = df.groupby(["algo", "constraint_level", "attack_type", "attack_level"]).mean(numeric_only=True)
    print(grouped.round(4))

    df.to_csv("experiment_summary_detailed.csv", index=False)
    grouped.to_csv("experiment_summary_grouped.csv")

    print("\nAnalysis completed!")
    print("Saved:")
    print("   • experiment_summary_detailed.csv")
    print("   • experiment_summary_grouped.csv")


if __name__ == "__main__":
    analyze_results("results_attacks_v2_G1MobileBase")