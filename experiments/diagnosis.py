import pandas as pd
from pathlib import Path

parsed_dir = Path("parsed_results")

if not parsed_dir.exists():
    print("parsed_results folder not found!")
    exit()

print("=== DIAGNOSIS: Are the experiments actually different? ===\n")

files = sorted(parsed_dir.glob("*_timestep.csv"))

for f in files[:10]:  
    df = pd.read_csv(f)
    name = f.stem.replace("_timestep", "")
    
    min_dist = df["min_dist_to_env"].min() if "min_dist_to_env" in df.columns else "N/A"
    mean_dist = df["min_dist_to_env"].mean() if "min_dist_to_env" in df.columns else "N/A"
    infeas_rate = (df["infeasible"].mean()*100 if "infeasible" in df.columns else 
                  df.get("infeasibility", pd.Series([0])).mean()*100)
    
    print(f"{name:60} | Steps: {len(df):4} | Min Dist: {min_dist:.4f} | "
          f"Mean Dist: {mean_dist:.4f} | Infeas Rate: {infeas_rate:.2f}%")

print(f"\nTotal files found: {len(files)}")