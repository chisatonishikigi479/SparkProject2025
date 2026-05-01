import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_one_experiment(folder_path: Path):
    """Create detailed plots for one experiment"""
    npz_file = folder_path / "data.npz"
    if not npz_file.exists():
        return

    data = np.load(npz_file, allow_pickle=True)
    name = folder_path.name

    fig = plt.figure(figsize=(16, 12))

    # 1. Min Distance to Environment Over Time
    plt.subplot(3, 1, 1)
    if "min_dist_to_env" in data:
        plt.plot(data["min_dist_to_env"], color='blue', linewidth=2)
        plt.title(f"{name}\nMinimum Distance to Obstacles Over Time")
        plt.ylabel("Distance")
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label="Collision")
        plt.legend()
        plt.grid(True)
    else:
        plt.title("Min Distance to Environment (No data)")

    # 2. Slack Usage (if available)
    plt.subplot(3, 1, 2)
    slack_key = next((k for k in data.files if "slack" in k.lower()), None)
    if slack_key:
        plt.plot(data[slack_key], color='purple', linewidth=2)
        plt.title(f"Slack Usage Over Time ({slack_key})")
        plt.ylabel("Slack Value")
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.grid(True)
    else:
        plt.title("Slack Usage (No data)")

    # 3. Infeasibility Flag
    plt.subplot(3, 1, 3)
    infeas_key = next((k for k in data.files if "infeas" in k.lower()), None)
    if infeas_key:
        plt.plot(data[infeas_key], color='red', linewidth=2)
        plt.title(f"Infeasibility Over Time ({infeas_key})")
        plt.ylabel("Infeasible (1 = Yes)")
        plt.grid(True)
    else:
        plt.title("Infeasibility (No data)")

    plt.suptitle(f"Experiment: {name}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    output_dir = Path("individual_experiment_plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Plotted: {name}")


# ====================== MAIN ======================
if __name__ == "__main__":
    # Search in common locations
    search_dirs = [Path("results_attacks_v2_G1MobileBase"), 
                   Path("results_attacks"), 
                   Path(".")]
    
    experiment_folders = []
    for d in search_dirs:
        if d.exists():
            experiment_folders.extend([f for f in d.iterdir() if f.is_dir() and ("ExtB" in f.name or "attack" in f.name.lower())])
            break

    print(f"Found {len(experiment_folders)} experiments to plot.\n")

    for folder in experiment_folders:
        plot_one_experiment(folder)

    print(f"\n🎉 All plots saved in 'individual_experiment_plots/' folder!")