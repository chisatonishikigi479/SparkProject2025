import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def plot_single_experiment(timestep_csv_path: Path):
    """Create detailed plots for one single experiment"""
    
    df = pd.read_csv(timestep_csv_path)
    name = timestep_csv_path.stem.replace("_timestep", "")
    
    print(f"Plotting: {name}")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(3, 2, 1)
    if "min_dist_to_env" in df.columns:
        ax1.plot(df["min_dist_to_env"], linewidth=2, color='blue')
        ax1.set_title("Minimum Distance to Obstacles Over Time")
        ax1.set_ylabel("Distance")
        ax1.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, label="Collision Threshold")
        ax1.legend()
    
    ax2 = plt.subplot(3, 2, 2)
    slack_col = None
    for col in ["slack", "slack_value", "mean_slack", "slack_var"]:
        if col in df.columns:
            slack_col = col
            break
    
    if slack_col:
        ax2.plot(df[slack_col], linewidth=2, color='purple')
        ax2.set_title(f"Slack Usage Over Time ({slack_col})")
        ax2.set_ylabel("Slack Value")
        ax2.axhline(y=0, color='gray', linestyle='--')
    else:
        ax2.text(0.5, 0.5, "No Slack Data", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Slack Usage")
    
    ax3 = plt.subplot(3, 2, 3)
    infeas_col = None
    for col in ["infeasible", "infeasibility", "qp_infeasible", "infeas"]:
        if col in df.columns:
            infeas_col = col
            break
    
    if infeas_col:
        ax3.plot(df[infeas_col], linewidth=2, color='red')
        ax3.set_title("Infeasibility Flag Over Time (1 = Infeasible)")
        ax3.set_ylabel("Infeasible")
    else:
        ax3.text(0.5, 0.5, "No Infeasibility Data", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Infeasibility")
    
    ax4 = plt.subplot(3, 2, 4)
    if "dist_goal_right_arm" in df.columns:
        ax4.plot(df["dist_goal_right_arm"], label="Right Arm", color='green')
        ax4.plot(df["dist_goal_left_arm"], label="Left Arm", color='orange')
        ax4.set_title("Distance to Goal (Arms)")
        ax4.set_ylabel("Distance")
        ax4.legend()
    
    ax5 = plt.subplot(3, 2, 5)
    if infeas_col and slack_col:
        ax5.plot(df[infeas_col].cumsum(), label="Cumulative Infeasible Steps", color='red')
        ax5.plot(df[slack_col].cumsum(), label="Cumulative Slack", color='purple')
        ax5.set_title("Cumulative Metrics")
        ax5.set_ylabel("Cumulative Sum")
        ax5.legend()
    
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    stats = f"""Experiment: {name}
    
Total Steps: {len(df)}
Min Distance to Env: {df['min_dist_to_env'].min():.4f}
Mean Distance to Env: {df['min_dist_to_env'].mean():.4f}
"""
    if infeas_col:
        infeas_rate = df[infeas_col].mean() * 100
        stats += f"Infeasibility Rate: {infeas_rate:.2f}%\n"
    if slack_col:
        stats += f"Mean Slack: {df[slack_col].mean():.4f}\n"
        stats += f"Slack Active: {(df[slack_col] > 0).mean()*100:.2f}%"
    
    ax6.text(0.05, 0.95, stats, transform=ax6.transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.suptitle(f"Experiment: {name}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    output_path = Path("individual_plots") / f"{name}.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   → Saved: {output_path.name}")


if __name__ == "__main__":
    parsed_dir = Path("parsed_results")   # Change if your folder is different
    
    if not parsed_dir.exists():
        print(f"Folder '{parsed_dir}' not found!")
        print("Please run the parser first.")
    else:
        timestep_files = list(parsed_dir.glob("*_timestep.csv"))
        print(f"Found {len(timestep_files)} timestep files to plot...\n")
        
        for csv_file in timestep_files:
            plot_single_experiment(csv_file)
        
        print("\n🎉 All individual experiment plots completed!")
        print(f"Check the folder: individual_plots/")