import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

summary_file = "ExtensionB_Full_Summary.csv"

if not Path(summary_file).exists():
    print(f"{summary_file} not found. Please run the parser first!")
    exit()

df = pd.read_csv(summary_file)
print("Columns in your summary file:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())


infeas_col = None
slack_col = None

for name in ["infeasibility_rate", "infeas_rate", "infeasible_rate", "infeasible"]:
    if name in df.columns:
        infeas_col = name
        break

for name in ["mean_slack", "slack_mean", "slack", "mean_slack_value"]:
    if name in df.columns:
        slack_col = name
        break

if not infeas_col:
    print("Could not find infeasibility column. Using 0 as fallback.")
    df["infeasibility_rate"] = 0.0
    infeas_col = "infeasibility_rate"

if not slack_col:
    print("Could not find slack column. Using 0 as fallback.")
    df["mean_slack"] = 0.0
    slack_col = "mean_slack"

Path("plots").mkdir(exist_ok=True)

def save_plot(fig, name):
    fig.savefig(f"plots/{name}.png", dpi=300, bbox_inches='tight')
    print(f"Saved: plots/{name}.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(data=df, x="algo", y=infeas_col, hue="constraint_level", ax=axes[0])
axes[0].set_title("Infeasibility Rate: D1 vs D2")
axes[0].set_ylabel("Infeasibility Rate (%)")
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(data=df, x="algo", y=infeas_col, hue="attack_type", ax=axes[1])
axes[1].set_title("Infeasibility Rate under Attacks")
axes[1].set_ylabel("Infeasibility Rate (%)")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
save_plot(fig, "01_Infeasibility_Rate")

if slack_col in df.columns:
    relaxed = df[df['algo'].isin(['rssa', 'rcbf', 'rsss'])]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.barplot(data=relaxed, x="algo", y=slack_col, hue="constraint_level", ax=axes[0])
    axes[0].set_title("Mean Slack Usage")
    axes[0].set_ylabel("Mean Slack")
    
    sns.barplot(data=relaxed, x="algo", y="slack_active_rate" if "slack_active_rate" in df.columns else slack_col, 
                hue="attack_type", ax=axes[1])
    axes[1].set_title("Slack Active Rate under Attacks")
    axes[1].set_ylabel("Slack Active Rate (%)")
    
    plt.tight_layout()
    save_plot(fig, "02_Slack_Usage")

fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(data=df, x="algo", y="min_dist_to_env", hue="constraint_level", ax=ax)
ax.set_title("Minimum Distance to Obstacles (Higher = Safer)")
ax.set_ylabel("Min Distance")
ax.tick_params(axis='x', rotation=45)
save_plot(fig, "03_Min_Distance")

attack_df = df[df['attack_type'] != 'nominal']
if not attack_df.empty:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    sns.barplot(data=attack_df, x="attack_level", y=infeas_col, hue="algo", ax=axes[0,0])
    axes[0,0].set_title("Infeasibility Rate by Attack Strength")
    
    if slack_col:
        sns.barplot(data=attack_df, x="attack_level", y=slack_col, hue="algo", ax=axes[0,1])
        axes[0,1].set_title("Mean Slack by Attack Strength")
    
    sns.barplot(data=attack_df, x="attack_level", y="min_dist_to_env", hue="algo", ax=axes[1,0])
    axes[1,0].set_title("Min Distance by Attack Strength")
    
    plt.tight_layout()
    save_plot(fig, "04_Attack_Strength_Comparison")

print("\nAll available plots have been generated in the 'plots/' folder!")