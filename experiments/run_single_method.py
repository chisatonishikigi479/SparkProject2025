import sys
import os
import pandas as pd
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'spark'))

from configs.benchmark_config import get_config
from spark_pipeline import BenchmarkPipeline as Pipeline

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = f"results/cbf_planb_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("Running Strict CBF benchmark...\n")

cfg = get_config(
    safety_method="BasicControlBarrierFunction",
    min_distance=0.18,
    lambda_cbf=1.0
)

pipeline = Pipeline(cfg)
pipeline.run()

print("\nBenchmark finished!")


data = {
    'safety_method': 'CBF',
    'max_steps': getattr(cfg, 'max_steps', 150),
    'success_rate': None,
    'collision_rate': None,
    'avg_dist_to_goal': None,
    'min_safety_distance': None,
    'avg_safety_distance': None,
}

try:
    if hasattr(pipeline, 'logger') and hasattr(pipeline.logger, 'metrics'):
        m = pipeline.logger.metrics
        data['success_rate'] = m.get('success_rate', None)
        data['collision_rate'] = m.get('collision_rate', None)
        data['avg_dist_to_goal'] = m.get('dist_robot_to_goal', None)
        data['min_safety_distance'] = m.get('min_dist_robot_to_env', None)
        data['avg_safety_distance'] = m.get('dist_robot_to_env', None)

    elif hasattr(pipeline, 'metrics'):
        m = pipeline.metrics
        data['success_rate'] = m.get('success_rate', None)
        data['collision_rate'] = m.get('collision_rate', None)
        data['avg_dist_to_goal'] = m.get('dist_robot_to_goal', None)
        data['min_safety_distance'] = m.get('min_dist_robot_to_env', None)

    if hasattr(pipeline, 'task') and hasattr(pipeline.task, 'metrics'):
        print("Found metrics on task object")

except Exception as e:
    print(f"Warning: Could not extract metrics automatically: {e}")

df = pd.DataFrame([data])
csv_path = os.path.join(output_dir, "metrics_summary.csv")
df.to_csv(csv_path, index=False)

print("\nExtracted Metrics:")
print(df.to_string(index=False))

print(f"\nSaved to: {csv_path}")
print(f"Full folder: {output_dir}")