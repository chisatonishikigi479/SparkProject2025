"""
Run Single SPARK Benchmark with Strict CBF + Reliable Saving
"""

import sys
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'spark'))

from configs.benchmark_config import get_config
from spark_pipeline import BenchmarkPipeline as Pipeline

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = f"results/cbf_single_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("Starting SPARK Benchmark with Strict CBF")
print(f"Saving results to: {output_dir}\n")

cfg = get_config(
    safety_method="BasicControlBarrierFunction",
    min_distance=0.18,
    lambda_cbf=1.0
)

cfg.output_dir = output_dir
cfg.experiment_name = "Strict_CBF"

if hasattr(cfg, 'max_steps'):
    cfg.max_steps = 150
elif hasattr(cfg, 'runner') and hasattr(cfg.runner, 'max_steps'):
    cfg.runner.max_steps = 150

print("Starting pipeline...")

pipeline = Pipeline(cfg)
pipeline.run()

print("\nCBF Benchmark Finished!")
print(f"Results should be saved in: {output_dir}/")