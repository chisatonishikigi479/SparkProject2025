"""
Single Strict CBF run - Headless mode (no viewer, much faster + stable)
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

# Create output folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = f"results/cbf_headless_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("Starting Strict CBF benchmark in HEADLESS mode...\n")

cfg = get_config(
    safety_method="BasicControlBarrierFunction",
    min_distance=0.18,
    lambda_cbf=1.0
)

# Force headless mode + saving
cfg.output_dir = output_dir
cfg.experiment_name = "Strict_CBF"

# Disable all rendering / viewer to avoid the debug_object error
#cfg.rendering.enabled = False
if hasattr(cfg, 'visualization'):
    cfg.visualization.render = False
if hasattr(cfg, 'render'):
    cfg.render = False

# Try to disable viewer completely
if hasattr(cfg, 'use_viewer'):
    cfg.use_viewer = False
if hasattr(cfg, 'viewer'):
    cfg.viewer = False

print(f"Saving results to: {output_dir}")

pipeline = Pipeline(cfg)
pipeline.run()

print("\nBenchmark finished successfully!")
print(f"Results saved in: {output_dir}/")