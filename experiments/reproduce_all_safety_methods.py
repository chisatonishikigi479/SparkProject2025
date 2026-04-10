"""
Reproduce SPARK Benchmark - Compare all 5 safety methods + Save Results
"""

import sys
import os
import time
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'spark'))

from configs.benchmark_config import get_config
from spark_pipeline import BenchmarkPipeline as Pipeline

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = f"results/benchmark_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting SPARK Benchmark Comparison")
print(f"Results will be saved to: {output_dir}\n")

safety_methods = {
    "SSA": "BasicSafeSetAlgorithm",
    "CBF": "BasicControlBarrierFunction",
    "SSS": "BasicSublevelSafeSetAlgorithm",
    "PFM": "BasicPotentialFieldMethod",
    "SMA": "BasicSlidingModeAlgorithm"
}

results_summary = {}

for name, class_name in safety_methods.items():
    print(f"→ Running {name} ...")
    start_time = time.time()

    cfg = get_config(
        safety_method=class_name,
        min_distance=0.18,
        lambda_cbf=1.0,
    )
    
    if hasattr(cfg, 'max_steps'):
        cfg.max_steps = 75
    elif hasattr(cfg, 'runner') and hasattr(cfg.runner, 'max_steps'):
        cfg.runner.max_steps = 75

    cfg.output_dir = output_dir
    cfg.experiment_name = name   

    pipeline = Pipeline(cfg)
    pipeline.run()

    elapsed = time.time() - start_time
    print(f"   Finished {name} in {elapsed:.1f} seconds\n")

    results_summary[name] = f"{elapsed:.1f}s"

print("="*70)
print("ALL 5 SAFETY METHODS COMPLETED!")
print("Summary:")
for name, t in results_summary.items():
    print(f"   {name:>3} : {t}")

print(f"\nAll logs, metrics, and data saved in:")
print(f"   {output_dir}/")
print("   (Look for .csv files, logs, and metric summaries inside)")