"""
Single config file for SPARK benchmarks
"""

import sys
import os

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'spark'))

from spark_pipeline import G1BenchmarkPipelineConfig as PipelineConfig
from spark_pipeline import generate_benchmark_test_case

def get_config(safety_method="BasicControlBarrierFunction", min_distance=0.15, lambda_cbf=1.0):
    cfg = PipelineConfig()
    cfg = generate_benchmark_test_case(cfg, "G1FixedBase_D1_AG_DO_v0")

    # Set safety algorithm
    cfg.algo.safe_controller.safe_algo.class_name = safety_method

    # === Required for CBF ===
    if safety_method == "BasicControlBarrierFunction":
        cfg.algo.safe_controller.safe_algo.lambda_cbf = lambda_cbf

    # Safety margin
    cfg.algo.safe_controller.safety_index.min_distance["environment"] = min_distance

    # Metrics
    cfg.metric_selection.dist_robot_to_env = True
    cfg.metric_selection.dist_robot_to_goal = True
    cfg.metric_selection.success_rate = True
    cfg.metric_selection.collision_rate = True

    return cfg