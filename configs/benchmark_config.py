"""
SPARK benchmark config - with built-in output_dir support
"""

import sys
import os
import numpy as np
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'spark'))

from spark_pipeline import G1BenchmarkPipelineConfig as PipelineConfig
from spark_pipeline import generate_benchmark_test_case

def get_config(safety_method="BasicControlBarrierFunction", 
               min_distance=0.15, 
               lambda_cbf=1.0,
               output_dir=None):  
    
    cfg = PipelineConfig()
    cfg = generate_benchmark_test_case(cfg, "G1FixedBase_D1_AG_DO_v0")

    cfg.algo.safe_controller.safe_algo.class_name = safety_method

    cfg.algo.safe_controller.safe_algo.control_weight = np.ones(17).tolist()

    if safety_method == "BasicControlBarrierFunction":
        cfg.algo.safe_controller.safe_algo.lambda_cbf = lambda_cbf
    elif safety_method == "BasicSublevelSafeSetAlgorithm":
        cfg.algo.safe_controller.safe_algo.lambda_sss = 1.0

    cfg.algo.safe_controller.safety_index.min_distance["environment"] = min_distance

    cfg.metric_selection.dist_robot_to_env = True
    cfg.metric_selection.dist_robot_to_goal = True
    cfg.metric_selection.success_rate = True
    cfg.metric_selection.collision_rate = True

    cfg.max_steps = 150
    if hasattr(cfg, 'rendering'):
        cfg.rendering.enabled = False
    if hasattr(cfg, 'visualization'):
        cfg.visualization.render = False

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = f"results/cbf_single_{timestamp}"
    
    cfg.output_dir = output_dir
    cfg.experiment_name = "Strict_CBF"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    return cfg