"""
Single config file for SPARK benchmarks - fixed for control_weight mismatch
"""

import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'spark'))

from spark_pipeline import G1BenchmarkPipelineConfig as PipelineConfig
from spark_pipeline import generate_benchmark_test_case

def get_config(safety_method="BasicControlBarrierFunction", min_distance=0.15, lambda_cbf=1.0):
    cfg = PipelineConfig()
    cfg = generate_benchmark_test_case(cfg, "G1FixedBase_D1_AG_DO_v0")

    cfg.algo.safe_controller.safe_algo.class_name = safety_method

    if safety_method == "BasicControlBarrierFunction":
        cfg.algo.safe_controller.safe_algo.lambda_cbf = lambda_cbf
        
 
        cfg.algo.safe_controller.safe_algo.control_weight = np.ones(17).tolist()
        
    if safety_method in ["BasicControlBarrierFunction", "BasicSafeSetAlgorithm", 
                         "BasicSublevelSafeSetAlgorithm", "BasicSlidingModeAlgorithm"]:
        cfg.algo.safe_controller.safe_algo.control_weight = np.ones(17).tolist()



    if safety_method == "BasicControlBarrierFunction":
        cfg.algo.safe_controller.safe_algo.lambda_cbf = 1.0
    elif safety_method == "BasicSublevelSafeSetAlgorithm":
        cfg.algo.safe_controller.safe_algo.lambda_sss = 1.0
        
    cfg.algo.safe_controller.safety_index.min_distance["environment"] = min_distance

    # Metrics for goal tracking + safety
    cfg.metric_selection.dist_robot_to_env = True
    
    cfg.metric_selection.dist_robot_to_goal = True
    cfg.metric_selection.success_rate = True
    cfg.metric_selection.collision_rate = True
    cfg.enable_logger = True
    #limit to 150 steps for performance improvement
    cfg.max_steps = 150

    #make it not run at like 2 FPS
    if hasattr(cfg, 'rendering'):
        cfg.rendering.enabled = False
    if hasattr(cfg, 'visualization'):
        cfg.visualization.render = False
    if hasattr(cfg, 'render'):
        cfg.render = False
    return cfg