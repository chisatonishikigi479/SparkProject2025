
import os
import sys

'''
os.environ["MUJOCO_GL"] = "osmesa"      
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["DISPLAY"] = ""
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
'''

from spark_pipeline import BenchmarkPipeline as Pipeline
from spark_pipeline import G1BenchmarkPipelineConfig as PipelineConfig
from spark_pipeline import generate_benchmark_test_case


import numpy as np

def config_task_module(cfg: PipelineConfig, **kwargs):
    """Configure task-related settings."""
    cfg.env.task.max_episode_length = 500
    cfg.env.task.seed = 20    
    return cfg

def config_agent_module(cfg: PipelineConfig, **kwargs):
    """Configure agent-related settings."""
    cfg.env.agent.enable_viewer = True
    cfg.env.agent.use_sim_dynamics = False
    #cfg.env.agent.real_time_factor = 0.0
    return cfg

def config_policy_module(cfg: PipelineConfig, **kwargs):
    """Configure policy-related settings."""
    # Use the default policy configuration
    return cfg

def config_safety_module(cfg: PipelineConfig, **kwargs):
    safe_algo = kwargs.get("safe_algo", "bypass")
    attack_type = kwargs.get("attack_type")
    attack_level = kwargs.get("attack_level", "medium")

    match safe_algo:
        case "bypass":
            cfg.algo.safe_controller.safe_algo.class_name = "ByPassSafeControl"

        case "ssa":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSafeSetAlgorithm"   # ← Correct name
            cfg.algo.safe_controller.safe_algo.eta_ssa = kwargs.get("eta_ssa", 0.1)

        case "rssa":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.eta_ssa = kwargs.get("eta_ssa", 0.1)
            cfg.algo.safe_controller.safe_algo.slack_weight = kwargs.get("slack_weight", 1000)

        case "sss":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSublevelSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.lambda_sss = kwargs.get("lambda_sss", 10.0)

        case "rsss":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedSublevelSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.lambda_sss = kwargs.get("lambda_sss", 10.0)
            cfg.algo.safe_controller.safe_algo.slack_weight = kwargs.get("slack_weight", 1000)

        case "cbf":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicControlBarrierFunction"
            cfg.algo.safe_controller.safe_algo.lambda_cbf = kwargs.get("lambda_cbf", 10.0)

        case "rcbf":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedControlBarrierFunction"
            cfg.algo.safe_controller.safe_algo.lambda_cbf = kwargs.get("lambda_cbf", 10.0)
            cfg.algo.safe_controller.safe_algo.slack_weight = kwargs.get("slack_weight", 1000)

        case "pfm":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicPotentialFieldMethod"
            cfg.algo.safe_controller.safe_algo.c_pfm = kwargs.get("c_pfm", 1.0)

        case "sma":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSlidingModeAlgorithm"
            cfg.algo.safe_controller.safe_algo.c_sma = kwargs.get("c_sma", 1.0)

    # Control weight truncation
    if "FixedBase" in cfg.robot.cfg.class_name:
        if hasattr(cfg.algo.safe_controller.safe_algo, 'control_weight'):
            cfg.algo.safe_controller.safe_algo.control_weight = cfg.algo.safe_controller.safe_algo.control_weight[:-3]
    elif "RightArm" in cfg.robot.cfg.class_name:
        if hasattr(cfg.algo.safe_controller.safe_algo, 'control_weight'):
            cfg.algo.safe_controller.safe_algo.control_weight = cfg.algo.safe_controller.safe_algo.control_weight[3:10]

    safety_index = kwargs.get("safety_index", "si1")
    match safety_index:
        case "si1":
            cfg.algo.safe_controller.safety_index.class_name = "FirstOrderCollisionSafetyIndex"
        case "si2":
            cfg.algo.safe_controller.safety_index.class_name = "SecondOrderCollisionSafetyIndex"

    if attack_type:
        cfg.algo.safe_controller.attack_type = attack_type
        cfg.algo.safe_controller.attack_level = attack_level
        print(f"[Attack Configured] {attack_type} ({attack_level})")

    return cfg

def apply_manual_attacks(pipeline):
    """Apply manual attacks by monkey-patching the safety index"""
    if not hasattr(pipeline.cfg.algo.safe_controller, 'attack_type'):
        return

    attack_type = pipeline.cfg.algo.safe_controller.attack_type
    if not attack_type:
        return

    safety_index = pipeline.agent.safety_index 
    if attack_type == "perception_noise":
        std = getattr(pipeline.cfg.algo.safe_controller, 'perception_noise_std', 0.08)
        
        original_get_distances = safety_index.get_distances
        
        def noisy_get_distances(self, *args, **kwargs):
            dists = original_get_distances(*args, **kwargs)
            noise = np.random.normal(0, std, size=dists.shape)
            return np.maximum(dists + noise, 0.0)   
        
        safety_index.get_distances = noisy_get_distances.__get__(safety_index)
        print(f"   → Perception noise injected (std={std})")

    elif attack_type == "latency":
        delay = getattr(pipeline.cfg.algo.safe_controller, 'obstacle_latency_steps', 8)
        safety_index.latency_buffer = []
        safety_index.latency_delay = delay
        
        original_get_distances = safety_index.get_distances
        
        def delayed_get_distances(self, *args, **kwargs):
            current_dists = original_get_distances(*args, **kwargs)
            safety_index.latency_buffer.append(current_dists.copy())
            
            if len(safety_index.latency_buffer) > safety_index.latency_delay:
                return safety_index.latency_buffer.pop(0)
            else:
                return current_dists 
        
        safety_index.get_distances = delayed_get_distances.__get__(safety_index)
        print(f"   → Latency attack injected (delay={delay} steps)")

def config_pipeline(cfg: PipelineConfig, **kwargs):
    test_case_name = kwargs.get("test_case_name", "G1MobileBase_D1_WG_SO_v0")
    cfg = generate_benchmark_test_case(cfg, test_case_name)
    
    cfg.max_num_steps = 2000
    cfg.max_num_reset = -1
    cfg.enable_logger = True
    cfg.enable_safe_zone_render = False

    attack_type = kwargs.get("attack_type", None)     
    attack_level = kwargs.get("attack_level", "medium") 

    if attack_type == "perception_noise":
        noise_std = {"low": 0.02, "medium": 0.05, "high": 0.10}.get(attack_level, 0.05)
        cfg.env.task.perception_noise_std = noise_std          
        print(f"Perception noise attack enabled (std={noise_std})")

    elif attack_type == "latency":
        delay_steps = {"low": 2, "medium": 5, "high": 10}.get(attack_level, 5)
        cfg.env.task.obstacle_update_delay = delay_steps
        print(f"Latency attack enabled (delay={delay_steps} steps)")

    elif attack_type == "crowding":
        density_multiplier = {"low": 1.5, "medium": 2.0, "high": 3.0}.get(attack_level, 2.0)
        cfg.env.task.obstacle_density_multiplier = density_multiplier
        cfg.env.task.obstacle_min_distance = 0.15   # force tighter constraints
        print(f"Crowding attack enabled (density x{density_multiplier})")

    if "FixedBase" in getattr(cfg.env.task, 'task_name', ''):
        cfg.metric_selection.dist_goal_base = False
    else:
        cfg.metric_selection.dist_goal_base = True

    cfg.metric_selection.dist_self = True
    cfg.metric_selection.dist_robot_to_env = True
    cfg.metric_selection.dist_goal_arm = True
    cfg.metric_selection.seed = True
    cfg.metric_selection.done = True

    cfg.metric_selection.attack_type = True
    cfg.metric_selection.attack_level = True

    return cfg
    
def run(**kwargs):
    cfg = PipelineConfig()

    cfg = config_pipeline(cfg, **kwargs)
    cfg = config_task_module(cfg, **kwargs)
    cfg = config_agent_module(cfg, **kwargs)
    cfg = config_policy_module(cfg, **kwargs)
    cfg = config_safety_module(cfg, **kwargs)
    

    pipeline = Pipeline(cfg)
    
    apply_manual_attacks(pipeline)
    
    save_path = kwargs.get("save_path")
    print(f"→ Running | Algo={kwargs.get('safe_algo')} | Attack={kwargs.get('attack_type')} | Save={save_path}")
    
    try:
        pipeline.run(save_path=save_path)
        print(f"Finished: {save_path}")
    except Exception as e:
        print(f"Error during pipeline.run(): {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 80)
def run_adversarial_sweep(**base_kwargs):
    test_case = base_kwargs.get("test_case_name")
    safe_algo = base_kwargs.get("safe_algo")
    
    print(f"Starting adversarial sweep for {safe_algo} on task: {test_case}")

    attack_types = [None, "perception_noise", "latency", "crowding"]
    levels = ["low", "medium", "high"]

    for attack in attack_types:
        for level in levels if attack else [None]:  
            kwargs = base_kwargs.copy()
            kwargs["attack_type"] = attack
            if attack:
                kwargs["attack_level"] = level
                kwargs["save_path"] = f"results_{safe_algo}_attack_{attack}_{level}.json"
            else:
                kwargs["save_path"] = f"results_{safe_algo}_nominal.json"
            
            kwargs["enable_viewer"] = False
            
            run(**kwargs)

def run_benchmark_sweep(**base_kwargs):

    test_case = base_kwargs.get("test_case_name")
    safe_algo = base_kwargs.get("safe_algo")
    safety_index = base_kwargs.get("safety_index", "si1")
    
    print(f"Starting sweep for {safe_algo} with safety_index={safety_index} on task: {test_case}")

    
    if safe_algo == "ssa":
        param_list = [0.01, 0.05, 0.1, 0.2, 0.5]
        param_name = "eta_ssa"
        for value in param_list:
            kwargs = base_kwargs.copy()
            kwargs[param_name] = value
            kwargs["enable_viewer"] = False
            kwargs["save_path"] = f"results_{safe_algo}_{param_name}_{value}.json"
            run(**kwargs)

    elif safe_algo in ["rssa", "rcbf", "rsss"]:
        param_list = [1e2, 5e2, 1e3, 5e3, 1e4]
        param_name = "slack_weight"
        for value in param_list:
            kwargs = base_kwargs.copy()
            kwargs[param_name] = value
            kwargs["enable_viewer"] = False
            kwargs["save_path"] = f"results_{safe_algo}_{param_name}_{value}.json"
            run(**kwargs)

    elif safe_algo in ["sss", "cbf"]:
        param_list = [1.0, 5.0, 10.0, 20.0, 50.0]
        param_name = "lambda_sss" if safe_algo in ["sss", "rsss"] else "lambda_cbf"
        for value in param_list:
            kwargs = base_kwargs.copy()
            kwargs[param_name] = value
            kwargs["enable_viewer"] = False
            kwargs["save_path"] = f"results_{safe_algo}_{param_name}_{value}.json"
            run(**kwargs)

    elif safe_algo == "pfm":
        param_list = [0.1, 0.5, 1.0, 2.0, 5.0]
        param_name = "c_pfm"
        for value in param_list:
            kwargs = base_kwargs.copy()
            kwargs[param_name] = value
            kwargs["enable_viewer"] = False
            kwargs["save_path"] = f"results_{safe_algo}_{param_name}_{value}.json"
            run(**kwargs)

    elif safe_algo == "sma":
        param_list = [0.1, 0.5, 1.0, 2.0, 5.0]
        param_name = "c_sma"
        for value in param_list:
            kwargs = base_kwargs.copy()
            kwargs[param_name] = value
            kwargs["enable_viewer"] = False
            kwargs["save_path"] = f"results_{safe_algo}_{param_name}_{value}.json"
            run(**kwargs)

    else:
        print(f"No sweep defined for {safe_algo}. Running once with defaults.")
        kwargs = base_kwargs.copy()
        kwargs["enable_viewer"] = False
        kwargs["save_path"] = f"results_{safe_algo}_default.json"
        run(**kwargs)
        
def run_constraint_conflict_stress_test_v2(**base_kwargs):
    algos = ["ssa", "cbf", "rssa", "rcbf"]
    levels = ["D1", "D2"]                    
    seeds = [10, 20, 30, 40, 50]
    
    for algo in algos:
        for seed in seeds:
            for level in levels:
                run(
                    test_case_name=f"G1SportMode_{level}_WG_SO_v1",
                    safe_algo=algo,
                    safety_index="si1",
                    eta_ssa=0.1 if algo == "ssa" else None,
                    lambda_cbf=10.0 if algo == "cbf" else None,
                    slack_weight=1000 if "r" in algo else None,
                    seed=seed,
                    enable_viewer=False,
                    save_path=f"results_ExtB_{algo}_{level}_seed{seed}.json"
                )
            
        
def run_constraint_conflict_stress_test(**base_kwargs):
    """Run full stress test with D1/D2 + manual attacks"""
    base_name = base_kwargs.get("test_case_name", "G1SportMode")
    safety_index = base_kwargs.get("safety_index", "si1")
    
    algos = ["ssa", "cbf", "rssa", "rcbf"]
    levels = ["D1", "D2"]
    
    print("Starting Constraint Conflict Stress Test with Manual Attacks...\n")
    
    for level in levels:
        test_case = f"{base_name}_{level}_WG_SO_v1"
        print(f"\n=== Testing Constraint Level: {level} === ({test_case})")
        
        for algo in algos:
            for attack in [None, "perception_noise", "latency"]:
                for level_name in ["low", "medium", "high"] if attack else ["nominal"]:
                    
                    save_name = f"results_ExtB_{algo}_{level}_{attack or 'nominal'}_{level_name}.json"
                    
                    run(
                        test_case_name=test_case,
                        safe_algo=algo,
                        safety_index=safety_index,
                        attack_type=attack,
                        attack_level=level_name if attack else None,
                        seed=42,                  
                        enable_viewer=False,
                        save_path=save_name
                    )
                    print(f"   → {algo:6} | {level} | Attack: {attack or 'nominal':18} | {level_name}")
            
if __name__ == "__main__":
    print("=== SPARK G1 Benchmark - Constraint Conflict Stress Test with Manual Attacks ===\n")
    
    
    
    TASK_BASE = "G1SportMode"
    run_constraint_conflict_stress_test(
        test_case_name=TASK_BASE,
        safety_index="si1"
    )

    print("\n=== All experiments finished! ===\n")
    print("Next: Run the parser → then the plotting scripts.")
    
    
