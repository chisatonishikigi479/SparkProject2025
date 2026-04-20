import sys
import os

spark_root = "/mnt/c/Users/mikun/SparkProject2025/spark"
sys.path.insert(0, spark_root)
sys.path.insert(0, os.path.join(spark_root, "pipeline"))

from spark_pipeline import BenchmarkPipeline as Pipeline
from spark_pipeline import G1BenchmarkPipelineConfig as PipelineConfig
from spark_pipeline import generate_benchmark_test_case


def config_task_module(cfg: PipelineConfig, **kwargs):
    """Configure task-related settings."""
    cfg.env.task.max_episode_length = 500
    cfg.env.task.seed = 20
    return cfg


def config_agent_module(cfg: PipelineConfig, **kwargs):
    """Configure agent-related settings - FIXED to prevent negative sleep error."""
    enable_viewer = kwargs.get("enable_viewer", False)
    
    cfg.env.agent.enable_viewer = enable_viewer
    cfg.env.agent.use_sim_dynamics = False
    
    # Critical fixes for stability and speed
    cfg.env.agent.real_time_factor = 0.0 if not enable_viewer else 0.5   # 0 = run as fast as possible
    cfg.env.agent.sync_simulation = False
    
    # Optional: slightly larger dt for heavy safety methods
    # if not enable_viewer:
    #     cfg.env.agent.dt = 0.005
    
    return cfg


def config_policy_module(cfg: PipelineConfig, **kwargs):
    """Configure policy-related settings."""
    # Using default policy
    return cfg


def config_safety_module(cfg: PipelineConfig, **kwargs):
    """Configure safety-related settings with flexible parameters."""
    safe_algo = kwargs.get("safe_algo", "bypass")
    
    # --------------------- Safe Control Algorithm --------------------- #
    match safe_algo:
        case "bypass":
            cfg.algo.safe_controller.safe_algo.class_name = "ByPassSafeControl"
       
        case "ssa":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.eta_ssa = kwargs.get("eta_ssa", 0.1)
            cfg.algo.safe_controller.safe_algo.control_weight = kwargs.get("control_weight", [1.0] * 20)
       
        case "rssa":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.eta_ssa = kwargs.get("eta_ssa", 0.1)
            cfg.algo.safe_controller.safe_algo.slack_weight = kwargs.get("slack_weight", 1e3)
            cfg.algo.safe_controller.safe_algo.control_weight = kwargs.get("control_weight", [1.0] * 20)
       
        case "sss":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSublevelSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.lambda_sss = kwargs.get("lambda_sss", 10.0)
            cfg.algo.safe_controller.safe_algo.control_weight = kwargs.get("control_weight", [1.0] * 20)
           
        case "rsss":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedSublevelSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.lambda_sss = kwargs.get("lambda_sss", 10.0)
            cfg.algo.safe_controller.safe_algo.slack_weight = kwargs.get("slack_weight", 1e3)
            cfg.algo.safe_controller.safe_algo.control_weight = kwargs.get("control_weight", [1.0] * 20)
       
        case "cbf":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicControlBarrierFunction"
            cfg.algo.safe_controller.safe_algo.lambda_cbf = kwargs.get("lambda_cbf", 10.0)
            cfg.algo.safe_controller.safe_algo.control_weight = kwargs.get("control_weight", [1.0] * 20)
           
        case "rcbf":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedControlBarrierFunction"
            cfg.algo.safe_controller.safe_algo.lambda_cbf = kwargs.get("lambda_cbf", 10.0)
            cfg.algo.safe_controller.safe_algo.slack_weight = kwargs.get("slack_weight", 1e3)
            cfg.algo.safe_controller.safe_algo.control_weight = kwargs.get("control_weight", [1.0] * 20)
       
        case "pfm":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicPotentialFieldMethod"
            cfg.algo.safe_controller.safe_algo.c_pfm = kwargs.get("c_pfm", 1.0)
       
        case "sma":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSlidingModeAlgorithm"
            cfg.algo.safe_controller.safe_algo.c_sma = kwargs.get("c_sma", 1.0)

    # Truncate control_weight based on robot type
    if "FixedBase" in cfg.robot.cfg.class_name:
        cfg.algo.safe_controller.safe_algo.control_weight = cfg.algo.safe_controller.safe_algo.control_weight[:-3]
    elif "RightArm" in cfg.robot.cfg.class_name:
        cfg.algo.safe_controller.safe_algo.control_weight = cfg.algo.safe_controller.safe_algo.control_weight[3:10]

    # --------------------- Safety Index --------------------- #
    safety_index = kwargs.get("safety_index", "si1")
    match safety_index:
        case "si1":
            cfg.algo.safe_controller.safety_index.class_name = "FirstOrderCollisionSafetyIndex"
        case "si2":
            cfg.algo.safe_controller.safety_index.class_name = "SecondOrderCollisionSafetyIndex"
            cfg.algo.safe_controller.safety_index.phi_n = kwargs.get("phi_n", 1.0)
            cfg.algo.safe_controller.safety_index.phi_k = kwargs.get("phi_k", 1.0)
        case 'si2nn':
            cfg.algo.safe_controller.safety_index.class_name = "SecondOrderNNCollisionSafetyIndex"
            cfg.algo.safe_controller.safety_index.phi_n = kwargs.get("phi_n", 2)
            cfg.algo.safe_controller.safety_index.phi_k = kwargs.get("phi_k", 1)
            cfg.algo.safe_controller.safety_index.phi_nn_path = kwargs.get("phi_nn_path", "n_2_scalar.onnx")

    return cfg


def config_pipeline(cfg: PipelineConfig, **kwargs):
    """Configure pipeline settings."""
    cfg = generate_benchmark_test_case(cfg, kwargs.get("test_case_name", "G1MobileBase_D1_WG_SO_v0"))
    
    cfg.max_num_steps = 2000
    cfg.max_num_reset = -1
    cfg.enable_logger = True
    cfg.enable_safe_zone_render = False

    # ====================== Adversarial Attacks ======================
    attack_type = kwargs.get("attack_type")      # None, "perception_noise", "latency", "crowding"
    attack_level = kwargs.get("attack_level", "medium")

    if attack_type == "perception_noise":
        noise_std = {"low": 0.02, "medium": 0.05, "high": 0.10}.get(attack_level, 0.05)
        # Note: If these attributes don't exist yet, you may need to patch the task/safety_index later
        if hasattr(cfg.env.task, 'perception_noise_std'):
            cfg.env.task.perception_noise_std = noise_std
        print(f"[Attack] Perception noise (std={noise_std})")

    elif attack_type == "latency":
        delay_steps = {"low": 2, "medium": 5, "high": 10}.get(attack_level, 5)
        if hasattr(cfg.env.task, 'obstacle_update_delay'):
            cfg.env.task.obstacle_update_delay = delay_steps
        print(f"[Attack] Latency delay ({delay_steps} steps)")

    elif attack_type == "crowding":
        density = {"low": 1.5, "medium": 2.0, "high": 3.0}.get(attack_level, 2.0)
        if hasattr(cfg.env.task, 'obstacle_density_multiplier'):
            cfg.env.task.obstacle_density_multiplier = density
        if hasattr(cfg.env.task, 'obstacle_min_distance'):
            cfg.env.task.obstacle_min_distance = 0.15
        print(f"[Attack] Crowding (density x{density})")

    # ====================== Metrics ======================
    if "FixedBase" in getattr(cfg.env.task, 'task_name', ''):
        cfg.metric_selection.dist_goal_base = False
    else:
        cfg.metric_selection.dist_goal_base = True

    cfg.metric_selection.dist_self = True
    cfg.metric_selection.dist_robot_to_env = True
    cfg.metric_selection.dist_goal_arm = True
    cfg.metric_selection.seed = True
    cfg.metric_selection.done = True

    return cfg


def run(**kwargs):
    """Main execution function."""
    cfg = PipelineConfig()
    
    cfg = config_pipeline(cfg, **kwargs)
    cfg = config_task_module(cfg, **kwargs)
    cfg = config_agent_module(cfg, **kwargs)
    cfg = config_policy_module(cfg, **kwargs)
    cfg = config_safety_module(cfg, **kwargs)
    
    pipeline = Pipeline(cfg)
    
    save_path = kwargs.get("save_path", "results_default.json")
    safe_algo = kwargs.get("safe_algo", "unknown")
    attack = kwargs.get("attack_type", "nominal")
    
    print(f"→ Running | Algo={safe_algo} | Attack={attack} | Save={save_path}")
    
    try:
        pipeline.run(save_path=save_path)
        print(f"✓ Successfully finished: {save_path}")
    except Exception as e:
        print(f"❌ Error during pipeline.run(): {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 80)
    return


# ====================== SWEEP FUNCTIONS ======================

def run_benchmark_sweep(**base_kwargs):
    """Sweep different parameter values for one safety algorithm."""
    test_case = base_kwargs.get("test_case_name")
    safe_algo = base_kwargs.get("safe_algo")
    print(f"\n=== Starting parameter sweep for {safe_algo} ===")

    if safe_algo == "ssa":
        for eta in [0.01, 0.05, 0.1, 0.2, 0.5]:
            run(
                test_case_name=test_case,
                safe_algo=safe_algo,
                safety_index=base_kwargs.get("safety_index", "si1"),
                eta_ssa=eta,
                enable_viewer=False,
                save_path=f"results_{safe_algo}_eta_{eta}.json"
            )
    
    elif safe_algo in ["rssa", "rcbf", "rsss"]:
        for slack in [100, 500, 1000, 5000, 10000]:
            run(
                test_case_name=test_case,
                safe_algo=safe_algo,
                safety_index=base_kwargs.get("safety_index", "si1"),
                slack_weight=slack,
                enable_viewer=False,
                save_path=f"results_{safe_algo}_slack_{slack}.json"
            )
    
    elif safe_algo in ["sss", "cbf"]:
        param_name = "lambda_sss" if safe_algo in ["sss", "rsss"] else "lambda_cbf"
        for lam in [1.0, 5.0, 10.0, 20.0, 50.0]:
            run(
                test_case_name=test_case,
                safe_algo=safe_algo,
                safety_index=base_kwargs.get("safety_index", "si1"),
                **{param_name: lam},
                enable_viewer=False,
                save_path=f"results_{safe_algo}_{param_name}_{lam}.json"
            )
    
    else:
        # Default run for pfm, sma, bypass
        run(**base_kwargs, enable_viewer=False)


def run_adversarial_sweep(**base_kwargs):
    """Sweep over attack types and levels."""
    test_case = base_kwargs.get("test_case_name")
    safe_algo = base_kwargs.get("safe_algo")
    print(f"\n=== Starting ADVERSARIAL sweep for {safe_algo} ===")

    attack_types = [None, "perception_noise", "latency", "crowding"]
    
    for attack in attack_types:
        if attack is None:
            run(
                test_case_name=test_case,
                safe_algo=safe_algo,
                safety_index=base_kwargs.get("safety_index", "si1"),
                enable_viewer=False,
                save_path=f"results_{safe_algo}_nominal.json"
            )
        else:
            for level in ["low", "medium", "high"]:
                run(
                    test_case_name=test_case,
                    safe_algo=safe_algo,
                    safety_index=base_kwargs.get("safety_index", "si1"),
                    attack_type=attack,
                    attack_level=level,
                    enable_viewer=False,
                    save_path=f"results_{safe_algo}_attack_{attack}_{level}.json"
                )


# ====================== MAIN ======================

if __name__ == "__main__":
    TASK_CASE = "G1SportMode_D1_WG_SO_v1"
    
    print("=== SPARK G1 Benchmark Script Started ===\n")
    
    # === 1. Debug run with viewer (uncomment when you want to visually check) ===
    # run(
    #     test_case_name=TASK_CASE,
    #     safe_algo="ssa",
    #     safety_index="si1",
    #     eta_ssa=0.1,
    #     enable_viewer=True,
    #     save_path="debug_visual_ssa.json"
    # )

    # === 2. Normal parameter sweeps ===
    '''
    for algo in ["ssa", "rssa", "cbf", "rcbf", "sss"]:
        run_benchmark_sweep(
            test_case_name=TASK_CASE,
            safe_algo=algo,
            safety_index="si1"
        )
        '''

    for algo in ["ssa", "rssa", "rcbf"]:
        run_adversarial_sweep(
            test_case_name=TASK_CASE,
            safe_algo=algo,
            safety_index="si1"
    )

    print("\n=== All runs completed! ===")