from spark_pipeline import BenchmarkPipeline as Pipeline
from spark_pipeline import G1BenchmarkPipelineConfig as PipelineConfig
from spark_pipeline import generate_benchmark_test_case

def config_task_module(cfg: PipelineConfig, **kwargs):
    """Configure task-related settings."""
    cfg.env.task.max_episode_length = 500
    cfg.env.task.seed = 20    
    return cfg

def config_agent_module(cfg: PipelineConfig, **kwargs):
    """Configure agent-related settings."""
    cfg.env.agent.enable_viewer = True
    cfg.env.agent.use_sim_dynamics = False
    return cfg

def config_policy_module(cfg: PipelineConfig, **kwargs):
    """Configure policy-related settings."""
    # Use the default policy configuration
    return cfg

def config_safety_module(cfg: PipelineConfig, **kwargs):
    """Configure safety-related settings."""
    # --------------------- Config Safe Control Algorithm --------------------- #
    safe_algo = kwargs.get("safe_algo", "bypass")  # Default to 'bypass' if not provided
    match safe_algo:
        case "bypass":
            cfg.algo.safe_controller.safe_algo.class_name = "ByPassSafeControl"
        
        case "ssa":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.eta_ssa = 0.1
            cfg.algo.safe_controller.safe_algo.control_weight = [
                1.0, 1.0, 1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
                1.0, 1.0, 1.0,
            ]
        
        case "rssa":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.eta_ssa = 0.1
            cfg.algo.safe_controller.safe_algo.slack_weight = 1e3
            cfg.algo.safe_controller.safe_algo.control_weight = [
                1.0, 1.0, 1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
                1.0, 1.0, 1.0  # locomotion
            ]
        
        case "sss":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSublevelSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.lambda_sss = 10.0
            cfg.algo.safe_controller.safe_algo.control_weight = [
                1.0, 1.0, 1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
                1.0, 1.0, 1.0  # locomotion
            ]
            
        case "rsss":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedSublevelSafeSetAlgorithm"
            cfg.algo.safe_controller.safe_algo.lambda_sss = 10.0
            cfg.algo.safe_controller.safe_algo.slack_weight = 1e3
            cfg.algo.safe_controller.safe_algo.control_weight = [
                1.0, 1.0, 1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
                1.0, 1.0, 1.0  # locomotion
            ]
        
        case "cbf":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicControlBarrierFunction"
            cfg.algo.safe_controller.safe_algo.lambda_cbf = 10.0
            cfg.algo.safe_controller.safe_algo.control_weight = [
                1.0, 1.0, 1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
                1.0, 1.0, 1.0  # locomotion
            ]
            
        case "rcbf":
            cfg.algo.safe_controller.safe_algo.class_name = "RelaxedControlBarrierFunction"
            cfg.algo.safe_controller.safe_algo.lambda_cbf = 10.0
            cfg.algo.safe_controller.safe_algo.slack_weight = 1e3
            cfg.algo.safe_controller.safe_algo.control_weight = [
                1.0, 1.0, 1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
                1.0, 1.0, 1.0  # locomotion
            ]
        
        case "pfm":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicPotentialFieldMethod"
            cfg.algo.safe_controller.safe_algo.c_pfm = 1.0
        
        case "sma":
            cfg.algo.safe_controller.safe_algo.class_name = "BasicSlidingModeAlgorithm"
            cfg.algo.safe_controller.safe_algo.c_sma = 1.0
            
    if "FixedBase" in cfg.robot.cfg.class_name:
        cfg.algo.safe_controller.safe_algo.control_weight = cfg.algo.safe_controller.safe_algo.control_weight[:-3]
    elif "RightArm" in cfg.robot.cfg.class_name:
        cfg.algo.safe_controller.safe_algo.control_weight = cfg.algo.safe_controller.safe_algo.control_weight[3:10]

    # --------------------- Config Safe Control Index --------------------- #
    safety_index = kwargs.get("safety_index", "si1")  # Default to 'si1' if not provided
    match safety_index:
        case "si1":
            cfg.algo.safe_controller.safety_index.class_name = "FirstOrderCollisionSafetyIndex"
        case "si2":
            cfg.algo.safe_controller.safety_index.class_name = "SecondOrderCollisionSafetyIndex"
            cfg.algo.safe_controller.safety_index.phi_n = 1.0
            cfg.algo.safe_controller.safety_index.phi_k = 1.0
        case 'si2nn':
            cfg.algo.safe_controller.safety_index.class_name = "SecondOrderNNCollisionSafetyIndex"
            cfg.algo.safe_controller.safety_index.phi_n = 2,
            cfg.algo.safe_controller.safety_index.phi_k = 1,
            cfg.algo.safe_controller.safety_index.phi_nn_path = "n_2_scalar.onnx"

    return cfg

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
    
def run( **kwargs):
    cfg = PipelineConfig()
    
    cfg = config_pipeline(cfg, **kwargs)
    cfg = config_task_module(cfg, **kwargs)
    cfg = config_agent_module(cfg, **kwargs)
    cfg = config_policy_module(cfg, **kwargs)
    cfg = config_safety_module(cfg, **kwargs)
    
    pipeline = Pipeline(cfg)
    
    save_path = kwargs.get("save_path")
    print(f"Running {kwargs.get('safe_algo', 'unknown')} | save_path = {save_path}")
    
    try:
        pipeline.run(save_path=save_path)  
    except Exception as e:
        print(f"Error during pipeline.run(): {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Finished run. Expected results file: {save_path}\n")
    return
    
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
    levels = ["D1", "D2"]                    # D2 usually has more obstacles → more conflicts
    seeds = [10, 20, 30, 40, 50]
    
    for level in levels:
        for algo in algos:
            for seed in seeds:
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
    """
    Extension B: Constraint conflict stress test
    
    Systematically increases the number of safety constraints (by using D1 vs D2 test cases)
    until conflicts become common, then compares safety methods under:
    - Perception noise attacks
    - Latency attacks
    
    Metrics automatically logged: infeasibility rate, slack usage, collision rate,
    minimum distance to obstacles, and task success rate.
    """
    test_case_base = base_kwargs.get("test_case_name", "G1SportMode")
    safety_index = base_kwargs.get("safety_index", "si1")
    
    print("\n" + "="*80)
    print("EXTENSION B: CONSTRAINT CONFLICT STRESS TEST")
    print("Increasing number of safety constraints + noise/latency attacks")
    print("="*80)

    # Constraint levels: D1 = fewer obstacles (low conflict), D2 = more obstacles (high conflict)
    constraint_levels = ["D1", "D2"]          # Add "D3" if your test cases support it
    
    # Safety methods to compare (strict vs relaxed)
    algos = ["ssa", "cbf", "rssa", "rcbf"]    # you can add "sss", "rsss", "pfm", "sma"

    # Attack types we will test on top of high constraint levels
    attack_configs = [
        (None, None),                                      # Nominal (baseline)
        ("perception_noise", "low"),
        ("perception_noise", "medium"),
        ("perception_noise", "high"),
        ("latency", "low"),
        ("latency", "medium"),
        ("latency", "high"),
    ]

    for constraint_level in constraint_levels:
        # Construct the full test case name (e.g. G1SportMode_D2_WG_SO_v1)
        test_case = f"{test_case_base}_{constraint_level}_WG_SO_v1"
        
        print(f"\n→ Constraint level: {constraint_level}  (test_case = {test_case})")
        
        for algo in algos:
            print(f"   Running {algo} ...")
            
            for attack_type, attack_level in attack_configs:
                if attack_type is None:
                    save_name = f"results_ExtB_{algo}_{constraint_level}_nominal.json"
                    attack_str = "nominal"
                else:
                    save_name = f"results_ExtB_{algo}_{constraint_level}_{attack_type}_{attack_level}.json"
                    attack_str = f"{attack_type}-{attack_level}"
                
                run(
                    test_case_name=test_case,
                    safe_algo=algo,
                    safety_index=safety_index,
                    attack_type=attack_type,
                    attack_level=attack_level,
                    enable_viewer=False,
                    save_path=save_name
                )
                
                print(f"      └─ {attack_str} → {save_name}")

    print("\n" + "="*80)
    print("Extension B completed! All results saved with prefix 'results_ExtB_'")
    print("Next step: run the analysis script to compare infeasibility rate, slack usage, etc.")
    print("="*80)
            
if __name__ == "__main__":
    print("=== SPARK G1 Benchmark Script Started ===\n")
    
    run_constraint_conflict_stress_test_v2(
        test_case_name="G1SportMode",      
        safety_index="si1"
    )
    
    print("\n=== All runs completed! ===")
    
    
