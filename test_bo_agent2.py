import os
import shutil
import numpy as np
import pandas as pd
from scilink.agents.planning_agents.bo_agent import BOAgent
from scilink.agents.planning_agents.parser_utils import append_experiment_result

# =============================================================================
# 1. THE "LABORATORY" (4D Rosenbrock Function)
# =============================================================================
def rosenbrock_lab_simulator(x1: float, x2: float, x3: float, x4: float) -> float:
    """
    4D Rosenbrock Function.
    Global Minimum is at (1, 1, 1, 1) with value 0.
    We Negate it to make it a MAXIMIZATION problem (Target -> 0).
    
    Domain: [-2.0, 2.0] for all inputs.
    Features: Long, narrow curved valley. Hard to converge.
    """
    X = np.array([x1, x2, x3, x4])
    
    # Standard Rosenbrock sum: sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    val = 0
    for i in range(len(X) - 1):
        val += 100 * (X[i+1] - X[i]**2)**2 + (1 - X[i])**2
        
    # Invert for maximization (Map 0 -> 0, larger errors -> negative)
    # Ideally we want to get as close to 0 as possible from below.
    y = -val + np.random.normal(0, 1.0) # Add noise
    return round(y, 4)

def create_seed_data(filepath: str, n_points: int = 10):
    """
    Seeding with more points (10) because 4D space is much larger.
    """
    print(f"🌱 Seeding {filepath} with {n_points} random points (4D)...")
    data = []
    for _ in range(n_points):
        # Random inputs in [-2, 2]
        inputs = np.random.uniform(-2, 2, 4)
        y = rosenbrock_lab_simulator(*inputs)
        
        row = {f"x{i+1}": round(v, 4) for i, v in enumerate(inputs)}
        row["Yield"] = y
        data.append(row)
    
    pd.DataFrame(data).to_excel(filepath, index=False)

# =============================================================================
# 2. CAMPAIGN RUNNER
# =============================================================================
def run_rosenbrock_campaign():
    # --- Config ---
    DATA_FILE = "experiment_data_rosen.xlsx"
    ARTIFACTS_DIR = "./bo_artifacts_rosen"
    N_STEPS = 20 # More steps needed for this harder problem
    
    # Cleanup
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
    if os.path.exists(ARTIFACTS_DIR): shutil.rmtree(ARTIFACTS_DIR)
    if os.path.exists("bo_history.json"): os.remove("bo_history.json")

    # 1. Initialize
    create_seed_data(DATA_FILE, n_points=10)
    print("\n🤖 Initializing BO Agent (Gemini Pro)...")
    # Ensure GOOGLE_API_KEY is set
    agent = BOAgent(model_name="gemini-3-pro-preview") 

    # 2. Schema
    input_cols = ["x1", "x2", "x3", "x4"]
    bounds = [[-2.0, 2.0] for _ in range(4)]
    target_cols = ["Yield"]
    
    # We explicitly tell the agent about the valley structure in the objective
    objective_text = """
    Maximize Yield. 
    This is a 4D Rosenbrock landscape (inverted). 
    Theoretical Maximum is 0. 
    Expect a narrow, curved valley.
    """

    # 3. Loop
    for step in range(1, N_STEPS + 1):
        print(f"\n" + "="*60)
        print(f"🔄 ROSENBROCK STEP {step}/{N_STEPS}")
        print("="*60)
        
        result = agent.run_optimization_loop(
            data_path=DATA_FILE,
            objective_text=objective_text,
            input_cols=input_cols,
            input_bounds=bounds,
            target_cols=target_cols,
            output_dir=ARTIFACTS_DIR
        )
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            break
            
        rec = result["next_parameters"]
        conf = result.get("strategy", {}) 
        
        strat_type = conf.get("acquisition_strategy", {}).get("type")
        kernel = conf.get("model_config", {}).get("kernel")
        
        print(f"🧠 Rationale: {conf.get('rationale')}")
        print(f"⚙️  Settings: {strat_type} | {kernel}")
        print(f"👉 Suggestion: {rec}")
        print(f"📊 Plot: {result.get('plot_path')}")

        # Run Lab
        new_y = rosenbrock_lab_simulator(rec["x1"], rec["x2"], rec["x3"], rec["x4"])
        print(f"🧪 Result: {new_y}")
        
        append_experiment_result(DATA_FILE, rec, {"Yield": new_y})

    # 4. Final
    print(f"\n" + "="*60)
    final_df = pd.read_excel(DATA_FILE)
    best_val = final_df["Yield"].max()
    print(f"🏆 Best Yield Achieved: {best_val} (Target: 0.0)")
    print("="*60)

if __name__ == "__main__":
    run_rosenbrock_campaign()