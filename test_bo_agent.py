import os
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from scilink.agents.planning_agents.bo_agent import BOAgent
from scilink.agents.planning_agents.parser_utils import append_experiment_result


# =============================================================================
# 1. THE "LABORATORY" (Synthetic Ground Truth)
# =============================================================================
def branin_lab_simulator(x1: float, x2: float) -> float:
    """
    Simulates a physical experiment using the Branin function.
    We negate the result because Branin is usually minimized, but scientists 
    usually MAXIMIZE yield.
    
    Domain: x1=[-5, 10], x2=[0, 15]
    Global Maxima (shifted): approx 300.4
    """
    # Standard parameters
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    result = term1 + term2 + s
    
    # Invert for maximization, add offset, add simulated measurement noise
    y = 300 - result + np.random.normal(0, 0.5)
    return round(y, 4)

def create_seed_data(filepath: str, n_points: int = 5):
    """Creates initial random experiments to warm start the GP."""
    print(f"🌱 Seeding {filepath} with {n_points} random points...")
    data = []
    for _ in range(n_points):
        x1 = np.random.uniform(-5, 10)
        x2 = np.random.uniform(0, 15)
        y = branin_lab_simulator(x1, x2)
        data.append({"Temperature": round(x1, 2), "Pressure": round(x2, 2), "Yield": y})
    
    pd.DataFrame(data).to_excel(filepath, index=False)

# =============================================================================
# 2. THE CAMPAIGN RUNNER
# =============================================================================
def run_campaign():
    # --- Configuration ---
    DATA_FILE = "synthetic_experiment.xlsx"
    ARTIFACTS_DIR = "./bo_artifacts_test"
    N_STEPS = 5
    
    # Clean up previous runs
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
    if os.path.exists(ARTIFACTS_DIR): shutil.rmtree(ARTIFACTS_DIR)
    if os.path.exists("bo_history.json"): os.remove("bo_history.json")

    # 1. Initialize System
    create_seed_data(DATA_FILE, n_points=5)
    
    print("\n🤖 Initializing BO Agent...")
    # NOTE: Ensure you have GOOGLE_API_KEY env var set, or pass it here
    agent = BOAgent(model_name="gemini-3-pro-preview") 

    # 2. Schema Definition
    # This maps the physical world to the math world
    input_cols = ["Temperature", "Pressure"]
    target_cols = ["Yield"]
    bounds = [[-5.0, 10.0], [0.0, 15.0]]
    objective_text = "Maximize Yield. Theoretical maximum is around 300."

    # 3. Main Loop
    for step in range(1, N_STEPS + 1):
        print(f"\n" + "="*60)
        print(f"🔄 CAMPAIGN STEP {step}/{N_STEPS}")
        print("="*60)
        
        # A. AGENT THINKING & OPTIMIZATION
        result = agent.run_optimization_loop(
            data_path=DATA_FILE,
            objective_text=objective_text,
            input_cols=input_cols,
            input_bounds=bounds,
            target_cols=target_cols,
            output_dir=ARTIFACTS_DIR
        )
        
        if result.get("error"):
            print(f"❌ CRITICAL ERROR: {result['error']}")
            break
            
        # Parse Decision
        rec = result["next_parameters"]
        strategy = result.get("config", {}).get("acquisition_strategy", {})
        rationale = result.get("config", {}).get("rationale", "N/A")
        
        print(f"\n🧠 Agent Rationale: \"{rationale}\"")
        print(f"⚙️  Strategy Used: {strategy.get('type')} (Params: {strategy.get('params')})")
        print(f"👉 Recommendation: {rec}")
        print(f"📊 Diagnostics: {result.get('plot_path')}")

        # B. LAB EXECUTION (Simulated)
        print("\n🧪 Running Experiment in Simulator...")
        # Extract values strictly in order of input_cols to match function signature
        new_y = branin_lab_simulator(rec["Temperature"], rec["Pressure"])
        print(f"🎉 Result: Yield = {new_y}")
        
        # C. MEMORY UPDATE
        append_experiment_result(
            file_path=DATA_FILE,
            parameters=rec,
            results={"Yield": new_y}
        )

    # 4. Final Report
    print(f"\n" + "="*60)
    print("✅ CAMPAIGN FINISHED")
    final_df = pd.read_excel(DATA_FILE)
    best_row = final_df.loc[final_df["Yield"].idxmax()]
    print(f"🏆 Best Found: Yield={best_row['Yield']} at Temp={best_row['Temperature']}, Press={best_row['Pressure']}")
    print(f"📂 Final Data: {DATA_FILE}")
    print("="*60)

if __name__ == "__main__":
    run_campaign()