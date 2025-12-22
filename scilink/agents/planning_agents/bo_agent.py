import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import PIL.Image as PIL_Image

import google.generativeai as genai
from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from .parser_utils import parse_json_from_response 
from ...tools.bo_tools import get_optimizer
from .instruct import (
    BO_CONFIG_SOO_PROMPT,
    BO_CONFIG_MOO_PROMPT,
    BO_VISUAL_INSPECTION_PROMPT,
    BO_VISUAL_INSPECTION_MOO_PROMPT
)

class BOAgent:
    """
    Autonomous Agent for Bayesian Optimization (BO) designed for "Stop-and-Go" experimental loops.

    This agent acts as an AI research partner that plans your next set of experiments.
    It combines valid statistical modeling (Gaussian Processes) with LLM-based reasoning 
    to adaptively configure the optimization strategy based on your data trends.

    **DATA FORMATTING REQUIREMENTS:**
    --------------------------------
    The agent expects a "Tidy Data" format (Excel .xlsx or CSV .csv) where:
    1.  **Rows** represent individual experiments.
    2.  **Columns** represent input parameters (e.g., 'Temperature', 'Pressure') and 
        measured objectives (e.g., 'Yield', 'Purity').
    3.  **No Merged Cells:** Ensure the header is a single row containing clean variable names.
    4.  **Missing Data:** The agent requires complete data rows for the optimization columns. 
        Rows with NaNs in inputs/targets should be removed or imputed before running.

    **PERSISTENCE & WORKFLOW:**
    ---------------------------
    This agent is stateless and persistent. It is safe to shut down between experiments.
    
    1.  **Run Agent:** Call `run_optimization_loop` pointing to your current data file.
    2.  **Get Recommendations:** The agent saves a new batch of experiments to 
        `./bo_artifacts/batch_step_N.csv`.
    3.  **Shut Down:** You can close the program while you perform the experiments in the lab 
        (whether it takes 1 hour or 1 week).
    4.  **Update Data:** Once results are in, append them as new rows to your original 
        data file (.xlsx/.csv).
    5.  **Restart:** Run the agent again. It automatically re-reads the updated data 
        and the history file (`bo_history.json`) to pick up exactly where it left off.

    **ARGUMENTS:**
    --------------
    google_api_key (str): Google Gemini API Key.
    model_name (str): LLM model name (default: "gemini-3-pro-preview").
    local_model (str): Optional URL for local/OpenAI-compatible endpoints.
    """
    def __init__(self, 
                 google_api_key: str = None, 
                 model_name: str = "gemini-3-pro-preview", 
                 local_model: str = None):
        
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            logging.info(f"🏛️  BO Agent using OpenAI-compatible model: {model_name}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name, 
                api_key=google_api_key, 
                base_url=local_model
            )
            self.generation_config = None 
        else:
            logging.info(f"☁️  BO Agent using Google Gemini model: {model_name}")
            if google_api_key:
                genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        self.history_file = Path("./bo_history.json")

    def _load_history(self) -> List[Dict]:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f: return json.load(f)
        return []

    def _save_history(self, entry: Dict):
        history = self._load_history()
        history.append(entry)
        with open(self.history_file, 'w') as f: json.dump(history, f, indent=2)

    def _validate_config(self, config: Dict) -> Dict:
        clean = config.copy()
        m_conf = clean.get("model_config", {})
        if m_conf.get("kernel") not in ["matern_2.5", "matern_1.5", "rbf"]:
            m_conf["kernel"] = "matern_2.5"
        if m_conf.get("noise") not in ["fixed_low", "learnable", "high_noise"]:
            m_conf["noise"] = "fixed_low"
        clean["model_config"] = m_conf
        return clean

    def run_optimization_loop(self, data_path: str, objective_text: str, 
                              input_cols: List[str], input_bounds: List[List[float]], 
                              target_cols: List[str], output_dir: str = "./bo_artifacts",
                              batch_size: int = 1) -> Dict[str, Any]:
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # 1. Load Data
        try:
            df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
            for col in input_cols + target_cols:
                if col not in df.columns: return {"error": f"Column '{col}' not found in data."}
            X = df[input_cols].values
            y = df[target_cols].values
        except Exception as e:
            return {"error": f"Data load failed: {e}"}

        is_moo = len(target_cols) > 1
        history = self._load_history()

        # 2. Configure Strategy (LLM)
        trend_context = f"Last 5 strategies: {[h.get('config', {}).get('rationale', 'N/A') for h in history[-5:]]}" if history else "No history."
        
        # Select Prompt and inject context
        prompt_tmpl = BO_CONFIG_MOO_PROMPT if is_moo else BO_CONFIG_SOO_PROMPT
        prompt_parts = [
            prompt_tmpl,
            f"Objective: {objective_text}",
            f"Constraint: Fixed Batch Size = {batch_size}",
            f"Meta-Data Trend: {trend_context}",
            f"Data Summary:\n{df.describe().to_markdown()}"
        ]
        
        print(f"  - 🤖 BO Agent: Configuring strategy (Batch={batch_size})...")
        resp = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
        raw_config, parse_error = parse_json_from_response(resp)
        if parse_error: 
            return {"error": f"JSON Error: {parse_error}"}
        
        valid_config = self._validate_config(raw_config)
        valid_config["batch_size"] = batch_size # Lock in the user constraint

        # 3. Fit Model
        optimizer = get_optimizer(is_moo=is_moo)
        optimizer.fit(
            X, y, 
            bounds=input_bounds, 
            model_config=valid_config["model_config"],
            feature_names=input_cols
        )

        # 4. Recommend
        acq_conf = valid_config.get("acquisition_strategy", {})
        strategy_name = acq_conf.get("type", "pareto" if is_moo else "log_ei")
        
        print(f"  - 🚀 Optimizing {strategy_name}...")
        next_x_batch = optimizer.recommend(
            n_candidates=batch_size,
            strategy=strategy_name,
            params=acq_conf.get("params", {})
        )

        # 5. Diagnostics (Plot only first candidate)
        plot_path = f"{output_dir}/step_{len(history)+1}.png"
        if is_moo:
            optimizer.generate_diagnostics(save_path=plot_path)
        else:
            optimizer.generate_diagnostics(next_x_batch, df[target_cols[0]].values.tolist(), save_path=plot_path)

        # 6. Inspection
        print("  - 👀 BO Agent: Inspecting visuals...")
        visual_prompt = BO_VISUAL_INSPECTION_MOO_PROMPT if is_moo else BO_VISUAL_INSPECTION_PROMPT
        try:
            img = PIL_Image.open(plot_path)
            insp_resp = self.model.generate_content([visual_prompt, img], generation_config=self.generation_config)
            inspection, _ = parse_json_from_response(insp_resp)
        except Exception as e:
            inspection = {"status": "skipped", "reason": str(e)}

        # 7. Save History
        recommendations = []
        for row in next_x_batch:
            recommendations.append({k: float(v) for k, v in zip(input_cols, row)})
            
        log_entry = {
            "step": len(history) + 1, 
            "config": valid_config, 
            "recommendation_batch": recommendations, 
            "inspection": inspection
        }
        self._save_history(log_entry)

        # 8. Output
        if batch_size > 1:
            batch_csv = f"{output_dir}/batch_step_{len(history)+1}.csv"
            pd.DataFrame(recommendations).to_csv(batch_csv, index=False)
            print(f"  - 💾 Batch saved: {batch_csv}")

        return {
            "status": "success",
            "next_parameters": recommendations[0] if batch_size == 1 else recommendations,
            "strategy": valid_config,
            "plot_path": plot_path
        }