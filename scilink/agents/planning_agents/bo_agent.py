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
    BO_VISUAL_INSPECTION_PROMPT
)

class BOAgent:
    """
    Autonomous Agent for Bayesian Optimization.
    """
    def __init__(self, 
                 google_api_key: str = None, 
                 model_name: str = "gemini-3-pro-preview", 
                 local_model: str = None):
        
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        # --- LLM Backend Configuration ---
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
        """Sanitizes LLM output to ensure it matches the hardcoded tools."""
        clean = config.copy()
        m_conf = clean.get("model_config", {})
        
        # Kernel Validation
        if m_conf.get("kernel") not in ["matern_2.5", "matern_1.5", "rbf"]:
            m_conf["kernel"] = "matern_2.5"
            
        # Noise Validation
        if m_conf.get("noise") not in ["fixed_low", "learnable", "high_noise"]:
            m_conf["noise"] = "fixed_low"
            
        clean["model_config"] = m_conf
        return clean

    def run_optimization_loop(self, data_path: str, objective_text: str, 
                              input_cols: List[str], input_bounds: List[List[float]], 
                              target_cols: List[str], output_dir: str = "./bo_artifacts") -> Dict[str, Any]:
        
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

        # 2. Trend Analysis
        trend_context = "No history."
        if history:
            last_5 = history[-5:]
            trend_context = f"Last 5 strategies: {[h.get('config', {}).get('rationale', 'N/A') for h in last_5]}"

        # 3. Configure Strategy (LLM)
        print("  - 🤖 BO Agent: Configuring strategy...")
        prompt_tmpl = BO_CONFIG_MOO_PROMPT if is_moo else BO_CONFIG_SOO_PROMPT
        
        prompt_parts = [
            prompt_tmpl,
            f"Objective: {objective_text}",
            f"Meta-Data Trend: {trend_context}",
            f"Data Summary:\n{df.describe().to_markdown()}"
        ]
        
        resp = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
        
        raw_config, parse_error = parse_json_from_response(resp)
        if parse_error: 
            return {"error": f"Failed to parse strategy JSON: {parse_error}"}
        
        valid_config = self._validate_config(raw_config)
        
        # 4. Instantiate & Fit Tool
        optimizer = get_optimizer(is_moo=is_moo)
        optimizer.fit(
            X, y, 
            bounds=input_bounds, 
            model_config=valid_config["model_config"],
            feature_names=input_cols
        )

        # 5. Recommend Next Point
        acq_conf = valid_config.get("acquisition_strategy", {})
        
        print(f"  - 🚀 Optimizing {acq_conf.get('type')}...")
        next_x = optimizer.recommend(
            n_candidates=1,
            strategy=acq_conf.get("type", "log_ei" if not is_moo else "pareto"),
            params=acq_conf.get("params", {})
        )

        # 6. Generate Diagnostics (Visual Memory)
        plot_path = f"{output_dir}/step_{len(history)+1}.png"
        y_hist_list = df[target_cols[0]].values.tolist()
        
        if is_moo:
            optimizer.generate_diagnostics(save_path=plot_path)
        else:
            optimizer.generate_diagnostics(next_x, y_hist_list, save_path=plot_path)

        # 7. Visual Inspection (Multimodal)
        print("  - 👀 BO Agent: Inspecting visuals...")
        try:
            img = PIL_Image.open(plot_path)
            insp_prompt = [BO_VISUAL_INSPECTION_PROMPT, img]
            
            insp_resp = self.model.generate_content(insp_prompt, generation_config=self.generation_config)
            
            inspection, inspect_error = parse_json_from_response(insp_resp)
            if inspect_error:
                 inspection = {"status": "unknown", "reason": f"Parse error: {inspect_error}"}
                 
        except Exception as e:
            logging.warning(f"Visual inspection failed: {e}")
            inspection = {"status": "skipped", "reason": f"Error: {e}"}

        # 8. Commit to History
        rec_dict = {k: float(v) for k, v in zip(input_cols, next_x[0])}
        
        log_entry = {
            "step": len(history) + 1,
            "config": valid_config,
            "recommendation": rec_dict,
            "inspection": inspection
        }
        self._save_history(log_entry)

        return {
            "status": "success",
            "next_parameters": rec_dict,
            "strategy": valid_config,
            "plot_path": plot_path
        }
