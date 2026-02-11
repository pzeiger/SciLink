import pandas as pd
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import PIL.Image as PIL_Image

from ...auth import get_internal_proxy_key
from .parser_utils import parse_json_from_response 
from ...tools.bo_tools import get_optimizer
from .instruct import (
    BO_CONFIG_SOO_PROMPT,
    BO_CONFIG_MOO_PROMPT,
    BO_VISUAL_INSPECTION_PROMPT,
    BO_VISUAL_INSPECTION_MOO_PROMPT
)

from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from ._deprecation import normalize_params

from .base_agent import BaseAgent

class BOAgent(BaseAgent):
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

    Args:
        api_key: API key for the LLM provider.
        model_name: Model name. For public deployments, use LiteLLM format
            (e.g., "gemini/gemini-2.0-flash", "gpt-4o", "claude-sonnet-4-20250514").
        base_url: Base URL for internal proxy endpoint.
            When provided, uses OpenAI-compatible client.
            When None, uses LiteLLM for multi-provider support.
        output_dir: Output directory for artifacts.
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        output_dir: str = ".",
        # Deprecated
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        
        super().__init__(output_dir)
        self.agent_type = "bo"

        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="BOAgent"
        )
        
        if base_url:
            # INTERNAL PROXY
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
            
            logging.info(f"🏛️ BOAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # PUBLIC LITELLM
            logging.info(f"🌐 BOAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
        
        self.generation_config = None

        self.history_file = self.output_dir / "bo_history.json"

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Agent-specific state fields"""
        return {
            "objective": None,
            "data_path": None,
            "optimization_history": [],
            "current_config": None,
            "data_points_seen": 0
        }
        
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
            logging.warning(f"Invalid kernel '{m_conf.get('kernel')}', defaulting to 'matern_2.5'")
            m_conf["kernel"] = "matern_2.5"
        if m_conf.get("noise") not in ["fixed_low", "learnable", "high_noise"]:
            logging.warning(f"Invalid noise '{m_conf.get('noise')}', defaulting to 'fixed_low'")
            m_conf["noise"] = "fixed_low"
        clean["model_config"] = m_conf
        return clean

    def run_optimization_loop(self, data_path: str, objective_text: str, 
                             input_cols: List[str], input_bounds: List[List[float]], 
                             target_cols: List[str], output_dir: str = "./bo_artifacts",
                             batch_size: int = 1,
                             save_acq: bool = True,
                             plot_acq: bool = True) -> Dict[str, Any]:
        """
        Run one iteration of the Bayesian Optimization loop.
        
        Args:
            data_path: Path to the data file (.xlsx or .csv).
            objective_text: Natural language description of the optimization goal.
            input_cols: List of input column names.
            input_bounds: List of [min, max] bounds for each input.
            target_cols: List of target/objective column names.
            output_dir: Directory for saving artifacts.
            batch_size: Number of candidates to recommend.
            save_acq: If True, saves acquisition function landscape data to .npz file.
                Supported for single-objective only; ignored for multi-objective.
            plot_acq: If True, generates and saves a plot of the acquisition function.
                Supported for single-objective only; ignored for multi-objective.
            
        Returns:
            Dict with status, recommendations, strategy, plot paths, and optionally
            acquisition function plot/data paths (single-objective only).
        """
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialize state
        self._init_state(objective=objective_text, data_path=data_path)
        
        # 1. Load Data
        try:
            df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
            for col in input_cols + target_cols:
                if col not in df.columns: 
                    return {"error": f"Column '{col}' not found in data."}
            X = df[input_cols].values
            y = df[target_cols].values
            
            # Track data points
            self.state["data_points_seen"] = len(df)
            
        except Exception as e:
            return {"error": f"Data load failed: {e}"}

        is_moo = len(target_cols) > 1
        history = self._load_history()

        # 2. Configure Strategy (LLM)
        trend_context = f"Last 5 strategies: {[h.get('config', {}).get('rationale', 'N/A') for h in history[-5:]]}" if history else "No history."
        
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
        valid_config["batch_size"] = int(batch_size)
        
        # Store current config in state
        self.state["current_config"] = valid_config

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

        # 5. Diagnostics
        step_num = len(history) + 1
        plot_path = f"{output_dir}/step_{step_num}.png"
        if is_moo:
            optimizer.generate_diagnostics(save_path=plot_path)
        else:
            optimizer.generate_diagnostics(next_x_batch, df[target_cols[0]].values.tolist(), save_path=plot_path)

        # 5b. Acquisition Function Plot & Data (SOO only)
        acq_plot_path = None
        acq_data_path = None
        
        if not is_moo and plot_acq:
            print("  - 📊 BO Agent: Plotting acquisition function...")
            try:
                acq_plot_path = f"{output_dir}/acq_step_{step_num}.png"
                optimizer.plot_acquisition(
                    candidate_x=next_x_batch,
                    save_path=acq_plot_path
                )
                print(f"  - 💾 Acquisition plot saved: {acq_plot_path}")
            except RuntimeError as e:
                logging.warning(f"Could not plot acquisition function: {e}")
                acq_plot_path = None
        
        if not is_moo and save_acq:
            print("  - 💾 BO Agent: Saving acquisition function data...")
            try:
                acq_data_path = f"{output_dir}/acq_data_step_{step_num}.npz"
                acq_meta = optimizer.save_acquisition_data(
                    candidate_x=next_x_batch,
                    save_path=acq_data_path
                )
                print(f"  - 💾 Acquisition data saved: {acq_data_path} "
                      f"(keys: {len(acq_meta['keys'])})")
            except RuntimeError as e:
                logging.warning(f"Could not save acquisition data: {e}")
                acq_data_path = None

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
            "step": step_num, 
            "config": valid_config, 
            "recommendation_batch": recommendations, 
            "inspection": inspection,
        }
        # Include acquisition paths in history when available
        if acq_plot_path or acq_data_path:
            log_entry["acquisition"] = {
                "strategy": strategy_name,
                "plot_path": acq_plot_path,
                "data_path": acq_data_path,
            }
        self._save_history(log_entry)

        # 8. Output
        if batch_size > 1:
            batch_csv = f"{output_dir}/batch_step_{step_num}.csv"
            pd.DataFrame(recommendations).to_csv(batch_csv, index=False)
            print(f"  - 💾 Batch saved: {batch_csv}")

        result = {
            "status": "success",
            "next_parameters": recommendations[0] if batch_size == 1 else recommendations,
            "strategy": valid_config,
            "plot_path": plot_path,
        }
        if acq_plot_path:
            result["acq_plot_path"] = acq_plot_path
        if acq_data_path:
            result["acq_data_path"] = acq_data_path
        
        # Log this action to state
        self._log_action(
            action="run_optimization_loop",
            input_ctx={
                "data_path": data_path,
                "input_cols": input_cols,
                "target_cols": target_cols,
                "batch_size": batch_size,
                "save_acq": save_acq,
                "plot_acq": plot_acq,
            },
            result=result,
            rationale=valid_config.get("rationale")
        )
        
        return result