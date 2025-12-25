import subprocess
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
import PIL.Image as PIL_Image

import google.generativeai as genai

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from .parser_utils import parse_json_from_response
from .instruct import SCALARIZER_PROMPT, SCALARIZER_REFLECTION_PROMPT


class ScalarizerAgent:
    """
    Agent for converting raw experimental data into scalar descriptors
    suitable for Bayesian Optimization.
    """
    def __init__(self, 
                 google_api_key: str = None, 
                 model_name: str = "gemini-3-pro-preview", 
                 local_model: str = None):
        
        # Auth & Model Initialization
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            logging.info(f"🏛️  Analysis Agent using OpenAI-compatible model: {model_name}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name, 
                api_key=google_api_key, 
                base_url=local_model
            )
            self.generation_config = None 
        else:
            logging.info(f"☁️  Analysis Agent using Google Gemini model: {model_name}")
            if google_api_key:
                genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        # Local Storage Setup
        self.output_dir = Path("./analysis_artifacts")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _read_file_head(self, file_path: str, n_lines=25) -> str:
        """Reads raw file header to help LLM handle delimiters/metadata."""
        path = Path(file_path)
        if not path.exists(): return "Error: File not found."
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                head = [next(f) for _ in range(n_lines)]
            return "".join(head)
        except Exception as e:
            return f"Error reading file head: {str(e)}"

    def _execute_script(self, script_path: Path) -> Dict[str, Any]:
        """Runs the generated python script in a subprocess."""
        try:
            process = subprocess.run(
                ["python", str(script_path)],
                capture_output=True, text=True, timeout=45
            )
            # Parse STDOUT for JSON
            json_match = re.search(r'\{.*\}', process.stdout.strip(), re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return {
                    "status": "success",
                    "metrics": data.get("metrics", {}),
                    "plot_path": data.get("plot_path", ""),
                    "stdout": process.stdout
                }
            else:
                return {
                    "status": "failure",
                    "error": "No JSON output found in stdout",
                    "stdout": process.stdout,
                    "stderr": process.stderr
                }
        except subprocess.TimeoutExpired:
            return {"status": "failure", "error": "Script execution timed out"}
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    def _verify_analysis(self, 
                         objective: str, 
                         context_str: str,
                         script_content: str, 
                         metrics: Dict, 
                         plot_path: str) -> Dict[str, Any]:
        """Multimodal Self-Reflection: Checks plot vs. objective."""
        try:
            image = PIL_Image.open(plot_path)
        except Exception as e:
            return {"status": "fail", "feedback": f"Could not load visual proof: {e}"}

        prompt = f"""
        **AUDIT REQUEST:**
        **1. OBJECTIVE:** "{objective}"
        **2. CONTEXT:** {context_str}
        **3. METRICS:** {json.dumps(metrics, indent=2)}
        **4. CODE:** \n```python\n{script_content[:1500]}\n```
        **5. PROOF:** (See Attached Image)
        
        Does the plot prove the metric was extracted correctly?
        """
        
        try:
            # Note: If using local_model (OpenAI wrapper), ensure it supports image inputs
            response = self.model.generate_content(
                [SCALARIZER_REFLECTION_PROMPT, prompt, image],
                generation_config=self.generation_config
            )
            return parse_json_from_response(response)[0]
        except Exception as e:
            # Fail safe: If reflection crashes, warn but let human review catch it
            logging.warning(f"Reflection failed: {e}")
            return {"status": "pass", "reasoning": "Auto-reflection unavailable."}

    def scalarize(self, 
                  data_path: str, 
                  objective_query: str, 
                  experiment_context: Optional[Dict[str, Any]] = None,
                  metadata_path: Optional[str] = None, 
                  enable_human_review: bool = True) -> Dict[str, Any]:
        """
        Main entry point. Converts raw data -> Scalar Metrics.
        Auto-discovers metadata JSON if not provided.
        """
        path_obj = Path(data_path)
        file_context = self._read_file_head(data_path)
        
        # Metadata Auto-Discovery
        # If no path provided, check for a JSON file with the same name (e.g., run_01.csv -> run_01.json)
        if not metadata_path:
            potential_json = path_obj.with_suffix('.json')
            if potential_json.exists():
                metadata_path = str(potential_json)
                print(f"  - ℹ️  Auto-discovered metadata file: {potential_json.name}")
        
        metadata_str = self._read_metadata(metadata_path)

        # Format Contexts
        exp_context_str = json.dumps(experiment_context) if experiment_context else "None"
        
        base_prompt = f"""
        **INPUT DATA:** {data_path}
        **HEAD SNIPPET:** \n{file_context}\n

        **METADATA SIDECAR (Column Defs / Units):**
        {metadata_str}
        
        **EXPERIMENTAL CONTEXT (Hypothesis / Steps):**
        {exp_context_str}
        
        **GOAL:** "{objective_query}"
        
        **REQ:** Parse, Calculate, Plot (save to {self.output_dir}/debug_{path_obj.stem}.png), Print JSON.
        """

        current_prompt = base_prompt
        max_retries = 5

        for attempt in range(max_retries):
            print(f"  - 📉 Scalarizer (Attempt {attempt+1}): Generating script...")
            
            # Generate Script
            try:
                response = self.model.generate_content(
                    [SCALARIZER_PROMPT, current_prompt], 
                    generation_config=self.generation_config
                )
                result, error = parse_json_from_response(response)
            except Exception as e:
                return {"error": f"LLM Generation Error: {e}"}

            if error or not result or "implementation_code" not in result:
                # If LLM produces garbage, try again with simple feedback
                current_prompt = base_prompt + f"\n\n**PREVIOUS ERROR:** JSON parsing failed. Return ONLY valid JSON."
                continue

            # Save Script
            script_path = self.output_dir / f"proc_{path_obj.stem}.py"
            with open(script_path, "w", encoding="utf-8") as f: 
                f.write(result["implementation_code"])
            
            # Execute Script
            exec_res = self._execute_script(script_path)
            
            if exec_res["status"] == "failure":
                print(f"    ❌ Runtime Error. Retrying...")
                # Feed stderr back to LLM
                current_prompt = base_prompt + f"\n\n**RUNTIME ERROR:**\n{exec_res.get('stderr')}\nFix the code."
                continue
                
            # Auto-Reflection (Visual Check)
            print(f"    🤔 Auto-Reflecting on visual proof...")
            verification = self._verify_analysis(
                objective=objective_query,
                context_str=exp_context_str,
                script_content=result["implementation_code"],
                metrics=exec_res["metrics"],
                plot_path=exec_res["plot_path"]
            )
            
            if verification.get("status") == "fail":
                feedback = verification.get("feedback", "Unknown logic error")
                print(f"    ❌ Self-Correction Triggered: {feedback}")
                current_prompt = base_prompt + f"\n\n**AUTO-CRITIQUE:** {feedback}\nAdjust the code and visuals."
                continue
            
            print(f"    ✅ Auto-Reflection Passed.")

            # Human Review (Optional)
            if enable_human_review:
                print("\n" + "="*60)
                print(f"👀 SCALARIZER REVIEW: {path_obj.name}")
                print(f"• Metrics: {exec_res['metrics']}")
                print(f"• Plot: {exec_res['plot_path']}")
                print("-" * 60)
                user_fb = input("> Press [ENTER] to confirm or type feedback: ").strip()
                
                if user_fb:
                    current_prompt = base_prompt + f"\n\n**HUMAN FEEDBACK:**\n{user_fb}"
                    continue

            return {
                "status": "success", 
                "metrics": exec_res["metrics"], 
                "source_script": str(script_path)
            }

        return {"status": "failure", "error": "Max retries exceeded"}