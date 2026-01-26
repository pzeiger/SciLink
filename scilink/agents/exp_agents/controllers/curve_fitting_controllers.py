# controllers/curve_fitting_controllers.py

"""
Controllers for the curve fitting analysis pipeline.
"""

import json
import logging
import os
import numpy as np
from typing import Callable, Any

import base64
from datetime import datetime


class AnalyzeDataController:
    """Compute data statistics and create initial visualization."""

    def __init__(self, logger: logging.Logger, plot_fn: Callable):
        self.logger = logger
        self.plot_fn = plot_fn

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n🔍 --- Analyzing Data ---\n")

        try:
            data = state["curve_data"]

            if data.ndim == 1:
                x = np.arange(len(data))
                y = data
            elif data.shape[0] == 2:
                x, y = data[0], data[1]
            elif data.shape[1] == 2:
                x, y = data[:, 0], data[:, 1]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            state["data_statistics"] = {
                "n_points": len(x),
                "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
                "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
                "y_mean": float(np.nanmean(y)),
                "y_std": float(np.nanstd(y)),
                "has_nans": bool(np.any(np.isnan(data))),
            }

            plot_bytes = self.plot_fn(state["curve_data"], state.get("system_info", {}))
            state["original_plot_bytes"] = plot_bytes
            state["analysis_images"] = [{"label": "Raw Data", "data": plot_bytes}]

            self.logger.info(f"  Points: {state['data_statistics']['n_points']}")
            self.logger.info(f"  X: {state['data_statistics']['x_range']}")
            self.logger.info(f"  Y: {state['data_statistics']['y_range']}")

        except Exception as e:
            self.logger.error(f"❌ Data analysis failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Data analysis failed", "details": str(e)}

        return state


class PlanAnalysisController:
    """LLM examines data and plans the fitting approach."""

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        instructions: str,
        enable_human_feedback: bool = False,
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.instructions = instructions
        self.enable_human_feedback = enable_human_feedback

    def _display_plan(self, state: dict) -> None:
        """Display the proposed analysis plan."""
        print("\n" + "=" * 70)
        print("📋 PROPOSED ANALYSIS PLAN")
        print("=" * 70)
        
        if state.get("observations"):
            print(f"\n🔍 Observations:\n   {state['observations']}")
        
        print(f"\n📊 Approach:\n   {state.get('analysis_approach', 'N/A')}")
        print(f"\n📐 Physical Model:\n   {state.get('physical_model', 'N/A')}")
        print(f"\n🎯 Parameters to Extract:\n   {', '.join(state.get('parameters_to_extract', [])) or 'N/A'}")
        print(f"\n⚙️  Fitting Strategy:\n   {state.get('fitting_strategy', 'N/A')}")
        
        if state.get("literature_query"):
            print(f"\n📚 Literature Query:\n   {state['literature_query']}")
        
        print("\n" + "=" * 70)

    def _get_human_feedback(self, state: dict) -> dict:
        """Get human feedback on the proposed plan."""
        self._display_plan(state)
        
        feedback = input("\n🤔 Your feedback (or Enter to accept): ").strip()
        
        if feedback == "":
            print("✅ Plan accepted.")
            return state
        else:
            state["_refine_requested"] = True
            state["_refine_feedback"] = feedback
            return state

    def _plan_analysis(self, state: dict) -> dict:
        """Generate initial analysis plan using LLM."""
        prompt = [
            self.instructions,
            "\n## Data Plot",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ]

        if state.get("analysis_hints"):
            prompt.append(f"\n## User Guidance\n{state['analysis_hints']}")

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            raise ValueError(f"Failed to parse: {error}")

        state["observations"] = result.get("observations", "")
        state["analysis_approach"] = result.get("analysis_approach", "Curve fitting")
        state["physical_model"] = result.get("physical_model", "Appropriate model")
        state["parameters_to_extract"] = result.get("parameters_to_extract", [])
        state["fitting_strategy"] = result.get("fitting_strategy", "Standard fitting")
        state["literature_query"] = result.get("literature_query")

        return state

    def _refine_plan(self, state: dict, feedback: str) -> dict:
        """Refine the plan based on user feedback."""
        current_plan = (
            f"Observations: {state.get('observations', 'N/A')}\n"
            f"Approach: {state.get('analysis_approach', 'N/A')}\n"
            f"Physical Model: {state.get('physical_model', 'N/A')}\n"
            f"Parameters: {', '.join(state.get('parameters_to_extract', []))}\n"
            f"Strategy: {state.get('fitting_strategy', 'N/A')}"
        )

        prompt = [
            self.instructions,
            "\n## Data Plot",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
            f"\n## Current Plan\n{current_plan}",
            f"\n## User Feedback\nAdjust the plan based on this feedback: \"{feedback}\"",
        ]

        if state.get("analysis_hints"):
            prompt.append(f"\n## Original Guidance\n{state['analysis_hints']}")

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            self.logger.warning(f"Refinement failed: {error}. Keeping current plan.")
            return state

        state["observations"] = result.get("observations", state.get("observations", ""))
        state["analysis_approach"] = result.get("analysis_approach", state.get("analysis_approach"))
        state["physical_model"] = result.get("physical_model", state.get("physical_model"))
        state["parameters_to_extract"] = result.get("parameters_to_extract", state.get("parameters_to_extract", []))
        state["fitting_strategy"] = result.get("fitting_strategy", state.get("fitting_strategy"))
        state["literature_query"] = result.get("literature_query", state.get("literature_query"))

        return state

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n🧠 --- Planning Analysis ---\n")

        try:
            state = self._plan_analysis(state)

            self.logger.info(f"  Approach: {state['analysis_approach']}")
            self.logger.info(f"  Model: {state['physical_model']}")

            if self.enable_human_feedback:
                max_iterations = 5
                iteration = 0

                while iteration < max_iterations:
                    state = self._get_human_feedback(state)

                    if state.pop("_refine_requested", False):
                        feedback = state.pop("_refine_feedback", "")
                        self.logger.info(f"  Refining with feedback: {feedback}")
                        print("\n🔄 Refining plan...\n")
                        state = self._refine_plan(state, feedback)
                        iteration += 1
                    else:
                        break

                if iteration >= max_iterations:
                    self.logger.warning("  Max iterations reached.")
                    print("⚠️  Max refinements reached. Proceeding with current plan.")

        except Exception as e:
            self.logger.warning(f"⚠️ Planning failed: {e}, using fallback")
            state["observations"] = ""
            state["analysis_approach"] = "Fit the data with an appropriate model"
            state["physical_model"] = "To be determined"
            state["parameters_to_extract"] = []
            state["fitting_strategy"] = "Standard curve fitting"
            state["literature_query"] = None

        return state
    
class LiteratureSearchController:
    """Search literature if enabled and query provided."""

    def __init__(
        self,
        logger: logging.Logger,
        literature_agent: Any | None,
        output_dir: str,
    ):
        self.logger = logger
        self.literature_agent = literature_agent
        self.output_dir = output_dir

    def _save_results(self, query: str, report: str) -> dict:
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature")
            os.makedirs(lit_dir, exist_ok=True)

            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, "w") as f:
                f.write(query)
            saved_files["query_file"] = query_path

            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, "w") as f:
                f.write(report)
            saved_files["report_file"] = report_path
        except Exception as e:
            self.logger.warning(f"Failed to save literature: {e}")
        return saved_files

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        # Skip if no agent or no query
        if self.literature_agent is None:
            self.logger.info("\n📚 --- Skipping Literature (disabled) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        query = state.get("literature_query")
        if not query:
            self.logger.info("\n📚 --- Skipping Literature (no query needed) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        self.logger.info("\n📚 --- Searching Literature ---\n")
        self.logger.info(f"  Query: {query}")

        try:
            result = self.literature_agent.query_for_models(query)
            if result.get("status") == "success":
                state["literature_context"] = result["formatted_answer"]
                self.logger.info("  ✅ Success")
            else:
                state["literature_context"] = None
                self.logger.warning(f"  ⚠️ No results")

            state["literature_files"] = self._save_results(
                query, state["literature_context"] or f"No results: {result.get('message')}"
            )
        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}")
            state["literature_context"] = None
            state["literature_files"] = self._save_results(query, f"Error: {e}")

        return state


class ExecuteFittingController:
    """Generate and execute fitting script with self-correction."""

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        script_instructions: str,
        correction_instructions: str,
        quality_instructions: str,
        output_dir: str,
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.script_instructions = script_instructions
        self.correction_instructions = correction_instructions
        self.quality_instructions = quality_instructions
        self.output_dir = output_dir

    def _generate_script(self, state: dict) -> str:
        stats = state["data_statistics"]

        context_parts = []
        if state.get("literature_context"):
            context_parts.append(state["literature_context"])

        prompt = self.script_instructions.format(
            analysis_approach=state.get("analysis_approach", "Fit the data"),
            physical_model=state.get("physical_model", "Appropriate model"),
            parameters_to_extract=", ".join(state.get("parameters_to_extract", [])) or "relevant parameters",
            fitting_strategy=state.get("fitting_strategy", "Standard fitting"),
            context="\n".join(context_parts) or "Use your expertise.",
            data_path=state.get("processed_data_path") or state["data_path"],
            n_points=stats["n_points"],
            x_min=stats["x_range"][0],
            x_max=stats["x_range"][1],
            y_min=stats["y_range"][0],
            y_max=stats["y_range"][1],
        )

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Script generation failed: {error or 'no script'}")

        return result["script"]

    def _correct_script(self, state: dict, script: str, error_msg: str) -> str:
        prompt = self.correction_instructions.format(
            analysis_approach=state.get("analysis_approach", ""),
            physical_model=state.get("physical_model", ""),
            failed_script=script,
            error_message=error_msg,
        )

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Correction failed: {error or 'no script'}")

        if "diagnosis" in result:
            self.logger.info(f"  Diagnosis: {result['diagnosis']}")

        return result["script"]

    def _assess_quality(self, state: dict, plot_bytes: bytes, metrics: dict) -> dict:
        prompt = [
            self.quality_instructions.format(
                analysis_approach=state.get("analysis_approach", ""),
                physical_model=state.get("physical_model", ""),
                metrics=json.dumps(metrics, indent=2),
            ),
            "\n## Original Data",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Fit Result",
            {"mime_type": "image/png", "data": plot_bytes},
        ]

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, _ = self._parse(response)

        if not result:
            return {"is_acceptable": True, "quality_score": 0.5}

        is_ok = result.get("is_acceptable", True)
        if isinstance(is_ok, str):
            is_ok = is_ok.lower() == "true"
        result["is_acceptable"] = bool(is_ok)

        return result

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n⚙️ --- Executing Fitting ---\n")

        script = None
        last_error = ""
        exec_result = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            self.logger.info(f"  Attempt {attempt}/{self.MAX_ATTEMPTS}")

            try:
                if attempt == 1:
                    script = self._generate_script(state)
                else:
                    script = self._correct_script(state, script, last_error)

                exec_result = self.executor.execute_script(script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    self.logger.info("  ✅ Script executed")
                    break
                else:
                    last_error = exec_result.get("message", "Unknown error")
                    self.logger.warning(f"  ❌ Failed: {last_error[:150]}")

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"  ❌ Error: {e}")
        else:
            state["error_dict"] = {"error": "Script generation failed", "details": last_error}
            return state

        # Parse results
        fit_results = {}
        for line in (exec_result.get("stdout") or "").splitlines():
            if line.startswith("FIT_RESULTS_JSON:"):
                try:
                    fit_results = json.loads(line.replace("FIT_RESULTS_JSON:", "").strip())
                except json.JSONDecodeError as e:
                    self.logger.warning(f"  Could not parse results: {e}")
                break

        # Load visualization
        viz_path = os.path.join(self.output_dir, "fit_visualization.png")
        if not os.path.exists(viz_path):
            state["error_dict"] = {"error": "No fit_visualization.png generated"}
            return state

        with open(viz_path, "rb") as f:
            plot_bytes = f.read()

        # Assess quality
        quality = self._assess_quality(state, plot_bytes, fit_results.get("fit_quality", {}))
        self.logger.info(f"  Quality: {quality.get('quality_score', 'N/A')}, OK: {quality.get('is_acceptable')}")

        # Store results
        state["final_script"] = script
        state["final_plot_bytes"] = plot_bytes
        state["fit_results"] = fit_results
        state["quality_assessment"] = quality
        state["analysis_images"].append({
            "label": fit_results.get("model_type", "Fit"),
            "data": plot_bytes,
        })

        state["result_json"] = {
            "model_type": fit_results.get("model_type"),
            "fitting_parameters": fit_results.get("parameters", {}),
            "fit_quality": fit_results.get("fit_quality", {}),
            "summary": fit_results.get("summary"),
            "literature_files": state.get("literature_files"),
        }

        return state


class BuildInterpretationPromptController:
    """Assemble prompt for final interpretation."""

    def __init__(self, logger: logging.Logger, instructions: str):
        self.logger = logger
        self.instructions = instructions

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n📝 --- Building Interpretation Prompt ---\n")

        fit_results = state.get("fit_results", {})

        formatted = self.instructions.format(
            model_type=fit_results.get("model_type", "Curve fit"),
            summary=fit_results.get("summary", "Fitting complete"),
        )

        state["instruction_prompt"] = formatted
        state["final_prompt_parts"] = [
            formatted,
            "\n## Original Data",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Fit Result",
            {"mime_type": "image/png", "data": state["final_plot_bytes"]},
            "\n## Parameters\n" + json.dumps(fit_results.get("parameters", {}), indent=2),
            "\n## Fit Quality\n" + json.dumps(fit_results.get("fit_quality", {}), indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ]

        if state.get("literature_context"):
            state["final_prompt_parts"].extend(["\n## Literature", state["literature_context"]])

        return state


class RunCurvePreprocessingController:
    """Optional data preprocessing."""

    def __init__(self, logger: logging.Logger, preprocessor: Any, output_dir: str):
        self.logger = logger
        self.preprocessor = preprocessor
        self.output_dir = output_dir

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if self.preprocessor is None:
            return state

        self.logger.info("\n🛠️ --- Preprocessing ---\n")

        try:
            processed_data, data_quality = self.preprocessor.run_preprocessing(
                state["curve_data"], state.get("system_info", {})
            )

            state["curve_data"] = processed_data
            state["data_quality"] = data_quality

            pid = os.getpid()
            processed_path = os.path.join(self.output_dir, f"temp_processed_{pid}.npy")
            np.save(processed_path, processed_data)
            state["processed_data_path"] = processed_path

            self.logger.info(f"  ✅ Saved to {processed_path}")

        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Preprocessing failed", "details": str(e)}

        return state
    

class GenerateCurveFittingReportController:
    """
    [🛠️ Tool Step]
    Generates a human-readable HTML report for curve fitting analysis.
    """
    
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = output_dir

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Convert bytes to base64 string for HTML embedding."""
        return base64.b64encode(image_bytes).decode('utf-8')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
            
        self.logger.info("\n📄 --- Generating HTML Report ---\n")
        
        result_json = state.get("result_json", {})
        fit_results = state.get("fit_results", {})
        
        # Extract data
        detailed_analysis = result_json.get("detailed_analysis", "No analysis provided.")
        scientific_claims = result_json.get("scientific_claims", [])
        system_info = state.get("system_info", {})
        model_type = fit_results.get("model_type", result_json.get("model_type", "N/A"))
        parameters = fit_results.get("parameters", result_json.get("fitting_parameters", {}))
        fit_quality = fit_results.get("fit_quality", result_json.get("fit_quality", {}))
        caveats = result_json.get("caveats", "")
        
        # Get images
        original_plot = state.get("original_plot_bytes")
        fit_plot = state.get("final_plot_bytes")
        
        # Setup output
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CurveFitting_Report_{file_timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        # Format sections
        params_html = self._format_parameters(parameters)
        quality_html = self._format_fit_quality(fit_quality)

        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curve Fitting Analysis Report</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; 
            color: #333; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            background-color: #f4f4f9; 
        }}
        .container {{ 
            background-color: #fff; 
            padding: 40px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 2px solid #3498db; 
            padding-bottom: 10px; 
        }}
        h2 {{ 
            color: #2980b9; 
            margin-top: 30px; 
        }}
        h3 {{ 
            color: #16a085;
            margin-top: 20px;
        }}
        .metadata-box {{ 
            background-color: #ecf0f1; 
            padding: 15px; 
            border-radius: 5px; 
            border-left: 5px solid #3498db; 
            margin-bottom: 20px; 
        }}
        .model-box {{
            background-color: #e8f4fc;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #2980b9;
            margin-bottom: 15px;
        }}
        .analysis-text {{ 
            white-space: pre-wrap; 
            background-color: #fafafa; 
            padding: 20px; 
            border-radius: 5px; 
            border: 1px solid #eee;
            margin-top: 15px;
        }}
        .claim-card {{ 
            background-color: #e8f6f3; 
            border-left: 5px solid #1abc9c; 
            padding: 15px; 
            margin-bottom: 15px; 
            border-radius: 0 5px 5px 0;
        }}
        .claim-title {{ 
            font-weight: bold; 
            font-size: 1.1em; 
            color: #0e6655; 
        }}
        .image-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); 
            gap: 25px; 
            margin-top: 20px; 
        }}
        .image-card {{ 
            background: white; 
            border: 1px solid #ddd; 
            padding: 15px; 
            border-radius: 5px; 
            text-align: center; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.05); 
        }}
        .image-card img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 3px; 
        }}
        .image-label {{ 
            margin-top: 12px; 
            font-weight: bold; 
            color: #444; 
            font-size: 1em; 
            border-top: 1px solid #eee; 
            padding-top: 10px; 
        }}
        .params-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .params-table th, .params-table td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .params-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        .params-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .quality-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            margin-right: 10px;
        }}
        .quality-good {{
            background-color: #d4edda;
            color: #155724;
        }}
        .quality-ok {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .quality-poor {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .caveats {{
            background-color: #fff8e6;
            border-left: 5px solid #f0ad4e;
            padding: 15px;
            margin-top: 20px;
            border-radius: 0 5px 5px 0;
        }}
        .footer {{ 
            margin-top: 50px; 
            text-align: center; 
            color: #7f8c8d; 
            font-size: 0.8em; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Curve Fitting Analysis Report</h1>
        
        <!-- 0. Metadata -->
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Data Source:</strong> {state.get('data_path', 'N/A')}</p>
            <p><strong>Sample Info:</strong> {self._format_system_info(system_info)}</p>
        </div>

        <!-- 1. Scientific Analysis -->
        <h2>1. Scientific Analysis</h2>
        
        <h3>Fitting Model</h3>
        <div class="model-box">
            {model_type}
        </div>
        
        <h3>Fit Quality</h3>
        {quality_html}
        
        <h3>Interpretation</h3>
        <div class="analysis-text">{detailed_analysis}</div>

        <!-- 2. Visualizations -->
        <h2>2. Visualizations</h2>
        <div class="image-grid">
"""

        if original_plot:
            b64_original = self._image_to_base64(original_plot)
            html_content += f"""
            <div class="image-card">
                <img src="data:image/png;base64,{b64_original}" alt="Original Data">
                <div class="image-label">Original Data</div>
            </div>
"""

        if fit_plot:
            b64_fit = self._image_to_base64(fit_plot)
            html_content += f"""
            <div class="image-card">
                <img src="data:image/png;base64,{b64_fit}" alt="Fit Visualization">
                <div class="image-label">Fit Result with Residuals</div>
            </div>
"""

        html_content += f"""
        </div>

        <!-- 3. Fitted Parameters -->
        <h2>3. Fitted Parameters</h2>
        {params_html}

        <!-- 4. Scientific Claims -->
        <h2>4. Scientific Claims</h2>
"""

        if not scientific_claims:
            html_content += "<p>No specific claims generated.</p>"
        else:
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                html_content += f"""
        <div class="claim-card">
            <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
            <p><strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
            <p><strong>Research Question:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
            <p><strong>Keywords:</strong> {keywords_str}</p>
        </div>
"""

        # 5. Caveats (if any)
        if caveats:
            html_content += f"""
        <!-- 5. Caveats -->
        <h2>5. Caveats & Limitations</h2>
        <div class="caveats">
            {caveats}
        </div>
"""

        html_content += """
        <div class="footer">
            Generated by SciLink Curve Fitting Analysis Agent
        </div>
    </div>
</body>
</html>
"""

        # Write file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"  ✅ Report saved: {filepath}")
            state["report_path"] = filepath
        except Exception as e:
            self.logger.error(f"  ❌ Failed to write report: {e}")

        return state

    def _format_system_info(self, system_info: dict) -> str:
        """Format system info for display."""
        if not system_info:
            return "N/A"
        
        parts = []
        for key, value in system_info.items():
            if value:
                parts.append(f"{key}: {value}")
        
        return ", ".join(parts) if parts else "N/A"

    def _format_parameters(self, parameters: dict) -> str:
        """Format fitted parameters as HTML table."""
        if not parameters:
            return "<p>No parameters extracted.</p>"

        html = """
        <table class="params-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Uncertainty</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for component, params in parameters.items():
            if isinstance(params, dict):
                first_row = True
                for param_name, value in params.items():
                    if param_name.endswith("_err"):
                        continue
                    
                    err_key = f"{param_name}_err"
                    err_value = params.get(err_key, "—")
                    if isinstance(err_value, (int, float)):
                        err_value = f"± {err_value:.4g}"
                    
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4g}"
                    else:
                        value_str = str(value)
                    
                    component_display = component if first_row else ""
                    html += f"""
                <tr>
                    <td><strong>{component_display}</strong></td>
                    <td>{param_name}</td>
                    <td>{value_str}</td>
                    <td>{err_value}</td>
                </tr>
"""
                    first_row = False
            else:
                html += f"""
                <tr>
                    <td><strong>{component}</strong></td>
                    <td>—</td>
                    <td>{parameters[component]}</td>
                    <td>—</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""
        return html

    def _format_fit_quality(self, fit_quality: dict) -> str:
        """Format fit quality metrics with visual badges."""
        if not fit_quality:
            return "<p>No quality metrics available.</p>"

        r_squared = fit_quality.get("r_squared", fit_quality.get("r2"))
        rmse = fit_quality.get("rmse")
        chi_squared = fit_quality.get("chi_squared_reduced", fit_quality.get("reduced_chi_squared"))

        html = "<div>"

        if r_squared is not None:
            if r_squared >= 0.99:
                badge_class = "quality-good"
                label = "Excellent"
            elif r_squared >= 0.95:
                badge_class = "quality-ok"
                label = "Good"
            else:
                badge_class = "quality-poor"
                label = "Poor"
            
            html += f'<span class="quality-badge {badge_class}">{label}</span>'
            html += f"<strong>R² = {r_squared:.4f}</strong>"

        if rmse is not None:
            html += f" &nbsp;|&nbsp; <strong>RMSE = {rmse:.4g}</strong>"

        if chi_squared is not None:
            html += f" &nbsp;|&nbsp; <strong>χ²/DOF = {chi_squared:.3f}</strong>"

        html += "</div>"
        return html
