# controllers/curve_fitting_controllers.py

"""
Curve Fitting Controllers - Complete Module

This module contains:
1. Original controllers for single-spectrum analysis steps
2. Unified controllers that handle both single spectrum (n=1) and series (n>1) analysis

Key principle for series analysis: Single spectrum = Series of 1
"""

import subprocess
import json
import logging
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Any, Dict, List
import numpy as np


# ============================================================================
# ORIGINAL CONTROLLERS (for single-spectrum pipeline steps)
# ============================================================================

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
        
        # Skip for series (handled by UnifiedCurveReportController)
        if not state.get("is_single_spectrum", True):
            return state
            
        self.logger.info("\n📄 --- Generating HTML Report ---\n")
        
        result_json = state.get("result_json", {})
        fit_results = state.get("fit_results", {})
        synthesis_result = state.get("synthesis_result", {})
        
        # Extract data - prefer synthesis_result, fall back to result_json
        detailed_analysis = synthesis_result.get("detailed_analysis") or result_json.get("detailed_analysis", "No analysis provided.")
        scientific_claims = synthesis_result.get("scientific_claims") or result_json.get("scientific_claims", [])
        system_info = state.get("system_info", {})
        model_type = fit_results.get("model_type", result_json.get("model_type", "N/A"))
        parameters = fit_results.get("parameters", result_json.get("fitting_parameters", {}))
        fit_quality = fit_results.get("fit_quality", result_json.get("fit_quality", {}))
        caveats = synthesis_result.get("caveats") or result_json.get("caveats", "")
        
        # Get images
        original_plot = state.get("original_plot_bytes")
        fit_plot = state.get("final_plot_bytes")
        
        # Setup output
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CurveFitting_Report_{file_timestamp}.html"
        filepath = output_dir / filename

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
            state["report_path"] = str(filepath)
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


# ============================================================================
# UNIFIED CONTROLLERS (for series analysis support)
# ============================================================================

class HumanFeedbackRefinementController:
    """
    [👤 Human Step + 🧠 LLM Step]
    Facilitates human-in-the-loop parameter refinement for the first spectrum.
    
    Works identically for single spectra and series:
    - Single spectrum: Refine fitting, then process that one spectrum
    - Series: Refine fitting on first spectrum, then apply to all
    """
    
    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        instructions: str,
        output_dir: str,
        enable_human_feedback: bool = False,
        max_iterations: int = 5
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.instructions = instructions
        self.output_dir = Path(output_dir)
        self.enable_human_feedback = enable_human_feedback
        self.max_iterations = max_iterations

    def _display_plan(self, state: dict) -> None:
        """Display the proposed analysis plan."""
        is_single = state.get("is_single_spectrum", True)
        num_spectra = state.get("num_spectra", 1)
        
        print("\n" + "=" * 70)
        mode_str = "SINGLE SPECTRUM" if is_single else f"SERIES ({num_spectra} spectra)"
        print(f"📋 PROPOSED FITTING PLAN - {mode_str}")
        print("=" * 70)
        
        if state.get("observations"):
            print(f"\n🔍 Observations:\n   {state['observations']}")
        
        print(f"\n📊 Approach:\n   {state.get('analysis_approach', 'N/A')}")
        print(f"\n📐 Physical Model:\n   {state.get('physical_model', 'N/A')}")
        print(f"\n🎯 Parameters to Extract:\n   {', '.join(state.get('parameters_to_extract', [])) or 'N/A'}")
        print(f"\n⚙️  Fitting Strategy:\n   {state.get('fitting_strategy', 'N/A')}")
        
        if state.get("literature_query"):
            print(f"\n📚 Literature Query:\n   {state['literature_query']}")
        
        if not is_single:
            print(f"\n📦 **Note:** This fitting model will be LOCKED and applied to all {num_spectra} spectra.")
        
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
        
        # Add series context if applicable
        if not state.get("is_single_spectrum", True):
            prompt.append(f"\n## Series Context\nThis is the first spectrum in a series of {state.get('num_spectra', 1)}. "
                         "The fitting model you choose will be applied to ALL spectra in the series.")

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
        """Execute the human feedback refinement loop."""
        if state.get("error_dict"):
            return state

        is_single = state.get("is_single_spectrum", True)
        mode_str = "SINGLE SPECTRUM" if is_single else "SERIES"
        self.logger.info(f"\n🧠 --- Planning Analysis ({mode_str}) ---\n")

        try:
            state = self._plan_analysis(state)

            self.logger.info(f"  Approach: {state['analysis_approach']}")
            self.logger.info(f"  Model: {state['physical_model']}")

            if self.enable_human_feedback:
                iteration = 0

                while iteration < self.max_iterations:
                    state = self._get_human_feedback(state)

                    if state.pop("_refine_requested", False):
                        feedback = state.pop("_refine_feedback", "")
                        self.logger.info(f"  Refining with feedback: {feedback}")
                        print("\n🔄 Refining plan...\n")
                        state = self._refine_plan(state, feedback)
                        iteration += 1
                    else:
                        break

                if iteration >= self.max_iterations:
                    self.logger.warning("  Max iterations reached.")
                    print("⚠️  Max refinements reached. Proceeding with current plan.")

            # Store the locked fitting configuration for series processing
            state["locked_fitting_config"] = {
                "analysis_approach": state.get("analysis_approach"),
                "physical_model": state.get("physical_model"),
                "parameters_to_extract": state.get("parameters_to_extract", []),
                "fitting_strategy": state.get("fitting_strategy"),
            }
            
            self.logger.info("  ✅ Fitting configuration locked for series processing.")

        except Exception as e:
            self.logger.warning(f"⚠️ Planning failed: {e}, using fallback")
            state["observations"] = ""
            state["analysis_approach"] = "Fit the data with an appropriate model"
            state["physical_model"] = "To be determined"
            state["parameters_to_extract"] = []
            state["fitting_strategy"] = "Standard curve fitting"
            state["literature_query"] = None
            state["locked_fitting_config"] = None

        return state


class UnifiedSeriesProcessingController:
    """
    [🛠️ Tool Step]
    Processes ALL spectra using the locked fitting model.
    
    Key features:
    - Uses the same fitting model/script for all spectra
    - Works identically for n=1 or n>1
    - Collects parameters across series for trend analysis
    """
    
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
        plot_fn: Callable
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
        self.output_dir = Path(output_dir)
        self.plot_fn = plot_fn

    def _generate_fitting_script(self, state: dict, data_path: str, stats: dict) -> str:
        """Generate fitting script for a spectrum."""
        config = state.get("locked_fitting_config", {})
        
        context_parts = []
        if state.get("literature_context"):
            context_parts.append(state["literature_context"])

        prompt = self.script_instructions.format(
            analysis_approach=config.get("analysis_approach", "Fit the data"),
            physical_model=config.get("physical_model", "Appropriate model"),
            parameters_to_extract=", ".join(config.get("parameters_to_extract", [])) or "relevant parameters",
            fitting_strategy=config.get("fitting_strategy", "Standard fitting"),
            context="\n".join(context_parts) or "Use your expertise.",
            data_path=data_path,
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
        """Correct a failed script using LLM."""
        config = state.get("locked_fitting_config", {})
        
        prompt = self.correction_instructions.format(
            analysis_approach=config.get("analysis_approach", ""),
            physical_model=config.get("physical_model", ""),
            failed_script=script,
            error_message=error_msg,
        )

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Correction failed: {error or 'no script'}")

        if "diagnosis" in result:
            self.logger.info(f"    Diagnosis: {result['diagnosis']}")

        return result["script"]

    def _adapt_script_for_spectrum(self, base_script: str, data_path: str, output_prefix: str) -> str:
        """Adapt the base fitting script for a specific spectrum."""
        import re
        
        adapted = base_script
        
        # Replace output filename patterns
        adapted = adapted.replace('fit_visualization.png', f'{output_prefix}_fit.png')
        adapted = re.sub(r'spectrum_\d{4}_fit\.png', f'{output_prefix}_fit.png', adapted)
        adapted = re.sub(r'spectrum_\d{4}_T\d+K_fit\.png', f'{output_prefix}_fit.png', adapted)
        
        # Find and replace the data path in np.load() calls
        # Match patterns like: np.load("...temp_spectrum_0.npy") or np.load('...temp_spectrum_0.npy')
        adapted = re.sub(
            r'np\.load\s*\(\s*["\'].*?temp_spectrum_\d+\.npy["\']\s*\)',
            f'np.load("{data_path}")',
            adapted
        )
        
        # Also handle np.load(variable) where variable was assigned the path
        # Replace the path in any string that looks like a temp_spectrum path
        adapted = re.sub(
            r'(["\']).*?temp_spectrum_\d+\.npy\1',
            f'"{data_path}"',
            adapted
        )
        
        # Replace DATA_PATH variable assignments (preserve uppercase)
        adapted = re.sub(
            r'DATA_PATH\s*=\s*["\'].*?["\']',
            f'DATA_PATH = "{data_path}"',
            adapted
        )
        
        # Replace data_path variable assignments (preserve lowercase)
        adapted = re.sub(
            r'data_path\s*=\s*["\'].*?["\']',
            f'data_path = "{data_path}"',
            adapted
        )
        
        # Also handle file_path or filepath variants
        adapted = re.sub(
            r'file_path\s*=\s*["\'].*?temp_spectrum_\d+\.npy["\']',
            f'file_path = "{data_path}"',
            adapted
        )
        adapted = re.sub(
            r'filepath\s*=\s*["\'].*?temp_spectrum_\d+\.npy["\']',
            f'filepath = "{data_path}"',
            adapted
        )
        
        return adapted

    def _compute_statistics(self, curve_data: np.ndarray) -> dict:
        """Compute statistics for a spectrum."""
        if curve_data.ndim == 1:
            x = np.arange(len(curve_data))
            y = curve_data
        elif curve_data.shape[0] == 2:
            x, y = curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            x, y = curve_data[:, 0], curve_data[:, 1]
        else:
            raise ValueError(f"Unexpected data shape: {curve_data.shape}")

        return {
            "n_points": len(x),
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
            "has_nans": bool(np.any(np.isnan(curve_data))),
        }

    def _load_curve_data(self, data_path: str) -> np.ndarray:
        """Load curve data from file."""
        # Try different import paths for flexibility
        try:
            from ...tools.curve_fitting_tools import load_curve_data
            return load_curve_data(data_path)
        except ImportError:
            pass
        
        try:
            from ....tools.curve_fitting_tools import load_curve_data
            return load_curve_data(data_path)
        except ImportError:
            pass
        
        # Fallback: basic loading
        if data_path.endswith('.npy'):
            return np.load(data_path)
        elif data_path.endswith('.csv'):
            return np.loadtxt(data_path, delimiter=',')
        else:
            return np.loadtxt(data_path)

    def _fit_single_spectrum(
        self, 
        state: dict, 
        curve_data: np.ndarray, 
        data_path: str,
        spectrum_name: str,
        spectrum_idx: int,
        base_script: Optional[str] = None
    ) -> dict:
        """Fit a single spectrum and return results."""
        stats = self._compute_statistics(curve_data)
        
        # Create temp file for this spectrum's data
        temp_data_path = self.output_dir / f"temp_spectrum_{spectrum_idx}.npy"
        np.save(temp_data_path, curve_data)
        
        output_prefix = f"spectrum_{spectrum_idx:04d}"
        
        script = None
        last_error = ""
        exec_result = None
        
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                if base_script is not None and attempt == 1:
                    script = self._adapt_script_for_spectrum(base_script, str(temp_data_path), output_prefix)
                elif attempt == 1:
                    script = self._generate_fitting_script(state, str(temp_data_path), stats)
                else:
                    script = self._correct_script(state, script, last_error)

                exec_result = self.executor.execute_script(script, working_dir=str(self.output_dir))

                if exec_result.get("status") == "success":
                    break
                else:
                    last_error = exec_result.get("message", "Unknown error")
                    self.logger.warning(f"    ⚠️ Attempt {attempt} failed: {last_error[:100]}")

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"    ❌ Attempt {attempt} error: {e}")
        
        # Clean up temp file
        if temp_data_path.exists():
            try:
                os.remove(temp_data_path)
            except:
                pass
        
        if exec_result is None or exec_result.get("status") != "success":
            return {
                "index": spectrum_idx,
                "name": spectrum_name,
                "success": False,
                "error": last_error,
                "parameters": {},
                "fit_quality": {},
                "script": script
            }
        
        # Parse results from stdout
        fit_results = {}
        for line in (exec_result.get("stdout") or "").splitlines():
            if line.startswith("FIT_RESULTS_JSON:"):
                try:
                    fit_results = json.loads(line.replace("FIT_RESULTS_JSON:", "").strip())
                except json.JSONDecodeError:
                    pass
                break
        
        # Check for visualization
        viz_path = self.output_dir / f"{output_prefix}_fit.png"
        if not viz_path.exists():
            viz_path = self.output_dir / "fit_visualization.png"
        
        viz_bytes = None
        if viz_path.exists():
            with open(viz_path, "rb") as f:
                viz_bytes = f.read()
            # Rename to spectrum-specific name
            final_viz_path = self.output_dir / f"{output_prefix}_fit.png"
            if viz_path != final_viz_path:
                viz_path.rename(final_viz_path)
            viz_path = final_viz_path
        
        return {
            "index": spectrum_idx,
            "name": spectrum_name,
            "data_path": data_path,
            "success": True,
            "error": None,
            "model_type": fit_results.get("model_type"),
            "parameters": fit_results.get("parameters", {}),
            "fit_quality": fit_results.get("fit_quality", {}),
            "summary": fit_results.get("summary"),
            "visualization_path": str(viz_path) if viz_path.exists() else None,
            "visualization_bytes": viz_bytes,
            "statistics": stats,
            "script": script
        }

    def execute(self, state: dict) -> dict:
        """Process all spectra using the locked fitting model."""
        if state.get("error_dict"):
            return state

        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        mode_str = "SINGLE SPECTRUM" if is_single else f"SERIES ({num_spectra} spectra)"
        self.logger.info(f"\n\n⚙️ --- FITTING: {mode_str} --- ⚙️\n")
        
        spectrum_paths = state.get("spectrum_paths", [])
        spectrum_stack = state.get("spectrum_stack")
        
        series_results = []
        base_script = None
        
        for idx in range(num_spectra):
            if spectrum_stack is not None:
                curve_data = spectrum_stack[idx]
                spectrum_name = f"spectrum_{idx:04d}"
                data_path = f"stack_index_{idx}"
            else:
                data_path = spectrum_paths[idx]
                spectrum_name = Path(data_path).stem
                curve_data = self._load_curve_data(data_path)
            
            if is_single:
                self.logger.info(f"   Fitting: {spectrum_name}")
            else:
                self.logger.info(f"   [{idx + 1}/{num_spectra}] Fitting: {spectrum_name}")
            
            result = self._fit_single_spectrum(
                state=state,
                curve_data=curve_data,
                data_path=data_path,
                spectrum_name=spectrum_name,
                spectrum_idx=idx,
                base_script=base_script if idx > 0 else None
            )
            
            if idx == 0 and result["success"] and result.get("script"):
                base_script = result["script"]
                state["base_fitting_script"] = base_script
                self.logger.info("   📝 Base fitting script locked for series.")
            
            series_results.append(result)
            
            if result["success"]:
                self.logger.info(f"      ✅ {result.get('model_type', 'Fit')} - R²: {result.get('fit_quality', {}).get('r_squared', 'N/A')}")
            else:
                self.logger.error(f"      ❌ Failed: {result.get('error', 'Unknown')[:50]}")
        
        state["series_results"] = series_results
        
        # For single spectrum, populate legacy fields for backward compatibility
        if is_single and series_results and series_results[0]["success"]:
            first_result = series_results[0]
            state["fit_results"] = {
                "model_type": first_result.get("model_type"),
                "parameters": first_result.get("parameters", {}),
                "fit_quality": first_result.get("fit_quality", {}),
                "summary": first_result.get("summary"),
            }
            state["final_script"] = first_result.get("script")
            state["final_plot_bytes"] = first_result.get("visualization_bytes")
            
            if first_result.get("visualization_bytes"):
                state["analysis_images"].append({
                    "label": first_result.get("model_type", "Fit"),
                    "data": first_result["visualization_bytes"],
                })
        
        successful = sum(1 for r in series_results if r["success"])
        self.logger.info(f"\n✅ Fitting complete: {successful}/{num_spectra} successful.")
        
        # Save series results JSON
        results_path = self.output_dir / "series_fit_results.json"
        with open(results_path, 'w') as f:
            serializable_results = []
            for r in series_results:
                r_copy = {k: v for k, v in r.items() if k != "visualization_bytes"}
                serializable_results.append(r_copy)
            
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_spectra": num_spectra,
                "successful": successful,
                "is_single_spectrum": is_single,
                "locked_config": state.get("locked_fitting_config"),
                "results": serializable_results
            }, f, indent=2, default=str)
        
        state["series_results_path"] = str(results_path)
        
        return state


class ConditionalTrendAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates and executes custom Python script for trend analysis.
    
    CONDITIONAL:
    - For n>=2: Generates trend analysis script
    - For n=1: Skipped (no trends to analyze)
    """
    
    TREND_ANALYSIS_INSTRUCTIONS = '''You are analyzing a series of fitted spectra to identify trends.

**SERIES SUMMARY:**
{series_summary}

**SERIES METADATA:**
{series_metadata}

**JSON FILE STRUCTURE:**
The file 'series_fit_results.json' has this exact structure:
```json
{{
    "timestamp": "2026-01-27T...",
    "total_spectra": N,
    "successful": M,
    "is_single_spectrum": false,
    "locked_config": {{...}},
    "results": [
        {{
            "index": 0,
            "name": "spectrum_0000_T300K",
            "success": true,
            "model_type": "Sum of Lorentzian peaks",
            "parameters": {{
                "peak_0_center": 720.5,
                "peak_0_amplitude": 0.25,
                "peak_0_sigma": 12.3,
                "peak_1_center": 1100.2,
                "peak_1_amplitude": 0.55,
                ...
            }},
            "fit_quality": {{"r_squared": 0.95, "rmse": 0.02}},
            ...
        }},
        ...more results...
    ]
}}
```

**CRITICAL: USE THIS EXACT DATA EXTRACTION PATTERN:**
```python
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
with open('series_fit_results.json', 'r') as f:
    data = json.load(f)

results = data['results']  # This is a LIST of dictionaries

# Extract temperatures from spectrum names (e.g., "spectrum_0000_T300K" -> 300)
temperatures = []
for r in results:
    name = r['name']
    # Extract temperature from name like "spectrum_0000_T300K"
    if '_T' in name and 'K' in name:
        temp_str = name.split('_T')[-1].replace('K', '')
        temperatures.append(float(temp_str))
    else:
        temperatures.append(r['index'])  # fallback to index

temperatures = np.array(temperatures)

# Extract parameters - collect all parameter names first
all_params = set()
for r in results:
    if r['success'] and 'parameters' in r:
        all_params.update(r['parameters'].keys())

# Build parameter arrays
param_data = {{param: [] for param in all_params}}
for r in results:
    params = r.get('parameters', {{}})
    for param in all_params:
        param_data[param].append(params.get(param, np.nan))

# Convert to numpy arrays
for param in param_data:
    param_data[param] = np.array(param_data[param])

# Now plot - MAKE SURE TO ACTUALLY PLOT DATA
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... use axes[i,j].plot(temperatures, param_data['param_name'], 'o-') ...
```

Generate a complete Python script that:
1. Loads the fit results using the EXACT pattern shown above
2. Extracts ALL parameters across the series
3. Creates publication-quality visualizations with ACTUAL DATA POINTS
4. For each subplot, call ax.plot() or ax.scatter() with real data
5. Identifies and quantifies trends (linear fits with scipy.stats.linregress)
6. Saves visualizations as PNG files
7. Prints a summary of findings

IMPORTANT REQUIREMENTS:
- Use the exact data loading pattern shown above
- Always verify data is not empty before plotting: `if len(data) > 0:`
- Actually call plot/scatter functions with real arrays, not empty lists
- Include error handling for missing parameters
- Print debug info: `print(f"Found {{len(results)}} results, {{len(temperatures)}} temperatures")`

Output files to save:
- 'parameter_trends.png' - Main parameter evolution plots
- 'correlation_matrix.png' - Parameter correlations (optional)

Return JSON with:
{{
    "analysis_approach": "description of trend analysis approach",
    "key_metrics": ["list", "of", "metrics", "tracked"],
    "expected_outputs": ["parameter_trends.png"],
    "script": "full python script"
}}
'''

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        output_dir: str,
        max_corrections: int = 3
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.max_corrections = max_corrections

    def _generate_trend_script(self, state: dict) -> Optional[Dict]:
        """Generate custom trend analysis script using LLM."""
        series_results = state.get("series_results", [])
        series_metadata = state.get("series_metadata", {})
        
        param_summary = []
        for r in series_results:
            if r["success"]:
                param_summary.append({
                    "index": r["index"],
                    "name": r["name"],
                    "model_type": r.get("model_type"),
                    "parameters": r.get("parameters", {}),
                    "fit_quality": r.get("fit_quality", {}),
                })
        
        prompt = self.TREND_ANALYSIS_INSTRUCTIONS.format(
            series_summary=json.dumps(param_summary, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2)
        )
        
        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse(response)
            
            if error_dict and not (result_json and 'script' in result_json):
                return None
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error generating trend script: {e}")
            return None

    def _execute_script(self, script: str) -> tuple:
        """Execute the generated Python script."""
        script_path = self.output_dir / "trend_analysis.py"
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        result = self.executor.execute_script(script, working_dir=str(self.output_dir))
        
        success = result.get("status") == "success"
        return success, result.get("stdout", ""), result.get("message", "")

    def _correct_script(self, original_script: str, error_message: str, attempt: int) -> Optional[str]:
        """Use LLM to correct a failed script."""
        self.logger.info(f"   🔧 Attempting script correction (attempt {attempt})...")
        
        if len(error_message) > 1000:
            error_message = error_message[:500] + "\n...[truncated]...\n" + error_message[-500:]
        
        prompt = f"""Fix this Python script that failed:

**SCRIPT:**
```python
{original_script}
```

**ERROR:**
```
{error_message}
```

Return JSON with: {{"diagnosis": "...", "script": "corrected script"}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, _ = self._parse(response)
            
            if result_json:
                self.logger.info(f"   📋 Diagnosis: {result_json.get('diagnosis', 'N/A')}")
                return result_json.get("script")
            return None
            
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None

    def execute(self, state: dict) -> dict:
        """Execute trend analysis - conditional on series size."""
        if state.get("error_dict"):
            return state
        
        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        if is_single or num_spectra < 2:
            self.logger.info("\n📊 Trend analysis skipped (single spectrum mode).\n")
            state["trend_analysis_results"] = {
                "success": True,
                "skipped": True,
                "reason": "Single spectrum - no trend analysis applicable"
            }
            return state
        
        self.logger.info("\n\n📈 --- TREND ANALYSIS --- 📈\n")
        
        script_result = self._generate_trend_script(state)
        
        if not script_result or "script" not in script_result:
            self.logger.error("Failed to generate trend analysis script.")
            state["trend_analysis_results"] = {"success": False, "error": "Script generation failed"}
            return state
        
        approach = script_result.get('analysis_approach', 'unknown')
        metrics = script_result.get('key_metrics', [])
        self.logger.info(f"   📊 Approach: {approach}")
        self.logger.info(f"   📈 Metrics: {metrics}")
        
        script = script_result["script"]
        success, stdout, stderr = False, "", ""
        
        for attempt in range(self.max_corrections + 1):
            if attempt > 0:
                self.logger.info(f"   🔄 Execution attempt {attempt + 1}")
            
            success, stdout, stderr = self._execute_script(script)
            
            if success:
                self.logger.info("   ✅ Trend analysis completed!")
                break
            
            error_preview = stderr[:200] + "..." if len(stderr) > 200 else stderr
            self.logger.warning(f"   ⚠️ Script failed: {error_preview}")
            
            if attempt < self.max_corrections:
                corrected = self._correct_script(script, stderr, attempt + 1)
                if corrected:
                    script = corrected
                else:
                    break
        
        generated_files = []
        for ext in ['*.png', '*.csv', '*.json']:
            for f in self.output_dir.glob(ext):
                if f.name not in ['series_fit_results.json']:
                    generated_files.append(str(f))
        
        state["trend_analysis_results"] = {
            "success": success,
            "skipped": False,
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics"),
            "stdout": stdout,
            "stderr": stderr if not success else None,
            "generated_files": generated_files,
            "script_path": str(self.output_dir / "trend_analysis.py")
        }
        
        return state


class UnifiedCurveSynthesisController:
    """
    [🧠 LLM Step]
    Synthesizes findings into scientific claims.
    
    ADAPTIVE:
    - For n>=2: Cross-spectrum synthesis with trend interpretation
    - For n=1: Single-spectrum scientific interpretation
    """
    
    SERIES_SYNTHESIS_INSTRUCTIONS = '''You are synthesizing findings from a curve fitting analysis of a spectral series.

**SERIES OVERVIEW:**
- Total spectra: {num_spectra}
- Successful fits: {successful_fits}
- Fitting model: {model_type}

**INDIVIDUAL FIT SUMMARIES:**
{fit_summaries}

**TREND ANALYSIS RESULTS:**
{trend_results}

**SERIES METADATA:**
{series_metadata}

**SYSTEM INFORMATION:**
{system_info}

Provide a comprehensive scientific synthesis including:
1. Overall quality assessment of the series analysis
2. Key trends identified in the fitted parameters
3. Physical interpretation of parameter evolution
4. Scientific claims supported by the data
5. Caveats and limitations

Return JSON with:
{{
    "detailed_analysis": "comprehensive scientific interpretation",
    "scientific_claims": [
        {{
            "claim": "specific claim statement",
            "evidence": "supporting evidence from fits",
            "confidence": "high/medium/low",
            "scientific_impact": "why this matters",
            "source": "curve_fitting_analysis",
            "parameters_involved": ["list of parameter names supporting this claim"]
        }}
    ],
    "parameter_trends": {{
        "parameter_name": {{
            "trend": "increasing/decreasing/stable/oscillating",
            "interpretation": "physical meaning"
        }}
    }},
    "caveats": "limitations and considerations"
}}
'''

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        single_spectrum_instructions: str,
        output_dir: str
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.single_spectrum_instructions = single_spectrum_instructions
        self.output_dir = Path(output_dir)

    def _synthesize_single_spectrum(self, state: dict) -> dict:
        """Generate interpretation for single spectrum."""
        self.logger.info("\n\n🔬 --- SINGLE SPECTRUM INTERPRETATION --- 🔬\n")
        
        fit_results = state.get("fit_results", {})
        
        formatted = self.single_spectrum_instructions.format(
            model_type=fit_results.get("model_type", "Curve fit"),
            summary=fit_results.get("summary", "Fitting complete"),
        )
        
        prompt_parts = [
            formatted,
            "\n## Original Data",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
        ]
        
        if state.get("final_plot_bytes"):
            prompt_parts.extend([
                "\n## Fit Result",
                {"mime_type": "image/png", "data": state["final_plot_bytes"]},
            ])
        
        prompt_parts.extend([
            "\n## Parameters\n" + json.dumps(fit_results.get("parameters", {}), indent=2),
            "\n## Fit Quality\n" + json.dumps(fit_results.get("fit_quality", {}), indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ])
        
        if state.get("literature_context"):
            prompt_parts.extend(["\n## Literature", state["literature_context"]])
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse(response)
            
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Single spectrum synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state

    def _synthesize_series(self, state: dict) -> dict:
        """Synthesize findings across spectral series."""
        self.logger.info("\n\n🔬 --- SERIES SYNTHESIS --- 🔬\n")
        
        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        
        successful_fits = [r for r in series_results if r["success"]]
        fit_summaries = []
        for r in successful_fits[:10]:
            fit_summaries.append({
                "index": r["index"],
                "name": r["name"],
                "model": r.get("model_type"),
                "key_params": r.get("parameters", {}),
                "r_squared": r.get("fit_quality", {}).get("r_squared"),
            })
        
        model_type = successful_fits[0].get("model_type") if successful_fits else "Unknown"
        
        prompt = self.SERIES_SYNTHESIS_INSTRUCTIONS.format(
            num_spectra=state.get("num_spectra", 1),
            successful_fits=len(successful_fits),
            model_type=model_type,
            fit_summaries=json.dumps(fit_summaries, indent=2),
            trend_results=json.dumps(trend_results, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            system_info=json.dumps(state.get("system_info", {}), indent=2)
        )
        
        prompt_parts = [prompt]
        
        if trend_results.get("success") and trend_results.get("generated_files"):
            prompt_parts.append("\n\n**TREND VISUALIZATIONS:**")
            for file_path in trend_results["generated_files"][:5]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        prompt_parts.append(f"\n{Path(file_path).name}:")
                        prompt_parts.append({"mime_type": "image/png", "data": f.read()})
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse(response)
            
            if error_dict:
                self.logger.error(f"Series synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Series synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Series synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state

    def execute(self, state: dict) -> dict:
        """Execute synthesis - adapts to single vs series."""
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            return self._synthesize_single_spectrum(state)
        else:
            return self._synthesize_series(state)


class UnifiedCurveReportController:
    """
    [📄 Report Step]
    Generates final HTML report for series analysis.
    
    ADAPTIVE:
    - For n>=2: Full series report with trend analysis
    - For n=1: Skipped (handled by GenerateCurveFittingReportController)
    """
    
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = Path(output_dir)

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Convert bytes to base64 string for HTML embedding."""
        return base64.b64encode(image_bytes).decode('utf-8')

    def _extract_parameter_evolution(self, series_results: List[dict]) -> List[dict]:
        """Extract parameter values across the series for tabulation."""
        evolution = []
        for r in series_results:
            entry = {
                "index": r["index"],
                "name": r["name"],
                "success": r["success"],
            }
            if r["success"]:
                entry["r_squared"] = r.get("fit_quality", {}).get("r_squared")
                entry["parameters"] = r.get("parameters", {})
            evolution.append(entry)
        return evolution

    def execute(self, state: dict) -> dict:
        """Generate final reports."""
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            self.logger.info("\n📄 Single spectrum report handled by standard controller.\n")
            return state
        
        self._generate_series_report(state)
        return state

    def _generate_series_report(self, state: dict) -> None:
        """Generate comprehensive report for spectral series."""
        self.logger.info("\n📄 --- GENERATING SERIES REPORT --- 📄\n")
        
        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        synthesis = state.get("synthesis_result", {})
        series_metadata = state.get("series_metadata", {})
        locked_config = state.get("locked_fitting_config", {})
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_spectra = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        
        param_evolution = self._extract_parameter_evolution(series_results)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curve Fitting Series Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1400px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
        .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
        .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }}
        .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .trend-card {{ background-color: #fef9e7; border-left: 5px solid #f39c12; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 25px; margin-top: 20px; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; }}
        .params-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
        .params-table th, .params-table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .params-table th {{ background-color: #f8f9fa; font-weight: 600; color: #2c3e50; position: sticky; top: 0; }}
        .params-table tr:hover {{ background-color: #f5f5f5; }}
        .success-badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
        .success {{ background-color: #d4edda; color: #155724; }}
        .failed {{ background-color: #f8d7da; color: #721c24; }}
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
        .scrollable-table {{ max-height: 400px; overflow-y: auto; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Spectral Series Analysis Report</h1>
        
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Spectra Processed:</strong> {successful}/{num_spectra}</p>
            <p><strong>Series Type:</strong> {series_metadata.get('series_type', 'N/A')}</p>
            <p><strong>Fitting Model:</strong> {locked_config.get('physical_model', 'N/A')}</p>
        </div>

        <h2>1. Scientific Analysis</h2>
        <div class="analysis-text">{synthesis.get('detailed_analysis', 'No analysis available.')}</div>
"""

        # Parameter trends
        param_trends = synthesis.get('parameter_trends', {})
        if param_trends:
            html += "\n        <h2>2. Parameter Trends</h2>\n"
            for param_name, trend_info in param_trends.items():
                if isinstance(trend_info, dict):
                    html += f"""
        <div class="trend-card">
            <strong>{param_name}</strong><br>
            Trend: {trend_info.get('trend', 'N/A')}<br>
            <em>{trend_info.get('interpretation', '')}</em>
        </div>
"""

        # Trend visualizations
        if trend_results.get("success") and trend_results.get("generated_files"):
            html += "\n        <h2>3. Trend Visualizations</h2>\n        <div class=\"image-grid\">\n"
            for file_path in trend_results["generated_files"]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        b64 = self._image_to_base64(f.read())
                    name = Path(file_path).stem.replace('_', ' ').title()
                    html += f"""
            <div class="image-card">
                <img src="data:image/png;base64,{b64}" alt="{name}">
                <p>{name}</p>
            </div>
"""
            html += "        </div>\n"

        # Parameter evolution table
        if param_evolution:
            html += "\n        <h2>4. Parameter Evolution</h2>\n        <div class=\"scrollable-table\">\n            <table class=\"params-table\">\n                <thead>\n                    <tr>\n                        <th>Index</th>\n                        <th>Name</th>\n                        <th>Status</th>\n                        <th>R²</th>\n"
            
            all_params = set()
            for r in series_results:
                if r["success"]:
                    all_params.update(r.get("parameters", {}).keys())
            all_params = sorted(all_params)
            
            for param in all_params[:10]:
                html += f"                        <th>{param}</th>\n"
            
            html += "                    </tr>\n                </thead>\n                <tbody>\n"
            
            for r in series_results:
                status_class = "success" if r["success"] else "failed"
                status_text = "✓" if r["success"] else "✗"
                r_squared = r.get("fit_quality", {}).get("r_squared", "")
                if isinstance(r_squared, float):
                    r_squared = f"{r_squared:.4f}"
                
                html += f"                    <tr>\n                        <td>{r['index']}</td>\n                        <td>{r['name']}</td>\n                        <td><span class=\"success-badge {status_class}\">{status_text}</span></td>\n                        <td>{r_squared}</td>\n"
                
                params = r.get("parameters", {})
                for param in all_params[:10]:
                    val = params.get(param, "")
                    if isinstance(val, float):
                        val = f"{val:.4g}"
                    html += f"                        <td>{val}</td>\n"
                
                html += "                    </tr>\n"
            
            html += "                </tbody>\n            </table>\n        </div>\n"

        # Scientific claims
        scientific_claims = synthesis.get('scientific_claims', [])
        if scientific_claims:
            html += "\n        <h2>5. Scientific Claims</h2>\n"
            for i, claim in enumerate(scientific_claims, 1):
                confidence = claim.get('confidence', 'medium')
                html += f"""
        <div class="claim-card">
            <strong>Claim {i}:</strong> {claim.get('claim', 'N/A')}<br>
            <em>Evidence:</em> {claim.get('evidence', 'N/A')}<br>
            <em>Confidence:</em> {confidence}<br>
            <em>Impact:</em> {claim.get('scientific_impact', 'N/A')}
        </div>
"""

        # Caveats
        caveats = synthesis.get('caveats', '')
        if caveats:
            html += f"""
        <h2>6. Caveats & Limitations</h2>
        <div class="caveats">
            {caveats}
        </div>
"""

        html += """
        <div class="footer">
            Generated by SciLink Curve Fitting Series Analysis Agent
        </div>
    </div>
</body>
</html>
"""

        report_path = self.output_dir / "series_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        state["report_path"] = str(report_path)
        self.logger.info(f"   ✅ Report saved: {report_path}")