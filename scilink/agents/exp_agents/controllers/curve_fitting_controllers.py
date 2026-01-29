# controllers/curve_fitting_controllers.py

"""
Curve Fitting Controllers - Complete Module

This module contains:
1. Original controllers for single-spectrum analysis steps
2. Unified controllers that handle both single spectrum (n=1) and series (n>1) analysis

Key principle for series analysis: Single spectrum = Series of 1

Quality control features:
- Automatic model retry when R² is inadequate
- Statistical outlier detection for series
- Human feedback integration for unresolved quality issues
"""

# Set non-interactive backend BEFORE importing pyplot anywhere
import matplotlib
matplotlib.use('Agg')

import subprocess
import json
import logging
import os
import base64
import re
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
                self.logger.warning("  ⚠️ No results")

            state["literature_files"] = self._save_results(
                query, state["literature_context"] or f"No results: {result.get('message')}"
            )
        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}")
            state["literature_context"] = None
            state["literature_files"] = self._save_results(query, f"Error: {e}")

        return state


class GenerateCurveFittingReportController:
    """Generates a human-readable HTML report for curve fitting analysis."""
    
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = output_dir

    def _image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if not state.get("is_single_spectrum", True):
            return state
            
        self.logger.info("\n📄 --- Generating HTML Report ---\n")
        
        result_json = state.get("result_json", {})
        fit_results = state.get("fit_results", {})
        synthesis_result = state.get("synthesis_result", {})
        
        detailed_analysis = synthesis_result.get("detailed_analysis") or result_json.get("detailed_analysis", "No analysis provided.")
        scientific_claims = synthesis_result.get("scientific_claims") or result_json.get("scientific_claims", [])
        system_info = state.get("system_info", {})
        model_type = fit_results.get("model_type", result_json.get("model_type", "N/A"))
        parameters = fit_results.get("parameters", result_json.get("fitting_parameters", {}))
        fit_quality = fit_results.get("fit_quality", result_json.get("fit_quality", {}))
        caveats = synthesis_result.get("caveats") or result_json.get("caveats", "")
        
        quality_warning = None
        series_results = state.get("series_results", [])
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]
        
        original_plot = state.get("original_plot_bytes")
        fit_plot = state.get("final_plot_bytes")
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CurveFitting_Report_{file_timestamp}.html"
        filepath = output_dir / filename

        params_html = self._format_parameters(parameters)
        quality_html = self._format_fit_quality(fit_quality, quality_warning)

        html_content = self._build_html_report(
            timestamp, state, system_info, model_type, quality_html, 
            detailed_analysis, original_plot, fit_plot, params_html,
            scientific_claims, caveats
        )

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"  ✅ Report saved: {filepath}")
            state["report_path"] = str(filepath)
        except Exception as e:
            self.logger.error(f"  ❌ Failed to write report: {e}")

        return state

    def _build_html_report(self, timestamp, state, system_info, model_type, 
                          quality_html, detailed_analysis, original_plot, 
                          fit_plot, params_html, scientific_claims, caveats):
        """Build the complete HTML report."""
        system_info_str = self._format_system_info(system_info)
        
        images_html = ""
        if original_plot:
            b64_original = self._image_to_base64(original_plot)
            images_html += f'<div class="image-card"><img src="data:image/png;base64,{b64_original}" alt="Original Data"><div class="image-label">Original Data</div></div>'
        if fit_plot:
            b64_fit = self._image_to_base64(fit_plot)
            images_html += f'<div class="image-card"><img src="data:image/png;base64,{b64_fit}" alt="Fit Visualization"><div class="image-label">Fit Result with Residuals</div></div>'

        claims_html = ""
        if not scientific_claims:
            claims_html = "<p>No specific claims generated.</p>"
        else:
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                claims_html += f"""
        <div class="claim-card">
            <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
            <p><strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
            <p><strong>Research Question:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
            <p><strong>Keywords:</strong> {keywords_str}</p>
        </div>"""

        caveats_html = ""
        if caveats:
            caveats_html = f"""
        <h2>5. Caveats & Limitations</h2>
        <div class="caveats">{caveats}</div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curve Fitting Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
        .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        h3 {{ color: #16a085; margin-top: 20px; }}
        .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
        .model-box {{ background-color: #e8f4fc; padding: 15px; border-radius: 5px; border-left: 5px solid #2980b9; margin-bottom: 15px; }}
        .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }}
        .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin-top: 20px; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; }}
        .image-label {{ margin-top: 12px; font-weight: bold; color: #444; font-size: 1em; border-top: 1px solid #eee; padding-top: 10px; }}
        .params-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .params-table th, .params-table td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        .params-table th {{ background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }}
        .params-table tr:hover {{ background-color: #f5f5f5; }}
        .quality-badge {{ display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: bold; margin-right: 10px; }}
        .quality-good {{ background-color: #d4edda; color: #155724; }}
        .quality-ok {{ background-color: #fff3cd; color: #856404; }}
        .quality-poor {{ background-color: #f8d7da; color: #721c24; }}
        .quality-warning-box {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px 15px; margin-top: 10px; border-radius: 0 5px 5px 0; font-size: 0.9em; }}
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Curve Fitting Analysis Report</h1>
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Data Source:</strong> {state.get('data_path', 'N/A')}</p>
            <p><strong>Sample Info:</strong> {system_info_str}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <h3>Fitting Model</h3>
        <div class="model-box">{model_type}</div>
        <h3>Fit Quality</h3>
        {quality_html}
        <h3>Interpretation</h3>
        <div class="analysis-text">{detailed_analysis}</div>
        <h2>2. Visualizations</h2>
        <div class="image-grid">{images_html}</div>
        <h2>3. Fitted Parameters</h2>
        {params_html}
        <h2>4. Scientific Claims</h2>
        {claims_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Curve Fitting Analysis Agent</div>
    </div>
</body>
</html>"""

    def _format_system_info(self, system_info: dict) -> str:
        if not system_info:
            return "N/A"
        parts = [f"{k}: {v}" for k, v in system_info.items() if v]
        return ", ".join(parts) if parts else "N/A"

    def _format_parameters(self, parameters: dict) -> str:
        if not parameters:
            return "<p>No parameters extracted.</p>"

        rows = ""
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
                    rows += f"<tr><td><strong>{component_display}</strong></td><td>{param_name}</td><td>{value_str}</td><td>{err_value}</td></tr>"
                    first_row = False
            else:
                rows += f"<tr><td><strong>{component}</strong></td><td>—</td><td>{parameters[component]}</td><td>—</td></tr>"

        return f"""<table class="params-table">
            <thead><tr><th>Component</th><th>Parameter</th><th>Value</th><th>Uncertainty</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    def _format_fit_quality(self, fit_quality: dict, quality_warning: str = None) -> str:
        if not fit_quality:
            return "<p>No quality metrics available.</p>"

        r_squared = fit_quality.get("r_squared", fit_quality.get("r2"))
        rmse = fit_quality.get("rmse")
        chi_squared = fit_quality.get("chi_squared_reduced", fit_quality.get("reduced_chi_squared"))

        html = "<div>"
        if r_squared is not None:
            if r_squared >= 0.99:
                badge_class, label = "quality-good", "Excellent"
            elif r_squared >= 0.95:
                badge_class, label = "quality-ok", "Good"
            else:
                badge_class, label = "quality-poor", "Poor"
            html += f'<span class="quality-badge {badge_class}">{label}</span><strong>R² = {r_squared:.4f}</strong>'

        if rmse is not None:
            html += f" &nbsp;|&nbsp; <strong>RMSE = {rmse:.4g}</strong>"
        if chi_squared is not None:
            html += f" &nbsp;|&nbsp; <strong>χ²/DOF = {chi_squared:.3f}</strong>"
        html += "</div>"
        
        if quality_warning:
            html += f'<div class="quality-warning-box">⚠️ <strong>Note:</strong> {quality_warning}. Alternative models were attempted but could not improve fit quality significantly.</div>'
        
        return html


# ============================================================================
# UNIFIED CONTROLLERS (for series analysis support)
# ============================================================================

class HumanFeedbackRefinementController:
    """
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
        prompt = [
            self.instructions,
            "\n## Data Plot",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ]

        if state.get("analysis_hints"):
            prompt.append(f"\n## User Guidance\n{state['analysis_hints']}")
        
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
    Processes ALL spectra using the locked fitting model.
    
    Quality control features:
    - If R² < threshold, automatically tries alternative models (max_model_retries)
    - If still inadequate and human feedback enabled, asks for guidance
    - Otherwise proceeds with best available fit
    - For series: detects statistical outliers that may indicate interesting physics
    """
    
    MAX_ATTEMPTS = 3
    DEFAULT_R2_THRESHOLD = 0.95
    DEFAULT_MAX_MODEL_RETRIES = 3
    DEFAULT_OUTLIER_SIGMA = 2.0
    DEFAULT_MAX_VERIFICATION_ITERATIONS = 3

    ALTERNATIVE_MODELS_PROMPT = '''The current fitting model achieved R² = {r_squared:.4f}, which is below the quality threshold of {threshold}.

**Current Model:** {current_model}
**Current Parameters:** {current_params}

**Data Statistics:**
{data_stats}

**Original Analysis Approach:** {analysis_approach}

**IMPORTANT:** Examine the fit visualization provided below carefully. Look for:
1. Systematic deviations between the fit and data (not just noise)
2. Missing features (additional peaks, shoulders, asymmetry)
3. Incorrect baseline behavior
4. Wrong peak shape (too sharp, too broad, wrong tail behavior)

Based on your visual inspection and the numerical results, suggest an alternative fitting model. Consider:
1. Different functional forms (e.g., if using single Gaussian, try double Gaussian, Voigt, or Lorentzian)
2. Additional components (e.g., baseline correction, additional peaks, shoulders)
3. Different physical models appropriate for this type of spectroscopy

Return JSON with:
{{
    "diagnosis": "specific description of what you observe is wrong with the current fit based on the visualization",
    "alternative_model": "description of the new model to try",
    "fitting_strategy": "specific fitting approach for the new model",
    "parameters_to_extract": ["list", "of", "parameters"]
}}
'''

    HUMAN_FEEDBACK_PROMPT = '''## Fit Quality Issue

The automated fitting could not achieve adequate fit quality.

**Best Result:** R² = {best_r2:.4f} (threshold: {threshold})
**Models Tried:**
{models_tried}

**Options:**
1. Suggest a different model or approach
2. Adjust the R² threshold for this analysis (e.g., "threshold 0.90")
3. Accept the best available fit (type "accept")

Your guidance: '''

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
        plot_fn: Callable,
        r2_threshold: float = None,
        max_model_retries: int = None,
        enable_human_feedback: bool = False,
        outlier_sigma: float = None,
        max_verification_iterations: int = None,
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
        self.r2_threshold = r2_threshold if r2_threshold is not None else self.DEFAULT_R2_THRESHOLD
        self.max_model_retries = max_model_retries if max_model_retries is not None else self.DEFAULT_MAX_MODEL_RETRIES
        self.enable_human_feedback = enable_human_feedback
        self.outlier_sigma = outlier_sigma if outlier_sigma is not None else self.DEFAULT_OUTLIER_SIGMA
        self.max_verification_iterations = max_verification_iterations if max_verification_iterations is not None else self.DEFAULT_MAX_VERIFICATION_ITERATIONS

    def _generate_fitting_script(self, state: dict, data_path: str, stats: dict) -> str:
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
        adapted = base_script
        adapted = adapted.replace('fit_visualization.png', f'{output_prefix}_fit.png')
        adapted = re.sub(r'spectrum_\d{4}_fit\.png', f'{output_prefix}_fit.png', adapted)
        adapted = re.sub(r'spectrum_\d{4}_T\d+K_fit\.png', f'{output_prefix}_fit.png', adapted)
        adapted = re.sub(
            r'np\.load\s*\(\s*["\'"].*?temp_spectrum_\d+\.npy["\'"]\s*\)',
            f'np.load("{data_path}")',
            adapted
        )
        adapted = re.sub(r'(["\'"]).*?temp_spectrum_\d+\.npy\1', f'"{data_path}"', adapted)
        adapted = re.sub(r'DATA_PATH\s*=\s*["\'"].*?["\'"]', f'DATA_PATH = "{data_path}"', adapted)
        adapted = re.sub(r'data_path\s*=\s*["\'"].*?["\'"]', f'data_path = "{data_path}"', adapted)
        return adapted

    def _compute_statistics(self, curve_data: np.ndarray) -> dict:
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
        """Load curve data from file, handling various formats."""
        # Try using the project's load_curve_data function first
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
        
        # Fallback: handle common formats manually
        if data_path.endswith('.npy'):
            return np.load(data_path)
        elif data_path.endswith('.csv'):
            # Try to load CSV, handling potential headers
            try:
                # First try without header
                return np.loadtxt(data_path, delimiter=',')
            except ValueError:
                # If that fails, try skipping first row (header)
                try:
                    return np.loadtxt(data_path, delimiter=',', skiprows=1)
                except ValueError:
                    # Try pandas as last resort for complex CSVs
                    try:
                        import pandas as pd
                        df = pd.read_csv(data_path)
                        return df.values.T  # Transpose to get (2, n_points) shape
                    except ImportError:
                        # Re-raise the original error if pandas not available
                        raise ValueError(f"Could not parse CSV file: {data_path}. "
                                        "File may have headers or non-numeric data.")
        elif data_path.endswith('.txt'):
            # Try common text formats
            try:
                return np.loadtxt(data_path)
            except ValueError:
                try:
                    return np.loadtxt(data_path, skiprows=1)
                except ValueError:
                    # Try tab-delimited
                    try:
                        return np.loadtxt(data_path, delimiter='\t', skiprows=1)
                    except:
                        raise ValueError(f"Could not parse text file: {data_path}")
        else:
            # Generic attempt
            try:
                return np.loadtxt(data_path)
            except ValueError:
                return np.loadtxt(data_path, skiprows=1)

    def _fit_single_spectrum(
        self, 
        state: dict, 
        curve_data: np.ndarray, 
        data_path: str,
        spectrum_name: str,
        spectrum_idx: int,
        base_script: Optional[str] = None
    ) -> dict:
        stats = self._compute_statistics(curve_data)
        temp_data_path = self.output_dir / f"temp_spectrum_{spectrum_idx}.npy"
        np.save(temp_data_path, curve_data)
        output_prefix = f"spectrum_{spectrum_idx:04d}"
        
        # Clean up any existing visualization files for this spectrum to ensure fresh output
        for old_viz in [
            self.output_dir / f"{output_prefix}_fit.png",
            self.output_dir / "fit_visualization.png",
        ]:
            if old_viz.exists():
                try:
                    os.remove(old_viz)
                except:
                    pass
        
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
        
        fit_results = {}
        for line in (exec_result.get("stdout") or "").splitlines():
            if line.startswith("FIT_RESULTS_JSON:"):
                try:
                    fit_results = json.loads(line.replace("FIT_RESULTS_JSON:", "").strip())
                except json.JSONDecodeError:
                    pass
                break
        
        viz_path = self.output_dir / f"{output_prefix}_fit.png"
        if not viz_path.exists():
            viz_path = self.output_dir / "fit_visualization.png"
        
        viz_bytes = None
        if viz_path.exists():
            with open(viz_path, "rb") as f:
                viz_bytes = f.read()
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

    def _suggest_alternative_model(self, state: dict, current_result: dict) -> Optional[dict]:
        """Use LLM to suggest an alternative fitting model, showing it the actual poor fit."""
        config = state.get("locked_fitting_config", {})
        
        prompt_text = self.ALTERNATIVE_MODELS_PROMPT.format(
            r_squared=current_result.get("fit_quality", {}).get("r_squared", 0),
            threshold=self.r2_threshold,
            current_model=config.get("physical_model", "Unknown"),
            current_params=json.dumps(current_result.get("parameters", {}), indent=2),
            data_stats=json.dumps(current_result.get("statistics", {}), indent=2),
            analysis_approach=config.get("analysis_approach", ""),
        )
        
        # Build prompt with visualization if available
        prompt_parts = [prompt_text]
        
        # Include the actual fit visualization so LLM can see what went wrong
        if current_result.get("visualization_bytes"):
            prompt_parts.append("\n\n**CURRENT FIT VISUALIZATION (showing the poor fit):**")
            prompt_parts.append({
                "mime_type": "image/png", 
                "data": current_result["visualization_bytes"]
            })
            prompt_parts.append("\nExamine this fit carefully. Look at where the model deviates from the data and suggest a better model.")
        
        # Also include original data plot if available for comparison
        if state.get("original_plot_bytes"):
            prompt_parts.append("\n\n**ORIGINAL DATA:**")
            prompt_parts.append({
                "mime_type": "image/png",
                "data": state["original_plot_bytes"]
            })
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            if error or not result:
                return None
            return result
        except Exception as e:
            self.logger.error(f"Failed to get alternative model suggestion: {e}")
            return None

    FIT_VERIFICATION_PROMPT = '''You are a scientific data analysis expert reviewing a curve/spectral fit.

**TASK:** Examine this fit visualization and identify any issues that should be corrected.

**FIT STATISTICS:**
- R² = {r_squared:.4f}
- Model: {model_type}
- Number of components: {n_components}

**FITTED PARAMETERS:**
{parameters}

**CRITICAL: Examine the residual plot carefully.** Look for:

1. **Systematic patterns in residuals** (not random noise):
   - S-shaped patterns (positive-negative-positive) suggest a single component is fitting what should be multiple
   - Large localized residuals suggest missing components
   - Oscillating residuals suggest wrong functional form
   
2. **Spurious components:**
   - Components with very small amplitude fitting noise rather than real features
   - Components in regions with no visible features in the data
   
3. **Poor component placement:**
   - Components not centered on actual features
   - Components too broad or too narrow for the features they fit

4. **Baseline issues:**
   - Systematic slope in residuals
   - Curvature not captured

**IMPORTANT:** This is a general curve fitting tool - features may be peaks, edges, steps, oscillations, or other shapes depending on the data type. Focus on whether the model captures the data structure, not on specific peak assignments.

**Return JSON:**
{{
    "fit_acceptable": true/false,
    "issues_found": [
        {{
            "location": "description of where (e.g., 'around 2850-2950 region' or 'high-x region')",
            "problem": "specific issue observed",
            "evidence": "what you see in the residuals or fit",
            "suggested_fix": "how to address it"
        }}
    ],
    "spurious_components": ["list of component names/indices that appear to be fitting noise"],
    "missing_features": ["descriptions of data features not captured by the model"],
    "overall_assessment": "brief summary of fit quality and main issues",
    "recommended_action": "specific instruction for improving the fit (or 'none' if acceptable)"
}}

If the fit is acceptable (residuals appear random, all features captured, no spurious components), set fit_acceptable=true and issues_found=[].
'''

    def _verify_fit_with_llm(self, state: dict, fit_result: dict) -> Optional[dict]:
        """
        Use LLM to verify fit quality by examining the visualization.
        Returns verification result with any issues found, or None if verification fails.
        """
        if not fit_result.get("visualization_bytes"):
            self.logger.warning("      No visualization available for LLM verification")
            return None
        
        # Gather fit info
        r_squared = fit_result.get("fit_quality", {}).get("r_squared", 0)
        model_type = fit_result.get("model_type", "Unknown")
        parameters = fit_result.get("parameters", {})
        
        # Count components
        n_components = len(parameters) if isinstance(parameters, dict) else 0
        
        # Format parameters for prompt
        params_str = json.dumps(parameters, indent=2) if parameters else "No parameters extracted"
        
        prompt_text = self.FIT_VERIFICATION_PROMPT.format(
            r_squared=r_squared,
            model_type=model_type,
            n_components=n_components,
            parameters=params_str
        )
        
        prompt_parts = [
            prompt_text,
            "\n\n**FIT VISUALIZATION (examine carefully, especially the residual plot):**",
            {"mime_type": "image/png", "data": fit_result["visualization_bytes"]}
        ]
        
        # Also include original data if available for comparison
        if state.get("original_plot_bytes"):
            prompt_parts.append("\n\n**ORIGINAL DATA (for reference):**")
            prompt_parts.append({"mime_type": "image/png", "data": state["original_plot_bytes"]})
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"      LLM verification parse failed: {error}")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"      LLM verification failed: {e}")
            return None

    def _apply_llm_verification_feedback(self, state: dict, verification: dict) -> dict:
        """
        Apply LLM verification feedback to refine the fitting configuration.
        Returns updated config.
        """
        config = state.get("locked_fitting_config", {}).copy()
        
        recommended_action = verification.get("recommended_action", "")
        if not recommended_action or recommended_action.lower() == "none":
            return config
        
        # Build a refinement prompt based on verification results
        issues_summary = []
        for issue in verification.get("issues_found", []):
            issues_summary.append(f"- {issue.get('location', 'Unknown')}: {issue.get('problem', '')} -> {issue.get('suggested_fix', '')}")
        
        spurious = verification.get("spurious_components", [])
        missing = verification.get("missing_features", [])
        
        refinement_prompt = f"""Refine the fitting approach based on automated verification feedback.

**CURRENT APPROACH:**
- Model: {config.get('physical_model', 'Unknown')}
- Strategy: {config.get('fitting_strategy', 'Unknown')}

**VERIFICATION FINDINGS:**
{chr(10).join(issues_summary) if issues_summary else 'No specific issues listed'}

**SPURIOUS COMPONENTS TO REMOVE:** {', '.join(spurious) if spurious else 'None identified'}

**MISSING FEATURES TO ADD:** {', '.join(missing) if missing else 'None identified'}

**RECOMMENDED ACTION:** {recommended_action}

Return JSON with the refined fitting approach:
{{
    "physical_model": "updated model description incorporating the fixes",
    "fitting_strategy": "updated fitting strategy",
    "parameters_to_extract": ["list", "of", "parameters"],
    "analysis_approach": "updated approach"
}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[refinement_prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"      Could not parse refinement: {error}")
                return config
            
            # Update config with refinements
            config.update(result)
            return config
            
        except Exception as e:
            self.logger.error(f"      Refinement failed: {e}")
            return config

    def _get_human_feedback_for_poor_fit(self, state: dict, best_result: dict, all_attempts: List[dict]) -> Optional[dict]:
        models_tried = "\n".join([f"  - {a['model']}: R² = {a['r2']:.4f}" for a in all_attempts])
        
        print("\n" + "=" * 70)
        print("⚠️  FIT QUALITY BELOW THRESHOLD")
        print("=" * 70)
        
        if best_result.get("visualization_bytes"):
            viz_path = self.output_dir / "quality_review_fit.png"
            with open(viz_path, 'wb') as f:
                f.write(best_result["visualization_bytes"])
            print(f"[Best fit visualization saved to: {viz_path}]")
        
        prompt = self.HUMAN_FEEDBACK_PROMPT.format(
            best_r2=best_result.get("fit_quality", {}).get("r_squared", 0),
            threshold=self.r2_threshold,
            models_tried=models_tried,
        )
        print(prompt)
        
        feedback = input("\nYour input: ").strip()
        
        if not feedback:
            print("No feedback provided. Proceeding with best available fit.")
            return None
        
        if "accept" in feedback.lower() or "proceed" in feedback.lower():
            print("✓ Accepting best available fit.")
            return None
        
        if "threshold" in feedback.lower():
            try:
                match = re.search(r'(\d+\.?\d*)', feedback)
                if match:
                    new_threshold = float(match.group(1))
                    if new_threshold <= 1.0:
                        print(f"✓ Adjusting threshold to {new_threshold}")
                        return {"action": "adjust_threshold", "new_threshold": new_threshold}
            except:
                pass
        
        print("🔄 Will retry with your suggested approach...")
        return {"action": "retry", "feedback": feedback}

    def _get_user_feedback_on_fit(self, state: dict, fit_result: dict, r2: float) -> Optional[str]:
        """
        Show user the first spectrum fit and ask for optional feedback.
        Returns feedback string if user wants changes, None if satisfied.
        """
        print("\n" + "=" * 70)
        print("📊 FIRST SPECTRUM FIT RESULT - Review Before Processing Series")
        print("=" * 70)
        
        # Save and display fit visualization path
        review_viz_path = None
        if fit_result.get("visualization_bytes"):
            review_viz_path = self.output_dir / "first_spectrum_fit_review.png"
            with open(review_viz_path, 'wb') as f:
                f.write(fit_result["visualization_bytes"])
            print(f"\n[Fit visualization saved to: {review_viz_path}]")
        
        # Show fit summary
        print(f"\n📈 Model: {fit_result.get('model_type', 'N/A')}")
        print(f"📊 R² = {r2:.4f} (threshold: {self.r2_threshold})")
        
        params = fit_result.get("parameters", {})
        if params:
            print("\n📋 Fitted Parameters:")
            for comp, values in params.items():
                if isinstance(values, dict):
                    print(f"   {comp}:")
                    for k, v in values.items():
                        if not k.endswith('_err'):
                            if isinstance(v, float):
                                print(f"      {k}: {v:.4g}")
                            else:
                                print(f"      {k}: {v}")
        
        num_spectra = state.get("num_spectra", 1)
        print(f"\n⚠️  This fitting model will be applied to all {num_spectra} spectra in the series.")
        print("\n" + "-" * 70)
        print("Options:")
        print("  • Press Enter to accept this fit and proceed with series")
        print("  • Type feedback to modify the fitting approach (e.g., 'add baseline', ")
        print("    'use Voigt instead of Gaussian', 'fit two peaks instead of one')")
        print("-" * 70)
        
        feedback = input("\n🤔 Your feedback (or Enter to accept): ").strip()
        
        # Clean up the review file - it's only for user viewing during this step
        if review_viz_path and review_viz_path.exists():
            try:
                os.remove(review_viz_path)
            except:
                pass
        
        if not feedback:
            print("✅ Fit accepted. Proceeding with series...")
            return None
        
        return feedback

    def _ask_keep_user_guided_fit(self, user_r2: float, original_r2: float) -> bool:
        """Ask user whether to keep the user-guided fit even if R² is worse."""
        print("\n" + "-" * 70)
        print(f"⚠️  User-guided fit has lower R² ({user_r2:.4f}) than original ({original_r2:.4f})")
        print("-" * 70)
        print("Options:")
        print(f"  • Type 'keep' to use the user-guided fit anyway (R² = {user_r2:.4f})")
        print(f"  • Press Enter to revert to original fit (R² = {original_r2:.4f})")
        
        response = input("\nYour choice: ").strip().lower()
        
        if response == 'keep':
            print("✅ Keeping user-guided fit.")
            return True
        else:
            print("✅ Reverting to original fit.")
            return False

    def _refine_model_from_feedback(self, state: dict, feedback: str) -> dict:
        config = state.get("locked_fitting_config", {})
        prompt = f"""Refine the fitting approach based on user feedback.

**Current Approach:**
- Model: {config.get('physical_model', 'Unknown')}
- Strategy: {config.get('fitting_strategy', 'Unknown')}

**User Feedback:** {feedback}

Return JSON with:
{{
    "physical_model": "updated model description",
    "fitting_strategy": "updated fitting strategy",
    "parameters_to_extract": ["list", "of", "parameters"],
    "analysis_approach": "updated approach"
}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            if error or not result:
                return config
            updated = config.copy()
            updated.update(result)
            return updated
        except Exception as e:
            self.logger.error(f"Failed to refine model from feedback: {e}")
            return config

    def _fit_with_quality_control(self, state: dict, curve_data: np.ndarray, data_path: str, spectrum_name: str, spectrum_idx: int) -> dict:
        all_attempts = []
        best_result = None
        best_r2 = -1.0
        
        initial_model = state.get('locked_fitting_config', {}).get('physical_model', 'Initial model')
        self.logger.info(f"      Attempt 1: {initial_model}")
        
        result = self._fit_single_spectrum(
            state=state, curve_data=curve_data, data_path=data_path,
            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
        )
        
        if result["success"]:
            r2 = result.get("fit_quality", {}).get("r_squared", 0)
            all_attempts.append({"model": initial_model, "r2": r2, "result": result})
            
            if r2 > best_r2:
                best_r2 = r2
                best_result = result
            
            # For first spectrum, run LLM verification loop before human feedback
            if spectrum_idx == 0:
                # LLM verification loop - iteratively improve the fit
                for verification_iter in range(self.max_verification_iterations):
                    self.logger.info(f"      🔍 Verification iteration {verification_iter + 1}/{self.max_verification_iterations}...")
                    verification = self._verify_fit_with_llm(state, best_result)
                    
                    if verification is None:
                        self.logger.warning(f"      Verification failed, skipping")
                        break
                    
                    if verification.get("fit_acceptable", True):
                        self.logger.info(f"      ✅ Verification passed: {verification.get('overall_assessment', 'Fit acceptable')}")
                        break
                    
                    # LLM found issues - try to fix
                    issues_count = len(verification.get("issues_found", []))
                    self.logger.info(f"      ⚠️ Found {issues_count} issue(s): {verification.get('overall_assessment', '')}")
                    
                    # Log specific issues
                    for issue in verification.get("issues_found", [])[:3]:  # Show up to 3 issues
                        self.logger.info(f"         - {issue.get('location', '?')}: {issue.get('problem', '?')}")
                    
                    # Apply LLM's recommended fixes
                    refined_config = self._apply_llm_verification_feedback(state, verification)
                    
                    if refined_config == state.get("locked_fitting_config", {}):
                        self.logger.info(f"      No config changes suggested, stopping verification loop")
                        break
                    
                    # Clean up old visualization
                    old_viz_path = best_result.get("visualization_path")
                    if old_viz_path and Path(old_viz_path).exists():
                        try:
                            os.remove(old_viz_path)
                        except:
                            pass
                    
                    original_config = state.get("locked_fitting_config")
                    state["locked_fitting_config"] = refined_config
                    
                    # Refit with refined config
                    self.logger.info(f"      🔄 Refitting with verification feedback...")
                    verified_result = self._fit_single_spectrum(
                        state=state, curve_data=curve_data, data_path=data_path,
                        spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                    )
                    
                    if verified_result["success"]:
                        verified_r2 = verified_result.get("fit_quality", {}).get("r_squared", 0)
                        self.logger.info(f"      Iteration {verification_iter + 1} result: R² = {verified_r2:.4f} (was {best_r2:.4f})")
                        all_attempts.append({"model": f"LLM-verified-{verification_iter + 1}", "r2": verified_r2, "result": verified_result})
                        
                        if verified_r2 >= best_r2 - 0.01:  # Accept if not significantly worse (within 0.01)
                            best_r2 = verified_r2
                            best_result = verified_result
                            # Keep the refined config for next iteration
                        else:
                            # Verification made it significantly worse, revert and stop
                            state["locked_fitting_config"] = original_config
                            self.logger.info(f"      R² decreased significantly, reverting and stopping verification")
                            # Regenerate the previous best fit
                            best_result = self._fit_single_spectrum(
                                state=state, curve_data=curve_data, data_path=data_path,
                                spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                            )
                            break
                    else:
                        state["locked_fitting_config"] = original_config
                        self.logger.warning(f"      Verification-guided fit failed, reverting")
                        break
                else:
                    # Loop completed without breaking - max iterations reached
                    self.logger.info(f"      Reached max verification iterations ({self.max_verification_iterations})")
                
                # Now offer human feedback opportunity (with potentially improved fit)
                if self.enable_human_feedback:
                    user_feedback = self._get_user_feedback_on_fit(state, best_result, best_r2)
                    
                    if user_feedback:
                        # User wants to modify the approach
                        refined_config = self._refine_model_from_feedback(state, user_feedback)
                        original_config = state.get("locked_fitting_config")
                        state["locked_fitting_config"] = refined_config
                        
                        # Clean up old visualization file before refitting
                        old_viz_path = best_result.get("visualization_path")
                        if old_viz_path and Path(old_viz_path).exists():
                            try:
                                os.remove(old_viz_path)
                                self.logger.info(f"      Removed old visualization: {old_viz_path}")
                            except Exception as e:
                                self.logger.warning(f"      Could not remove old visualization: {e}")
                        
                        self.logger.info(f"      🔄 Refitting with user feedback...")
                        user_guided_result = self._fit_single_spectrum(
                            state=state, curve_data=curve_data, data_path=data_path,
                            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                        )
                        
                        if user_guided_result["success"]:
                            user_r2 = user_guided_result.get("fit_quality", {}).get("r_squared", 0)
                            self.logger.info(f"      User-guided fit: R² = {user_r2:.4f}")
                            all_attempts.append({"model": "User-guided", "r2": user_r2, "result": user_guided_result})
                            
                            if user_r2 > best_r2:
                                best_r2 = user_r2
                                best_result = user_guided_result
                                # Keep the refined config
                                self.logger.info(f"      📝 Updated config based on user feedback")
                            else:
                                # User-guided was worse, but user explicitly requested it - ask what to do
                                keep_user = self._ask_keep_user_guided_fit(user_r2, best_r2)
                                if not keep_user:
                                    state["locked_fitting_config"] = original_config
                                    self.logger.info(f"      Reverting to previous config (R² = {best_r2:.4f})")
                                    # Need to refit with original config to get the visualization back
                                    self.logger.info(f"      🔄 Regenerating previous fit...")
                                    best_result = self._fit_single_spectrum(
                                        state=state, curve_data=curve_data, data_path=data_path,
                                        spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                                    )
                                else:
                                    best_r2 = user_r2
                                    best_result = user_guided_result
                        else:
                            self.logger.warning(f"      User-guided fit failed, keeping previous")
                            state["locked_fitting_config"] = original_config
                            # Regenerate previous fit visualization since we deleted it
                            self.logger.info(f"      🔄 Regenerating previous fit...")
                            best_result = self._fit_single_spectrum(
                                state=state, curve_data=curve_data, data_path=data_path,
                                spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                            )
            
            if best_r2 >= self.r2_threshold:
                self.logger.info(f"      ✅ R² = {best_r2:.4f} (meets threshold {self.r2_threshold})")
                return best_result
            else:
                self.logger.warning(f"      ⚠️ R² = {best_r2:.4f} (below threshold {self.r2_threshold})")
        else:
            self.logger.error(f"      ❌ Initial fit failed: {result.get('error', 'Unknown')[:50]}")
            all_attempts.append({"model": initial_model, "r2": 0, "result": result})
        
        current_config = state.get("locked_fitting_config", {}).copy()
        
        for retry in range(self.max_model_retries):
            self.logger.info(f"      🔄 Attempting alternative model {retry + 1}/{self.max_model_retries}...")
            
            alternative = self._suggest_alternative_model(state, best_result or result)
            
            if not alternative:
                self.logger.warning("      Could not generate alternative model suggestion")
                break
            
            self.logger.info(f"      Diagnosis: {alternative.get('diagnosis', 'N/A')}")
            self.logger.info(f"      Trying: {alternative.get('alternative_model', 'N/A')}")
            
            temp_config = current_config.copy()
            temp_config["physical_model"] = alternative.get("alternative_model", temp_config.get("physical_model"))
            temp_config["fitting_strategy"] = alternative.get("fitting_strategy", temp_config.get("fitting_strategy"))
            temp_config["parameters_to_extract"] = alternative.get("parameters_to_extract", temp_config.get("parameters_to_extract", []))
            
            original_config = state.get("locked_fitting_config")
            state["locked_fitting_config"] = temp_config
            
            alt_result = self._fit_single_spectrum(
                state=state, curve_data=curve_data, data_path=data_path,
                spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
            )
            
            state["locked_fitting_config"] = original_config
            
            if alt_result["success"]:
                alt_r2 = alt_result.get("fit_quality", {}).get("r_squared", 0)
                all_attempts.append({
                    "model": alternative.get("alternative_model", f"Alternative {retry + 1}"),
                    "r2": alt_r2, "result": alt_result, "config": temp_config
                })
                
                if alt_r2 > best_r2:
                    best_r2 = alt_r2
                    best_result = alt_result
                    best_result["_winning_config"] = temp_config
                
                if alt_r2 >= self.r2_threshold:
                    self.logger.info(f"      ✅ R² = {alt_r2:.4f} (meets threshold with alternative model)")
                    if spectrum_idx == 0:
                        state["locked_fitting_config"] = temp_config
                        self.logger.info(f"      📝 Updated locked config to: {temp_config.get('physical_model')}")
                    return alt_result
                else:
                    self.logger.warning(f"      R² = {alt_r2:.4f} (still below threshold)")
            else:
                self.logger.warning(f"      Alternative model failed: {alt_result.get('error', 'Unknown')}")
                all_attempts.append({
                    "model": alternative.get("alternative_model", f"Alternative {retry + 1}"),
                    "r2": 0, "result": alt_result
                })
        
        self.logger.warning(f"      All {len(all_attempts)} attempts below threshold. Best R² = {best_r2:.4f}")
        
        if self.enable_human_feedback and spectrum_idx == 0:
            feedback_result = self._get_human_feedback_for_poor_fit(state, best_result, all_attempts)
            
            if feedback_result:
                if feedback_result.get("action") == "adjust_threshold":
                    self.r2_threshold = feedback_result["new_threshold"]
                    if best_r2 >= self.r2_threshold:
                        self.logger.info(f"      ✅ Best fit now meets adjusted threshold")
                        return best_result
                
                elif feedback_result.get("action") == "retry":
                    refined_config = self._refine_model_from_feedback(state, feedback_result["feedback"])
                    original_config = state.get("locked_fitting_config")
                    state["locked_fitting_config"] = refined_config
                    
                    human_guided_result = self._fit_single_spectrum(
                        state=state, curve_data=curve_data, data_path=data_path,
                        spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                    )
                    
                    if human_guided_result["success"]:
                        human_r2 = human_guided_result.get("fit_quality", {}).get("r_squared", 0)
                        self.logger.info(f"      Human-guided fit: R² = {human_r2:.4f}")
                        
                        if human_r2 > best_r2:
                            best_r2 = human_r2
                            best_result = human_guided_result
                            if spectrum_idx == 0:
                                state["locked_fitting_config"] = refined_config
                                self.logger.info(f"      📝 Updated locked config based on feedback")
                        else:
                            state["locked_fitting_config"] = original_config
                    else:
                        state["locked_fitting_config"] = original_config
        
        if best_result:
            best_result["quality_warning"] = f"R² = {best_r2:.4f} below threshold {self.r2_threshold}"
            best_result["attempted_models"] = [a["model"] for a in all_attempts]
            self.logger.warning(f"      ⚠️ Proceeding with best available fit (R² = {best_r2:.4f})")
            
            if spectrum_idx == 0 and best_result.get("_winning_config"):
                state["locked_fitting_config"] = best_result["_winning_config"]
                self.logger.info(f"      📝 Locked best-performing config for series")
            
            return best_result
        else:
            return {
                "index": spectrum_idx, "name": spectrum_name, "success": False,
                "error": "All fitting attempts failed", "attempts": len(all_attempts),
                "parameters": {}, "fit_quality": {},
            }

    def _detect_outliers(self, series_results: List[dict]) -> List[dict]:
        r2_values = []
        for r in series_results:
            if r["success"]:
                r2 = r.get("fit_quality", {}).get("r_squared")
                if r2 is not None:
                    r2_values.append(r2)
        
        if len(r2_values) < 3:
            return []
        
        r2_array = np.array(r2_values)
        mean_r2 = np.mean(r2_array)
        std_r2 = np.std(r2_array)
        
        flagged = []
        
        for r in series_results:
            if not r["success"]:
                flagged.append({
                    "index": r["index"], "name": r["name"], "reason": "fit_failed",
                    "r_squared": None, "series_mean": float(mean_r2), "series_std": float(std_r2),
                    "deviation_sigma": None,
                    "recommendation": "Check data quality and consider manual inspection. The fitting script failed to execute successfully."
                })
                continue
            
            r2 = r.get("fit_quality", {}).get("r_squared")
            if r2 is None:
                continue
            
            below_threshold = r2 < self.r2_threshold
            
            if std_r2 > 0.001:
                deviation_sigma = (mean_r2 - r2) / std_r2
                is_outlier = deviation_sigma > self.outlier_sigma
            else:
                deviation_sigma = 0
                is_outlier = False
            
            if below_threshold or is_outlier:
                if is_outlier and not below_threshold:
                    reason = "statistical_outlier"
                    recommendation = "Fit quality significantly worse than series average. Possible causes: phase transition, sample change, or instrument artifact. Consider detailed inspection - may indicate interesting physics."
                elif below_threshold and not is_outlier:
                    reason = "below_threshold"
                    recommendation = "Fit quality below threshold but consistent with series. The chosen model may not be optimal for this data type."
                else:
                    reason = "outlier_and_below_threshold"
                    recommendation = "Significant fit quality issue. This spectrum behaves differently from others in the series. Strongly recommend manual review - could indicate interesting physics, phase transition, or data quality issue."
                
                flagged.append({
                    "index": r["index"], "name": r["name"], "reason": reason,
                    "r_squared": float(r2), "series_mean": float(mean_r2), "series_std": float(std_r2),
                    "deviation_sigma": float(deviation_sigma) if deviation_sigma else None,
                    "recommendation": recommendation
                })
        
        return flagged

    def _generate_outlier_report(self, flagged: List[dict], series_results: List[dict]) -> str:
        if not flagged:
            return ""
        
        lines = ["", "=" * 70, "⚠️  FLAGGED SPECTRA - REQUIRE ATTENTION", "=" * 70, ""]
        
        total = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        r2_values = [r.get("fit_quality", {}).get("r_squared", 0) for r in series_results if r["success"]]
        
        if r2_values:
            lines.append(f"Series statistics: {successful}/{total} successful fits")
            lines.append(f"R² range: {min(r2_values):.4f} - {max(r2_values):.4f}")
            lines.append(f"R² mean ± std: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
            lines.append(f"Quality threshold: {self.r2_threshold}")
            lines.append(f"Outlier detection: {self.outlier_sigma}σ below mean")
            lines.append("")
        
        by_reason = {}
        for f in flagged:
            reason = f["reason"]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(f)
        
        reason_labels = {
            "fit_failed": "❌ Failed Fits",
            "statistical_outlier": "📊 Statistical Outliers (possible interesting physics)",
            "below_threshold": "⚠️ Below Threshold",
            "outlier_and_below_threshold": "🔴 Critical: Outlier + Below Threshold"
        }
        
        for reason, items in by_reason.items():
            lines.append(f"\n{reason_labels.get(reason, reason)} ({len(items)} spectra):")
            lines.append("-" * 50)
            
            for f in items:
                lines.append(f"  • {f['name']} (index {f['index']})")
                if f["r_squared"] is not None:
                    lines.append(f"    R² = {f['r_squared']:.4f} (series mean: {f['series_mean']:.4f})")
                    if f["deviation_sigma"] is not None:
                        lines.append(f"    Deviation: {f['deviation_sigma']:.1f}σ below mean")
                lines.append(f"    → {f['recommendation']}")
                lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        mode_str = "SINGLE SPECTRUM" if is_single else f"SERIES ({num_spectra} spectra)"
        self.logger.info(f"\n\n⚙️ --- FITTING: {mode_str} --- ⚙️\n")
        self.logger.info(f"   R² threshold: {self.r2_threshold}")
        self.logger.info(f"   Max model retries: {self.max_model_retries}")
        if not is_single:
            self.logger.info(f"   Outlier detection: {self.outlier_sigma}σ")
        
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
            
            if idx == 0:
                result = self._fit_with_quality_control(
                    state=state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=spectrum_name, spectrum_idx=idx,
                )
                
                if result["success"] and result.get("script"):
                    base_script = result["script"]
                    state["base_fitting_script"] = base_script
                    self.logger.info("   📝 Base fitting script locked for series.")
            else:
                result = self._fit_single_spectrum(
                    state=state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=spectrum_name, spectrum_idx=idx, base_script=base_script
                )
            
            series_results.append(result)
            
            if result["success"]:
                r2 = result.get("fit_quality", {}).get("r_squared")
                r2_str = f"R²: {r2:.4f}" if r2 else "R²: N/A"
                self.logger.info(f"      ✅ {result.get('model_type', 'Fit')} - {r2_str}")
            else:
                self.logger.error(f"      ❌ Failed: {result.get('error', 'Unknown')[:50]}")
        
        flagged_spectra = []
        if num_spectra > 1:
            flagged_spectra = self._detect_outliers(series_results)
            
            if flagged_spectra:
                report = self._generate_outlier_report(flagged_spectra, series_results)
                self.logger.warning(report)
                
                flagged_indices = {f["index"] for f in flagged_spectra}
                for r in series_results:
                    if r["index"] in flagged_indices:
                        flag_info = next(f for f in flagged_spectra if f["index"] == r["index"])
                        r["flagged"] = True
                        r["flag_reason"] = flag_info["reason"]
                        r["flag_recommendation"] = flag_info["recommendation"]
                        r["deviation_sigma"] = flag_info.get("deviation_sigma")
                
                flagged_report_path = self.output_dir / "flagged_spectra.json"
                with open(flagged_report_path, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "r2_threshold": self.r2_threshold,
                        "outlier_sigma": self.outlier_sigma,
                        "total_spectra": num_spectra,
                        "flagged_count": len(flagged_spectra),
                        "flagged_spectra": flagged_spectra
                    }, f, indent=2)
                
                state["flagged_spectra_path"] = str(flagged_report_path)
        
        state["series_results"] = series_results
        state["flagged_spectra"] = flagged_spectra
        
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
        flagged_count = len(flagged_spectra)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"✅ Fitting complete: {successful}/{num_spectra} successful")
        if flagged_count > 0:
            self.logger.warning(f"⚠️  Flagged for review: {flagged_count} spectra")
        self.logger.info(f"{'='*60}\n")
        
        results_path = self.output_dir / "series_fit_results.json"
        with open(results_path, 'w') as f:
            serializable_results = []
            for r in series_results:
                r_copy = {k: v for k, v in r.items() if k not in ("visualization_bytes", "_winning_config")}
                serializable_results.append(r_copy)
            
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_spectra": num_spectra,
                "successful": successful,
                "flagged_count": flagged_count,
                "is_single_spectrum": is_single,
                "quality_settings": {
                    "r2_threshold": self.r2_threshold,
                    "max_model_retries": self.max_model_retries,
                    "outlier_sigma": self.outlier_sigma,
                },
                "locked_config": state.get("locked_fitting_config"),
                "results": serializable_results
            }, f, indent=2, default=str)
        
        state["series_results_path"] = str(results_path)
        
        return state


class ConditionalTrendAnalysisController:
    """Generates and executes custom Python script for trend analysis. Only for n>=2."""
    
    TREND_ANALYSIS_INSTRUCTIONS = '''You are analyzing a series of fitted spectra/curves to identify trends.

**SERIES SUMMARY:**
{series_summary}

**SERIES METADATA:**
{series_metadata}

**FLAGGED SPECTRA:**
{flagged_info}

**CRITICAL REQUIREMENTS:**
1. DO NOT use plt.show() anywhere in the script - only save figures with plt.savefig()
2. DO NOT include individual spectrum fit visualizations - only create parameter trend dashboard
3. Use plt.close('all') after saving each figure to free memory

**VISUALIZATION SCOPE - TRENDS ONLY:**
Create a SINGLE dashboard figure showing how fitted PARAMETERS evolve across the series.
DO NOT recreate individual spectrum fits - those already exist separately.
The dashboard should show:
- Parameter values (y-axis) vs series variable like temperature/time/index (x-axis)
- Error bars if uncertainties are available
- Fit quality (R²) evolution
- Mark flagged spectra with distinct markers

**FIGURE REQUIREMENTS:**
- Create ONE summary dashboard figure (parameter_trends.png)
- 2x2 or 2x3 subplot layout with 4-6 most important parameters
- Clean, publication-quality appearance
- Mark flagged spectra with red X markers
- Include linear regression trend lines where appropriate
- NO plt.show() calls
- Use plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
- Call plt.close('all') at the end

**DATA EXTRACTION PATTERN:**
```python
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - REQUIRED
import matplotlib.pyplot as plt

# Load data
with open('series_fit_results.json', 'r') as f:
    data = json.load(f)

results = data['results']

# Extract series variable and parameters...
# Create figure with subplots...
# Plot parameter trends (NOT individual fits)...

plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')  # REQUIRED - prevent memory leaks and display
```

Return JSON with:
{{
    "analysis_approach": "brief description",
    "key_metrics": ["list", "of", "parameters", "tracked"],
    "flagged_handling": "how flagged spectra are marked",
    "expected_outputs": ["parameter_trends.png"],
    "script": "full python script - NO plt.show()"
}}
'''

    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings,
                 parse_fn: Callable, executor: Any, output_dir: str, max_corrections: int = 3):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.max_corrections = max_corrections

    def _generate_trend_script(self, state: dict) -> Optional[Dict]:
        series_results = state.get("series_results", [])
        series_metadata = state.get("series_metadata", {})
        flagged_spectra = state.get("flagged_spectra", [])
        
        param_summary = []
        for r in series_results:
            if r["success"]:
                summary = {"index": r["index"], "name": r["name"], "model_type": r.get("model_type"),
                          "parameters": r.get("parameters", {}), "fit_quality": r.get("fit_quality", {})}
                if r.get("flagged"):
                    summary["flagged"] = True
                    summary["flag_reason"] = r.get("flag_reason")
                param_summary.append(summary)
        
        flagged_info = json.dumps(flagged_spectra, indent=2) if flagged_spectra else "No spectra were flagged."
        
        prompt = self.TREND_ANALYSIS_INSTRUCTIONS.format(
            series_summary=json.dumps(param_summary, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            flagged_info=flagged_info
        )
        
        try:
            response = self.model.generate_content(contents=[prompt], generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, error_dict = self._parse(response)
            if error_dict and not (result_json and 'script' in result_json):
                return None
            return result_json
        except Exception as e:
            self.logger.error(f"Error generating trend script: {e}")
            return None

    def _execute_script(self, script: str) -> tuple:
        # Remove any plt.show() calls that might have slipped through
        script = re.sub(r'plt\.show\s*\(\s*\)', '# plt.show() removed', script)
        
        # Ensure matplotlib backend is set at the top
        if 'matplotlib.use' not in script:
            script = "import matplotlib\nmatplotlib.use('Agg')\n" + script
        
        script_path = self.output_dir / "trend_analysis.py"
        with open(script_path, 'w') as f:
            f.write(script)
        result = self.executor.execute_script(script, working_dir=str(self.output_dir))
        return result.get("status") == "success", result.get("stdout", ""), result.get("message", "")

    def _correct_script(self, original_script: str, error_message: str, attempt: int) -> Optional[str]:
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
            response = self.model.generate_content(contents=[prompt], generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, _ = self._parse(response)
            if result_json:
                self.logger.info(f"   📋 Diagnosis: {result_json.get('diagnosis', 'N/A')}")
                return result_json.get("script")
            return None
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        if is_single or num_spectra < 2:
            self.logger.info("\n📊 Trend analysis skipped (single spectrum mode).\n")
            state["trend_analysis_results"] = {"success": True, "skipped": True, "reason": "Single spectrum - no trend analysis applicable"}
            return state
        
        self.logger.info("\n\n📈 --- TREND ANALYSIS --- 📈\n")
        
        flagged_count = len(state.get("flagged_spectra", []))
        if flagged_count > 0:
            self.logger.info(f"   Note: {flagged_count} flagged spectra will be highlighted in visualizations")
        
        script_result = self._generate_trend_script(state)
        
        if not script_result or "script" not in script_result:
            self.logger.error("Failed to generate trend analysis script.")
            state["trend_analysis_results"] = {"success": False, "error": "Script generation failed"}
            return state
        
        self.logger.info(f"   📊 Approach: {script_result.get('analysis_approach', 'unknown')}")
        self.logger.info(f"   📈 Metrics: {script_result.get('key_metrics', [])}")
        
        script = script_result["script"]
        success, stdout, stderr = False, "", ""
        
        for attempt in range(self.max_corrections + 1):
            if attempt > 0:
                self.logger.info(f"   🔄 Execution attempt {attempt + 1}")
            
            success, stdout, stderr = self._execute_script(script)
            
            if success:
                self.logger.info("   ✅ Trend analysis completed!")
                break
            
            self.logger.warning(f"   ⚠️ Script failed: {stderr[:200]}...")
            
            if attempt < self.max_corrections:
                corrected = self._correct_script(script, stderr, attempt + 1)
                if corrected:
                    script = corrected
                else:
                    break
        
        generated_files = []
        # Only include trend analysis outputs, NOT individual fit images or review files
        for f in self.output_dir.glob('*.png'):
            # Exclude individual fit visualizations (they have _fit.png suffix or spectrum_ prefix with _fit)
            fname = f.name
            if '_fit.png' in fname:
                continue
            if fname.startswith('spectrum_') and fname.endswith('.png'):
                continue
            # Exclude other known non-trend files
            if fname in ['quality_review_fit.png', 'first_spectrum_fit_review.png']:
                continue
            generated_files.append(str(f))
        
        # Include any CSV/JSON outputs from trend analysis
        for f in self.output_dir.glob('*.csv'):
            if f.name not in ['series_fit_results.json', 'flagged_spectra.json']:
                generated_files.append(str(f))
        
        state["trend_analysis_results"] = {
            "success": success, "skipped": False,
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics"),
            "flagged_handling": script_result.get("flagged_handling"),
            "stdout": stdout, "stderr": stderr if not success else None,
            "generated_files": generated_files,
            "script_path": str(self.output_dir / "trend_analysis.py")
        }
        
        return state


class UnifiedCurveSynthesisController:
    """Synthesizes findings into scientific claims. Adapts to single vs series."""
    
    SERIES_SYNTHESIS_INSTRUCTIONS = '''You are synthesizing findings from a curve fitting analysis of a spectral series.

**SERIES OVERVIEW:**
- Total spectra: {num_spectra}
- Successful fits: {successful_fits}
- Fitting model: {model_type}
- Flagged spectra: {flagged_count}

**INDIVIDUAL FIT SUMMARIES:**
{fit_summaries}

**FLAGGED SPECTRA (require attention):**
{flagged_summary}

**TREND ANALYSIS RESULTS:**
{trend_results}

**SERIES METADATA:**
{series_metadata}

**SYSTEM INFORMATION:**
{system_info}

Provide comprehensive scientific synthesis including:
1. Overall quality assessment
2. Key trends in fitted parameters
3. Physical interpretation of parameter evolution
4. **Analysis of flagged spectra** - what might explain why these fit poorly?
5. Scientific claims supported by the data
6. Caveats and limitations

Return JSON with:
{{
    "detailed_analysis": "comprehensive scientific interpretation",
    "scientific_claims": [
        {{
            "claim": "specific claim statement",
            "scientific_impact": "why this matters",
            "has_anyone_question": "research question formulation",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "parameter_trends": {{
        "parameter_name": {{"trend": "increasing/decreasing/stable", "interpretation": "physical meaning"}}
    }},
    "flagged_spectra_analysis": {{
        "summary": "interpretation of why spectra were flagged",
        "possible_causes": ["list of explanations"],
        "recommended_followup": ["suggested investigations"],
        "scientific_significance": "whether outliers represent interesting physics"
    }},
    "caveats": "limitations and considerations"
}}
'''

    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings,
                 parse_fn: Callable, single_spectrum_instructions: str, output_dir: str):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.single_spectrum_instructions = single_spectrum_instructions
        self.output_dir = Path(output_dir)

    def _synthesize_single_spectrum(self, state: dict) -> dict:
        self.logger.info("\n\n🔬 --- SINGLE SPECTRUM INTERPRETATION --- 🔬\n")
        
        fit_results = state.get("fit_results", {})
        series_results = state.get("series_results", [])
        
        quality_warning = None
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]
        
        formatted = self.single_spectrum_instructions.format(
            model_type=fit_results.get("model_type", "Curve fit"),
            summary=fit_results.get("summary", "Fitting complete"),
        )
        
        prompt_parts = [formatted, "\n## Original Data", {"mime_type": "image/png", "data": state["original_plot_bytes"]}]
        
        if state.get("final_plot_bytes"):
            prompt_parts.extend(["\n## Fit Result", {"mime_type": "image/png", "data": state["final_plot_bytes"]}])
        
        prompt_parts.extend([
            "\n## Parameters\n" + json.dumps(fit_results.get("parameters", {}), indent=2),
            "\n## Fit Quality\n" + json.dumps(fit_results.get("fit_quality", {}), indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ])
        
        if quality_warning:
            prompt_parts.append(f"\n## Quality Warning\n{quality_warning}\nNote: Alternative models were attempted but this was the best fit achieved.")
        
        if state.get("literature_context"):
            prompt_parts.extend(["\n## Literature", state["literature_context"]])
        
        try:
            response = self.model.generate_content(contents=prompt_parts, generation_config=self.generation_config, safety_settings=self.safety_settings)
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
        self.logger.info("\n\n🔬 --- SERIES SYNTHESIS --- 🔬\n")
        
        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        flagged_spectra = state.get("flagged_spectra", [])
        
        successful_fits = [r for r in series_results if r["success"]]
        
        fit_summaries = []
        for r in successful_fits[:15]:
            summary = {"index": r["index"], "name": r["name"], "model": r.get("model_type"),
                      "key_params": r.get("parameters", {}), "r_squared": r.get("fit_quality", {}).get("r_squared")}
            if r.get("flagged"):
                summary["flagged"] = True
                summary["flag_reason"] = r.get("flag_reason")
            fit_summaries.append(summary)
        
        flagged_summary = json.dumps(flagged_spectra, indent=2) if flagged_spectra else "No spectra were flagged."
        model_type = successful_fits[0].get("model_type") if successful_fits else "Unknown"
        
        prompt = self.SERIES_SYNTHESIS_INSTRUCTIONS.format(
            num_spectra=state.get("num_spectra", 1),
            successful_fits=len(successful_fits),
            model_type=model_type,
            flagged_count=len(flagged_spectra),
            fit_summaries=json.dumps(fit_summaries, indent=2),
            flagged_summary=flagged_summary,
            trend_results=json.dumps(trend_results, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            system_info=json.dumps(state.get("system_info", {}), indent=2)
        )
        
        prompt_parts = [prompt]
        
        if flagged_spectra:
            prompt_parts.append("\n\n**FLAGGED SPECTRA VISUALIZATIONS:**")
            flagged_indices = {f["index"] for f in flagged_spectra}
            included_count = 0
            for r in series_results:
                if r["index"] in flagged_indices and r.get("visualization_bytes") and included_count < 5:
                    prompt_parts.append(f"\n{r['name']} (flagged: {r.get('flag_reason', 'unknown')}):")
                    prompt_parts.append({"mime_type": "image/png", "data": r["visualization_bytes"]})
                    included_count += 1
        
        if trend_results.get("success") and trend_results.get("generated_files"):
            prompt_parts.append("\n\n**TREND VISUALIZATIONS:**")
            for file_path in trend_results["generated_files"][:5]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        prompt_parts.append(f"\n{Path(file_path).name}:")
                        prompt_parts.append({"mime_type": "image/png", "data": f.read()})
        
        try:
            response = self.model.generate_content(contents=prompt_parts, generation_config=self.generation_config, safety_settings=self.safety_settings)
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
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            return self._synthesize_single_spectrum(state)
        else:
            return self._synthesize_series(state)


class UnifiedCurveReportController:
    """Generates final HTML report for series analysis. Only for n>=2."""
    
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = Path(output_dir)

    def _image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def _generate_flagged_spectra_section(self, flagged_spectra: List[dict], series_results: List[dict], synthesis: dict) -> str:
        if not flagged_spectra:
            return ""
        
        flagged_analysis = synthesis.get("flagged_spectra_analysis", {})
        
        html = f"""
        <h2>⚠️ Flagged Spectra</h2>
        <div class="flagged-summary">
            <p><strong>{len(flagged_spectra)} spectra flagged for review</strong></p>
            <p>{flagged_analysis.get("summary", "Some spectra showed anomalous fitting behavior.")}</p>
        </div>
"""
        
        causes = flagged_analysis.get("possible_causes", [])
        if causes:
            html += "<h3>Possible Causes</h3><ul>"
            for cause in causes:
                html += f"<li>{cause}</li>"
            html += "</ul>"
        
        followup = flagged_analysis.get("recommended_followup", [])
        if followup:
            html += "<h3>Recommended Follow-up</h3><ul>"
            for item in followup:
                html += f"<li>{item}</li>"
            html += "</ul>"
        
        significance = flagged_analysis.get("scientific_significance", "")
        if significance:
            html += f"<h3>Scientific Significance</h3><p>{significance}</p>"
        
        html += '<h3>Flagged Spectra Details</h3><div class="flagged-grid">'
        
        badge_colors = {
            "fit_failed": ("#dc3545", "Failed"),
            "statistical_outlier": ("#fd7e14", "Outlier"),
            "below_threshold": ("#ffc107", "Low R²"),
            "outlier_and_below_threshold": ("#dc3545", "Critical"),
        }
        
        for f in flagged_spectra:
            result = next((r for r in series_results if r["index"] == f["index"]), None)
            color, label = badge_colors.get(f["reason"], ("#6c757d", "Flagged"))
            
            html += f'<div class="flagged-card" style="border-color: {color};">'
            html += f'<div class="flagged-card-header"><strong>{f["name"]}</strong>'
            html += f'<span class="flagged-badge" style="background-color: {color};">{label}</span></div>'
            
            if f.get("r_squared") is not None:
                html += f'<p><strong>R²:</strong> {f["r_squared"]:.4f} (series mean: {f["series_mean"]:.4f})</p>'
                if f.get("deviation_sigma") is not None:
                    html += f'<p><strong>Deviation:</strong> {f["deviation_sigma"]:.1f}σ below mean</p>'
            
            html += f'<p class="flagged-recommendation">{f["recommendation"]}</p>'
            
            if result and result.get("visualization_path") and Path(result["visualization_path"]).exists():
                with open(result["visualization_path"], 'rb') as img_f:
                    b64 = self._image_to_base64(img_f.read())
                html += f'<img src="data:image/png;base64,{b64}" alt="{f["name"]}">'
            
            html += '</div>'
        
        html += '</div>'
        return html

    def _generate_individual_fits_section(self, series_results: List[dict], num_spectra: int) -> str:
        results_with_viz = [(i, r) for i, r in enumerate(series_results) 
                           if r.get("visualization_path") and Path(r["visualization_path"]).exists()]
        
        if not results_with_viz:
            return ""
        
        failed_indices = {i for i, r in enumerate(series_results) if not r["success"]}
        flagged_indices = {i for i, r in enumerate(series_results) if r.get("flagged")}
        priority_indices = failed_indices | flagged_indices
        
        if num_spectra <= 10:
            indices_to_show = set(range(num_spectra))
            section_note = ""
        elif num_spectra <= 30:
            indices_to_show = set(range(min(3, num_spectra))) | set(range(max(0, num_spectra - 3), num_spectra))
            if num_spectra > 6:
                step = (num_spectra - 6) // 5
                for i in range(3, num_spectra - 3, max(1, step)):
                    if len(indices_to_show) < 10:
                        indices_to_show.add(i)
            indices_to_show.update(priority_indices)
            not_shown = num_spectra - len(indices_to_show)
            section_note = f"<p><em>Showing {len(indices_to_show)} of {num_spectra} fits. {not_shown} fits not displayed.</em></p>"
        else:
            indices_to_show = {0, 1, num_spectra - 2, num_spectra - 1}
            indices_to_show.update(list(priority_indices)[:10])
            section_note = f"<p><em>Large series ({num_spectra} spectra): Showing boundary fits and flagged/failed spectra.</em></p>"
        
        indices_to_show = sorted(indices_to_show)
        
        html = f"\n        <h2>Individual Fit Results</h2>\n{section_note}"
        html += '        <div class="image-grid" style="grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));">\n'
        
        for idx in indices_to_show:
            if idx >= len(series_results):
                continue
            r = series_results[idx]
            viz_path = r.get("visualization_path")
            
            if viz_path and Path(viz_path).exists():
                with open(viz_path, 'rb') as f:
                    b64 = self._image_to_base64(f.read())
                
                if not r["success"]:
                    status, status_color = "✗ FAILED", "#e74c3c"
                elif r.get("flagged"):
                    status, status_color = f"⚠ {r.get('flag_reason', 'Flagged')}", "#fd7e14"
                else:
                    status, status_color = "✓", "#27ae60"
                
                r_squared = r.get("fit_quality", {}).get("r_squared", 0)
                r2_str = f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else ""
                
                html += f'''
            <div class="image-card" style="border-left: 4px solid {status_color};">
                <img src="data:image/png;base64,{b64}" alt="{r['name']}">
                <div style="margin-top: 8px;">
                    <strong>{r['name']}</strong><br>
                    <span style="color: {status_color};">{status}</span> {r2_str}
                </div>
            </div>
'''
        
        html += "        </div>\n"
        return html

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            self.logger.info("\n📄 Single spectrum report handled by standard controller.\n")
            return state
        
        self._generate_series_report(state)
        return state

    def _generate_series_report(self, state: dict) -> None:
        self.logger.info("\n📄 --- GENERATING SERIES REPORT --- 📄\n")
        
        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        synthesis = state.get("synthesis_result", {})
        series_metadata = state.get("series_metadata", {})
        locked_config = state.get("locked_fitting_config", {})
        flagged_spectra = state.get("flagged_spectra", [])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_spectra = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        flagged_count = len(flagged_spectra)
        
        # Quality status indicator
        if flagged_count == 0:
            quality_indicator = '<span class="quality-indicator quality-good">✓ All fits acceptable</span>'
        elif flagged_count <= num_spectra * 0.1:
            quality_indicator = f'<span class="quality-indicator quality-warning">⚠ {flagged_count} spectra flagged</span>'
        else:
            quality_indicator = f'<span class="quality-indicator quality-critical">⚠ {flagged_count} spectra flagged ({100*flagged_count/num_spectra:.0f}%)</span>'
        
        # Build trend visualizations HTML
        trend_viz_html = ""
        if trend_results.get("success") and trend_results.get("generated_files"):
            trend_viz_html = '<h2>3. Trend Visualizations</h2><div class="image-grid">'
            for file_path in trend_results["generated_files"]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        b64 = self._image_to_base64(f.read())
                    name = Path(file_path).stem.replace('_', ' ').title()
                    trend_viz_html += f'<div class="image-card"><img src="data:image/png;base64,{b64}" alt="{name}"><div class="image-label">{name}</div></div>'
            trend_viz_html += '</div>'
        
        # Parameter trends HTML
        param_trends_html = ""
        param_trends = synthesis.get('parameter_trends', {})
        if param_trends:
            param_trends_html = "<h2>2. Parameter Trends</h2>"
            for param_name, trend_info in param_trends.items():
                if isinstance(trend_info, dict):
                    param_trends_html += f'<div class="trend-card"><strong>{param_name}</strong><br>Trend: {trend_info.get("trend", "N/A")}<br><em>{trend_info.get("interpretation", "")}</em></div>'
        
        # Scientific claims HTML
        claims_html = ""
        scientific_claims = synthesis.get('scientific_claims', [])
        if scientific_claims:
            claims_html = "<h2>5. Scientific Claims</h2>"
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                claims_html += f'''<div class="claim-card">
            <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
            <p><strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
            <p><strong>Research Question:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
            <p><strong>Keywords:</strong> {keywords_str}</p>
        </div>'''
        
        # Caveats HTML
        caveats_html = ""
        caveats = synthesis.get('caveats', '')
        if caveats:
            caveats_html = f'<h2>6. Caveats & Limitations</h2><div class="caveats">{caveats}</div>'
        
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
        h3 {{ color: #16a085; margin-top: 20px; }}
        .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
        .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }}
        .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
        .trend-card {{ background-color: #fef9e7; border-left: 5px solid #f39c12; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 25px; margin-top: 20px; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; }}
        .image-label {{ margin-top: 12px; font-weight: bold; color: #444; }}
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
        .quality-indicator {{ display: inline-block; padding: 5px 12px; border-radius: 15px; font-weight: bold; font-size: 0.9em; }}
        .quality-good {{ background-color: #d4edda; color: #155724; }}
        .quality-warning {{ background-color: #fff3cd; color: #856404; }}
        .quality-critical {{ background-color: #f8d7da; color: #721c24; }}
        .flagged-summary {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin-bottom: 20px; border-radius: 0 5px 5px 0; }}
        .flagged-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-top: 15px; }}
        .flagged-card {{ background: white; border: 2px solid #ffc107; border-radius: 8px; padding: 15px; }}
        .flagged-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .flagged-badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.85em; color: white; }}
        .flagged-card img {{ max-width: 100%; margin-top: 10px; border-radius: 4px; }}
        .flagged-recommendation {{ margin: 10px 0; font-size: 0.9em; color: #666; }}
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
            <p><strong>Quality Status:</strong> {quality_indicator}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <div class="analysis-text">{synthesis.get('detailed_analysis', 'No analysis available.')}</div>
        {param_trends_html}
        {trend_viz_html}
        {self._generate_individual_fits_section(series_results, num_spectra)}
        {self._generate_flagged_spectra_section(flagged_spectra, series_results, synthesis) if flagged_spectra else ''}
        {claims_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Curve Fitting Series Analysis Agent</div>
    </div>
</body>
</html>"""

        report_path = self.output_dir / "series_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        state["report_path"] = str(report_path)
        self.logger.info(f"   ✅ Report saved: {report_path}")