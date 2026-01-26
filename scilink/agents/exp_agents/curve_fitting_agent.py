# curve_fitting_agent.py

"""
CurveFittingAgent - LLM-driven spectroscopic curve fitting.
"""

import os
import logging
from pathlib import Path

from .base_agent import BaseAnalysisAgent
from .human_feedback import SimpleFeedbackMixin
from ...executors import ScriptExecutor
from ..lit_agents.literature_agent import FittingModelLiteratureAgent
from .preprocess import CurvePreprocessingAgent
from .pipelines.curve_fitting_pipelines import create_curve_fitting_pipeline
from ...tools.curve_fitting_tools import load_curve_data, plot_curve_to_bytes
from ._deprecation import normalize_params


logger = logging.getLogger(__name__)


class CurveFittingAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for spectroscopic curve fitting.

    Args:
        api_key: LLM API key
        model_name: LLM model name
        base_url: LLM API base URL
        output_dir: Output directory
        futurehouse_api_key: FutureHouse API key for literature
        use_literature: Enable literature search (default: False)
        run_preprocessing: Enable data preprocessing
        enable_human_feedback: Enable feedback loop
        executor_timeout: Script timeout in seconds

    Example:
        agent = CurveFittingAgent(api_key="...", use_literature=True)
        result = agent.analyze(
            "spectrum.csv",
            system_info={"sample": "TiO2"},
            hints="Focus on the band gap"
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: str | None = None,
        output_dir: str = "curve_analysis_output",
        # Deprecated
        google_api_key: str | None = None,
        local_model: str | None = None,
        # Agent config
        futurehouse_api_key: str | None = None,
        use_literature: bool = False,
        run_preprocessing: bool = True,
        enable_human_feedback: bool = True,
        executor_timeout: int = 60,
        max_wait_time: int = 1000,
        **kwargs,
    ):
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="CurveFittingAgent"
        )

        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
        )

        self.agent_type = "curve_fitting"
        self.use_literature = use_literature
        self.output_dir = Path(self.output_dir).resolve()

        self.executor = ScriptExecutor(timeout=executor_timeout, enforce_sandbox=False)

        # Optional preprocessor
        self.run_preprocessing = run_preprocessing
        self.preprocessor = None
        if run_preprocessing:
            self.preprocessor = CurvePreprocessingAgent(
                api_key=self.api_key,
                model_name=model_name,
                base_url=self.base_url,
                output_dir=os.path.join(self.output_dir, "preprocessing"),
                executor_timeout=executor_timeout,
            )

        # Optional literature agent
        self.literature_agent = None
        if use_literature:
            lit_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")
            if lit_key:
                try:
                    self.literature_agent = FittingModelLiteratureAgent(
                        api_key=lit_key, max_wait_time=max_wait_time
                    )
                    logger.info("Literature agent initialized")
                except Exception as e:
                    logger.error(f"Literature agent failed: {e}")
            else:
                logger.warning("use_literature=True but no API key provided")

        # Create pipeline
        self.pipeline = create_curve_fitting_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            plot_fn=plot_curve_to_bytes,
            executor=self.executor,
            output_dir=str(self.output_dir),
            preprocessor=self.preprocessor,
            literature_agent=self.literature_agent,
            enable_human_feedback=enable_human_feedback
        )

    def _run_analysis_pipeline(
        self, data_path: str, system_info: dict, analysis_hints: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """Run the analysis pipeline."""
        processed_data_path = None

        try:
            self.logger.info(f"--- Starting analysis: {data_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)

            curve_data = load_curve_data(data_path)

            state = {
                "data_path": data_path,
                "system_info": system_info,
                "curve_data": curve_data,
                "analysis_hints": analysis_hints,
                "analysis_images": [],
                "result_json": {},
                "error_dict": None,
            }

            for controller in self.pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Failed at {controller.__class__.__name__}")
                    break

            processed_data_path = state.get("processed_data_path")
            self.logger.info("--- Analysis complete ---")
            return state.get("result_json"), state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            return None, {"error": "File not found", "details": data_path}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"Unexpected error: {e}")
            return None, {"error": "Unexpected error", "details": str(e)}
        finally:
            if processed_data_path and os.path.exists(processed_data_path):
                try:
                    os.remove(processed_data_path)
                except Exception:
                    pass

    def analyze(
        self,
        data_path: str,
        system_info: dict | None = None,
        hints: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Analyze spectroscopic data.

        Args:
            data_path: Path to data file (.npy, .csv, .txt)
            system_info: Sample/experiment metadata
            hints: Optional guidance for the analysis

        Returns:
            Dict with status, model_type, fitting_parameters, fit_quality,
            detailed_analysis, scientific_claims, literature_files
        """
        self._init_state(data_path=data_path, system_info=system_info)
        self.logger.info(f"Analyzing: {data_path}")

        result_json, error_dict = self._run_analysis_pipeline(
            data_path, system_info or {}, analysis_hints=hints
        )

        if error_dict:
            self._log_action("curve_fit", {"data": data_path}, {"error": error_dict})
            return {"status": "error", "message": error_dict.get("error"), "details": error_dict}

        if not result_json:
            return {"status": "error", "message": "No results returned"}

        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis"),
            "scientific_claims": self._validate_scientific_claims(
                result_json.get("scientific_claims", [])
            ),
            "fitting_parameters": result_json.get("fitting_parameters"),
            "literature_files": result_json.get("literature_files"),
        }

        final_result = self._apply_feedback_if_enabled(initial_result, system_info=system_info)

        final_result["status"] = "success"
        final_result["model_type"] = result_json.get("model_type")
        final_result["fitting_parameters"] = result_json.get("fitting_parameters")
        final_result["fit_quality"] = result_json.get("fit_quality")
        final_result["literature_files"] = result_json.get("literature_files")

        self._log_action(
            "curve_fit",
            {"data": data_path},
            final_result,
            rationale=f"Model: {result_json.get('model_type', 'unknown')}",
        )

        return final_result

    def analyze_for_claims(self, data_path: str, system_info: dict | None = None, **kwargs) -> dict:
        """Backwards-compatible alias for analyze()."""
        return self.analyze(data_path, system_info, **kwargs)

    def _get_claims_instruction_prompt(self) -> str:
        from .instruct import FITTING_INTERPRETATION_INSTRUCTIONS
        return FITTING_INTERPRETATION_INSTRUCTIONS

    def _get_measurement_recommendations_prompt(self) -> str:
        from .instruct import CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
        return CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS