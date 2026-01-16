import os
import logging

from .base_agent import BaseAnalysisAgent
from .human_feedback import SimpleFeedbackMixin
from ...executors import ScriptExecutor
from ..lit_agents.literature_agent import FittingModelLiteratureAgent
from .instruct import (
    FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS,
    CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)
from .preprocess import CurvePreprocessingAgent
from .pipelines.curve_fitting_pipelines import create_curve_fitting_pipeline
from ...tools.curve_fitting_tools import load_curve_data
from ._deprecation import normalize_params


logger = logging.getLogger(__name__)

class CurveFittingAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing 1D curves via an automated, modular,
    literature-informed fitting pipeline.
    """

    def __init__(self, 
                 # New standard params
                 api_key: str | None = None,
                 model_name: str = "gemini-3-pro-preview", 
                 base_url: str | None = None,
                 output_dir: str = "curve_analysis_output",
                 # Deprecated params
                 google_api_key: str | None = None, 
                 local_model: str = None,
                 # Agent specific params
                 futurehouse_api_key: str = None,  
                 run_preprocessing: bool = True,
                 enable_human_feedback: bool = True, 
                 executor_timeout: int = 60,  
                 max_wait_time: int = 1000, 
                 **kwargs):
        
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="CurveFittingAgent"
        )
        
        # Initialize Base
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )

        self.agent_type = "curve_fitting"

        self.output_dir = self.output_dir.resolve()
        
        # --- Dependency Initialization ---
        self.executor = ScriptExecutor(timeout=executor_timeout, enforce_sandbox=False)
        self.literature_agent = None
        effective_api_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")

        self.run_preprocessing = run_preprocessing
        if self.run_preprocessing:  
            # Pass normalized params to preprocessor sub-agent
            self.preprocessor = CurvePreprocessingAgent(
                api_key=self.api_key,  
                model_name=model_name,  
                base_url=self.base_url,
                output_dir=os.path.join(self.output_dir, "preprocessing"),
                executor_timeout=executor_timeout
            )
        else:
            self.preprocessor = None

        if effective_api_key:
            try:
                self.literature_agent = FittingModelLiteratureAgent(api_key=effective_api_key, max_wait_time=max_wait_time)
                logger.info("FittingModelLiteratureAgent initialized successfully.")
            except Exception as e:
                logger.error(f"Unexpected error initializing FittingModelLiteratureAgent: {e}", exc_info=True)
                self.literature_agent = None
        else:
            logger.warning("FutureHouse API key not provided. Literature search disabled.")
            
        # --- Pipeline Initialization ---
        self.pipeline = create_curve_fitting_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config, # This is None from BaseAgent
            safety_settings=self.safety_settings,
            settings={}, 
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            preprocessor=self.preprocessor,
            literature_agent=self.literature_agent,
            executor=self.executor,
            output_dir=self.output_dir 
        )

    def _run_analysis_pipeline(
        self, 
        data_path: str, 
        system_info: dict, 
        instruction_prompt: str
    ) -> tuple[dict | None, dict | None]:
        """
        The agent's main execution engine.
        It prepares the initial state and runs the loaded pipeline.
        """
        
        # This path is stateful and must be cleaned up
        processed_data_path = None
        
        try:
            self.logger.info(f"--- Starting analysis pipeline for {data_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            # Load initial data using the new tool
            curve_data = load_curve_data(data_path)

            # Create the initial state
            state = {
                "data_path": data_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "curve_data": curve_data, # Initial data
                "analysis_images": [],
                "result_json": {}, # Initialize as dict per user request
                "error_dict": None
            }

            # Run the Pipeline
            for controller in self.pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at {controller.__class__.__name__}. Stopping.")
                    break
            
            # Store path for cleanup
            processed_data_path = state.get("processed_data_path")

            self.logger.info(f"--- Analysis pipeline finished. ---")
            return state.get("result_json"), state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Data file not found: {data_path}")
            return None, {"error": "Data file not found", "details": f"Path: {data_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"An unexpected error occurred during the analysis pipeline: {e}")
            return None, {"error": "An unexpected error occurred", "details": str(e)}
        finally:
            # Always clean up the temporary processed data file
            if processed_data_path and os.path.exists(processed_data_path):
                try:
                    os.remove(processed_data_path)
                    self.logger.debug(f"Cleaned up temporary file: {processed_data_path}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up temp file {processed_data_path}: {e}")

    def analyze_for_claims(self, data_path: str, system_info: dict = None, **kwargs) -> dict:
        
        # 1. Init
        self._init_state(data_path=data_path, system_info=system_info)
        self.logger.info(f"Starting curve analysis: {data_path}")

        # 2. Run Pipeline
        result_json, error_dict = self._run_analysis_pipeline(
            data_path, 
            system_info, 
            FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS
        )

        if error_dict:
            self._log_action("curve_fit", {"data": data_path}, {"error": error_dict})
            return {"status": "error", "message": error_dict.get("error")}
        if not result_json:
            return {"status": "error", "message": "Analysis failed unexpectedly, no results returned."}

        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis"),
            "scientific_claims": self._validate_scientific_claims(result_json.get("scientific_claims", [])),
            # Pass these through so they aren't lost during simple feedback cycles
            "fitting_parameters": result_json.get("fitting_parameters"),
            "literature_files": result_json.get("literature_files")
        }
        
        final_result = self._apply_feedback_if_enabled(initial_result, system_info=system_info)
        
        # Add back the other metadata populated by the controllers
        final_result["status"] = "success"
        final_result["fitting_parameters"] = result_json.get("fitting_parameters")
        final_result["literature_files"] = result_json.get("literature_files")

        # 5. Log Action
        model_type = "unknown"
        if final_result.get("fitting_parameters"):
             model_type = final_result["fitting_parameters"].get("model_type", "unknown")

        self._log_action(
            action="curve_fit",
            input_ctx={"data": data_path},
            result=final_result,
            rationale=f"Fitted model: {model_type}"
        )

        return final_result

    def _get_claims_instruction_prompt(self) -> str:
        return FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS