from .base_agent import BaseAnalysisAgent
from .human_feedback import SimpleFeedbackMixin
from .instruct import (
    SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS, 
    SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .pipelines.sam_pipelines import create_sam_pipeline
from ._deprecation import normalize_params

from ...tools.image_processor import (
    load_image, 
    preprocess_image, 
    convert_numpy_to_jpeg_bytes
)


class SAMMicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """    
    This agent executes a modular pipeline to analyze microscopy
    images using the Segment Anything Model (SAM).
    
    Configuration (`sam_settings`):
    ---------------------------------
    - SAM_ENABLED (bool): Master switch.
    - refinement_cycles (int): Number of LLM-driven tuning loops.
    - save_visualizations (bool): Whether the tool should save plots.
    - model_type (str): 'vit_h', 'vit_l', 'vit_b'
    - checkpoint_path (str): Path to the SAM .pth model file.
    - ... (other atomai ParticleAnalyzer settings)
    """

    def __init__(self,
                 api_key: str | None = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: str | None = None,
                 # Deprecated params
                 google_api_key: str | None = None,
                 local_model: str = None,
                 # Agent specific params
                 sam_settings: dict | None = None,
                 enable_human_feedback: bool = False,
                 output_dir: str = "sam_output"): # Default output directory
        
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="SAMMicroscopyAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key, 
            model_name=model_name, 
            base_url=self.base_url, 
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "sam_microscopy"

        # 1. Resolve main output directory to absolute path
        self.output_dir = self.output_dir.resolve()
        
        # 2. Define sub-directories nested INSIDE the main output_dir
        viz_dir = self.output_dir / "sam_visualizations"
        data_dir = self.output_dir / "sam_analysis"
        
        # 3. Create them immediately
        viz_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        # 4. Prepare Settings and INJECT paths
        self.settings = sam_settings if sam_settings else {}
        if 'SAM_ENABLED' not in self.settings:
            self.settings['SAM_ENABLED'] = True 
        
        # Overwrite defaults to force nesting
        self.settings['visualization_dir'] = str(viz_dir)
        self.settings['output_dir'] = str(data_dir)

        # --- Pipeline Initialization ---
        self.pipeline = []
        if self.settings.get('SAM_ENABLED', True):
            self.pipeline = create_sam_pipeline(
                model=self.model,
                logger=self.logger,
                generation_config=self.generation_config, 
                safety_settings=self.safety_settings,
                settings=self.settings, # Now contains nested paths
                parse_fn=self._parse_llm_response,
                store_fn=self._store_analysis_images
            )
            self.logger.info(f"SAMMicroscopyAnalysisAgent initialized. Outputs will be saved to: {self.output_dir}")
        else:
             self.logger.warning("SAMMicroscopyAnalysisAgent initialized, but 'SAM_ENABLED' is False.")

    def _get_initial_state_fields(self) -> dict:
        return {
            "current_image": None,
            "pipeline_type": "sam",
            "particles_detected": 0
        }

    def _run_analysis_pipeline(
        self, 
        image_path: str, 
        system_info: dict, 
        instruction_prompt: str, 
        additional_context: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        The agent's main execution engine.
        It prepares the initial state and runs the loaded pipeline.
        """
        if not self.pipeline:
             return None, {"error": "SAMAnalysisAgent pipeline is not configured (SAM_ENABLED=False?)."}
        
        try:
            # --- 1. Common State Initialization ---
            self.logger.info(f"--- Starting analysis pipeline for {image_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            loaded_image = load_image(image_path)
            nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(system_info, loaded_image.shape)
            
            preprocessed_img_array, _ = preprocess_image(loaded_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)

            state = {
                "image_path": image_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "additional_top_level_context": additional_context,
                "image_blob": {"mime_type": "image/jpeg", "data": image_bytes},
                "preprocessed_image_array": preprocessed_img_array,
                "nm_per_pixel": nm_per_pixel,
                "fov_in_nm": fov_in_nm,
                "analysis_images": [
                    {"label": "Primary Microscopy Image", "data": image_bytes}
                ],
                "result_json": None,
                "error_dict": None
            }

            # --- 2. Run the Pipeline ---
            for controller in self.pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}. Stopping execution.")
                    break

            # --- 3. Return Final Results ---
            self.logger.info(f"--- Analysis pipeline finished. ---")
            return state.get("result_json"), state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"An unexpected error occurred during the analysis pipeline: {e}")
            return None, {"error": "An unexpected error occurred", "details": str(e)}

    def analyze_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims.
        """
        # 1. Initialize State
        self._init_state(current_image=image_path, system_info=system_info)

        # 2. Run Pipeline
        result_json, error_dict = self._run_analysis_pipeline(
            image_path, 
            system_info, 
            SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict: 
            self._log_action("analyze_for_claims", {"image": image_path}, {"error": error_dict})
            return error_dict

        if result_json is None: 
            return {"error": "Analysis for claims failed unexpectedly."}

        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))
        
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis complete, but no text was returned."), 
            "scientific_claims": valid_claims
        }
        
        # 3. Apply Feedback
        final_result = self._apply_feedback_if_enabled(
            initial_result, 
            image_path=image_path, 
            system_info=system_info
        )

        # 4. Log Success
        self._log_action(
            action="analyze_for_claims",
            input_ctx={"image": image_path, "system_info": system_info},
            result=final_result,
            rationale="SAM microscopy analysis pipeline completed."
        )

        return final_result
    
    def _get_claims_instruction_prompt(self) -> str:
        return SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS