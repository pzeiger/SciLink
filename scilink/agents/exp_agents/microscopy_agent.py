"""
Microscopy Analysis Agent - Unified Single Image + Series Support.

DROP-IN REPLACEMENT for existing MicroscopyAnalysisAgent.

BACKWARD COMPATIBILITY:
- All existing methods work exactly as before
- Same constructor signature  
- Same return formats

NEW CAPABILITIES:
- analyze_series() for time-series data
- analyze_image() unified method that auto-detects single vs series

Usage:
    # Existing code works unchanged:
    agent = MicroscopyAnalysisAgent(api_key="...")
    result = agent.analyze_for_claims("image.png", system_info)
    
    # New series capability:
    result = agent.analyze_series("path/to/series/", system_info)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Union
from datetime import datetime

from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent
from .human_feedback import SimpleFeedbackMixin
from .instruct import (
    MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    MICROSCOPY_CLAIMS_INSTRUCTIONS,
    MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .pipelines.microscopy_pipelines import create_fftnmf_pipeline
from .controllers.microscopy_controllers import (
    SeriesLoaderController,
    FirstFrameAnalysisController, 
    UserFeedbackController,
    SeriesBatchController,
    SummaryScriptController,
    ReportGenerationController
)

from ...tools.image_processor import (
    load_image, 
    preprocess_image, 
    convert_numpy_to_jpeg_bytes
)
from ._deprecation import normalize_params


class MicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """    
    Unified Microscopy Analysis Agent - Single Image + Series Support.
    
    This agent executes a modular pipeline of "controllers".
    Fully backward compatible with existing code.
    
    Configuration (`fft_nmf_settings`):
    ---------------------------------
    - FFT_NMF_ENABLED (bool): Master switch for Sliding FFT/NMF.
    - output_dir (str): Where to save NMF numpy arrays.
    - visualization_dir (str): Where to save NMF plots.
    """

    def __init__(self,
                 api_key: str | None = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: str | None = None,
                 # Deprecated params
                 google_api_key: str | None = None,
                 local_model: str = None,
                 # Agent specific params
                 fft_nmf_settings: dict | None = None,
                 enable_human_feedback: bool = True,
                 output_dir: str = "microscopy_analysis_output"):
        
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="MicroscopyAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key, 
            model_name=model_name, 
            base_url=self.base_url, 
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
        )

        self.agent_type = "microscopy"
        self.output_dir = self.output_dir.resolve()
        
        # Define sub-directories
        viz_dir = self.output_dir / "fft_nmf_visualizations"
        data_dir = self.output_dir / "analysis_output"
        
        # Agent-Specific Settings
        self.settings = fft_nmf_settings if fft_nmf_settings else {}
        self.settings['visualization_dir'] = str(viz_dir)
        self.settings['output_dir'] = str(data_dir)
        
        viz_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        self._recommendation_agent = None
        
        # Single-image pipeline (existing)
        self.pipeline = create_fftnmf_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images
        )
        
        self.logger.info(f"MicroscopyAnalysisAgent initialized with {len(self.pipeline)} controllers.")

    def _get_initial_state_fields(self) -> dict:
        return {
            "current_image": None,
            "pipeline_type": "general",
            "analysis_results": [],
            "locked_params": None
        }
    
    # =========================================================================
    # EXISTING METHODS (UNCHANGED - BACKWARD COMPATIBLE)
    # =========================================================================
    
    def _run_analysis_pipeline(
        self, 
        image_path: str, 
        system_info: dict, 
        instruction_prompt: str, 
        additional_context: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        Main execution engine for SINGLE IMAGE analysis.
        (UNCHANGED - backward compatible)
        """
        try:
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

            for controller in self.pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at {controller.__class__.__name__}")
                    break

            self.logger.info(f"--- Analysis pipeline finished. ---")
            return state.get("result_json"), state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"Unexpected error during analysis: {e}")
            return None, {"error": "An unexpected error occurred", "details": str(e)}

    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        """
        Analyze microscopy image for DFT structure recommendations.
        (UNCHANGED - backward compatible)
        """
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")
            if not self._recommendation_agent:
                self._recommendation_agent = RecommendationAgent(
                    api_key=self.api_key, 
                    model_name=self.model_name, 
                    base_url=self.base_url
                )
            return self._recommendation_agent.generate_dft_recommendations_from_text(
                cached_detailed_analysis=cached_detailed_analysis,
                additional_prompt_context=additional_prompt_context,
                system_info=system_info
            )
        
        elif image_path:
            self.logger.info("Generating DFT recommendations via modular pipeline.")
            result_json, error_dict = self._run_analysis_pipeline(
                image_path, 
                system_info, 
                MICROSCOPY_ANALYSIS_INSTRUCTIONS, 
                additional_prompt_context
            )
            
            if error_dict: return error_dict
            if result_json is None: return {"error": "Analysis failed unexpectedly."}

            recommendations = result_json.get("structure_recommendations", [])
            sorted_recs = self._validate_structure_recommendations(recommendations)
            
            if not sorted_recs:
                self.logger.warning("Pipeline ran but LLM returned no valid recommendations.")

            return {
                "analysis_summary_or_reasoning": result_json.get("detailed_analysis", "Analysis complete."), 
                "recommendations": sorted_recs
            }
        
        else:
            return {"error": "Either image_path or (cached_detailed_analysis...) must be provided."}

    def analyze_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image for scientific claims.
        (UNCHANGED - backward compatible)
        """
        self._init_state(current_image=image_path, system_info=system_info)

        result_json, error_dict = self._run_analysis_pipeline(
            image_path, 
            system_info, 
            MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict: 
            self._log_action("analyze_for_claims", {"image": image_path}, {"error": error_dict})
            return error_dict

        if result_json is None: 
            return {"error": "Analysis failed."}

        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))
        
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis complete."),
            "scientific_claims": valid_claims
        }
        
        final_result = self._apply_feedback_if_enabled(
            initial_result, 
            image_path=image_path, 
            system_info=system_info
        )

        self._log_action(
            action="analyze_for_claims",
            input_ctx={"image": image_path, "system_info": system_info},
            result=final_result,
            rationale="Standard microscopy analysis pipeline completed."
        )

        return final_result
    
    def _get_claims_instruction_prompt(self) -> str:
        return MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS

    # =========================================================================
    # NEW SERIES ANALYSIS METHODS
    # =========================================================================
    
    def analyze_series(
        self,
        series_input: Union[str, np.ndarray],
        system_info: Optional[dict] = None,
        feedback_callback: Optional[Callable] = None,
        preset_params: Optional[dict] = None,
    ) -> dict:
        """
        Analyze an image series with interactive parameter tuning.
        
        NEW METHOD - Extends agent for time-series data.
        
        Workflow:
        1. Load image series (directory, TIFF stack, or 3D array)
        2. Analyze first frame to estimate optimal parameters
        3. Present results and collect user feedback (if interactive)
        4. Apply locked parameters to process all frames
        5. Generate custom analysis script
        
        Args:
            series_input: One of:
                - Path to directory containing image files
                - Path to multi-page TIFF file
                - 3D numpy array (Time, Height, Width)
                - 2D numpy array (treated as single frame)
            
            system_info: Dictionary with metadata:
                - nm_per_pixel: Scale factor
                - acquisition_rate: Frame rate (fps)
                - sample_description: Text description
            
            feedback_callback: Optional function for custom feedback.
                Signature: callback(state: dict) -> dict with:
                    - "action": "accept" | "modify" | "cancel"
                    - "modifications": dict of param changes
            
            preset_params: Skip first-frame analysis, use these params:
                - window_size_nm: float
                - n_components: int
            
            interactive: If False, auto-accept LLM parameters
        
        Returns:
            dict with:
                - "status": "success" | "cancelled" | "error"
                - "n_frames_processed": int
                - "components": np.ndarray (n_comps, h, w)
                - "abundances": np.ndarray (Time, n_comps, grid_h, grid_w)
                - "analysis_script_path": str
                - "summary": dict
                - "locked_params": dict
                - "error": dict (if status == "error")
        """
        self.logger.info("Starting series analysis...")
        
        system_info = self._handle_system_info(system_info or {})
        
        try:
            # Choose pipeline
            if preset_params:
                self.logger.info("Using preset parameters - skipping first-frame analysis")
                
                class PresetParamsController:
                    def __init__(self, params):
                        self.params = params
                    def execute(self, state):
                        state["locked_params"] = self.params
                        state["first_frame_results"] = {"llm_params": self.params}
                        return state
                
                pipeline = [
                    SeriesLoaderController(self.logger),
                    PresetParamsController(preset_params),
                    SeriesBatchController(self.logger, self.settings),
                    SummaryScriptController(self.model, self.logger, self.generation_config, 
                        self.safety_settings, self._parse_llm_response, self.settings),
                    ReportGenerationController(
                        self.model, 
                        self.logger, 
                        self.generation_config, 
                        self.safety_settings, 
                        self._parse_llm_response, 
                        self.settings
                    )
                ]
            else:
                pipeline = [
                    SeriesLoaderController(self.logger),
                    FirstFrameAnalysisController(self.model, self.logger, self.generation_config, 
                                                  self.safety_settings, self.settings),
                    UserFeedbackController(self.logger, self.settings, feedback_callback),
                    SeriesBatchController(self.logger, self.settings),
                    SummaryScriptController(self.model, self.logger, self.generation_config, 
                        self.safety_settings, self._parse_llm_response,self.settings),
                    ReportGenerationController(
                        self.model, 
                        self.logger, 
                        self.generation_config, 
                        self.safety_settings, 
                        self._parse_llm_response, 
                        self.settings
                    )
                ]
            
            # Calculate nm_per_pixel
            nm_per_pixel = self._get_series_nm_per_pixel(system_info, series_input)
            
            # Initialize state
            state = {
                "series_input": series_input,
                "system_info": system_info,
                "nm_per_pixel": nm_per_pixel,
                "analysis_images": [],
                "error_dict": None
            }
            
            # Run pipeline
            for controller in pipeline:
                state = controller.execute(state)
                
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed: {state['error_dict']}")
                    return {"status": "error", "error": state["error_dict"]}
                
                if state.get("batch_cancelled"):
                    self.logger.info("Analysis cancelled by user.")
                    return {
                        "status": "cancelled",
                        "first_frame_results": state.get("first_frame_results")
                    }
            
            return {
                "status": "success",
                "n_frames_processed": state.get("n_frames", 0),
                "components": state.get("series_components"),
                "abundances": state.get("series_abundances"),
                "analysis_script_path": state.get("analysis_script_path"),
                "summary": state.get("analysis_summary"),
                "locked_params": state.get("locked_params"),
                "first_frame_results": state.get("first_frame_results")
            }
            
        except Exception as e:
            self.logger.exception(f"Series analysis failed: {e}")
            return {
                "status": "error",
                "error": {"message": str(e), "type": type(e).__name__}
            }
    
    def analyze_image(
        self,
        image_input: Union[str, np.ndarray],
        system_info: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        Unified analysis method - auto-detects single image vs series.
        
        NEW METHOD - Convenience wrapper.
        
        Args:
            image_input: Image path, directory, or numpy array
            system_info: Metadata dictionary
            **kwargs: Additional arguments
        
        Returns:
            Analysis results dictionary
        """
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return self._analyze_single_array(image_input, system_info)
            elif image_input.ndim == 3:
                return self.analyze_series(image_input, system_info, **kwargs)
            else:
                return {"status": "error", "error": f"Invalid array dimensions: {image_input.ndim}"}
        
        elif isinstance(image_input, str):
            if os.path.isdir(image_input):
                return self.analyze_series(image_input, system_info, **kwargs)
            elif image_input.lower().endswith(('.tif', '.tiff')):
                from skimage import io
                test_load = io.imread(image_input)
                if test_load.ndim == 3 and test_load.shape[0] > 1:
                    return self.analyze_series(image_input, system_info, **kwargs)
                else:
                    return self.analyze_for_claims(image_input, system_info)
            else:
                return self.analyze_for_claims(image_input, system_info)
        
        return {"status": "error", "error": f"Unsupported input type: {type(image_input)}"}
    
    def analyze_series_with_preset(
        self,
        series_input: Union[str, np.ndarray],
        window_size_nm: float,
        n_components: int = 4,
        system_info: Optional[dict] = None
    ) -> dict:
        """
        Analyze series with preset parameters (no feedback loop).
        
        NEW METHOD - For batch processing with known parameters.
        """
        return self.analyze_series(
            series_input=series_input,
            system_info=system_info,
            preset_params={
                "window_size_nm": window_size_nm,
                "n_components": n_components
            }
        )
    
    def load_series_results(self) -> tuple:
        """
        Load previously saved series results.
        
        NEW METHOD.
        
        Returns:
            (components, abundances) or (None, None)
        """
        data_dir = Path(self.settings['output_dir'])
        comp_path = data_dir / "series_components.npy"
        abun_path = data_dir / "series_abundances.npy"
        
        if comp_path.exists() and abun_path.exists():
            return np.load(comp_path), np.load(abun_path)
        
        self.logger.warning("No saved series results found")
        return None, None
    
    # =========================================================================
    # SERIES HELPER METHODS
    # =========================================================================
    
    def _get_series_nm_per_pixel(
        self, 
        system_info: dict, 
        series_input: Union[str, np.ndarray]
    ) -> float:
        """Get nm_per_pixel for series analysis."""
        if "nm_per_pixel" in system_info:
            return float(system_info["nm_per_pixel"])
        
        if "fov_nm" in system_info and isinstance(series_input, np.ndarray):
            shape = series_input.shape[-2:]
            return system_info["fov_nm"] / ((shape[0] + shape[1]) / 2)
        
        self.logger.warning("nm_per_pixel not specified, using 1.0")
        return 1.0
    
    def _analyze_single_array(self, array: np.ndarray, system_info) -> dict:
        """Analyze a single 2D array."""
        import tempfile
        from skimage import io
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            arr_norm = ((array - array.min()) / (array.max() - array.min() + 1e-8) * 255).astype(np.uint8)
            io.imsave(f.name, arr_norm)
            result = self.analyze_for_claims(f.name, system_info)
            os.unlink(f.name)
            return result