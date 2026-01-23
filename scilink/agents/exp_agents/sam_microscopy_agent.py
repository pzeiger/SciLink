"""
SAM Batch Analysis Agent with Human Feedback and Custom Analysis

This module extends the SAMMicroscopyAnalysisAgent to support:
1. Human-in-the-loop feedback for parameter refinement on the first image
2. Batch processing of image series with parameter transfer
3. LLM-driven custom code generation for trend analysis
4. Comprehensive result storage for all individual images
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import json
from pathlib import Path
import numpy as np
from typing import List

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
from ...tools.sam import (
    run_sam_analysis,
    calculate_sam_statistics,
)

class SAMMicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Segment Anything Model Analysis Agent supporting:
    - Single image analysis (backward compatible with SAMMicroscopyAnalysisAgent)
    - Human-in-the-loop parameter refinement on the first image
    - Batch processing of image series with parameter transfer
    - LLM-driven custom code generation for trend analysis
    - Comprehensive result storage for all individual images
    
    This agent is a drop-in replacement for SAMMicroscopyAnalysisAgent.
    Single image calls use the original pipeline; batch calls use extended features.
    
    Configuration (`sam_settings`):
    ---------------------------------
    - SAM_ENABLED (bool): Master switch.
    - enable_human_feedback (bool): Enable interactive parameter refinement.
    - max_feedback_iterations (int): Max refinement iterations for first image.
    - max_script_corrections (int): Max attempts to fix custom analysis script.
    - refinement_cycles (int): Number of automatic LLM-driven tuning loops.
    - save_visualizations (bool): Whether to save plots.
    - model_type (str): 'vit_h', 'vit_l', 'vit_b'
    - checkpoint_path (str): Path to the SAM .pth model file.
    """
    
    def __init__(self,
                 api_key: str | None = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: str | None = None,
                 google_api_key: str | None = None,
                 local_model: str = None,
                 sam_settings: dict | None = None,
                 enable_human_feedback: bool = False,  # Default False for backward compat
                 output_dir: str = "sam_output"):  # Match original default
        
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="SAMBatchAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "sam_microscopy"  # Keep same type for compatibility
        
        # Resolve output directory
        self.output_dir = self.output_dir.resolve()
        
        # Define sub-directories (matching original structure)
        self.viz_dir = self.output_dir / "sam_visualizations"
        self.data_dir = self.output_dir / "sam_analysis"
        self.scripts_dir = self.output_dir / "scripts"
        
        # Create directories
        for d in [self.viz_dir, self.data_dir, self.scripts_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Prepare settings
        self.settings = sam_settings if sam_settings else {}
        self.settings.setdefault('SAM_ENABLED', True)
        self.settings.setdefault('enable_human_feedback', enable_human_feedback)
        self.settings.setdefault('max_feedback_iterations', 3)
        self.settings.setdefault('max_script_corrections', 3)
        self.settings.setdefault('save_visualizations', True)
        self.settings['visualization_dir'] = str(self.viz_dir)
        self.settings['output_dir'] = str(self.data_dir)
        
        # Initialize single-image pipeline (for backward compatibility)
        self.pipeline = []
        if self.settings.get('SAM_ENABLED', True):
            self.pipeline = create_sam_pipeline(
                model=self.model,
                logger=self.logger,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                settings=self.settings,
                parse_fn=self._parse_llm_response,
                store_fn=self._store_analysis_images
            )
            self.logger.info(f"SAMBatchAnalysisAgent initialized. Outputs: {self.output_dir}")
        else:
            self.logger.warning("SAMBatchAnalysisAgent initialized, but 'SAM_ENABLED' is False.")
    
    def _get_initial_state_fields(self) -> dict:
        return {
            "current_image": None,
            "pipeline_type": "sam_batch",
            "particles_detected": 0,
            "batch_mode": False
        }
    
    def _initialize_sam_params(self) -> dict:
        """Get initial SAM parameters from settings."""
        return {
            "checkpoint_path": self.settings.get('checkpoint_path', None),
            "model_type": self.settings.get('model_type', 'vit_h'),
            "device": self.settings.get('device', 'auto'),
            "use_clahe": self.settings.get('use_clahe', False),
            "sam_parameters": self.settings.get('sam_parameters', 'default'),
            "min_area": self.settings.get('min_area', 500),
            "max_area": self.settings.get('max_area', 50000),
            "use_pruning": self.settings.get('use_pruning', True),
            "pruning_iou_threshold": self.settings.get('pruning_iou_threshold', 0.5)
        }
    
    def analyze_image_series(
        self,
        image_paths: List[str] | None = None,
        image_stack: np.ndarray | None = None,
        system_info: dict | str | None = None,
        series_metadata: dict | None = None
    ) -> dict:
        """
        Analyze a series of images with human feedback on the first image.
        
        Args:
            image_paths: List of paths to images in the series (mutually exclusive with image_stack)
            image_stack: 3D numpy array of shape (n, h, w) containing image stack (mutually exclusive with image_paths)
            system_info: System/sample information
            series_metadata: Optional metadata about the series (type, time points, etc.)
        
        Returns:
            Dictionary containing batch results, custom analysis, and synthesis
        """
        from .pipelines.sam_pipelines import create_sam_batch_pipeline
        
        # ============================================================
        # Input Validation
        # ============================================================
        if image_paths is None and image_stack is None:
            return {"error": "Must provide either image_paths or image_stack"}
        
        if image_paths is not None and image_stack is not None:
            return {"error": "Provide either image_paths OR image_stack, not both"}
        
        # Determine input type and count
        if image_stack is not None:
            if not isinstance(image_stack, np.ndarray):
                return {"error": "image_stack must be a numpy array"}
            if image_stack.ndim != 3:
                return {"error": f"image_stack must be 3D (n, h, w), got {image_stack.ndim}D"}
            num_images = image_stack.shape[0]
            input_type = "numpy_array"
            self.logger.info(f"Input: 3D numpy array with shape {image_stack.shape}")
        else:
            if not image_paths:
                return {"error": "image_paths list is empty"}
            num_images = len(image_paths)
            input_type = "file_paths"
            self.logger.info(f"Input: {num_images} file paths")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 SAM BATCH ANALYSIS - {num_images} images")
        self.logger.info(f"{'='*80}\n")
        
        # ============================================================
        # Initialize State
        # ============================================================
        
        # Load and preprocess first image for initial analysis
        if image_stack is not None:
            first_image = image_stack[0]
            first_image_name = "frame_0000"
        else:
            first_image = load_image(image_paths[0])
            first_image_name = Path(image_paths[0]).stem
        
        preprocessed_img, _ = preprocess_image(first_image)
        image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img)
        
        nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(
            self._handle_system_info(system_info), first_image.shape
        )
        
        # Build initial state dict
        state = {
            # Input data
            "image_paths": image_paths,
            "image_stack": image_stack,
            "input_type": input_type,
            "num_images": num_images,
            
            # System info
            "system_info": self._handle_system_info(system_info),
            "series_metadata": series_metadata or {},
            
            # First image (preprocessed)
            "image_path": image_paths[0] if image_paths else first_image_name,
            "first_image_name": first_image_name,
            "preprocessed_image_array": preprocessed_img,
            "image_blob": {"mime_type": "image/jpeg", "data": image_bytes},
            
            # Spatial calibration
            "nm_per_pixel": nm_per_pixel,
            "fov_in_nm": fov_in_nm,
            
            # Settings and params
            "settings": self.settings,
            "enable_human_feedback": self.settings.get('enable_human_feedback', False),
            "current_params": self._initialize_sam_params(),
            
            # Initial SAM analysis on first image
            "sam_result": None,
            "summary_stats": None,
        }
        
        # ============================================================
        # Run Initial SAM Analysis on First Image
        # ============================================================
        self.logger.info("📍 Running initial SAM analysis on first image...\n")
        
        try:
            sam_result = run_sam_analysis(preprocessed_img, params=state["current_params"])
            state["sam_result"] = sam_result
            
            summary_stats = calculate_sam_statistics(
                sam_result=sam_result,
                image_path=state["image_path"],
                preprocessed_image_shape=preprocessed_img.shape,
                nm_per_pixel=nm_per_pixel
            )
            state["summary_stats"] = summary_stats
            
            self.logger.info(f"   Initial detection: {sam_result['total_count']} particles\n")
            
        except Exception as e:
            self.logger.error(f"Initial SAM analysis failed: {e}")
            return {"error": f"Initial SAM analysis failed: {e}"}
        
        # ============================================================
        # Create and Execute Batch Pipeline
        # ============================================================
        batch_pipeline = create_sam_batch_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings,
            parse_fn=self._parse_llm_response
        )
        
        # Execute pipeline steps
        for i, controller in enumerate(batch_pipeline, 1):
            step_name = controller.__class__.__name__
            self.logger.info(f"\n📍 STEP {i}: {step_name}\n")
            
            try:
                state = controller.execute(state)
                
                # Check for errors
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {step_name}: {state['error_dict']}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Pipeline step {step_name} raised exception: {e}")
                state["error_dict"] = {"error": f"Pipeline step failed: {step_name}", "details": str(e)}
                break
        
        # ============================================================
        # Compile Final Results
        # ============================================================
        final_results = {
            "batch_summary": {
                "total_images": num_images,
                "successful": sum(1 for r in state.get("batch_results", []) if r["success"]),
                "input_type": input_type,
                "parameters_used": state.get("final_params_for_batch", {})
            },
            "individual_results": state.get("batch_results", []),
            "custom_analysis": state.get("custom_analysis_results", {}),
            "synthesis": state.get("synthesis_result", {}),
            "output_directory": str(self.output_dir)
        }
        
        # Save final results JSON
        final_path = self.output_dir / "final_batch_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ BATCH ANALYSIS COMPLETE")
        self.logger.info(f"   Results saved to: {final_path}")
        self.logger.info(f"{'='*80}\n")
        
        # Log action
        self._log_action(
            action="analyze_image_series",
            input_ctx={
                "num_images": num_images,
                "input_type": input_type,
                "series_metadata": series_metadata
            },
            result=final_results.get("batch_summary"),
            rationale="SAM batch analysis with human feedback completed."
        )
        
        return final_results
    
    def _run_single_image_pipeline(
        self,
        image_path: str,
        system_info: dict,
        instruction_prompt: str,
        additional_context: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        Run single-image analysis using the original pipeline.
        This maintains full backward compatibility with SAMMicroscopyAnalysisAgent.
        """
        if not self.pipeline:
            return None, {"error": "SAMAnalysisAgent pipeline is not configured (SAM_ENABLED=False?)."}
        
        try:
            self.logger.info(f"--- Starting single-image analysis for {image_path} ---")
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

            # Run the original pipeline
            for controller in self.pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}.")
                    break

            self.logger.info(f"--- Single-image analysis finished. ---")
            return state.get("result_json"), state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"An unexpected error occurred: {e}")
            return None, {"error": "An unexpected error occurred", "details": str(e)}

    def analyze_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze a single microscopy image to generate scientific claims.
        
        This method maintains FULL backward compatibility with SAMMicroscopyAnalysisAgent.
        It uses the original single-image pipeline, NOT the batch pipeline.
        
        For batch processing with human feedback, use analyze_image_series() instead.
        """
        # 1. Initialize State
        self._init_state(current_image=image_path, system_info=system_info)

        # 2. Run the ORIGINAL single-image pipeline
        result_json, error_dict = self._run_single_image_pipeline(
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
        
        # 3. Apply Feedback (using SimpleFeedbackMixin)
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
            rationale="SAM microscopy analysis completed."
        )

        return final_result

    def analyze_for_recommendations(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze a single microscopy image to generate measurement recommendations.
        Backward compatible with SAMMicroscopyAnalysisAgent.
        """
        self._init_state(current_image=image_path, system_info=system_info)

        result_json, error_dict = self._run_single_image_pipeline(
            image_path,
            system_info,
            SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
        )

        if error_dict:
            self._log_action("analyze_for_recommendations", {"image": image_path}, {"error": error_dict})
            return error_dict

        if result_json is None:
            return {"error": "Analysis for recommendations failed unexpectedly."}

        self._log_action(
            action="analyze_for_recommendations",
            input_ctx={"image": image_path, "system_info": system_info},
            result=result_json,
            rationale="SAM measurement recommendations completed."
        )

        return result_json
    
    def _get_claims_instruction_prompt(self) -> str:
        return SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
    
    def _run_analysis_pipeline(
        self,
        image_path: str,
        system_info: dict,
        instruction_prompt: str,
        additional_context: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        Alias for _run_single_image_pipeline for backward compatibility.
        """
        return self._run_single_image_pipeline(
            image_path, system_info, instruction_prompt, additional_context
        )


