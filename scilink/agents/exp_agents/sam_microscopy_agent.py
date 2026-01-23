"""
SAM Microscopy Analysis Agent - Unified Batch-First Architecture

This module implements a unified SAM analysis agent where ALL analysis
follows the batch processing pattern. Single image analysis is simply
a batch of 1.

Key Design Principles:
1. Single image = Batch of 1 (same pipeline, same controllers)
2. Model caching for all cases (loads SAM once per analysis call)
3. Consistent output structure regardless of batch size
4. Human feedback loop available for all cases
5. Conditional trend analysis (skipped for n < 2)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import json
from pathlib import Path
import numpy as np
from typing import List, Optional, Union

from .base_agent import BaseAnalysisAgent
from .human_feedback import SimpleFeedbackMixin
from .instruct import (
    SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .pipelines.sam_pipelines import create_unified_sam_pipeline
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
    Unified Segment Anything Model Analysis Agent.
    
    ALL analysis follows the batch processing pattern:
    - Single image analysis = batch of 1
    - Multiple images = standard batch processing
    - Numpy array stack = batch processing
    
    This eliminates code duplication and ensures consistent behavior.
    
    Features:
    - Human-in-the-loop parameter refinement (optional)
    - SAM model caching (loaded once per analysis call)
    - Conditional trend analysis (for n >= 2 images)
    - Consistent output structure for all cases
    - HTML report generation
    
    Configuration (`sam_settings`):
    ---------------------------------
    - SAM_ENABLED (bool): Master switch.
    - enable_human_feedback (bool): Enable interactive parameter refinement.
    - max_feedback_iterations (int): Max refinement iterations.
    - max_script_corrections (int): Max attempts to fix custom analysis script.
    - save_visualizations (bool): Whether to save plots.
    - model_type (str): 'vit_h', 'vit_l', 'vit_b'
    - checkpoint_path (str): Path to the SAM .pth model file.
    
    Backward Compatibility:
    -----------------------
    The legacy methods `analyze_for_claims()` and `analyze_for_recommendations()`
    are preserved but now internally use the unified batch pipeline.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: str | None = None,
        google_api_key: str | None = None,
        local_model: str = None,
        sam_settings: dict | None = None,
        enable_human_feedback: bool = False,
        output_dir: str = "sam_output"
    ):
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, 
            source="SAMMicroscopyAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "sam_microscopy"
        
        # Resolve output directory
        self.output_dir = self.output_dir.resolve()
        
        # Define sub-directories
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
        
        if self.settings.get('SAM_ENABLED', True):
            self.logger.info(f"SAMMicroscopyAnalysisAgent initialized. Outputs: {self.output_dir}")
        else:
            self.logger.warning("SAMMicroscopyAnalysisAgent initialized, but 'SAM_ENABLED' is False.")
    
    def _get_initial_state_fields(self) -> dict:
        """Return initial state fields for the agent."""
        return {
            "current_image": None,
            "pipeline_type": "sam_unified",
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
    
    # =========================================================================
    # UNIFIED ANALYSIS METHOD
    # =========================================================================
    
    def analyze(
        self,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        image_stack: Optional[np.ndarray] = None,
        system_info: Optional[Union[dict, str]] = None,
        series_metadata: Optional[dict] = None,
        analysis_mode: str = "claims"
    ) -> dict:
        """
        Unified analysis method - handles single images and batches identically.
        
        Single image analysis is internally converted to a batch of 1.
        
        Args:
            image_path: Single image path (convenience parameter)
            image_paths: List of image paths
            image_stack: 3D numpy array (n, h, w)
            system_info: System/sample information
            series_metadata: Optional metadata about the series
            analysis_mode: "claims" or "recommendations" (for output formatting)
        
        Returns:
            Dictionary containing analysis results with consistent structure
        
        Examples:
            # Single image
            result = agent.analyze(image_path="sample.tif")
            
            # Multiple images
            result = agent.analyze(image_paths=["img1.tif", "img2.tif"])
            
            # Numpy stack
            result = agent.analyze(image_stack=my_stack)
        """
        # ============================================================
        # Input Normalization: Convert single image to batch of 1
        # ============================================================
        if image_path is not None:
            if image_paths is not None or image_stack is not None:
                return {"error": "Provide only one of: image_path, image_paths, or image_stack"}
            image_paths = [image_path]
            self.logger.info(f"Single image mode: treating as batch of 1")
        
        # Validate inputs
        if image_paths is None and image_stack is None:
            return {"error": "Must provide image_path, image_paths, or image_stack"}
        
        if image_paths is not None and image_stack is not None:
            return {"error": "Provide either image_paths OR image_stack, not both"}
        
        # Determine input type and count
        if image_stack is not None:
            if not isinstance(image_stack, np.ndarray):
                return {"error": "image_stack must be a numpy array"}
            if image_stack.ndim == 2:
                # Single 2D image provided as array - convert to 3D
                image_stack = image_stack[np.newaxis, :, :]
                self.logger.info("Single 2D array provided, converted to shape (1, h, w)")
            if image_stack.ndim != 3:
                return {"error": f"image_stack must be 2D or 3D, got {image_stack.ndim}D"}
            num_images = image_stack.shape[0]
            input_type = "numpy_array"
        else:
            if not image_paths:
                return {"error": "image_paths list is empty"}
            num_images = len(image_paths)
            input_type = "file_paths"
        
        is_single_image = (num_images == 1)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 SAM ANALYSIS - {num_images} image{'s' if num_images > 1 else ''}")
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
            "is_single_image": is_single_image,
            
            # Analysis mode
            "analysis_mode": analysis_mode,
            
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
            
            # Results placeholders
            "sam_result": None,
            "summary_stats": None,
            "analysis_images": [
                {"label": "Primary Microscopy Image", "data": image_bytes}
            ],
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
        # Create and Execute Unified Pipeline
        # ============================================================
        pipeline = create_unified_sam_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images
        )
        
        # Execute pipeline steps
        for i, controller in enumerate(pipeline, 1):
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
        final_results = self._compile_results(state)
        
        # Save final results JSON
        final_path = self.output_dir / "analysis_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ ANALYSIS COMPLETE")
        self.logger.info(f"   Results saved to: {final_path}")
        self.logger.info(f"{'='*80}\n")
        
        # Log action
        self._log_action(
            action="analyze",
            input_ctx={
                "num_images": num_images,
                "input_type": input_type,
                "analysis_mode": analysis_mode,
                "series_metadata": series_metadata
            },
            result=final_results.get("summary"),
            rationale="SAM analysis completed."
        )
        
        return final_results
    
    def _compile_results(self, state: dict) -> dict:
        """
        Compile results into a consistent output structure.
        
        For single images, provides backward-compatible structure.
        For batches, provides full batch results.
        """
        is_single = state.get("is_single_image", False)
        num_images = state.get("num_images", 1)
        batch_results = state.get("batch_results", [])
        
        # Common structure
        results = {
            "summary": {
                "total_images": num_images,
                "successful": sum(1 for r in batch_results if r.get("success", False)),
                "input_type": state.get("input_type"),
                "parameters_used": state.get("final_params_for_batch", state.get("current_params", {})),
                "is_single_image": is_single
            },
            "output_directory": str(self.output_dir)
        }
        
        if is_single:
            # Single image: provide simplified structure for backward compatibility
            synthesis = state.get("synthesis_result", {})
            
            results["detailed_analysis"] = synthesis.get(
                "detailed_analysis", 
                "Analysis complete."
            )
            results["scientific_claims"] = synthesis.get("scientific_claims", [])
            
            # Also include the raw data for those who want it
            if batch_results:
                results["statistics"] = batch_results[0].get("statistics", {})
                results["particle_count"] = batch_results[0].get("particle_count", 0)
                results["visualization_path"] = batch_results[0].get("visualization_path")
        else:
            # Batch: full structure
            results["individual_results"] = batch_results
            results["custom_analysis"] = state.get("custom_analysis_results", {})
            results["synthesis"] = state.get("synthesis_result", {})
        
        return results
    
    # =========================================================================
    # BACKWARD COMPATIBLE METHODS
    # =========================================================================
    
    def analyze_for_claims(
        self, 
        image_path: str, 
        system_info: Optional[Union[dict, str]] = None
    ) -> dict:
        """
        Analyze a single microscopy image to generate scientific claims.
        
        BACKWARD COMPATIBLE: This method now uses the unified pipeline internally.
        
        Args:
            image_path: Path to the image file
            system_info: System/sample information
        
        Returns:
            Dictionary with 'detailed_analysis' and 'scientific_claims'
        """
        self._init_state(current_image=image_path, system_info=system_info)
        
        result = self.analyze(
            image_path=image_path,
            system_info=system_info,
            analysis_mode="claims"
        )
        
        if "error" in result:
            self._log_action("analyze_for_claims", {"image": image_path}, {"error": result})
            return result
        
        # Extract backward-compatible structure
        claims_result = {
            "detailed_analysis": result.get("detailed_analysis", ""),
            "scientific_claims": result.get("scientific_claims", [])
        }
        
        # Validate claims
        valid_claims = self._validate_scientific_claims(claims_result.get("scientific_claims", []))
        claims_result["scientific_claims"] = valid_claims
        
        self._log_action(
            action="analyze_for_claims",
            input_ctx={"image": image_path, "system_info": system_info},
            result=claims_result,
            rationale="SAM microscopy analysis completed (unified pipeline)."
        )
        
        return claims_result
    
    def analyze_for_recommendations(
        self, 
        image_path: str, 
        system_info: Optional[Union[dict, str]] = None
    ) -> dict:
        """
        Analyze a single microscopy image to generate measurement recommendations.
        
        BACKWARD COMPATIBLE: This method now uses the unified pipeline internally.
        
        Args:
            image_path: Path to the image file
            system_info: System/sample information
        
        Returns:
            Dictionary with recommendations
        """
        self._init_state(current_image=image_path, system_info=system_info)
        
        result = self.analyze(
            image_path=image_path,
            system_info=system_info,
            analysis_mode="recommendations"
        )
        
        if "error" in result:
            self._log_action("analyze_for_recommendations", {"image": image_path}, {"error": result})
            return result
        
        self._log_action(
            action="analyze_for_recommendations",
            input_ctx={"image": image_path, "system_info": system_info},
            result=result,
            rationale="SAM measurement recommendations completed (unified pipeline)."
        )
        
        return result
    
    def analyze_image_series(
        self,
        image_paths: Optional[List[str]] = None,
        image_stack: Optional[np.ndarray] = None,
        system_info: Optional[Union[dict, str]] = None,
        series_metadata: Optional[dict] = None
    ) -> dict:
        """
        Analyze a series of images.
        
        BACKWARD COMPATIBLE: Delegates to unified analyze() method.
        
        Args:
            image_paths: List of paths to images
            image_stack: 3D numpy array (n, h, w)
            system_info: System/sample information
            series_metadata: Optional metadata about the series
        
        Returns:
            Dictionary containing batch results
        """
        return self.analyze(
            image_paths=image_paths,
            image_stack=image_stack,
            system_info=system_info,
            series_metadata=series_metadata,
            analysis_mode="claims"
        )
    
    # =========================================================================
    # INSTRUCTION PROMPTS (for pipeline compatibility)
    # =========================================================================
    
    def _get_claims_instruction_prompt(self) -> str:
        return SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS