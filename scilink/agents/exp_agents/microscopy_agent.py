"""
Microscopy Analysis Agent - Unified Batch-First Architecture

This module implements a unified microscopy analysis agent where ALL analysis
follows the batch processing pattern. Single image analysis is simply
a batch of 1.

Key Design Principles:
1. Single image = Batch of 1 (same pipeline, same controllers)
2. Model caching for all cases (loads FFT/NMF analyzer once per analysis call)
3. Consistent output structure regardless of batch size
4. Human feedback loop available for all cases
5. Conditional trend analysis (skipped for n < 2)

BACKWARD COMPATIBILITY:
- All existing methods work exactly as before
- Same constructor signature  
- Same return formats

NEW CAPABILITIES:
- Unified analyze() method that auto-detects single vs series
- Consistent pipeline for all input types
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Union
from datetime import datetime

from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent
from .human_feedback import SimpleFeedbackMixin
from .instruct import (
    MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    MICROSCOPY_CLAIMS_INSTRUCTIONS,
    MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .pipelines.microscopy_pipelines import create_unified_microscopy_pipeline
from ._deprecation import normalize_params

from ...tools.image_processor import (
    load_image, 
    preprocess_image, 
    convert_numpy_to_jpeg_bytes
)


class MicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Unified Microscopy Analysis Agent.
    
    ALL analysis follows the batch processing pattern:
    - Single image analysis = batch of 1
    - Multiple images = standard batch processing
    - Numpy array stack = batch processing
    
    This eliminates code duplication and ensures consistent behavior.
    
    Features:
    - Human-in-the-loop parameter refinement (optional)
    - FFT/NMF analyzer caching (loaded once per analysis call)
    - Conditional trend analysis (for n >= 2 images)
    - Consistent output structure for all cases
    - HTML report generation
    
    Configuration (`fft_nmf_settings`):
    ---------------------------------
    - FFT_NMF_ENABLED (bool): Master switch for Sliding FFT/NMF.
    - enable_human_feedback (bool): Enable interactive parameter refinement.
    - max_feedback_iterations (int): Max refinement iterations.
    - max_script_corrections (int): Max attempts to fix custom analysis script.
    - save_visualizations (bool): Whether to save plots.
    - output_dir (str): Where to save NMF numpy arrays.
    - visualization_dir (str): Where to save NMF plots.
    
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
        # Deprecated params
        google_api_key: str | None = None,
        local_model: str = None,
        # Agent specific params
        fft_nmf_settings: dict | None = None,
        enable_human_feedback: bool = False,
        output_dir: str = "microscopy_analysis_output"
    ):
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, 
            source="MicroscopyAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "microscopy"
        
        # Resolve output directory
        self.output_dir = self.output_dir.resolve()
        
        # Define sub-directories
        self.viz_dir = self.output_dir / "fft_nmf_visualizations"
        self.data_dir = self.output_dir / "analysis_output"
        self.scripts_dir = self.output_dir / "scripts"
        
        # Create directories
        for d in [self.viz_dir, self.data_dir, self.scripts_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Prepare settings
        self.settings = fft_nmf_settings if fft_nmf_settings else {}
        self.settings.setdefault('FFT_NMF_ENABLED', True)
        self.settings.setdefault('enable_human_feedback', enable_human_feedback)
        self.settings.setdefault('max_feedback_iterations', 3)
        self.settings.setdefault('max_script_corrections', 3)
        self.settings.setdefault('save_visualizations', True)
        self.settings['visualization_dir'] = str(self.viz_dir)
        self.settings['output_dir'] = str(self.data_dir)
        
        self._recommendation_agent = None
        
        if self.settings.get('FFT_NMF_ENABLED', True):
            self.logger.info(f"MicroscopyAnalysisAgent initialized. Outputs: {self.output_dir}")
        else:
            self.logger.warning("MicroscopyAnalysisAgent initialized, but 'FFT_NMF_ENABLED' is False.")
    
    def _get_initial_state_fields(self) -> dict:
        """Return initial state fields for the agent."""
        return {
            "current_image": None,
            "pipeline_type": "microscopy_unified",
            "analysis_results": [],
            "batch_mode": False,
            "locked_params": None
        }
    
    def _initialize_fft_nmf_params(self) -> dict:
        """Get initial FFT/NMF parameters from settings."""
        return {
            "window_size_nm": self.settings.get('window_size_nm', None),  # Auto-detect
            "n_components": self.settings.get('n_components', 4),
            "use_clahe": self.settings.get('use_clahe', False),
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
        analysis_mode: str = "claims",
        preset_params: Optional[dict] = None,
        feedback_callback: Optional[Callable] = None
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
            preset_params: Skip first-frame analysis, use these params directly
            feedback_callback: Optional function for custom feedback
        
        Returns:
            Dictionary containing analysis results with consistent structure
        
        Examples:
            # Single image
            result = agent.analyze(image_path="sample.tif")
            
            # Multiple images
            result = agent.analyze(image_paths=["img1.tif", "img2.tif"])
            
            # Numpy stack
            result = agent.analyze(image_stack=my_stack)
            
            # With preset parameters (skip feedback)
            result = agent.analyze(
                image_paths=paths, 
                preset_params={"window_size_nm": 10.0, "n_components": 4}
            )
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
        self.logger.info(f"🔬 MICROSCOPY ANALYSIS - {num_images} image{'s' if num_images > 1 else ''}")
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
        
        # Normalize float images
        if first_image.dtype in [np.float32, np.float64, float]:
            frame_min, frame_max = first_image.min(), first_image.max()
            if frame_max > frame_min:
                first_image_uint8 = ((first_image - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                first_image_uint8 = np.zeros_like(first_image, dtype=np.uint8)
        else:
            first_image_uint8 = first_image
        
        preprocessed_img, _ = preprocess_image(first_image_uint8)
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
            
            # For series compatibility
            "series_data": image_stack if image_stack is not None else None,
            "n_frames": num_images,
            "first_frame": first_image,
            
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
            "enable_human_feedback": self.settings.get('enable_human_feedback', False) and preset_params is None,
            "current_params": preset_params or self._initialize_fft_nmf_params(),
            "preset_params": preset_params,
            "feedback_callback": feedback_callback,
            
            # Results placeholders
            "fft_components": None,
            "fft_abundances": None,
            "llm_params": None,
            "analysis_images": [
                {"label": "Primary Microscopy Image", "data": image_bytes}
            ],
        }
        
        # If preset params provided, inject them
        if preset_params:
            state["locked_params"] = preset_params
            state["first_frame_results"] = {"llm_params": preset_params}
        
        # ============================================================
        # Create and Execute Unified Pipeline
        # ============================================================
        pipeline = create_unified_microscopy_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            preset_params=preset_params
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
                
                # Check for cancellation
                if state.get("batch_cancelled"):
                    self.logger.info("Analysis cancelled by user.")
                    return {
                        "status": "cancelled",
                        "first_frame_results": state.get("first_frame_results")
                    }
                    
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
            rationale="Microscopy analysis completed."
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
            "status": "success" if not state.get("error_dict") else "error",
            "summary": {
                "total_images": num_images,
                "successful": sum(1 for r in batch_results if r.get("success", False)) if batch_results else (1 if not state.get("error_dict") else 0),
                "input_type": state.get("input_type"),
                "parameters_used": state.get("locked_params", state.get("current_params", {})),
                "is_single_image": is_single
            },
            "output_directory": str(self.output_dir)
        }
        
        if state.get("error_dict"):
            results["error"] = state["error_dict"]
            return results
        
        if is_single:
            # Single image: provide simplified structure for backward compatibility
            synthesis = state.get("synthesis_result", {})
            result_json = state.get("result_json", {})
            
            # Use synthesis if available, else fall back to result_json
            results["detailed_analysis"] = (
                synthesis.get("detailed_analysis") or 
                result_json.get("detailed_analysis") or 
                "Analysis complete."
            )
            results["scientific_claims"] = (
                synthesis.get("scientific_claims") or 
                result_json.get("scientific_claims") or 
                []
            )
            
            # Include component interpretations if available
            if synthesis.get("component_interpretations"):
                results["component_interpretations"] = synthesis["component_interpretations"]
            
            # Also include the raw data for those who want it
            if batch_results:
                results["statistics"] = batch_results[0].get("statistics", {})
                results["n_components"] = batch_results[0].get("n_components", 0)
                results["visualization_path"] = batch_results[0].get("visualization_path")
            elif state.get("fft_components") is not None:
                results["n_components"] = state["fft_components"].shape[0]
        else:
            # Batch: full structure
            results["n_frames_processed"] = state.get("n_frames", 0)
            results["components"] = state.get("series_components")
            results["abundances"] = state.get("series_abundances")
            results["individual_results"] = batch_results
            results["custom_analysis"] = state.get("custom_analysis_results", {})
            results["synthesis"] = state.get("synthesis_result", {})
            results["analysis_script_path"] = state.get("analysis_script_path")
            results["locked_params"] = state.get("locked_params")
            results["first_frame_results"] = state.get("first_frame_results")
        
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
        
        if result.get("status") == "error" or "error" in result:
            self._log_action("analyze_for_claims", {"image": image_path}, {"error": result})
            return result.get("error", result)
        
        # Extract backward-compatible structure
        claims_result = {
            "detailed_analysis": result.get("detailed_analysis", ""),
            "scientific_claims": result.get("scientific_claims", [])
        }
        
        # Validate claims
        valid_claims = self._validate_scientific_claims(claims_result.get("scientific_claims", []))
        claims_result["scientific_claims"] = valid_claims
        
        # Apply feedback if enabled
        final_result = self._apply_feedback_if_enabled(
            claims_result,
            image_path=image_path,
            system_info=system_info
        )
        
        self._log_action(
            action="analyze_for_claims",
            input_ctx={"image": image_path, "system_info": system_info},
            result=final_result,
            rationale="Microscopy analysis completed (unified pipeline)."
        )
        
        return final_result
    
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
        
        if result.get("status") == "error" or "error" in result:
            self._log_action("analyze_for_recommendations", {"image": image_path}, {"error": result})
            return result.get("error", result)
        
        self._log_action(
            action="analyze_for_recommendations",
            input_ctx={"image": image_path, "system_info": system_info},
            result=result,
            rationale="Microscopy recommendations completed (unified pipeline)."
        )
        
        return result
    
    def analyze_microscopy_image_for_structure_recommendations(
        self,
        image_path: str | None = None,
        system_info: dict | str | None = None,
        additional_prompt_context: str | None = None,
        cached_detailed_analysis: str | None = None
    ):
        """
        Analyze microscopy image for DFT structure recommendations.
        (BACKWARD COMPATIBLE)
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
            self.logger.info("Generating DFT recommendations via unified pipeline.")
            result = self.analyze(
                image_path=image_path, 
                system_info=system_info,
                analysis_mode="recommendations"
            )
            
            if result.get("status") == "error":
                return result.get("error", result)

            recommendations = result.get("structure_recommendations", [])
            sorted_recs = self._validate_structure_recommendations(recommendations)
            
            return {
                "analysis_summary_or_reasoning": result.get("detailed_analysis", "Analysis complete."), 
                "recommendations": sorted_recs
            }
        
        else:
            return {"error": "Either image_path or (cached_detailed_analysis...) must be provided."}
    
    def analyze_series(
        self,
        series_input: Union[str, np.ndarray],
        system_info: Optional[dict] = None,
        feedback_callback: Optional[Callable] = None,
        preset_params: Optional[dict] = None,
    ) -> dict:
        """
        Analyze an image series with interactive parameter tuning.
        
        BACKWARD COMPATIBLE: Delegates to unified analyze() method.
        
        Args:
            series_input: Directory path, TIFF path, or 3D numpy array
            system_info: Metadata dictionary
            feedback_callback: Optional function for custom feedback
            preset_params: Skip first-frame analysis, use these params
        
        Returns:
            Dictionary containing batch results
        """
        # Convert series_input to appropriate format
        if isinstance(series_input, np.ndarray):
            return self.analyze(
                image_stack=series_input,
                system_info=system_info,
                preset_params=preset_params,
                feedback_callback=feedback_callback,
                analysis_mode="claims"
            )
        elif isinstance(series_input, str):
            if os.path.isdir(series_input):
                # Load directory
                image_paths = self._load_image_paths_from_directory(series_input)
                return self.analyze(
                    image_paths=image_paths,
                    system_info=system_info,
                    preset_params=preset_params,
                    feedback_callback=feedback_callback,
                    analysis_mode="claims"
                )
            elif series_input.lower().endswith(('.tif', '.tiff')):
                # Load TIFF stack
                from skimage import io
                stack = io.imread(series_input)
                if stack.ndim == 2:
                    stack = stack[np.newaxis, :, :]
                return self.analyze(
                    image_stack=stack,
                    system_info=system_info,
                    preset_params=preset_params,
                    feedback_callback=feedback_callback,
                    analysis_mode="claims"
                )
            else:
                return {"error": f"Unsupported file type: {series_input}"}
        else:
            return {"error": f"Unsupported input type: {type(series_input)}"}
    
    def analyze_image(
        self,
        image_input: Union[str, np.ndarray],
        system_info: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        Unified analysis method - auto-detects single image vs series.
        
        BACKWARD COMPATIBLE: Convenience wrapper.
        
        Args:
            image_input: Image path, directory, or numpy array
            system_info: Metadata dictionary
            **kwargs: Additional arguments
        
        Returns:
            Analysis results dictionary
        """
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return self.analyze(image_stack=image_input, system_info=system_info, **kwargs)
            elif image_input.ndim == 3:
                return self.analyze(image_stack=image_input, system_info=system_info, **kwargs)
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
                    return self.analyze(image_path=image_input, system_info=system_info, **kwargs)
            else:
                return self.analyze(image_path=image_input, system_info=system_info, **kwargs)
        
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
        
        BACKWARD COMPATIBLE.
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
        
        Returns:
            (components, abundances) or (None, None)
        """
        comp_path = self.data_dir / "series_components.npy"
        abun_path = self.data_dir / "series_abundances.npy"
        
        if comp_path.exists() and abun_path.exists():
            return np.load(comp_path), np.load(abun_path)
        
        self.logger.warning("No saved series results found")
        return None, None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _load_image_paths_from_directory(self, directory: str) -> List[str]:
        """Load image paths from a directory."""
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        files = sorted([
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.lower().endswith(valid_ext)
        ])
        return files
    
    def _get_claims_instruction_prompt(self) -> str:
        return MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS