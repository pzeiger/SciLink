"""
Hyperspectral Analysis Agent
"""


import os
import numpy as np
import cv2
from typing import Dict, Any
from collections import deque

from .base_agent import BaseAnalysisAgent, AnalysisInput
from .instruct import (
    SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
    SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)
from .human_feedback import SimpleFeedbackMixin
from .preprocess import HyperspectralPreprocessingAgent
from .pipelines.hyperspectral_pipelines import (
    create_hyperspectral_iteration_pipeline,
    create_hyperspectral_synthesis_pipeline
)
from ...tools.image_processor import load_image, convert_numpy_to_jpeg_bytes
from ...executors import require_sandbox_approval

from ._deprecation import normalize_params


class HyperspectralAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Hyperspectral Analysis Agent with recursive "survey-then-focus" loop.
    
    This agent analyzes hyperspectral/spectroscopic data using NMF decomposition
    and LLM-guided interpretation.
    
    Features:
        - Automatic component number selection via elbow method
        - Recursive refinement with spatial/spectral zooming
        - Optional structure image correlation
        - Human-in-the-loop feedback
        - HTML report generation
    
    Example:
        agent = HyperspectralAnalysisAgent(api_key="...")
        
        # Single file
        result = agent.analyze("spectrum.npy")
        
        # With metadata
        result = agent.analyze(
            "spectrum.npy",
            system_info={"sample": "TiO2", "technique": "EELS"}
        )
        
        # Get measurement recommendations
        recommendations = agent.recommend_measurements(analysis_result=result)
    """
    
    MAX_REFINEMENT_ITERATIONS = 2

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: str | None = None,
        output_dir: str = "hyperspectral_analysis_output",
        # Deprecated params
        google_api_key: str | None = None,
        local_model: str | None = None,
        # Agent specific params
        spectral_unmixing_settings: dict | None = None,
        run_preprocessing: bool = True,
        enable_human_feedback: bool = True
    ):
        
        if not require_sandbox_approval(
            context="HyperspectralAnalysisAgent (hyperspectral analysis)"
        ):
            raise RuntimeError(
                "HyperspectralAnalysisAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )
        
        # Normalize params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model,
            source="HyperspectralAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )

        self.agent_type = "hyperspectral"
        
        # Settings
        default_settings = {
            'method': 'nmf',
            'n_components': 4,
            'normalize': True,
            'enabled': True,
            'auto_components': True,
            'min_auto_components': 2,
            'max_auto_components': 8,
            'enable_human_feedback': enable_human_feedback
        }
        self.spectral_settings = spectral_unmixing_settings if spectral_unmixing_settings else default_settings
        self.spectral_settings['run_preprocessing'] = run_preprocessing
        self.spectral_settings['output_dir'] = str(self.output_dir)
        self.spectral_settings['feedback_depths'] = [0]
        
        # Sub-agent initialization
        preprocess_dir = self.output_dir / "preprocessing"
        self.preprocessor = HyperspectralPreprocessingAgent(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=str(preprocess_dir)
        )

        # Pipeline initialization
        pipeline_args = {
            "model": self.model,
            "logger": self.logger,
            "generation_config": self.generation_config,
            "safety_settings": self.safety_settings,
            "settings": self.spectral_settings,
            "parse_fn": self._parse_llm_response,
        }

        self.iteration_pipeline = create_hyperspectral_iteration_pipeline(
            **pipeline_args,
            preprocessor=self.preprocessor
        )
        self.synthesis_pipeline = create_hyperspectral_synthesis_pipeline(
            **pipeline_args,
            store_fn=self._store_analysis_images
        )
        
        self.logger.info(f"HyperspectralAnalysisAgent initialized. Output: {self.output_dir}")

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        return {
            "data_path": None,
            "analysis_depth": 0,
            "components_found": []
        }

    # =========================================================================
    # PRIMARY ENTRY POINT
    # =========================================================================

    def analyze(
        self,
        data: AnalysisInput,
        system_info: Dict[str, Any] | str | None = None,
        # Hyperspectral-specific options
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        hints: str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Primary analysis entry point for hyperspectral data.

        Args:
            data: Input data. Can be:
                - str: Path to .npy hyperspectral data file
                - List[str]: Batch processing (not supported)
                - np.ndarray: Direct array (not supported)
            system_info: Metadata dictionary or path to metadata file
            structure_image_path: Optional path to structural reference image
            structure_system_info: Optional metadata for structure image
            hints: Optional user guidance to steer analysis (e.g., "focus on
                the Ti L-edge around 460 eV"). The agent will prioritize these
                suggestions but still report other significant features.
            **kwargs: Additional options
        
        Returns:
            dict containing:
                - "status": "success" | "error"
                - "detailed_analysis": str
                - "scientific_claims": list[dict]
                - "output_directory": str
                - "error": dict (when status="error")
        
        Examples:
            # Single file
            result = agent.analyze("spectrum.npy")
            
            # With metadata and structure image
            result = agent.analyze(
                "spectrum.npy",
                system_info={"sample": "TiO2", "technique": "EELS"},
                structure_image_path="stem_image.png"
            )
        """
        # Parse input
        data_path, data_paths, data_array, error = self._parse_data_input(data)
        
        if error:
            return {
                "status": "error",
                "error": error,
                "output_directory": str(self.output_dir)
            }
        
        # Batch processing not supported
        if data_paths is not None:
            return {
                "status": "error",
                "error": {
                    "error": "Batch processing not supported",
                    "details": "HyperspectralAnalysisAgent processes one file at a time. Pass a single file path."
                },
                "output_directory": str(self.output_dir)
            }
        
        # Direct array not supported
        if data_array is not None:
            return {
                "status": "error",
                "error": {
                    "error": "Direct array input not supported",
                    "details": "Save array to .npy file and pass the file path."
                },
                "output_directory": str(self.output_dir)
            }
        
        # Initialize and Run Pipeline
        self._init_state(data_path=data_path, metadata=system_info)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 HYPERSPECTRAL ANALYSIS")
        self.logger.info(f"   Data: {data_path}")
        self.logger.info(f"{'='*80}\n")
        
        # Run the analysis pipeline
        result_json, error_dict = self._run_analysis_pipeline(
            data_path=data_path,
            system_info=system_info,
            instruction_prompt=SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info,
            hints=hints
        )
        
        # Handle Errors
        if error_dict:
            self._log_action("analyze", {"data": data_path}, {"error": error_dict})
            return {
                "status": "error",
                "error": error_dict,
                "output_directory": str(self.output_dir)
            }
        
        if result_json is None:
            return {
                "status": "error",
                "error": {
                    "error": "Analysis failed",
                    "details": "Pipeline returned no results"
                },
                "output_directory": str(self.output_dir)
            }
        
        # Process Successful Results
        valid_claims = self._validate_scientific_claims(
            result_json.get("scientific_claims", [])
        )
        
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis not provided."),
            "scientific_claims": valid_claims
        }
        
        # Apply feedback if enabled
        final_result = self._apply_feedback_if_enabled(
            initial_result,
            system_info=self._handle_system_info(system_info)
        )
        
        # Regenerate report if feedback changed results
        if self.enable_human_feedback and final_result != initial_result:
            self.logger.info("🔄 Feedback applied. Regenerating HTML report...")
            self._regenerate_report_with_feedback(final_result, system_info, data_path)
        
        # Build Response
        response = {
            "status": "success",
            "detailed_analysis": final_result.get("detailed_analysis"),
            "scientific_claims": final_result.get("scientific_claims", []),
            "output_directory": str(self.output_dir)
        }
        
        self._log_action(
            action="analyze",
            input_ctx={"data": data_path},
            result=response,
            rationale="Hyperspectral analysis completed."
        )
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ ANALYSIS COMPLETE")
        self.logger.info(f"   Output: {self.output_dir}")
        self.logger.info(f"{'='*80}\n")
        
        return response

    # =========================================================================
    # BACKWARD COMPATIBLE METHODS
    # =========================================================================
    
    def analyze_for_claims(
        self,
        data_path: str,
        metadata_path: Dict[str, Any] | str | None = None,
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        hints: str | None = None
    ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data to generate scientific claims.

        BACKWARD COMPATIBLE: Delegates to analyze().
        """
        result = self.analyze(
            data_path,
            system_info=metadata_path,
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info,
            hints=hints
        )
        
        if result.get("status") == "success":
            return {
                "detailed_analysis": result.get("detailed_analysis", ""),
                "scientific_claims": result.get("scientific_claims", [])
            }
        else:
            return result.get("error", result)
    
    def analyze_hyperspectral_data(
        self,
        data_path: str,
        metadata_path: str,
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        hints: str | None = None
    ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data for materials characterization.

        BACKWARD COMPATIBLE: Delegates to analyze().
        """
        return self.analyze_for_claims(
            data_path=data_path,
            metadata_path=metadata_path,
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info,
            hints=hints
        )

    # =========================================================================
    # INSTRUCTION PROMPTS
    # =========================================================================
    
    def _get_claims_instruction_prompt(self) -> str:
        return SPECTROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _regenerate_report_with_feedback(
        self,
        final_result: Dict[str, Any],
        system_info: Any,
        data_path: str
    ) -> None:
        """Regenerate HTML report after feedback modifications."""
        stored_images = self._get_stored_analysis_images()
        
        report_state = {
            "result_json": final_result,
            "system_info": self._handle_system_info(system_info),
            "analysis_images": stored_images,
            "image_path": data_path
        }
        
        from .controllers.hyperspectral_controllers import GenerateHTMLReportController
        report_gen = GenerateHTMLReportController(self.logger, self.spectral_settings)
        report_gen.execute(report_state)
        
        self.logger.info("✅ Refined HTML report generated.")

    def _load_hyperspectral_data(self, data_path: str) -> np.ndarray:
        """Load hyperspectral data from numpy array."""
        try:
            if not data_path.endswith('.npy'):
                raise ValueError(f"Expected .npy file, got: {data_path}")
            
            data = np.load(data_path)
            self.logger.info(f"Loaded hyperspectral data: shape {data.shape}")
            
            if data.ndim == 2:
                self.logger.warning("2D data detected, reshaping to (1, 1, n_channels)")
                data = data.reshape(1, 1, -1)
            elif data.ndim != 3:
                raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    def _run_analysis_pipeline(
        self,
        data_path: str,
        system_info: Dict[str, Any] | str | None,
        instruction_prompt: str,
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        hints: str | None = None
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        """
        Main execution engine using Queue-Based Branching architecture.
        """
        try:
            self.logger.info(f"--- Starting analysis pipeline for {data_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            # Load data
            original_hspy_data = self._load_hyperspectral_data(data_path)

            # Handle structure image
            structure_image_blob = None
            if structure_image_path and os.path.exists(structure_image_path):
                try:
                    img = load_image(structure_image_path)
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    structure_image_blob = {
                        "mime_type": "image/jpeg",
                        "data": convert_numpy_to_jpeg_bytes(img)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not load structure image: {e}")
            
            # Initialize task queue
            initial_task = {
                "data": original_hspy_data,
                "system_info": system_info,
                "title": "Global_Analysis",
                "parent_reasoning": None,
                "depth": 0
            }
            
            task_queue = deque([initial_task])
            all_completed_results = []
            task_counter = 0
            
            # Process tasks
            while task_queue:
                current_task = task_queue.popleft()
                task_counter += 1
                
                if current_task["depth"] > self.MAX_REFINEMENT_ITERATIONS:
                    self.logger.info(f"Skipping '{current_task['title']}': max depth reached")
                    continue

                self.logger.info(f"\n=== TASK {task_counter}: {current_task['title']} (Depth {current_task['depth']}) ===\n")

                iteration_state = {
                    "data_path": data_path,
                    "hspy_data": current_task["data"],
                    "original_hspy_data": original_hspy_data,
                    "system_info": current_task["system_info"],
                    "instruction_prompt": instruction_prompt,
                    "settings": self.spectral_settings.copy(),
                    "iteration_title": current_task["title"],
                    "parent_refinement_reasoning": current_task["parent_reasoning"],
                    "current_depth": current_task["depth"],
                    "structure_image_path": structure_image_path,
                    "structure_system_info": self._handle_system_info(structure_system_info),
                    "structure_image_blob": structure_image_blob,
                    "analysis_hints": hints,
                    "analysis_images": [],
                    "error_dict": None
                }

                if current_task["depth"] > 0:
                    iteration_state["settings"]['run_preprocessing'] = False

                # Run iteration pipeline
                for controller in self.iteration_pipeline:
                    iteration_state = controller.execute(iteration_state)
                    if iteration_state.get("error_dict"):
                        self.logger.error(f"Pipeline failed at {controller.__class__.__name__}")
                        break
                
                if iteration_state.get("error_dict"):
                    continue

                # Store results
                result_summary = {
                    "iteration_title": iteration_state.get("iteration_title"),
                    "iteration_analysis_text": iteration_state.get("result_json", {}).get("detailed_analysis", ""),
                    "analysis_images": iteration_state.get("analysis_images", []),
                    "refinement_decision": iteration_state.get("refinement_decision", {}),
                    "depth": current_task["depth"],
                    "custom_analysis_metadata": iteration_state.get("custom_analysis_metadata")
                }
                all_completed_results.append(result_summary)

                # Process new tasks
                for t in iteration_state.get("new_tasks", []):
                    task_queue.append({
                        "data": t["data"],
                        "system_info": t["system_info"],
                        "title": t["title"],
                        "parent_reasoning": t["parent_reasoning"],
                        "depth": t["source_depth"]
                    })

            # Run synthesis
            self.logger.info(f"\n=== Synthesizing {len(all_completed_results)} analyses ===\n")
            
            synthesis_state = {
                "all_iteration_results": all_completed_results,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "analysis_hints": hints,
                "result_json": None,
                "error_dict": None
            }

            for controller in self.synthesis_pipeline:
                synthesis_state = controller.execute(synthesis_state)
                if synthesis_state.get("error_dict"):
                    self.logger.error(f"Synthesis failed at {controller.__class__.__name__}")
                    break

            self.logger.info("--- Analysis pipeline finished ---")
            return synthesis_state.get("result_json"), synthesis_state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            return None, {"error": "File not found", "details": f"Path: {data_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"Unexpected error: {e}")
            return None, {"error": "Unexpected error", "details": str(e)}