"""
SAM Analysis Controllers - Unified Architecture

This module contains unified controllers that handle both single image (n=1)
and batch (n>1) analysis identically. The key principle is:

    Single image = Batch of 1

All controllers adapt their behavior based on state["is_single_image"]
and state["num_images"], but use the same code paths.
"""

import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Any, Dict
import numpy as np

from PIL import Image

from ..instruct import (
    SAM_BATCH_REFINEMENT_INSTRUCTIONS,
    SAM_BATCH_SYNTHESIS_INSTRUCTIONS,
    SAM_BATCH_CUSTOM_ANALYSIS_INSTRUCTIONS,
    SAM_SINGLE_IMAGE_SYNTHESIS_INSTRUCTIONS,  # New instruction for single images
)

from ....tools.image_processor import (
    load_image,
    preprocess_image,
    convert_numpy_to_jpeg_bytes
)
from ....tools.sam import (
    get_or_create_sam_model,
    run_sam_analysis,
    visualize_sam_results,
    calculate_sam_statistics,
    save_sam_visualization
)


# ============================================================================
# HUMAN FEEDBACK REFINEMENT CONTROLLER
# (Same for single and batch - refines parameters on first image)
# ============================================================================

class HumanFeedbackRefinementController:
    """
    [👤 Human Step + 🧠 LLM Step]
    Facilitates human-in-the-loop parameter refinement for the first image.
    
    Works identically for single images and batches:
    - Single image: Refine params, then process that one image
    - Batch: Refine params on first image, then apply to all
    """
    
    def __init__(
        self, 
        model, 
        logger: logging.Logger, 
        generation_config, 
        safety_settings, 
        parse_fn: Callable, 
        settings: dict
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.max_refinement_iterations = settings.get('max_feedback_iterations', 3)
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _display_analysis_for_review(self, state: dict, iteration: int) -> None:
        """Display current analysis results for human review."""
        sam_result = state.get("sam_result")
        sam_stats = state.get("summary_stats", {})
        is_single = state.get("is_single_image", False)

        overlay_img = visualize_sam_results(sam_result)
        
        review_viz_path = self.output_dir / f"review_iteration_{iteration}.png"
        review_viz_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay_img).save(review_viz_path)

        print("\n" + "=" * 80)
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({state.get('num_images', 1)} images)"
        print(f"🔬 SAM ANALYSIS REVIEW - {mode_str} - Iteration {iteration}")
        print("=" * 80)

        print(f"\n🖼️  **Visualization saved to:** {review_viz_path}")
        print(f"   Open this file to see the segmentation overlay.\n")
        
        print(f"\n📊 **Detection Summary (First Image):**")
        print(f"   - Total particles detected: {sam_stats.get('total_particles', 0)}")
        
        mean_area = sam_stats.get('mean_area_pixels', 'N/A')
        std_area = sam_stats.get('std_area_pixels', 'N/A')
        if isinstance(mean_area, (int, float)):
            print(f"   - Mean area: {mean_area:.2f} px²")
        if isinstance(std_area, (int, float)):
            print(f"   - Area std: {std_area:.2f} px²")
        
        print(f"\n⚙️ **Current Parameters:**")
        current_params = state.get("current_params", {})
        for key, value in current_params.items():
            if key not in ['checkpoint_path', 'device']:
                print(f"   - {key}: {value}")
        
        if not is_single:
            print(f"\n📦 **Note:** These parameters will be applied to all {state.get('num_images', 1)} images.")
        
        print("-" * 80)
    
    def _get_llm_assessment(self, state: dict) -> dict:
        """Get LLM's assessment of the current segmentation quality."""
        self.logger.info("   🧠 Getting LLM assessment of segmentation quality...")
        
        original_image_bytes = state["image_blob"]["data"]
        sam_result = state.get("sam_result")
        sam_stats = state.get("summary_stats", {})
        
        overlay_img = visualize_sam_results(sam_result)
        overlay_bytes = convert_numpy_to_jpeg_bytes(overlay_img)
        
        prompt_parts = [
            SAM_BATCH_REFINEMENT_INSTRUCTIONS,
            "\n\n**ORIGINAL MICROSCOPY IMAGE:**",
            {"mime_type": "image/jpeg", "data": original_image_bytes},
            "\n\n**CURRENT SEGMENTATION RESULT:**",
            {"mime_type": "image/jpeg", "data": overlay_bytes},
            f"\n\n**MORPHOLOGICAL STATISTICS:**\n{json.dumps(sam_stats, indent=2)}",
            f"\n\n**CURRENT PARAMETERS:**\n{json.dumps(state.get('current_params', {}), indent=2)}"
        ]
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM assessment failed: {error_dict}")
                return {"needs_refinement": False, "reasoning": "Could not assess"}
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error in LLM assessment: {e}")
            return {"needs_refinement": False, "reasoning": str(e)}
    
    def _collect_human_feedback(self, llm_assessment: dict) -> dict:
        """Collect human feedback on the analysis."""
        print("\n🤖 **LLM Assessment:**")
        evaluation = llm_assessment.get("evaluation", {})
        print(f"   - Coverage: {evaluation.get('coverage_score', 'N/A')}/10")
        print(f"   - Accuracy: {evaluation.get('accuracy_score', 'N/A')}/10")
        print(f"   - Overall Quality: {evaluation.get('overall_quality', 'N/A')}")
        print(f"\n   Reasoning: {llm_assessment.get('reasoning', 'N/A')}")
        
        if llm_assessment.get("needs_refinement"):
            print("\n   📝 LLM recommends refinement with these parameters:")
            rec_params = llm_assessment.get("recommended_parameters", {})
            for key, value in rec_params.items():
                print(f"      - {key}: {value}")
        
        print("\n" + "-" * 80)
        print("👤 **Your Options:**")
        print("   [1] Accept current results (proceed to processing)")
        print("   [2] Accept LLM's recommended parameters")
        print("   [3] Provide feedback in natural language")
        
        try:
            choice = input("\nYour choice [1/2/3]: ").strip()
        except KeyboardInterrupt:
            self.logger.warning("User interrupted. Accepting current results.")
            return {"action": "accept", "params": None}
        
        if choice == '1' or choice == '':
            return {"action": "accept", "params": None}
        elif choice == '2':
            return {"action": "use_llm", "params": llm_assessment.get("recommended_parameters", {})}
        elif choice == '3':
            return self._collect_natural_language_feedback()
        else:
            print("Invalid choice. Accepting current results.")
            return {"action": "accept", "params": None}
    
    def _collect_natural_language_feedback(self) -> dict:
        """Collect natural language feedback and convert to parameters via LLM."""
        print("\n📝 **Describe what you'd like to change:**")
        print("   Examples:")
        print("   - 'Detect smaller particles, current minimum is too high'")
        print("   - 'The contrast is low, try enhancing it'")
        print("   - 'Increase min_area to 150'")
        
        try:
            user_feedback = input("\nYour feedback: ").strip()
        except KeyboardInterrupt:
            self.logger.warning("User interrupted. Accepting current results.")
            return {"action": "accept", "params": None}
        
        if not user_feedback:
            return {"action": "accept", "params": None}
        
        params = self._convert_feedback_to_params(user_feedback)
        
        if params:
            return {"action": "custom", "params": params}
        else:
            print("Could not interpret feedback. Accepting current results.")
            return {"action": "accept", "params": None}
    
    def _convert_feedback_to_params(self, user_feedback: str) -> dict:
        """Use LLM to convert natural language feedback to SAM parameters."""
        self.logger.info("   🧠 Converting feedback to parameters...")
        
        current_params = {
            "use_clahe": self.settings.get('use_clahe', False),
            "sam_parameters": self.settings.get('sam_parameters', 'default'),
            "min_area": self.settings.get('min_area', 500),
            "max_area": self.settings.get('max_area', 50000),
            "pruning_iou_threshold": self.settings.get('pruning_iou_threshold', 0.5)
        }
        
        prompt = f"""Convert user feedback into SAM parameter adjustments.

**Current Parameters:**
{json.dumps(current_params, indent=2)}

**Available Parameters:**
- use_clahe (true/false): Enable contrast enhancement
- sam_parameters ("default"/"sensitive"/"ultra-permissive"): Detection sensitivity
- min_area (integer): Minimum particle size in pixels
- max_area (integer): Maximum particle size in pixels
- pruning_iou_threshold (0.0-1.0): Overlap threshold for removing duplicates

**User Feedback:**
"{user_feedback}"

Return JSON with ONLY the parameters to change:
{{"min_area": 200, "sam_parameters": "sensitive"}}
"""

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict or not result_json:
                return None
            
            print(f"\n   ✅ Interpreted as: {json.dumps(result_json, indent=2)}")
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error converting feedback: {e}")
            return None
    
    def execute(self, state: dict) -> dict:
        """Execute the human feedback refinement loop."""
        if not state.get('enable_human_feedback', False):
            self.logger.info("Human feedback disabled. Using initial parameters.")
            state["final_params_for_batch"] = state.get("current_params", {})
            return state
        
        is_single = state.get("is_single_image", False)
        mode_str = "SINGLE IMAGE" if is_single else "BATCH"
        self.logger.info(f"\n\n👤 --- HUMAN FEEDBACK REFINEMENT ({mode_str}) --- 👤\n")
        
        iteration = 0
        while iteration < self.max_refinement_iterations:
            iteration += 1
            
            self._display_analysis_for_review(state, iteration)
            llm_assessment = self._get_llm_assessment(state)
            feedback = self._collect_human_feedback(llm_assessment)
            
            if feedback["action"] == "accept":
                self.logger.info("✅ User accepted current results.")
                state["refinement_complete"] = True
                state["final_params_for_batch"] = state.get("current_params", {})
                break
            
            elif feedback["action"] in ["use_llm", "custom"]:
                new_params = feedback["params"]
                if new_params:
                    current_params = state.get("current_params", {})
                    current_params.update(new_params)
                    state["current_params"] = current_params
                    
                    self.logger.info(f"🔄 Re-running SAM analysis with updated parameters...")
                    
                    try:
                        image_array = state["preprocessed_image_array"]
                        sam_result = run_sam_analysis(image_array, params=current_params)
                        state["sam_result"] = sam_result
                        
                        summary_stats = calculate_sam_statistics(
                            sam_result=sam_result,
                            image_path=state["image_path"],
                            preprocessed_image_shape=image_array.shape,
                            nm_per_pixel=state.get("nm_per_pixel")
                        )
                        state["summary_stats"] = summary_stats
                        
                        self.logger.info(f"✅ Re-analysis complete. {sam_result['total_count']} particles detected.")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Re-analysis failed: {e}")
                        break
        
        if iteration >= self.max_refinement_iterations:
            self.logger.warning(f"⚠️ Max iterations reached. Using current parameters.")
            state["final_params_for_batch"] = state.get("current_params", {})
        
        return state


# ============================================================================
# UNIFIED BATCH PROCESSING CONTROLLER
# (Processes all images - whether n=1 or n>1)
# ============================================================================

class UnifiedBatchProcessingController:
    """
    [🛠️ Tool Step]
    Processes ALL images using the refined parameters.
    
    Key features:
    - Loads SAM model ONCE and reuses for all images
    - Works identically for n=1 or n>1
    - Supports both file paths and numpy array stack inputs
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.save_visualizations = settings.get('save_visualizations', True)
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
    
    def _get_image_and_name(self, state: dict, idx: int) -> tuple:
        """Get image array and name for a given index."""
        image_stack = state.get("image_stack")
        image_paths = state.get("image_paths")
        
        if image_stack is not None:
            image_array = image_stack[idx]
            image_name = f"frame_{idx:04d}"
            return image_array, image_name
        elif image_paths:
            image_path = image_paths[idx]
            loaded_image = load_image(image_path)
            image_name = Path(image_path).stem
            return loaded_image, image_name
        else:
            raise ValueError("No image data found in state")
    
    def execute(self, state: dict) -> dict:
        """Process all images using refined parameters."""
        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", False)
        input_type = state.get("input_type", "unknown")
        
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({num_images} images)"
        self.logger.info(f"\n\n🔄 --- PROCESSING: {mode_str} --- 🔄\n")
        
        if num_images == 0:
            state["batch_results"] = []
            return state
        
        # Get refined parameters
        batch_params = state.get("final_params_for_batch", state.get("current_params", {}))
        
        self.logger.info(f"📦 Processing with parameters:")
        for key, value in batch_params.items():
            if key not in ['checkpoint_path', 'device']:
                self.logger.info(f"   - {key}: {value}")
        
        # Load SAM model ONCE
        self.logger.info("\n🧠 Loading SAM model...")
        try:
            sam_analyzer = get_or_create_sam_model(batch_params)
            self.logger.info("✅ SAM model loaded and cached.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load SAM model: {e}")
            state["batch_results"] = []
            state["error_dict"] = {"error": "SAM model loading failed", "details": str(e)}
            return state
        
        nm_per_pixel = state.get("nm_per_pixel")
        batch_results = []
        
        for idx in range(num_images):
            try:
                raw_image, image_name = self._get_image_and_name(state, idx)
                
                if is_single:
                    self.logger.info(f"   Processing: {image_name}")
                else:
                    self.logger.info(f"   [{idx + 1}/{num_images}] Processing: {image_name}")
                
                preprocessed_img, _ = preprocess_image(raw_image)
                
                # Reuse cached model
                sam_result = run_sam_analysis(
                    preprocessed_img, 
                    params=batch_params,
                    analyzer=sam_analyzer
                )
                
                # Get image path for stats
                image_paths = state.get("image_paths")
                image_path_for_stats = image_paths[idx] if image_paths else f"frame_{idx:04d}"
                
                summary_stats = calculate_sam_statistics(
                    sam_result=sam_result,
                    image_path=str(image_path_for_stats),
                    preprocessed_image_shape=preprocessed_img.shape,
                    nm_per_pixel=nm_per_pixel
                )
                
                # Save visualization
                viz_path = None
                if self.save_visualizations:
                    overlay_img = visualize_sam_results(sam_result, preprocessed_img)
                    viz_dir = self.output_dir / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    viz_path = viz_dir / f"overlay_{idx:04d}_{image_name}.png"
                    save_sam_visualization(
                        overlay_img, str(viz_path), idx,
                        sam_result['total_count'], batch_params, self.logger
                    )
                
                result_entry = {
                    "index": idx,
                    "image_path": str(image_paths[idx]) if image_paths else None,
                    "image_name": image_name,
                    "visualization_path": str(viz_path) if viz_path else None,
                    "particle_count": sam_result['total_count'],
                    "statistics": summary_stats,
                    "success": True,
                    "error": None
                }
                
                batch_results.append(result_entry)
                self.logger.info(f"      ✅ Detected {sam_result['total_count']} particles")
                
            except Exception as e:
                self.logger.error(f"      ❌ Failed: {e}")
                batch_results.append({
                    "index": idx,
                    "image_path": None,
                    "image_name": f"frame_{idx:04d}",
                    "visualization_path": None,
                    "particle_count": 0,
                    "statistics": {},
                    "success": False,
                    "error": str(e)
                })
        
        # Save batch results JSON
        results_path = self.output_dir / "batch_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_images": num_images,
                "is_single_image": is_single,
                "successful": sum(1 for r in batch_results if r["success"]),
                "parameters_used": batch_params,
                "input_type": input_type,
                "results": batch_results
            }, f, indent=2, default=str)
        
        state["batch_results"] = batch_results
        state["batch_results_path"] = str(results_path)
        
        successful = sum(1 for r in batch_results if r["success"])
        self.logger.info(f"\n✅ Processing complete: {successful}/{num_images} successful.")
        
        return state


# ============================================================================
# CONDITIONAL CUSTOM ANALYSIS CONTROLLER
# (Only runs for n>=2, skipped for single images)
# ============================================================================

class ConditionalCustomAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates and executes custom Python script for trend analysis.
    
    CONDITIONAL:
    - For n>=2: Generates trend analysis script
    - For n=1: Skipped (no trends to analyze)
    """
    
    def __init__(
        self, 
        model, 
        logger: logging.Logger, 
        generation_config, 
        safety_settings, 
        parse_fn: Callable, 
        settings: dict
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
        self.max_correction_attempts = settings.get('max_script_corrections', 3)
    
    def execute(self, state: dict) -> dict:
        """Execute custom analysis - conditional on batch size."""
        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", False)
        
        if is_single or num_images < 2:
            self.logger.info("\n📊 Custom trend analysis skipped (single image mode).\n")
            state["custom_analysis_results"] = {
                "success": True,
                "skipped": True,
                "reason": "Single image - no trend analysis applicable"
            }
            return state
        
        self.logger.info("\n\n🧠 --- CUSTOM ANALYSIS SCRIPT GENERATION --- 🧠\n")
        
        # Full custom analysis logic for batches
        batch_results = state.get("batch_results", [])
        
        script_result = self._generate_analysis_script(state)
        
        if not script_result or "script" not in script_result:
            self.logger.error("Failed to generate analysis script.")
            state["custom_analysis_results"] = {"success": False, "error": "Script generation failed"}
            return state
        
        approach = script_result.get('analysis_approach', 'unknown')
        metrics = script_result.get('key_metrics_to_track', [])
        self.logger.info(f"   📊 Analysis approach: {approach}")
        self.logger.info(f"   📈 Key metrics: {metrics}")
        
        script = script_result["script"]
        success, stdout, stderr = False, "", ""
        
        for attempt in range(self.max_correction_attempts + 1):
            if attempt > 0:
                self.logger.info(f"   🔄 Execution attempt {attempt + 1}")
            
            success, stdout, stderr = self._execute_script(script)
            
            if success:
                self.logger.info("   ✅ Script executed successfully!")
                break
            
            error_preview = stderr[:200] + "..." if len(stderr) > 200 else stderr
            self.logger.warning(f"   ⚠️ Script failed: {error_preview}")
            
            if attempt < self.max_correction_attempts:
                corrected = self._correct_script(script, stderr, attempt + 1)
                if corrected:
                    script = corrected
                else:
                    break
        
        generated_files = []
        for ext in ['*.png', '*.csv']:
            generated_files.extend(self.output_dir.glob(ext))
        
        state["custom_analysis_results"] = {
            "success": success,
            "skipped": False,
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics_to_track"),
            "reasoning": script_result.get("reasoning"),
            "stdout": stdout,
            "stderr": stderr if not success else None,
            "generated_files": [str(f) for f in generated_files],
            "script_path": str(self.output_dir / "custom_analysis.py")
        }
        
        return state
    
    def _generate_analysis_script(self, state: dict) -> Optional[Dict]:
        """Generate custom analysis script using LLM."""
        self.logger.info("   🧠 Generating custom analysis script...")
        
        batch_results = state.get("batch_results", [])
        series_metadata = state.get("series_metadata", {})
        
        particle_counts = []
        mean_areas = []
        
        for r in batch_results:
            if r["success"]:
                particle_counts.append(r["particle_count"])
                stats = r.get("statistics", {})
                mean_area = stats.get("mean_area_pixels")
                mean_areas.append(round(mean_area, 2) if mean_area else 0)
        
        time_points = series_metadata.get("time_points") or list(range(len(batch_results)))
        
        summary = {
            "total_images": len(batch_results),
            "successful": sum(1 for r in batch_results if r["success"]),
            "particle_counts": particle_counts,
            "mean_areas": mean_areas,
            "series_type": series_metadata.get("series_type", "unknown"),
            "time_points": time_points,
        }
        
        prompt_parts = [
            SAM_BATCH_CUSTOM_ANALYSIS_INSTRUCTIONS,
            f"\n\n**BATCH SUMMARY:**\n```json\n{json.dumps(summary, indent=2)}\n```",
        ]
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict and not (result_json and 'script' in result_json):
                return None
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error generating script: {e}")
            return None
    
    def _execute_script(self, script: str) -> tuple:
        """Execute the generated Python script."""
        script_path = self.output_dir / "custom_analysis.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(
                ['python', str(script_path)],
                cwd=str(self.output_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            return (result.returncode == 0), result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Script execution timed out"
        except Exception as e:
            return False, "", str(e)
    
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
            result_json, _ = self._parse_llm_response(response)
            
            if result_json:
                self.logger.info(f"   📋 Diagnosis: {result_json.get('diagnosis', 'N/A')}")
                return result_json.get("script")
            return None
            
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None


# ============================================================================
# UNIFIED SYNTHESIS CONTROLLER
# (Adapts behavior based on single vs batch)
# ============================================================================

class UnifiedSynthesisController:
    """
    [🧠 LLM Step]
    Synthesizes findings into scientific claims.
    
    ADAPTIVE:
    - For n>=2: Cross-image synthesis with trend analysis
    - For n=1: Single-image scientific interpretation
    """
    
    def __init__(
        self, 
        model, 
        logger: logging.Logger, 
        generation_config, 
        safety_settings, 
        parse_fn: Callable, 
        settings: dict,
        store_fn: Callable = None
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self._store_analysis_images = store_fn or (lambda *args, **kwargs: None)
    
    def execute(self, state: dict) -> dict:
        """Execute synthesis - adapts to single vs batch."""
        is_single = state.get("is_single_image", False)
        
        if is_single:
            return self._synthesize_single_image(state)
        else:
            return self._synthesize_batch(state)
    
    def _synthesize_single_image(self, state: dict) -> dict:
        """Generate scientific interpretation for a single image."""
        self.logger.info("\n\n🔬 --- SINGLE IMAGE SYNTHESIS --- 🔬\n")
        
        batch_results = state.get("batch_results", [])
        if not batch_results or not batch_results[0].get("success"):
            state["synthesis_result"] = {"error": "No successful analysis to synthesize"}
            return state
        
        result = batch_results[0]
        stats = result.get("statistics", {})
        
        # Build prompt for single-image analysis
        prompt_parts = [
            SAM_SINGLE_IMAGE_SYNTHESIS_INSTRUCTIONS,
            f"\n\n**IMAGE ANALYSIS RESULTS:**",
            f"- Image: {result.get('image_name', 'unknown')}",
            f"- Particle count: {result.get('particle_count', 0)}",
            f"\n\n**MORPHOLOGICAL STATISTICS:**\n{json.dumps(stats, indent=2)}",
            f"\n\n**SYSTEM INFORMATION:**\n{json.dumps(state.get('system_info', {}), indent=2)}",
        ]
        
        # Add visualization if available
        viz_path = result.get("visualization_path")
        if viz_path and Path(viz_path).exists():
            with open(viz_path, 'rb') as f:
                prompt_parts.append("\n\n**SEGMENTATION VISUALIZATION:**")
                prompt_parts.append({"mime_type": "image/png", "data": f.read()})
        
        # Add original image
        if state.get("image_blob"):
            prompt_parts.append("\n\n**ORIGINAL IMAGE:**")
            prompt_parts.append(state["image_blob"])
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Single image synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state
    
    def _synthesize_batch(self, state: dict) -> dict:
        """Synthesize findings across multiple images."""
        self.logger.info("\n\n🔬 --- BATCH SYNTHESIS --- 🔬\n")
        
        batch_results = state.get("batch_results", [])
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        
        stats_summary = []
        for r in batch_results:
            if r["success"]:
                stats_summary.append({
                    "index": r["index"],
                    "name": r["image_name"],
                    "particles": r["particle_count"],
                    "mean_area": r["statistics"].get("mean_area_pixels"),
                })
        
        prompt_parts = [
            SAM_BATCH_SYNTHESIS_INSTRUCTIONS,
            f"\n\n**INDIVIDUAL ANALYSIS SUMMARY:**\n{json.dumps(stats_summary, indent=2)}",
            f"\n\n**CUSTOM ANALYSIS RESULTS:**\n{json.dumps(custom_results, indent=2)}",
            f"\n\n**SERIES METADATA:**\n{json.dumps(series_metadata, indent=2)}"
        ]
        
        # Add generated plots
        if custom_results.get("success") and custom_results.get("generated_files"):
            prompt_parts.append("\n\n**TREND ANALYSIS VISUALIZATIONS:**")
            for file_path in custom_results["generated_files"][:5]:
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
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Batch synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state


# ============================================================================
# UNIFIED REPORT GENERATION CONTROLLER
# (Generates appropriate report for single vs batch)
# ============================================================================

class UnifiedReportGenerationController:
    """
    [📄 Report Step]
    Generates final HTML report and JSON summary.
    
    ADAPTIVE:
    - For n>=2: Full batch report with trends
    - For n=1: Simplified single-image report
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
    
    def execute(self, state: dict) -> dict:
        """Generate final reports."""
        is_single = state.get("is_single_image", False)
        
        self.logger.info("\n\n📄 --- GENERATING REPORT --- 📄\n")
        
        if is_single:
            self._generate_single_image_report(state)
        else:
            self._generate_batch_report(state)
        
        return state
    
    def _generate_single_image_report(self, state: dict) -> None:
        """Generate report for single image analysis."""
        import base64
        
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        result = batch_results[0] if batch_results else {}
        
        detailed_analysis = synthesis.get("detailed_analysis", "No analysis available.")
        scientific_claims = synthesis.get("scientific_claims", [])
        
        # Embed visualization
        viz_path = result.get("visualization_path")
        viz_b64 = None
        if viz_path and Path(viz_path).exists():
            with open(viz_path, 'rb') as f:
                viz_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SAM Analysis Report - Single Image</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f4f4f9; }}
        .container {{ background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .stats-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin: 20px 0; }}
        .analysis-text {{ white-space: pre-wrap; background: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; }}
        .claim-card {{ background: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin: 15px 0; border-radius: 0 5px 5px 0; }}
        .viz-img {{ max-width: 100%; border-radius: 8px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 SAM Analysis Report</h1>
    <p><strong>Date:</strong> {timestamp} | <strong>Image:</strong> {result.get('image_name', 'unknown')}</p>
    
    <div class="stats-box">
        <p><strong>Particles Detected:</strong> {result.get('particle_count', 0)}</p>
        <p><strong>Mean Area:</strong> {result.get('statistics', {}).get('mean_area_pixels', 'N/A')} px²</p>
    </div>
"""
        
        if viz_b64:
            html += f"""
    <h2>Segmentation Result</h2>
    <img class="viz-img" src="data:image/png;base64,{viz_b64}" alt="Segmentation">
"""
        
        html += f"""
    <h2>Scientific Analysis</h2>
    <div class="analysis-text">{detailed_analysis}</div>
"""
        
        if scientific_claims:
            html += "<h2>Key Findings</h2>\n"
            for i, claim in enumerate(scientific_claims, 1):
                html += f"""<div class="claim-card">
    <strong>Finding {i}:</strong> {claim.get('claim', 'N/A')}<br>
    <em>Impact:</em> {claim.get('scientific_impact', 'N/A')}
</div>\n"""
        
        html += """
    <div class="footer">Generated by SAM Analysis Agent (Unified Architecture)</div>
</div>
</body>
</html>"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated report: {report_path}")
    
    def _generate_batch_report(self, state: dict) -> None:
        """Generate report for batch analysis."""
        import base64
        
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        
        detailed_analysis = synthesis.get("detailed_analysis", "No synthesis available.")
        scientific_claims = synthesis.get("scientific_claims", [])
        
        # Collect PNG files
        png_files = sorted(self.output_dir.glob("*.png"))
        embedded_images = []
        for png_path in png_files:
            if png_path.name.startswith("review_iteration"):
                continue
            if png_path.exists():
                with open(png_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    embedded_images.append({"name": png_path.stem, "data": b64})
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_images = len(batch_results)
        successful = sum(1 for r in batch_results if r.get("success"))
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SAM Batch Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f4f4f9; }}
        .container {{ background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .metadata-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin: 20px 0; }}
        .analysis-text {{ white-space: pre-wrap; background: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; }}
        .claim-card {{ background: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin: 15px 0; border-radius: 0 5px 5px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center; }}
        .image-card img {{ max-width: 100%; border-radius: 4px; }}
        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 SAM Batch Analysis Report</h1>
    
    <div class="metadata-box">
        <p><strong>Date:</strong> {timestamp}</p>
        <p><strong>Images Processed:</strong> {successful}/{num_images}</p>
        <p><strong>Analysis Approach:</strong> {custom_results.get("approach", "time_series")}</p>
        <p><strong>Series Type:</strong> {series_metadata.get("series_type", "unknown")}</p>
    </div>

    <h2>Scientific Analysis</h2>
    <div class="analysis-text">{detailed_analysis}</div>
"""
        
        if embedded_images:
            html += """
    <h2>Visualizations</h2>
    <div class="image-grid">
"""
            for img in embedded_images[:6]:
                html += f"""        <div class="image-card">
            <img src="data:image/png;base64,{img['data']}" alt="{img['name']}">
            <p>{img['name'].replace('_', ' ').title()}</p>
        </div>
"""
            html += "    </div>\n"
        
        if scientific_claims:
            html += "    <h2>Key Scientific Claims</h2>\n"
            for i, claim in enumerate(scientific_claims, 1):
                html += f"""    <div class="claim-card">
        <strong>Claim {i}:</strong> {claim.get('claim', 'N/A')}<br>
        <em>Impact:</em> {claim.get('scientific_impact', 'N/A')}
    </div>\n"""
        
        html += """
    <div class="footer">Generated by SAM Batch Analysis Agent (Unified Architecture)</div>
</div>
</body>
</html>"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated batch report: {report_path}")