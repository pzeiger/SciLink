"""
SAM Analysis Controllers

This module contains all controllers for SAM-based microscopy analysis:
- Single-image pipeline controllers (refinement loop, stats, prompt building, interpretation)
- Batch processing controllers (human feedback, batch processing, custom analysis, synthesis)
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
    SAM_ANALYSIS_REFINE_INSTRUCTIONS
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
# SINGLE-IMAGE PIPELINE CONTROLLERS (used by original SAM pipeline)
# ============================================================================

class RunFinalInterpretationController:
    """
    [🧠 LLM Step]
    A controller that takes the 'final_prompt_parts' from the state,
    runs the LLM, and stores the 'result_json' and 'error_dict' back in the state.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn

    def execute(self, state: dict) -> dict:
        self.logger.info("🧠 LLM Step: Generating final scientific interpretation...")
        prompt = state.get("final_prompt_parts")
        
        if not prompt:
            self.logger.error("Pipeline reached final step, but no 'final_prompt_parts' in state.")
            state["error_dict"] = {"error": "Pipeline failed to build final prompt"}
            return state
        
        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            state["result_json"] = result_json
            state["error_dict"] = error_dict
            
            if not error_dict:
                self.logger.info("✅ LLM Step Complete: Final analysis generated.")
            else:
                self.logger.error(f"❌ LLM Step Failed: {error_dict.get('details')}")

        except Exception as e:
            self.logger.exception(f"❌ LLM Step Failed: {e}")
            state["result_json"] = None
            state["error_dict"] = {"error": "Final LLM analysis failed", "details": str(e)}

        return state


class StoreAnalysisResultsController:
    """
    [🛠️ Tool Step]
    A controller that takes the 'analysis_images' from the state
    and saves them using the agent's 'store_fn' for the feedback loop.
    """
    def __init__(self, logger: logging.Logger, store_fn: Callable):
        self.logger = logger
        self._store_analysis_images = store_fn

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Tool Step: Storing analysis images for feedback...")
        
        if state.get("error_dict"):
            self.logger.warning("Skipping storage: An error occurred in the pipeline.")
            return state

        try:
            analysis_metadata = {
                "image_path": state.get("image_path"),
                "system_info": state.get("system_info"),
                "num_stored_images": len(state.get("analysis_images", []))
            }
            self._store_analysis_images(state.get("analysis_images", []), analysis_metadata)
            self.logger.info("✅ Tool Step Complete: Analysis images stored.")
        except Exception as e:
            self.logger.error(f"❌ Tool Step Failed: Could not store analysis images: {e}")
            
        return state


class IterativeFeedbackController:
    """
    [🧠 LLM/User Step] 
    Facilitates human-in-the-loop validation and refinement for single-image analysis.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable, settings: dict, refinement_instruction: str):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.refinement_instruction = refinement_instruction
        self.feedback_depths = settings.get('feedback_depths', [0])

    def execute(self, state: dict) -> dict:
        if not state.get('settings', {}).get('enable_human_feedback', False):
            self.logger.info("Feedback skipped: Human feedback not enabled for this agent.")
            return state
        
        current_depth = state.get("current_depth", -1)
        
        if current_depth not in self.feedback_depths:
            self.logger.info(f"Feedback skipped: Current depth ({current_depth}) is not in allowed list {self.feedback_depths}.")
            return state

        decision = state.get("refinement_decision")
        if not decision:
            self.logger.warning("Feedback skipped: 'refinement_decision' missing from state.")
            return state

        self.logger.info("\n\n👤 --- USER STEP: REVIEW ANALYSIS PLAN --- 👤\n")
        
        iteration_title = state.get("iteration_title", "Current Analysis")
        analysis_text = state.get("result_json", {}).get("detailed_analysis", "No analysis text provided.")
        targets = decision.get("targets", [])
        
        print("\n" + "="*80)
        print(f"🎯 ANALYSIS STEP REVIEW: {iteration_title}")
        print("="*80)
        print("\n**SUMMARY OF CURRENT ANALYSIS:**")
        print(analysis_text)
        print("-" * 80)
        
        print(f"🧠 LLM's Proposed Plan: Refinement Needed = **{decision.get('refinement_needed', False)}**")
        print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        print(f"\nTargeted Actions ({len(targets)} found):")
        
        if not targets:
            print("  (No specific targets were generated.)")
        
        for i, t in enumerate(targets, 1):
            t_type = t.get('type', 'N/A')
            t_value = t.get('value', 'N/A')
            t_desc = t.get('description', 'No description provided.')
            
            print(f"  {i}. Type: {t_type:<15} | Value: {str(t_value):<15}")
            print(f"      Description: {t_desc}")
        
        print("-" * 80)
        
        try:
            user_feedback = input("\n🤔 Your feedback to adjust the targets/plan (or press Enter to accept): ").strip()
        except KeyboardInterrupt:
            self.logger.warning("User interrupted feedback. Accepting original decision.")
            return state

        if not user_feedback:
            self.logger.info("✅ User accepted original refinement decision.")
            return state
        
        self.logger.info("🔄 Refining decision using full scientific context...")
        
        system_info_json = json.dumps(state.get("system_info", {}), indent=2)

        prompt_parts = [
            f"You are an expert reviewer. Use the HUMAN EXPERT FEEDBACK to produce a **REVISED** and definitive version of the analysis plan JSON object. The human input overrides the initial automated logic.",
            
            f"\n\n--- LLM'S ORIGINAL DECISION JSON ---\n{json.dumps(decision, indent=2)}",
            f"\n\n--- CURRENT ITERATION'S DETAILED ANALYSIS ---\n{analysis_text}",
            f"\n\n--- CURRENT SYSTEM METADATA ---\n{system_info_json}",
            f"\n\n--- HUMAN EXPERT FEEDBACK ---\n\"{user_feedback}\"",
            "\n\n--- VISUAL CONTEXT (Plots from Current Analysis) ---\n"
        ]

        for img in state.get("analysis_images", []):
            image_bytes = img.get('data') or img.get('bytes')
            if image_bytes:
                prompt_parts.append(f"\n{img['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
        
        prompt_parts.append(f"\n\n### DECISION RULES\n{self.refinement_instruction}")

        prompt_parts.append("""

### REVISION REQUIREMENTS
You MUST re-analyze the original targets and the human feedback, then generate a single, complete, and definitive JSON object.

Your task is to provide the FINAL list of executable tasks. Do NOT embed descriptions or reasonings outside of the specified keys.
                            
You are FORBIDDEN from returning `refinement_needed: true` with an empty `targets` list. If refinement is needed, at least one target is required.
                            
Output must strictly adhere to the JSON format defined above.
""")

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            refined_json, error_dict = self._parse_llm_response(response)

            if refined_json and not error_dict:
                state["refinement_decision"] = refined_json
                self.logger.info(f"✅ Refinement success. Final decision: {refined_json.get('reasoning', 'No reasoning').strip()}")
                
                new_targets = refined_json.get("targets", [])
                print(f"\n✅ REFINED: New plan established based on feedback. ({len(new_targets)} targets created)")
            else:
                self.logger.error("❌ LLM failed to produce a valid refinement JSON. Retaining original decision.")
                print("\n❌ Refinement failed due to bad LLM output. Retaining original plan.")

        except Exception as e:
            self.logger.error(f"❌ Error during LLM refinement call: {e}")
            print("\n❌ Critical error during refinement. Retaining original plan.")
            
        return state


class RunSAMRefinementLoopController:
    """
    [🛠️ Tool Step + 🧠 LLM Step]
    Runs the entire SAM analysis and refinement loop for single-image pipeline.
    Contains its own internal LLM calls for automatic parameter tuning.
    """
    def __init__(self, model, logger, generation_config, safety_settings, settings: dict, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.settings = settings
        self._parse_llm_response = parse_fn
        self.refinement_cycles = self.settings.get('refinement_cycles', 0)
        self.save_visualizations = self.settings.get('save_visualizations', True)

    def _llm_get_refinement_params(self, original_image_bytes, overlay_image_bytes, particle_count, current_params) -> dict | None:
        """
        (Private) LLM call for automatic parameter refinement.
        """
        try:
            self.logger.info("   (Loop 🧠: Calling LLM for refinement parameters...)")
            
            prompt_parts = [SAM_ANALYSIS_REFINE_INSTRUCTIONS]
            prompt_parts.append(f"\n\nCurrent Analysis Results:")
            prompt_parts.append(f"- Particle count: {particle_count}")
            prompt_parts.append(f"- Current parameters: {json.dumps(current_params, indent=2)}")
            prompt_parts.append(f"\n\n**ORIGINAL MICROSCOPY IMAGE (for reference):**")
            prompt_parts.append({"mime_type": "image/jpeg", "data": original_image_bytes})
            prompt_parts.append(f"\n\n**CURRENT SEGMENTATION RESULT:**")
            prompt_parts.append({"mime_type": "image/jpeg", "data": overlay_image_bytes})
            prompt_parts.append("\n\n**ANALYSIS TASK:**")
            prompt_parts.append("Compare the segmentation result against the original image. Provide refined parameters to improve accuracy.")
            
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"   (Loop 🧠: ❌ LLM refinement call failed: {error_dict})")
                return None
            
            reasoning = result_json.get("reasoning", "No reasoning provided")
            new_parameters = result_json.get("parameters", {})
            
            self.logger.info(f"   (Loop 🧠: LLM refinement reasoning: {reasoning})")
            
            expected_keys = {"use_clahe", "sam_parameters", "min_area", "max_area", "pruning_iou_threshold"}
            if not all(key in new_parameters for key in expected_keys):
                self.logger.warning(f"   (Loop 🧠: ❌ Invalid parameter set from LLM, missing keys.")
                return None
            
            new_parameters["use_pruning"] = True
            return new_parameters
            
        except Exception as e:
            self.logger.error(f"   (Loop 🧠: ❌ Error in LLM refinement: {e})")
            return None

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n🛠️ --- CALLING TOOL: PARTICLE ANALYZER WITH SEGMENT ANYTHING MODEL --- 🛠️\n")
        
        try:
            image_array = state["preprocessed_image_array"]
            original_image_bytes = state["image_blob"]["data"]
            
            current_params = {
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
            
            self.logger.info(f"   (Loop 🛠️: Running initial SAM analysis...)")
            sam_result = run_sam_analysis(image_array, params=current_params)
            state["sam_result"] = sam_result
            
            if self.save_visualizations:
                initial_overlay = visualize_sam_results(sam_result)
                save_sam_visualization(initial_overlay, "initial", 0, sam_result['total_count'], current_params, self.logger)
            
            for cycle in range(self.refinement_cycles):
                self.logger.info(f"   (Loop 🔄: Starting refinement cycle {cycle + 1}/{self.refinement_cycles}...)")
                
                current_overlay_img = visualize_sam_results(sam_result)
                current_overlay_bytes = convert_numpy_to_jpeg_bytes(current_overlay_img)
                
                new_params = self._llm_get_refinement_params(
                    original_image_bytes,
                    current_overlay_bytes,
                    sam_result['total_count'],
                    current_params
                )
                
                if new_params is None or new_params == current_params:
                    self.logger.info("   (Loop 🔄: No valid parameter changes suggested. Stopping refinement.)")
                    break
                
                current_params.update(new_params)
                
                self.logger.info(f"   (Loop 🛠️: Re-running analysis with new params...)")
                sam_result = run_sam_analysis(image_array, params=current_params)
                state["sam_result"] = sam_result
                
                if self.save_visualizations:
                    refined_overlay = visualize_sam_results(sam_result)
                    save_sam_visualization(refined_overlay, "refined", cycle + 1, sam_result['total_count'], current_params, self.logger)
            
            self.logger.info("✅ SAM Workflow Complete.")

        except Exception as e:
            self.logger.error(f"❌ SAM Workflow Failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "SAM Analysis Workflow failed", "details": str(e)}

        return state


class CalculateSAMStatsController:
    """
    [🛠️ Tool Step]
    Takes the final 'sam_result' from the state, calculates stats,
    and puts them in 'summary_stats'.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Tool Step: Calling 'calculate_sam_statistics'...")
        sam_result = state.get("sam_result")
        
        if sam_result is None:
            self.logger.warning("   (Tool Info: No 'sam_result' in state. Skipping stats.)")
            state["summary_stats"] = {"total_particles": 0, "error": "Analysis failed in previous step."}
            return state

        try:
            summary_stats = calculate_sam_statistics(
                sam_result=sam_result,
                image_path=state["image_path"],
                preprocessed_image_shape=state["preprocessed_image_array"].shape,
                nm_per_pixel=state.get("nm_per_pixel")
            )
            state["summary_stats"] = summary_stats
            self.logger.info("✅ Tool Step Complete: Statistics calculated.")
        except Exception as e:
            self.logger.error(f"❌ Tool Step Failed: Stats calculation failed: {e}")
            state["summary_stats"] = {"total_particles": 0, "error": str(e)}
            
        return state


class BuildSAMPromptController:
    """
    [📝 Prep Step]
    Builds the final prompt, adding the SAM overlay and stats.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("📝 Prep Step: Building final prompt with SAM results...")
        
        prompt_parts = [state["instruction_prompt"]]
        
        if state.get("additional_top_level_context"):
            prompt_parts.append(f"\n\n## Special Considerations:\n{state['additional_top_level_context']}\n")
            
        prompt_parts.append("\n\nPrimary Microscopy Image:\n")
        prompt_parts.append(state["image_blob"])

        sam_result = state.get("sam_result")
        sam_stats = state.get("summary_stats")

        if sam_result is not None and sam_stats is not None:
            final_overlay_img = visualize_sam_results(sam_result)
            overlay_bytes = convert_numpy_to_jpeg_bytes(final_overlay_img)
            
            prompt_parts.append("\n\nSupplemental SAM Particle Segmentation Analysis:")
            prompt_parts.append(f"Detected {sam_stats.get('total_particles', 0)} particles.")
            
            prompt_parts.append("\n**Morphological Statistics Summary:**")
            for key, value in sam_stats.items():
                if isinstance(value, (int, float, str, list)):
                    prompt_parts.append(f"- {key}: {value}")
                elif isinstance(value, dict):
                    prompt_parts.append(f"- {key}: {json.dumps(value)}")

            prompt_parts.append("\nSAM Particle Segmentation Overlay (particles outlined in red):")
            prompt_parts.append({"mime_type": "image/jpeg", "data": overlay_bytes})
            
            state["analysis_images"].append({
                "label": "SAM Particle Segmentation Overlay",
                "data": overlay_bytes
            })
        else:
            prompt_parts.append("\n\n(No supplemental SAM analysis was run or it failed)")

        prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(state['system_info'], indent=2)}")
        prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        self.logger.info("✅ Prep Step Complete: Final prompt is ready.")
        return state


# ============================================================================
# BATCH PROCESSING CONTROLLERS
# ============================================================================

class HumanFeedbackRefinementController:
    """
    [👤 Human Step + 🧠 LLM Step]
    Facilitates human-in-the-loop parameter refinement for the first image in a batch.
    
    Options:
    1. Accept current results
    2. Accept LLM's recommended parameters  
    3. Provide feedback in natural language (LLM interprets)
    """
    
    def __init__(self, model, logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
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

        overlay_img = visualize_sam_results(sam_result)
        
        review_viz_path = self.output_dir / f"review_iteration_{iteration}.png"
        review_viz_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay_img).save(review_viz_path)

        print("\n" + "=" * 80)
        print(f"🔬 SAM ANALYSIS REVIEW - Iteration {iteration}")
        print("=" * 80)

        print(f"\n🖼️  **Visualization saved to:** {review_viz_path}")
        print(f"   Open this file to see the segmentation overlay.\n")
        
        print(f"\n📊 **Detection Summary:**")
        print(f"   - Total particles detected: {sam_stats.get('total_particles', 0)}")
        
        mean_area = sam_stats.get('mean_area_pixels', 'N/A')
        std_area = sam_stats.get('std_area_pixels', 'N/A')
        if isinstance(mean_area, (int, float)):
            print(f"   - Mean area: {mean_area:.2f} px²")
        else:
            print(f"   - Mean area: {mean_area}")
        if isinstance(std_area, (int, float)):
            print(f"   - Area std: {std_area:.2f} px²")
        else:
            print(f"   - Area std: {std_area}")
        
        if 'mean_area_nm2' in sam_stats:
            mean_area_nm = sam_stats.get('mean_area_nm2', 'N/A')
            if isinstance(mean_area_nm, (int, float)):
                print(f"   - Mean area (calibrated): {mean_area_nm:.2f} nm²")
        
        print(f"\n⚙️ **Current Parameters:**")
        current_params = state.get("current_params", {})
        for key, value in current_params.items():
            if key not in ['checkpoint_path', 'device']:
                print(f"   - {key}: {value}")
        
        print("-" * 80)
    
    def _get_llm_assessment(self, state: dict) -> dict:
        """Get LLM's assessment of the current segmentation quality."""
        from ..instruct import SAM_BATCH_REFINEMENT_INSTRUCTIONS
        
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
        print(f"   - False Positives: {evaluation.get('false_positive_rate', 'N/A')}")
        print(f"   - Overall Quality: {evaluation.get('overall_quality', 'N/A')}")
        print(f"\n   Reasoning: {llm_assessment.get('reasoning', 'N/A')}")
        
        if llm_assessment.get("needs_refinement"):
            print("\n   📝 LLM recommends refinement with these parameters:")
            rec_params = llm_assessment.get("recommended_parameters", {})
            for key, value in rec_params.items():
                print(f"      - {key}: {value}")
        
        print("\n" + "-" * 80)
        print("👤 **Your Options:**")
        print("   [1] Accept current results (proceed to batch processing)")
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
            print("No feedback provided. Accepting current results.")
            return {"action": "accept", "params": None}
        
        # Convert natural language to parameters via LLM
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
        
        prompt = f"""You are an expert in image segmentation. Convert the user's natural language feedback into SAM parameter adjustments.

**Current Parameters:**
{json.dumps(current_params, indent=2)}

**Available Parameters:**
- `use_clahe` (true/false): Enable contrast enhancement. Use for low-contrast images.
- `sam_parameters` ("default"/"sensitive"/"ultra-permissive"): Detection sensitivity. "sensitive" finds more objects, "ultra-permissive" maximizes detection.
- `min_area` (integer, pixels): Minimum particle size. Lower = detect smaller particles.
- `max_area` (integer, pixels): Maximum particle size. Lower = reject large merged detections.
- `pruning_iou_threshold` (0.0-1.0): Overlap threshold for removing duplicates. Lower = more aggressive merging.

**User Feedback:**
"{user_feedback}"

**Task:**
Return a JSON object with ONLY the parameters that should be changed based on the feedback.

Example response:
{{"min_area": 200, "sam_parameters": "sensitive"}}

Return only the JSON object, no explanation."""

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict or not result_json:
                self.logger.warning("Failed to parse LLM parameter conversion.")
                return None
            
            # Validate and show the interpreted parameters
            print(f"\n   ✅ Interpreted as: {json.dumps(result_json, indent=2)}")
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error converting feedback: {e}")
            return None
    
    def execute(self, state: dict) -> dict:
        """Execute the human feedback refinement loop."""
        if not state.get('enable_human_feedback', False):
            self.logger.info("Human feedback disabled. Skipping refinement loop.")
            state["final_params_for_batch"] = state.get("current_params", {})
            return state
        
        self.logger.info("\n\n👤 --- HUMAN FEEDBACK REFINEMENT LOOP --- 👤\n")
        
        iteration = 0
        while iteration < self.max_refinement_iterations:
            iteration += 1
            
            # Display current results
            self._display_analysis_for_review(state, iteration)
            
            # Get LLM assessment
            llm_assessment = self._get_llm_assessment(state)
            
            # Collect human feedback
            feedback = self._collect_human_feedback(llm_assessment)
            
            if feedback["action"] == "accept":
                self.logger.info("✅ User accepted current results.")
                state["refinement_complete"] = True
                state["final_params_for_batch"] = state.get("current_params", {})
                break
            
            elif feedback["action"] in ["use_llm", "custom"]:
                new_params = feedback["params"]
                if new_params:
                    # Update parameters
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
            self.logger.warning(f"⚠️ Max iterations ({self.max_refinement_iterations}) reached. Using current parameters.")
            state["final_params_for_batch"] = state.get("current_params", {})
        
        return state


class BatchImageProcessingController:
    """
    [🛠️ Tool Step]
    Processes a series of images using the refined parameters from the first image.
    
    NOW WITH MODEL CACHING - loads SAM model once and reuses for all images.
    This reduces batch processing time by approximately 10x on CPU.
    
    Supports both:
    - List of file paths: ["img1.tif", "img2.tif", ...]
    - 3D numpy array: shape (n, h, w) where n is number of images
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.save_visualizations = settings.get('save_visualizations', True)
        self.output_dir = Path(settings.get('output_dir', 'sam_batch_output'))
    
    def _get_image_and_name(self, state: dict, idx: int) -> tuple:
        """
        Get image array and name for a given index.
        Handles both file paths and numpy array stack inputs.
        
        Returns:
            Tuple of (image_array, image_name)
        """
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
            raise ValueError("No image data found in state (neither image_stack nor image_paths)")
    
    def _get_num_images(self, state: dict) -> int:
        """Get the total number of images to process."""
        image_stack = state.get("image_stack")
        image_paths = state.get("image_paths")
        
        if image_stack is not None:
            return image_stack.shape[0]
        elif image_paths:
            return len(image_paths)
        return 0
    
    def execute(self, state: dict) -> dict:
        """
        Process all images in the batch using refined parameters.
        
        Key improvement: Loads the SAM model ONCE and reuses it for all images.
        """
        self.logger.info("\n\n🔄 --- BATCH IMAGE PROCESSING --- 🔄\n")
        
        num_images = self._get_num_images(state)
        if num_images == 0:
            self.logger.warning("No images provided for batch processing.")
            state["batch_results"] = []
            return state
        
        input_type = "numpy array" if state.get("image_stack") is not None else "file paths"
        self.logger.info(f"📦 Input type: {input_type}")
        
        # Get refined parameters from first image analysis
        batch_params = state.get("final_params_for_batch", {})
        if not batch_params:
            self.logger.warning("No refined parameters found. Using default settings.")
            batch_params = state.get("current_params", {})
        
        self.logger.info(f"📦 Processing {num_images} images with refined parameters:")
        for key, value in batch_params.items():
            if key not in ['checkpoint_path', 'device']:
                self.logger.info(f"   - {key}: {value}")
        
        # ============================================================
        # KEY OPTIMIZATION: Load SAM model ONCE before the loop
        # ============================================================
        self.logger.info("\n🧠 Loading SAM model (once for entire batch)...")
        try:
            sam_analyzer = get_or_create_sam_model(batch_params)
            self.logger.info("✅ SAM model loaded and cached.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load SAM model: {e}")
            state["batch_results"] = []
            state["error_dict"] = {"error": "SAM model loading failed", "details": str(e)}
            return state
        
        # Get spatial calibration if available
        nm_per_pixel = state.get("nm_per_pixel")
        
        batch_results = []
        
        for idx in range(num_images):
            try:
                # Load and preprocess image
                raw_image, image_name = self._get_image_and_name(state, idx)
                self.logger.info(f"\n   [{idx + 1}/{num_images}] Processing: {image_name}")
                
                preprocessed_img, _ = preprocess_image(raw_image)
                
                # ============================================================
                # KEY CHANGE: Pass the cached analyzer to avoid reloading
                # ============================================================
                sam_result = run_sam_analysis(
                    preprocessed_img, 
                    params=batch_params,
                    analyzer=sam_analyzer  # Reuse cached model!
                )
                
                # Get image path for statistics metadata
                image_paths = state.get("image_paths")
                if image_paths and idx < len(image_paths):
                    image_path_for_stats = image_paths[idx]
                else:
                    image_path_for_stats = f"frame_{idx:04d}"
                
                # Calculate statistics
                summary_stats = calculate_sam_statistics(
                    sam_result=sam_result,
                    image_path=str(image_path_for_stats),
                    preprocessed_image_shape=preprocessed_img.shape,
                    nm_per_pixel=nm_per_pixel
                )
                
                # Save visualization if enabled
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
                
                # Build result entry
                result_entry = {
                    "index": idx,
                    "image_path": str(image_paths[idx]) if image_paths and idx < len(image_paths) else None,
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
                    "image_path": str(state.get("image_paths", [None])[idx]) if state.get("image_paths") else None,
                    "image_name": f"frame_{idx:04d}",
                    "visualization_path": None,
                    "particle_count": 0,
                    "statistics": {},
                    "success": False,
                    "error": str(e)
                })
        
        # Save batch results to JSON
        results_path = self.output_dir / "batch_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_images": num_images,
                "successful": sum(1 for r in batch_results if r["success"]),
                "parameters_used": batch_params,
                "input_type": input_type,
                "results": batch_results
            }, f, indent=2, default=str)
        
        state["batch_results"] = batch_results
        state["batch_results_path"] = str(results_path)
        
        successful = sum(1 for r in batch_results if r["success"])
        self.logger.info(f"\n✅ Batch processing complete: {successful}/{num_images} images processed successfully.")
        
        return state


class CustomAnalysisScriptController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates and executes a custom Python script for trend analysis.
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
        self.output_dir = Path(settings.get('output_dir', 'sam_batch_output'))
        self.max_correction_attempts = settings.get('max_script_corrections', 3)
    
    def _generate_analysis_script(self, state: dict) -> Optional[Dict[str, Any]]:
        """Generate custom analysis script using LLM."""
        from ..instruct import SAM_BATCH_CUSTOM_ANALYSIS_INSTRUCTIONS
        
        self.logger.info("   🧠 Generating custom analysis script...")
        
        batch_results = state.get("batch_results", [])
        series_metadata = state.get("series_metadata", {})
        
        # Build summary statistics
        particle_counts = []
        mean_areas = []
        
        for r in batch_results:
            if r["success"]:
                particle_counts.append(r["particle_count"])
                stats = r.get("statistics", {})
                mean_area = stats.get("mean_area_pixels")
                if mean_area is not None:
                    mean_areas.append(round(mean_area, 2))
                else:
                    mean_areas.append(0)
        
        time_points = series_metadata.get("time_points") or list(range(len(batch_results)))
        
        summary = {
            "total_images": len(batch_results),
            "successful": sum(1 for r in batch_results if r["success"]),
            "particle_counts": particle_counts,
            "mean_areas": mean_areas,
            "series_type": series_metadata.get("series_type", "unknown"),
            "time_points": time_points,
            "time_unit": series_metadata.get("time_unit", "frames"),
        }
        
        # Load representative visualizations
        viz_images = []
        for r in batch_results[:3]:
            viz_path = r.get("visualization_path")
            if viz_path and Path(viz_path).exists():
                try:
                    with open(viz_path, 'rb') as f:
                        viz_images.append(f.read())
                except Exception:
                    pass
        
        # Build prompt
        prompt_parts = [
            SAM_BATCH_CUSTOM_ANALYSIS_INSTRUCTIONS,
            f"\n\n**BATCH SUMMARY:**\n```json\n{json.dumps(summary, indent=2)}\n```",
            f"\n\n**SERIES METADATA:**\n```json\n{json.dumps(series_metadata, indent=2)}\n```",
        ]
        
        # Add schema documentation (concise)
        prompt_parts.append("""

**batch_results.json STRUCTURE:**
```json
{
  "results": [
    {
      "index": <int>,
      "particle_count": <int>,
      "statistics": {
        "mean_area_pixels": <float>,
        "mean_area_nm2": <float>,
        "mean_circularity": <float>,
        "mean_solidity": <float>,
        ...
      },
      "success": <bool>
    },
    ...
  ]
}
```

Extract data like this:
```python
data = json.load(open('batch_results.json'))
results = data['results']
counts = [r['particle_count'] for r in results if r['success']]
areas = [r['statistics']['mean_area_pixels'] for r in results if r['success']]
```
""")
        
        # Add visualizations
        if viz_images:
            prompt_parts.append("\n\n**REPRESENTATIVE VISUALIZATIONS:**")
            for i, viz_bytes in enumerate(viz_images):
                prompt_parts.append(f"\nImage {i + 1}:")
                prompt_parts.append({"mime_type": "image/png", "data": viz_bytes})
        
        prompt_parts.append("\n\nReturn your JSON response now:")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                if result_json and 'script' in result_json:
                    return result_json
                self.logger.error(f"Script generation failed: {error_dict}")
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
        
        self.logger.info(f"   📝 Script saved to: {script_path}")
        
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
            return False, "", "Script execution timed out (5 minutes)"
        except Exception as e:
            return False, "", str(e)
    
    def _correct_script(self, original_script: str, error_message: str, attempt: int) -> Optional[str]:
        """Use LLM to correct a failed script."""
        self.logger.info(f"   🔧 Attempting script correction (attempt {attempt})...")
        
        if len(error_message) > 1000:
            error_message = error_message[:500] + "\n...[truncated]...\n" + error_message[-500:]
        
        prompt = f"""Fix this Python script that failed to execute.

**SCRIPT:**
```python
{original_script}
```

**ERROR:**
```
{error_message}
```

**REMINDER - batch_results.json structure:**
```python
data = json.load(open('batch_results.json'))
results = data['results']  # array of result objects
counts = [r['particle_count'] for r in results if r['success']]
areas = [r['statistics']['mean_area_pixels'] for r in results if r['success']]
```

Return JSON with:
{{
  "diagnosis": "what caused the error",
  "script": "corrected Python script"
}}
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
            
            self.logger.info(f"   📋 Diagnosis: {result_json.get('diagnosis', 'N/A')}")
            return result_json.get("script")
            
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None
    
    def execute(self, state: dict) -> dict:
        """Generate and execute custom analysis script."""
        self.logger.info("\n\n🧠 --- CUSTOM ANALYSIS SCRIPT GENERATION --- 🧠\n")
        
        batch_results = state.get("batch_results", [])
        
        if not batch_results or len(batch_results) < 2:
            self.logger.warning("Insufficient batch results for trend analysis.")
            state["custom_analysis_results"] = {
                "success": False,
                "error": "Need at least 2 images for trend analysis"
            }
            return state
        
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
                self.logger.info(f"   🔄 Execution attempt {attempt + 1}/{self.max_correction_attempts + 1}")
            
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
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics_to_track"),
            "reasoning": script_result.get("reasoning"),
            "stdout": stdout,
            "stderr": stderr if not success else None,
            "generated_files": [str(f) for f in generated_files],
            "script_path": str(self.output_dir / "custom_analysis.py")
        }
        
        if success:
            self.logger.info(f"   📁 Generated {len(generated_files)} output files")
            
            # Save JSON summary (report generation happens after synthesis)
            self._save_analysis_summary(state, script_result)
        
        return state
    
    def _save_analysis_summary(self, state: dict, script_result: dict) -> None:
        """Save structured JSON summary of the analysis."""
        batch_results = state.get("batch_results", [])
        custom_results = state.get("custom_analysis_results", {})
        
        # Extract key metrics from batch results
        metrics_over_time = []
        for r in batch_results:
            if r["success"]:
                stats = r.get("statistics", {})
                metrics_over_time.append({
                    "index": r["index"],
                    "image_name": r.get("image_name"),
                    "particle_count": r["particle_count"],
                    "mean_area_pixels": stats.get("mean_area_pixels"),
                    "mean_area_nm2": stats.get("mean_area_nm2"),
                    "mean_circularity": stats.get("mean_circularity"),
                    "mean_solidity": stats.get("mean_solidity"),
                    "mean_aspect_ratio": stats.get("mean_aspect_ratio"),
                })
        
        summary = {
            "analysis_info": {
                "timestamp": state.get("batch_results", [{}])[0].get("statistics", {}).get("parameters_used", {}),
                "approach": script_result.get("analysis_approach"),
                "metrics_tracked": script_result.get("key_metrics_to_track"),
                "reasoning": script_result.get("reasoning"),
            },
            "batch_overview": {
                "total_images": len(batch_results),
                "successful": sum(1 for r in batch_results if r["success"]),
            },
            "metrics_over_time": metrics_over_time,
            "generated_plots": custom_results.get("generated_files", []),
            "script_output": custom_results.get("stdout", ""),
        }
        
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"   📄 Saved analysis summary: {summary_path}")
    


class ReportGenerationController:
    """
    [📄 Report Step]
    Generates final HTML report and JSON summary after synthesis is complete.
    Should be the LAST controller in the pipeline.
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'sam_batch_output'))
    
    def execute(self, state: dict) -> dict:
        """Generate final reports using synthesis results."""
        self.logger.info("\n\n📄 --- GENERATING FINAL REPORTS --- 📄\n")
        
        custom_results = state.get("custom_analysis_results", {})
        
        if not custom_results.get("success"):
            self.logger.warning("Skipping report generation: custom analysis was not successful.")
            return state
        
        # Generate HTML report
        self._generate_html_report(state)
        
        # Update generated files list
        report_path = self.output_dir / "analysis_report.html"
        summary_path = self.output_dir / "analysis_summary.json"
        
        if "generated_files" not in custom_results:
            custom_results["generated_files"] = []
        
        if report_path.exists():
            custom_results["generated_files"].append(str(report_path))
        if summary_path.exists():
            custom_results["generated_files"].append(str(summary_path))
        
        return state
    
    def _generate_html_report(self, state: dict) -> None:
        """Generate a professional HTML report with embedded figures and scientific synthesis."""
        import base64
        from datetime import datetime
        
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        synthesis_result = state.get("synthesis_result", {})
        
        # Extract synthesis content
        detailed_analysis = synthesis_result.get("detailed_analysis", "No synthesis available.")
        scientific_claims = synthesis_result.get("scientific_claims", [])
        
        # Collect PNG files
        png_files = sorted(self.output_dir.glob("*.png"))
        
        # Build embedded images
        embedded_images = []
        for png_path in png_files:
            if png_path.name.startswith("review_iteration"):
                continue
            if png_path.exists():
                with open(png_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    embedded_images.append({
                        "name": png_path.stem,
                        "data": b64
                    })
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM Batch Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f9;
        }}
        .container {{
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        .metadata-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #3498db;
            margin-bottom: 20px;
        }}
        .metadata-box p {{
            margin: 5px 0;
        }}
        .analysis-text {{
            white-space: pre-wrap;
            background-color: #fafafa;
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #eee;
            font-size: 0.95em;
            line-height: 1.8;
        }}
        .claim-card {{
            background-color: #e8f6f3;
            border-left: 5px solid #1abc9c;
            padding: 15px 20px;
            margin-bottom: 15px;
            border-radius: 0 5px 5px 0;
        }}
        .claim-title {{
            font-weight: bold;
            font-size: 1.05em;
            color: #0e6655;
            margin-bottom: 8px;
        }}
        .claim-card p {{
            margin: 5px 0;
        }}
        .keyword-tag {{
            display: inline-block;
            background: #d5f5e3;
            color: #1e8449;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-right: 5px;
            margin-top: 3px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .image-card {{
            background: white;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .image-label {{
            margin-top: 12px;
            font-weight: 600;
            color: #444;
            border-top: 1px solid #eee;
            padding-top: 10px;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.8em;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 SAM Batch Analysis Report</h1>
        
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Analysis Approach:</strong> {custom_results.get("approach", "time_series")}</p>
            <p><strong>Series Type:</strong> {series_metadata.get("series_type", "unknown")}</p>
            <p><strong>Metrics Tracked:</strong> {", ".join(custom_results.get("metrics_tracked", [])) or "particle_count, mean_area"}</p>
        </div>

        <h2>1. Scientific Analysis</h2>
        <div class="analysis-text">{detailed_analysis}</div>
"""

        # Visualizations section
        if embedded_images:
            html += """
        <h2>2. Visualizations</h2>
        <div class="image-grid">
"""
            for img in embedded_images:
                display_name = img['name'].replace('_', ' ').replace('-', ' ').title()
                html += f"""            <div class="image-card">
                <img src="data:image/png;base64,{img['data']}" alt="{img['name']}" loading="lazy">
                <div class="image-label">{display_name}</div>
            </div>
"""
            html += "        </div>\n"

        # Scientific claims section
        if scientific_claims:
            html += "        <h2>3. Key Scientific Claims</h2>\n"
            for i, claim in enumerate(scientific_claims, 1):
                claim_text = claim.get('claim', 'N/A')
                impact = claim.get('scientific_impact', 'N/A')
                evidence = claim.get('supporting_evidence', '')
                question = claim.get('has_anyone_question', 'N/A')
                keywords = claim.get('keywords', [])
                
                html += f"""        <div class="claim-card">
            <div class="claim-title">Claim {i}: {claim_text}</div>
            <p><strong>Scientific Impact:</strong> {impact}</p>
"""
                if evidence:
                    html += f"            <p><strong>Supporting Evidence:</strong> {evidence}</p>\n"
                html += f"            <p><strong>Research Question:</strong> <em>{question}</em></p>\n"
                if keywords:
                    html += "            <div><strong>Keywords:</strong> "
                    html += " ".join(f'<span class="keyword-tag">{kw}</span>' for kw in keywords)
                    html += "</div>\n"
                html += "        </div>\n"

        html += """
        <div class="footer">Generated by SAM Batch Analysis Agent</div>
    </div>
</body>
</html>
"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated HTML report: {report_path}")


class BatchSynthesisController:
    """
    [🧠 LLM Step]
    Synthesizes findings from batch analysis into cohesive scientific claims.
    """
    
    def __init__(self, model, logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
    
    def execute(self, state: dict) -> dict:
        """Synthesize batch analysis findings."""
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
                    "std_area": r["statistics"].get("std_area_pixels")
                })
        
        prompt_parts = [
            SAM_BATCH_SYNTHESIS_INSTRUCTIONS,
            f"\n\n**INDIVIDUAL ANALYSIS SUMMARY:**\n{json.dumps(stats_summary, indent=2)}",
            f"\n\n**CUSTOM ANALYSIS RESULTS:**\n{json.dumps(custom_results, indent=2)}",
            f"\n\n**SERIES METADATA:**\n{json.dumps(series_metadata, indent=2)}"
        ]
        
        if custom_results and custom_results.get("success") and custom_results.get("generated_files"):
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