"""
Microscopy Analysis Controllers - Unified Architecture

This module contains unified controllers that handle both single image (n=1)
and batch (n>1) analysis identically. The key principle is:

    Single image = Batch of 1

All controllers adapt their behavior based on state["is_single_image"]
and state["num_images"], but use the same code paths.
"""

import subprocess
import json
import logging
import sys
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Dict
import numpy as np

from PIL import Image

from ..instruct import (
    FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS,
    SINGLE_IMAGE_ANALYSIS_INSTRUCTIONS,
    SERIES_ANALYSIS_INSTRUCTIONS,
)

from ....tools.image_processor import (
    load_image,
    preprocess_image,
    convert_numpy_to_jpeg_bytes,
    normalize_and_convert_to_image_bytes,
    calculate_global_fft
)
from ....tools.fft_nmf import SlidingFFTNMF


# ============================================================================
# INITIAL FFT ANALYSIS CONTROLLER
# ============================================================================

class InitialFFTAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Analyzes the first frame with LLM-guided parameters.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if state.get("locked_params"):
            self.logger.info("📋 Preset parameters provided, skipping initial analysis.")
            return state
        
        is_single = state.get("is_single_image", False)
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({state.get('num_images', 1)} images)"
        self.logger.info(f"\n\n🔬 --- INITIAL FFT ANALYSIS ({mode_str}) --- 🔬\n")
        
        # Get LLM parameter suggestions
        self.logger.info("🧠 LLM Step: Reasoning about FFT/NMF parameters...")
        
        image_blob = state["image_blob"]
        system_info = state.get("system_info", {})
        
        prompt_parts = [FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)
        if system_info:
            prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(system_info, indent=2)}")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            llm_params = json.loads(response.text)
            state["llm_params"] = llm_params
            state["current_params"] = llm_params
            
            print("\n" + "="*60)
            print("🧠 LLM REASONING")
            print(f"   Explanation: {llm_params.get('explanation', 'N/A')}")
            print(f"   Params: window_size_nm={llm_params.get('window_size_nm')}, n_components={llm_params.get('n_components')}")
            print("="*60 + "\n")
            
        except Exception as e:
            self.logger.error(f"❌ LLM parameter estimation failed: {e}")
            llm_params = {"window_size_nm": 10.0, "n_components": 4, "explanation": "Using defaults"}
            state["llm_params"] = llm_params
            state["current_params"] = llm_params
        
        # Run Global FFT
        self.logger.info("🛠️ Running Global FFT...")
        try:
            image_array = state["preprocessed_image_array"]
            output_dir = self.settings.get("visualization_dir", ".")
            
            base_name = os.path.splitext(os.path.basename(str(state.get("image_path", "image"))))[0]
            safe_name = "".join(c if c.isalnum() else "_" for c in base_name)
            filepath = os.path.join(output_dir, f"{safe_name}_global_fft.png")
            
            global_fft_image = calculate_global_fft(image_array, save_path=filepath)
            state["global_fft_image"] = global_fft_image
            self.logger.info("✅ Global FFT complete.")
        except Exception as e:
            self.logger.error(f"❌ Global FFT failed: {e}")
            state["global_fft_image"] = None
        
        # Run FFT/NMF
        self.logger.info("🛠️ Running Sliding FFT + NMF...")
        
        ws_nm = llm_params.get("window_size_nm")
        nc = llm_params.get("n_components", 4)
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        if ws_nm and nm_per_pixel and nm_per_pixel > 0:
            ws_pixels = int(round(ws_nm / nm_per_pixel))
            good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
            ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        else:
            ws_pixels = 64
        
        step = max(1, ws_pixels // 4)
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels, window_size_y=ws_pixels,
                window_step_x=step, window_step_y=step, components=nc
            )
            
            image_array = state["preprocessed_image_array"]
            components, abundances = analyzer.analyze(image_array, output_dir=None)
            
            state["fft_components"] = components
            state["fft_abundances"] = abundances
            state["summary_stats"] = self._compute_stats(components, abundances)
            
            self.logger.info(f"✅ FFT/NMF complete. {nc} components extracted.")
            
        except Exception as e:
            self.logger.error(f"❌ FFT/NMF failed: {e}")
            state["fft_components"] = None
            state["fft_abundances"] = None
        
        state["first_frame_results"] = {
            "components": state.get("fft_components"),
            "abundances": state.get("fft_abundances"),
            "llm_params": llm_params
        }
        
        return state
    
    def _compute_stats(self, components: np.ndarray, abundances: np.ndarray) -> dict:
        if components is None or abundances is None:
            return {}
        
        stats = {"n_components": components.shape[0], "components": []}
        
        for i in range(components.shape[0]):
            abun = abundances[i]
            stats["components"].append({
                "index": i + 1,
                "abundance_mean": float(np.mean(abun)),
                "abundance_std": float(np.std(abun)),
                "spatial_coverage": float(np.sum(abun > np.mean(abun)) / abun.size)
            })
        
        return stats


# ============================================================================
# HUMAN FEEDBACK REFINEMENT CONTROLLER
# ============================================================================

class HumanFeedbackRefinementController:
    """
    [👤 Human Step]
    Facilitates human-in-the-loop parameter refinement for the first image.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.max_refinement_iterations = settings.get('max_feedback_iterations', 3)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if not state.get('enable_human_feedback', False):
            self.logger.info("Human feedback disabled. Using current parameters.")
            state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
            return state
        
        if state.get("preset_params"):
            self.logger.info("Preset parameters provided. Skipping feedback loop.")
            state["locked_params"] = state.get("preset_params")
            return state
        
        is_single = state.get("is_single_image", False)
        mode_str = "SINGLE IMAGE" if is_single else "BATCH"
        self.logger.info(f"\n\n👤 --- HUMAN FEEDBACK REFINEMENT ({mode_str}) --- 👤\n")
        
        iteration = 0
        while iteration < self.max_refinement_iterations:
            iteration += 1
            
            self._display_analysis_for_review(state, iteration)
            feedback = self._collect_human_feedback(state)
            
            if feedback["action"] == "accept":
                self.logger.info("✅ User accepted current results.")
                state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
                break
            
            elif feedback["action"] == "cancel":
                self.logger.info("❌ User cancelled analysis.")
                state["batch_cancelled"] = True
                return state
            
            elif feedback["action"] == "modify" and feedback.get("params"):
                state = self._rerun_analysis(state, feedback["params"])
        
        if iteration >= self.max_refinement_iterations:
            self.logger.warning(f"⚠️ Max iterations reached. Using current parameters.")
            state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
        
        return state
    
    def _display_analysis_for_review(self, state: dict, iteration: int) -> None:
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        llm_params = state.get("llm_params", {})
        is_single = state.get("is_single_image", False)
        
        review_viz_path = self.output_dir / f"review_iteration_{iteration}.png"
        
        if components is not None and abundances is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = components.shape[0]
            fig, axes = plt.subplots(2, n_comps, figsize=(4*n_comps, 8))
            if n_comps == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_comps):
                axes[0, i].imshow(components[i], cmap='viridis')
                axes[0, i].set_title(f'Component {i+1}')
                axes[0, i].axis('off')
                axes[1, i].imshow(abundances[i], cmap='hot')
                axes[1, i].set_title(f'Abundance {i+1}')
                axes[1, i].axis('off')
            
            plt.suptitle(f'FFT/NMF Analysis - Iteration {iteration}', fontsize=14)
            plt.tight_layout()
            plt.savefig(review_viz_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print("\n" + "=" * 80)
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({state.get('num_images', 1)} images)"
        print(f"🔬 FFT/NMF ANALYSIS REVIEW - {mode_str} - Iteration {iteration}")
        print("=" * 80)
        print(f"\n🖼️  Visualization saved to: {review_viz_path}")
        if components is not None:
            print(f"\n📊 Components extracted: {components.shape[0]}")
        print(f"\n⚙️ Parameters: window_size_nm={llm_params.get('window_size_nm', 'auto')}, n_components={llm_params.get('n_components', 4)}")
        if not is_single:
            print(f"\n📦 Note: These parameters will be applied to all {state.get('num_images', 1)} images.")
        print("-" * 80)
    
    def _collect_human_feedback(self, state: dict) -> dict:
        llm_params = state.get("llm_params", {})
        
        print("\n👤 Options: [1] Accept  [2] Modify parameters  [c] Cancel")
        
        try:
            choice = input("\nChoice [1/2/c]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return {"action": "accept"}
        
        if choice == '1' or choice == '':
            return {"action": "accept"}
        elif choice == 'c':
            return {"action": "cancel"}
        elif choice == '2':
            mods = {}
            try:
                ws = input(f"   Window size (nm) [{llm_params.get('window_size_nm', 'auto')}]: ").strip()
                if ws:
                    mods['window_size_nm'] = float(ws)
                nc = input(f"   Components [{llm_params.get('n_components', 4)}]: ").strip()
                if nc:
                    mods['n_components'] = int(nc)
            except (KeyboardInterrupt, EOFError, ValueError):
                pass
            return {"action": "modify", "params": mods} if mods else {"action": "accept"}
        return {"action": "accept"}
    
    def _rerun_analysis(self, state: dict, new_params: dict) -> dict:
        self.logger.info(f"🔄 Re-running FFT/NMF with updated parameters...")
        
        llm_params = state.get("llm_params", {}).copy()
        llm_params.update(new_params)
        state["llm_params"] = llm_params
        state["current_params"] = llm_params
        
        ws_nm = llm_params.get("window_size_nm")
        nc = llm_params.get("n_components", 4)
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        if ws_nm and nm_per_pixel and nm_per_pixel > 0:
            ws_pixels = int(round(ws_nm / nm_per_pixel))
            good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
            ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        else:
            ws_pixels = 64
        
        step = max(1, ws_pixels // 4)
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels, window_size_y=ws_pixels,
                window_step_x=step, window_step_y=step, components=nc
            )
            components, abundances = analyzer.analyze(state["preprocessed_image_array"], output_dir=None)
            state["fft_components"] = components
            state["fft_abundances"] = abundances
            state["first_frame_results"] = {"components": components, "abundances": abundances, "llm_params": llm_params}
            self.logger.info(f"✅ Re-analysis complete. {nc} components extracted.")
        except Exception as e:
            self.logger.error(f"❌ Re-analysis failed: {e}")
        
        return state


# ============================================================================
# UNIFIED BATCH PROCESSING CONTROLLER
# ============================================================================

class UnifiedBatchProcessingController:
    """
    [🛠️ Tool Step]
    Processes ALL images using the refined parameters.
    Single image = batch of 1.
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.save_visualizations = settings.get('save_visualizations', True)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", False)
        
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({num_images} images)"
        self.logger.info(f"\n\n🔄 --- PROCESSING: {mode_str} --- 🔄\n")
        
        # For single image, use existing results
        if is_single and state.get("fft_components") is not None:
            self.logger.info("📋 Using results from initial analysis (single image).")
            stats = state.get("summary_stats", {})
            state["batch_results"] = [{
                "index": 0,
                "image_path": state.get("image_path"),
                "image_name": state.get("first_image_name", "image"),
                "n_components": state["fft_components"].shape[0],
                "statistics": stats,
                "success": True,
                "error": None
            }]
            if self.save_visualizations:
                self._save_visualization(state, 0)
            return state
        
        # Batch processing
        locked_params = state.get("locked_params", state.get("current_params", {}))
        
        ws_nm = locked_params.get("window_size_nm", 10.0)
        n_components = locked_params.get("n_components", 4)
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        if ws_nm and nm_per_pixel and nm_per_pixel > 0:
            ws_pixels = int(round(ws_nm / nm_per_pixel))
            good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
            ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        else:
            ws_pixels = 64
        
        step = max(1, ws_pixels // 4)
        
        self.logger.info(f"📦 Processing with: window={ws_pixels}px, components={n_components}")
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels, window_size_y=ws_pixels,
                window_step_x=step, window_step_y=step, components=n_components
            )
        except Exception as e:
            state["error_dict"] = {"error": "Analyzer creation failed", "details": str(e)}
            return state
        
        batch_results = []
        all_abundances = []
        
        for idx in range(num_images):
            try:
                raw_image, image_name = self._get_image(state, idx)
                self.logger.info(f"   [{idx + 1}/{num_images}] Processing: {image_name}")
                
                if raw_image.dtype in [np.float32, np.float64]:
                    img_min, img_max = raw_image.min(), raw_image.max()
                    if img_max > img_min:
                        raw_image = ((raw_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        raw_image = np.zeros_like(raw_image, dtype=np.uint8)
                
                preprocessed_img, _ = preprocess_image(raw_image)
                components, abundances = analyzer.analyze(preprocessed_img, output_dir=None)
                all_abundances.append(abundances)
                
                batch_results.append({
                    "index": idx, "image_name": image_name,
                    "n_components": components.shape[0],
                    "statistics": {"components": [{"index": i+1, "mean": float(np.mean(abundances[i]))} for i in range(components.shape[0])]},
                    "success": True, "error": None
                })
                self.logger.info(f"      ✅ Extracted {components.shape[0]} components")
                
            except Exception as e:
                self.logger.error(f"      ❌ Failed: {e}")
                batch_results.append({"index": idx, "image_name": f"frame_{idx:04d}", "success": False, "error": str(e)})
        
        if num_images > 1 and all_abundances:
            state["series_abundances"] = np.stack(all_abundances, axis=0)
            state["series_components"] = components
            np.save(self.output_dir / "series_components.npy", state["series_components"])
            np.save(self.output_dir / "series_abundances.npy", state["series_abundances"])
        
        state["batch_results"] = batch_results
        state["batch_params"] = {"window_size_pixels": ws_pixels, "window_size_nm": ws_nm, "n_components": n_components, "n_frames": num_images}
        
        return state
    
    def _get_image(self, state: dict, idx: int) -> tuple:
        image_stack = state.get("image_stack")
        image_paths = state.get("image_paths")
        
        if image_stack is not None:
            return image_stack[idx], f"frame_{idx:04d}"
        elif image_paths:
            return load_image(image_paths[idx]), Path(image_paths[idx]).stem
        raise ValueError("No image data found")
    
    def _save_visualization(self, state: dict, idx: int) -> None:
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        if components is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_comps = components.shape[0]
        fig, axes = plt.subplots(2, n_comps, figsize=(4*n_comps, 8))
        if n_comps == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_comps):
            axes[0, i].imshow(components[i], cmap='viridis')
            axes[0, i].set_title(f'Component {i+1}')
            axes[0, i].axis('off')
            axes[1, i].imshow(abundances[i], cmap='hot')
            axes[1, i].set_title(f'Abundance {i+1}')
            axes[1, i].axis('off')
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(viz_dir / f"fft_nmf_{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# CONDITIONAL CUSTOM ANALYSIS CONTROLLER
# ============================================================================

class ConditionalCustomAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates trend analysis script for n>=2, skipped for n=1.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.max_correction_attempts = settings.get('max_script_corrections', 3)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        if state.get("is_single_image", False) or state.get("num_images", 1) < 2:
            self.logger.info("\n📊 Custom trend analysis skipped (single image mode).\n")
            state["custom_analysis_results"] = {"success": True, "skipped": True, "reason": "Single image"}
            return state
        
        self.logger.info("\n\n🧠 --- CUSTOM ANALYSIS SCRIPT GENERATION --- 🧠\n")
        
        # Generate and execute script (simplified)
        script = self._fallback_script()
        script_path = self.output_dir / "analyze_results.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(['python', str(script_path)], capture_output=True, text=True, timeout=300, cwd=str(self.output_dir))
            success = result.returncode == 0
            if success:
                print(result.stdout)
        except Exception as e:
            success = False
        
        state["custom_analysis_results"] = {"success": success, "skipped": False, "script_path": str(script_path)}
        state["analysis_script_path"] = str(script_path)
        
        return state
    
    def _fallback_script(self) -> str:
        return f'''#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTPUT_DIR = Path("{self.output_dir}")

def main():
    components = np.load(OUTPUT_DIR / "series_components.npy")
    abundances = np.load(OUTPUT_DIR / "series_abundances.npy")
    print(f"Loaded: components {{components.shape}}, abundances {{abundances.shape}}")
    
    n_comps = components.shape[0]
    mean_ab = abundances.mean(axis=(2, 3))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_comps):
        ax.plot(mean_ab[:, i], 'o-', label=f'Component {{i+1}}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Abundance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "abundance_timeseries.png", dpi=150)
    plt.close()
    
    trends = {{}}
    for i in range(n_comps):
        data = mean_ab[:, i]
        slope = np.polyfit(range(len(data)), data, 1)[0] if len(data) > 1 else 0
        trends[f"component_{{i+1}}"] = {{"mean": float(np.mean(data)), "slope": float(slope)}}
    
    with open(OUTPUT_DIR / "trends.json", 'w') as f:
        json.dump(trends, f, indent=2)
    print("Saved: trends.json, abundance_timeseries.png")

if __name__ == "__main__":
    main()
'''


# ============================================================================
# UNIFIED SYNTHESIS CONTROLLER
# ============================================================================

class UnifiedSynthesisController:
    """
    [🧠 LLM Step]
    Synthesizes findings - adapts for single vs batch.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict, store_fn: Callable = None):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self._store_analysis_images = store_fn or (lambda *a, **k: None)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        is_single = state.get("is_single_image", False)
        
        if is_single:
            return self._synthesize_single(state)
        else:
            return self._synthesize_batch(state)
    
    def _synthesize_single(self, state: dict) -> dict:
        self.logger.info("\n\n🔬 --- SINGLE IMAGE SYNTHESIS --- 🔬\n")
        
        prompt_parts = [SINGLE_IMAGE_ANALYSIS_INSTRUCTIONS]
        
        if state.get("image_blob"):
            prompt_parts.append("\n\n## Primary Image\n")
            prompt_parts.append(state["image_blob"])
        
        system_info = state.get("system_info", {})
        if system_info:
            prompt_parts.append(f"\n\n## System Info\n```json\n{json.dumps(system_info, indent=2)}\n```")
        
        stats = state.get("summary_stats", {})
        if stats:
            prompt_parts.append(f"\n\n## Statistics\n```json\n{json.dumps(stats, indent=2)}\n```")
        
        prompt_parts.append("\n\nProvide analysis as JSON only.")
        
        try:
            response = self.model.generate_content(contents=prompt_parts, generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                state["synthesis_result"] = self._fallback_single(state)
            else:
                state["synthesis_result"] = result_json
                state["result_json"] = result_json
                self.logger.info("✅ Single image synthesis complete.")
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = self._fallback_single(state)
        
        return state
    
    def _synthesize_batch(self, state: dict) -> dict:
        """Synthesize findings across multiple images - includes trend analysis results."""
        self.logger.info("\n\n🔬 --- BATCH SYNTHESIS --- 🔬\n")
        
        batch_results = state.get("batch_results", [])
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        batch_params = state.get("batch_params", {})
        
        # Build summary of individual results
        stats_summary = []
        for r in batch_results:
            if r.get("success"):
                stats_summary.append({
                    "index": r["index"],
                    "name": r["image_name"],
                    "n_components": r.get("n_components", 0),
                    "statistics": r.get("statistics", {})
                })
        
        # ============================================================
        # CRITICAL: Load the trends.json generated by custom analysis
        # ============================================================
        trends_data = {}
        trends_path = self.output_dir / "trends.json"
        if trends_path.exists():
            try:
                with open(trends_path, 'r') as f:
                    trends_data = json.load(f)
                self.logger.info(f"   📊 Loaded trend analysis from {trends_path}")
            except Exception as e:
                self.logger.warning(f"   Failed to load trends.json: {e}")
        
        # Build prompt
        prompt_parts = [SERIES_ANALYSIS_INSTRUCTIONS]
        
        # Add batch parameters
        prompt_parts.append(f"\n\n**ANALYSIS PARAMETERS:**")
        prompt_parts.append(f"- Frames analyzed: {batch_params.get('n_frames', len(batch_results))}")
        prompt_parts.append(f"- NMF components: {batch_params.get('n_components', 'N/A')}")
        prompt_parts.append(f"- Window size: {batch_params.get('window_size_nm', 'auto')} nm ({batch_params.get('window_size_pixels', 'auto')} px)")
        
        # Add individual frame statistics (condensed)
        prompt_parts.append(f"\n\n**INDIVIDUAL FRAME SUMMARY:**\n{json.dumps(stats_summary[:10], indent=2)}")  # Limit to first 10
        if len(stats_summary) > 10:
            prompt_parts.append(f"\n... and {len(stats_summary) - 10} more frames")
        
        # ============================================================
        # CRITICAL: Include the actual trend analysis results
        # ============================================================
        if trends_data:
            prompt_parts.append(f"\n\n**TREND ANALYSIS RESULTS (from automated script):**\n```json\n{json.dumps(trends_data, indent=2)}\n```")
        
        # Include script stdout if available (contains printed summary)
        if custom_results.get("stdout"):
            stdout_text = custom_results["stdout"]
            # Truncate if too long
            if len(stdout_text) > 2000:
                stdout_text = stdout_text[:2000] + "\n... [truncated]"
            prompt_parts.append(f"\n\n**SCRIPT OUTPUT:**\n```\n{stdout_text}\n```")
        
        # Add series metadata
        if series_metadata:
            prompt_parts.append(f"\n\n**SERIES METADATA:**\n{json.dumps(series_metadata, indent=2)}")
        
        # ============================================================
        # CRITICAL: Include generated visualization images
        # ============================================================
        visualization_files = []
        for pattern in ["*.png", "abundance_timeseries.png", "components.png", "correlation_matrix.png"]:
            visualization_files.extend(self.output_dir.glob(pattern))
        
        # Deduplicate and filter
        seen = set()
        unique_viz_files = []
        for f in visualization_files:
            if f.name not in seen and not f.name.startswith("review_iteration"):
                seen.add(f.name)
                unique_viz_files.append(f)
        
        if unique_viz_files:
            prompt_parts.append("\n\n**TREND ANALYSIS VISUALIZATIONS:**")
            for viz_path in unique_viz_files[:5]:  # Limit to 5 images
                if viz_path.exists():
                    try:
                        with open(viz_path, 'rb') as f:
                            img_b64 = base64.b64encode(f.read()).decode('utf-8')
                            prompt_parts.append(f"\n{viz_path.name}:")
                            prompt_parts.append({"mime_type": "image/png", "data": img_b64})
                            self.logger.info(f"   📈 Added visualization: {viz_path.name}")
                    except Exception as e:
                        self.logger.warning(f"   Failed to load {viz_path}: {e}")
        
        # Add NMF components visualization
        components = state.get("series_components")
        if components is not None:
            prompt_parts.append("\n\n**NMF FREQUENCY COMPONENTS:**")
            for i in range(min(components.shape[0], 4)):
                comp_bytes = self._array_to_png_bytes(components[i])
                if comp_bytes:
                    prompt_parts.append(f"\nComponent {i+1}:")
                    prompt_parts.append({"mime_type": "image/png", "data": comp_bytes})
        
        prompt_parts.append("\n\nBased on ALL the above information (individual statistics, trend analysis, and visualizations), provide a comprehensive scientific synthesis as a JSON object. Output ONLY the JSON.")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = self._fallback_batch(state, trends_data)
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Batch synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = self._fallback_batch(state, trends_data)
        
        return state
    
    def _fallback_single(self, state: dict) -> dict:
        n_comps = state.get("fft_components").shape[0] if state.get("fft_components") is not None else 0
        return {
            "detailed_analysis": f"FFT/NMF extracted {n_comps} components.",
            "scientific_claims": [{"claim": f"Image shows {n_comps} structural components.", "scientific_impact": "Complex microstructure identified."}]
        }
    
    def _fallback_batch(self, state: dict, trends_data: dict = None) -> dict:
        """Fallback for batch synthesis - incorporates trends if available."""
        n_frames = state.get("num_images", 0)
        n_comps = 0
        if state.get("series_components") is not None:
            n_comps = state["series_components"].shape[0]
        
        # Build trend summary from trends_data if available
        trend_summary = ""
        if trends_data:
            trend_parts = []
            for comp_name, comp_data in trends_data.items():
                if isinstance(comp_data, dict):
                    trend = comp_data.get("trend", "stable")
                    slope = comp_data.get("slope", 0)
                    trend_parts.append(f"{comp_name}: {trend} (slope={slope:.4f})")
            if trend_parts:
                trend_summary = f" Trends: {'; '.join(trend_parts)}."
        
        return {
            "detailed_analysis": f"Time-series analysis of {n_frames} frames with {n_comps} NMF components.{trend_summary} See visualizations for detailed temporal evolution.",
            "temporal_interpretation": "Quantitative trends available in the generated analysis plots and trends.json file.",
            "scientific_claims": [{
                "claim": f"Time-series FFT/NMF analysis of {n_frames} microscopy frames reveals {n_comps} distinct evolving structural frequency components.",
                "scientific_impact": "Temporal decomposition enables tracking of structural dynamics and phase evolution.",
                "has_anyone_question": f"Has anyone used sliding-window FFT/NMF decomposition for in-situ microscopy time-series to track structural evolution?",
                "keywords": ["in-situ microscopy", "FFT", "NMF", "time-series", "structural dynamics"]
            }]
        }
    
    def _array_to_png_bytes(self, array: np.ndarray) -> Optional[str]:
        """Convert numpy array to base64 PNG string."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(array, cmap='viridis')
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
            
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            self.logger.warning(f"Array to PNG conversion failed: {e}")
            return None


# ============================================================================
# UNIFIED REPORT GENERATION CONTROLLER
# ============================================================================

class UnifiedReportGenerationController:
    """
    [📄 Report Step]
    Generates HTML report - adapts for single vs batch.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        is_single = state.get("is_single_image", False)
        self.logger.info("\n\n📄 --- GENERATING REPORT --- 📄\n")
        
        if is_single:
            self._generate_single_image_report(state)
        else:
            self._generate_batch_report(state)
        
        return state
    
    def _generate_single_image_report(self, state: dict) -> None:
        """Generate report for single image analysis - matches SAM agent style."""
        
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
    <title>FFT/NMF Analysis Report - Single Image</title>
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
    <h1>🔬 FFT/NMF Analysis Report</h1>
    <p><strong>Date:</strong> {timestamp} | <strong>Image:</strong> {result.get('image_name', 'unknown')}</p>
    
    <div class="stats-box">
        <p><strong>Components Extracted:</strong> {result.get('n_components', 0)}</p>
    </div>
"""
        
        if viz_b64:
            html += f"""
    <h2>Decomposition Result</h2>
    <img class="viz-img" src="data:image/png;base64,{viz_b64}" alt="FFT/NMF Decomposition">
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
    <div class="footer">Generated by Microscopy Analysis Agent (Unified Architecture)</div>
</div>
</body>
</html>"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated report: {report_path}")
    
    def _generate_batch_report(self, state: dict) -> None:
        """Generate report for batch analysis - matches SAM agent style."""
        
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        batch_params = state.get("batch_params", {})
        
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
        num_images = state.get("num_images", len(batch_results))
        successful = sum(1 for r in batch_results if r.get("success"))
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FFT/NMF Batch Analysis Report</title>
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
    <h1>🔬 FFT/NMF Batch Analysis Report</h1>
    
    <div class="metadata-box">
        <p><strong>Date:</strong> {timestamp}</p>
        <p><strong>Images Processed:</strong> {successful}/{num_images}</p>
        <p><strong>Analysis Approach:</strong> {custom_results.get("approach", "time_series")}</p>
        <p><strong>Window Size:</strong> {batch_params.get("window_size_nm", "auto")} nm ({batch_params.get("window_size_pixels", "auto")} px)</p>
        <p><strong>NMF Components:</strong> {batch_params.get("n_components", "N/A")}</p>
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
    <div class="footer">Generated by Microscopy Batch Analysis Agent (Unified Architecture)</div>
</div>
</body>
</html>"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated batch report: {report_path}")