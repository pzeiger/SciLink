"""
FFT/NMF Controllers for Microscopy Analysis.

Single Image Controllers:
- GetFFTParamsController
- RunFFTNMFController  
- RunGlobalFFTController
- BuildFFTNMFPromptController
- FinalLLMAnalysisController

Series Controllers:
- SeriesLoaderController
- FirstFrameAnalysisController
- UserFeedbackController
- SeriesBatchController
- SummaryScriptController
- ReportGenerationController
"""

import os
import sys
import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
import base64

from ..instruct import FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS
from ..instruct import SERIES_REPORT_ANALYSIS_INSTRUCTIONS
from ....tools.image_processor import normalize_and_convert_to_image_bytes, calculate_global_fft
from ....tools.fft_nmf import SlidingFFTNMF



# =============================================================================
# SINGLE IMAGE CONTROLLERS
# =============================================================================

class GetFFTParamsController:
    """[🧠 LLM Step] Asks an LLM to suggest FFT/NMF parameters."""
    
    def __init__(self, model, logger, generation_config, safety_settings):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings

    def execute(self, state: dict) -> dict:
        self.logger.info("🧠 LLM Step: Reasoning about FFT/NMF parameters...")
        image_blob = state["image_blob"]
        system_info = state["system_info"]
        
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
            result = json.loads(response.text)
            state["llm_params"] = result

            print("\n" + "="*60)
            print("🧠 LLM REASONING")
            print(f"   Explanation: {result.get('explanation', 'N/A')}")
            print(f"   Params: window_size_nm={result.get('window_size_nm')}, n_components={result.get('n_components')}")
            print("="*60 + "\n")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: {e}")
            state["llm_params"] = {}
            
        return state


class RunFFTNMFController:
    """[🛠️ Tool Step] Runs the FFT/NMF analysis using SlidingFFTNMF."""
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Running Sliding FFT + NMF...")
        llm_params = state.get("llm_params", {})
        
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
                window_size_x=ws_pixels,
                window_size_y=ws_pixels,
                window_step_x=step,
                window_step_y=step,
                components=nc
            )
            
            image_array = state["preprocessed_image_array"]
            components, abundances = analyzer.analyze(image_array, output_dir=None)
            
            state["fft_components"] = components
            state["fft_abundances"] = abundances
            self.logger.info("✅ FFT/NMF complete.")
            
        except Exception as e:
            self.logger.error(f"❌ FFT/NMF failed: {e}")
            state["fft_components"] = None
            state["fft_abundances"] = None
            
        return state


class RunGlobalFFTController:
    """[🛠️ Tool Step] Calculates global FFT of the image."""
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
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
            
        return state


class BuildFFTNMFPromptController:
    """[📝 Prep Step] Builds the final prompt with analysis results."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("📝 Building final prompt...")
        
        prompt_parts = [state["instruction_prompt"]]
        
        if state.get("additional_top_level_context"):
            prompt_parts.append(f"\n\n## Special Considerations:\n{state['additional_top_level_context']}\n")
            
        prompt_parts.append("\n\nPrimary Microscopy Image:\n")
        prompt_parts.append(state["image_blob"])

        global_fft = state.get("global_fft_image")
        if global_fft is not None:
            try:
                fft_bytes = normalize_and_convert_to_image_bytes(global_fft, log_scale=False)
                prompt_parts.append("\n\nGlobal FFT:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": fft_bytes})
                state["analysis_images"].append({"label": "Global FFT", "data": fft_bytes})
            except Exception as e:
                self.logger.error(f"Failed to add Global FFT: {e}")

        components = state.get("fft_components")
        abundances = state.get("fft_abundances")

        if components is not None and abundances is not None:
            prompt_parts.append("\n\nSliding FFT + NMF Results:")
            for i in range(components.shape[0]):
                try:
                    comp_bytes = normalize_and_convert_to_image_bytes(components[i], log_scale=True)
                    abun_bytes = normalize_and_convert_to_image_bytes(abundances[i])
                    
                    prompt_parts.append(f"\nComponent {i+1}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": comp_bytes})
                    prompt_parts.append(f"\nAbundance Map {i+1}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": abun_bytes})
                    
                    state["analysis_images"].append({"label": f"Abundance {i+1}", "data": abun_bytes})
                except Exception as e:
                    self.logger.error(f"Failed to add NMF result {i+1}: {e}")

        prompt_parts.append(f"\n\nSystem Info:\n{json.dumps(state['system_info'], indent=2)}")
        prompt_parts.append("\n\nProvide your analysis in JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        return state


class FinalLLMAnalysisController:
    """[🧠 LLM Step] Executes the final LLM call with the built prompt."""
    
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn, store_fn):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.parse_fn = parse_fn
        self.store_fn = store_fn
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
            
        self.logger.info("🧠 Final LLM Analysis: Generating structured analysis...")
        
        prompt_parts = state.get("final_prompt_parts")
        if not prompt_parts:
            state["error_dict"] = {"error": "No prompt parts found"}
            return state
        
        if self.store_fn and state.get("analysis_images"):
            self.store_fn(state["analysis_images"])
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result = self.parse_fn(response.text) if self.parse_fn else self._parse_json(response.text)
            
            if result is None:
                state["error_dict"] = {"error": "Failed to parse LLM response"}
            else:
                state["result_json"] = result
                self.logger.info("✅ Final LLM Analysis complete.")
                
        except Exception as e:
            self.logger.error(f"❌ LLM Analysis failed: {e}")
            state["error_dict"] = {"error": "LLM analysis failed", "details": str(e)}
        
        return state
    
    def _parse_json(self, text: str) -> Optional[dict]:
        """Fallback JSON parser."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None


# =============================================================================
# SERIES CONTROLLERS
# =============================================================================

class SeriesLoaderController:
    """[📂 Load Step] Load image series from directory, TIFF, or array."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def execute(self, state: dict) -> dict:
        self.logger.info("📂 Loading image series...")
        series_input = state.get("series_input")
        
        try:
            if isinstance(series_input, np.ndarray):
                if series_input.ndim == 2:
                    series_data = series_input[np.newaxis, :, :]
                elif series_input.ndim == 3:
                    series_data = series_input
                else:
                    raise ValueError(f"Array must be 2D or 3D, got {series_input.ndim}D")
                state["series_source"] = "array"
                
            elif isinstance(series_input, str):
                if os.path.isdir(series_input):
                    series_data = self._load_directory(series_input)
                    state["series_source"] = "directory"
                elif series_input.lower().endswith(('.tif', '.tiff')):
                    series_data = self._load_tiff(series_input)
                    state["series_source"] = "tiff"
                else:
                    raise ValueError(f"Unsupported: {series_input}")
            else:
                raise TypeError(f"Expected str or ndarray, got {type(series_input)}")
            
            state["series_data"] = series_data
            state["n_frames"] = series_data.shape[0]
            state["frame_shape"] = series_data.shape[1:]
            state["first_frame"] = series_data[0]
            
            self.logger.info(f"✅ Loaded: {state['n_frames']} frames, shape {state['frame_shape']}")
            
        except Exception as e:
            self.logger.error(f"❌ Load failed: {e}")
            state["error_dict"] = {"error": "Load failed", "details": str(e)}
            
        return state
    
    def _load_directory(self, directory: str) -> np.ndarray:
        from skimage import io, color
        
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        files = sorted([f for f in os.listdir(directory) if f.lower().endswith(valid_ext)])
        
        if not files:
            raise ValueError(f"No images in {directory}")
        
        frames = []
        for f in files:
            img = io.imread(os.path.join(directory, f))
            if img.ndim == 3:
                img = color.rgb2gray(img[:, :, :3])
            frames.append(img)
        
        return np.stack(frames, axis=0)
    
    def _load_tiff(self, filepath: str) -> np.ndarray:
        from skimage import io, color
        
        stack = io.imread(filepath)
        if stack.ndim == 2:
            stack = stack[np.newaxis, :, :]
        elif stack.ndim == 4:
            stack = np.array([color.rgb2gray(f[:, :, :3]) for f in stack])
        return stack


class FirstFrameAnalysisController:
    """[🔬 Analysis Step] Analyze first frame with LLM-guided params."""
    
    def __init__(self, model, logger, generation_config, safety_settings, settings):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.settings = settings
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
            
        self.logger.info("🔬 Analyzing first frame...")
        
        from ....tools.image_processor import convert_numpy_to_jpeg_bytes, preprocess_image
        
        first_frame = state["first_frame"]
        
        if first_frame.dtype in [np.float32, np.float64, float]:
            frame_min, frame_max = first_frame.min(), first_frame.max()
            if frame_max > frame_min:
                first_frame = ((first_frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                first_frame = np.zeros_like(first_frame, dtype=np.uint8)
        
        preprocessed, _ = preprocess_image(first_frame)
        image_bytes = convert_numpy_to_jpeg_bytes(preprocessed)
        
        state["preprocessed_image_array"] = preprocessed
        state["image_blob"] = {"mime_type": "image/jpeg", "data": image_bytes}
        state["image_path"] = "first_frame"
        
        GetFFTParamsController(self.model, self.logger, self.generation_config, self.safety_settings).execute(state)
        RunGlobalFFTController(self.logger, self.settings).execute(state)
        RunFFTNMFController(self.logger, self.settings).execute(state)
        
        state["first_frame_results"] = {
            "components": state.get("fft_components"),
            "abundances": state.get("fft_abundances"),
            "llm_params": state.get("llm_params", {})
        }
        
        self.logger.info("✅ First frame analysis complete.")
        return state


class UserFeedbackController:
    """[👤 Feedback Step] Collect user feedback on parameters."""
    
    def __init__(self, logger, settings, feedback_callback=None):
        self.logger = logger
        self.settings = settings
        self.feedback_callback = feedback_callback
        self.max_iterations = settings.get('max_feedback_iterations', 3)
    
    def _display_results(self, state: dict, iteration: int) -> None:
        """Display current results for review."""
        llm_params = state.get("first_frame_results", {}).get("llm_params", {})
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        
        # Save visualization
        output_dir = Path(self.settings.get("visualization_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        review_path = output_dir / f"review_iteration_{iteration}.png"
        
        if components is not None and abundances is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = components.shape[0]
            fig, axes = plt.subplots(2, n_comps, figsize=(4*n_comps, 8))
            
            for i in range(n_comps):
                axes[0, i].imshow(components[i], cmap='viridis')
                axes[0, i].set_title(f'Component {i+1}')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(abundances[i], cmap='hot')
                axes[1, i].set_title(f'Abundance {i+1}')
                axes[1, i].axis('off')
            
            plt.suptitle(f'FFT/NMF Analysis - Iteration {iteration}', fontsize=14)
            plt.tight_layout()
            plt.savefig(review_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print("\n" + "=" * 70)
        print(f"🔬 FFT/NMF ANALYSIS REVIEW - Iteration {iteration}")
        print("=" * 70)
        print(f"\n🖼️  Visualization saved to: {review_path}")
        print(f"\n📊 Results:")
        if components is not None:
            print(f"   Components: {components.shape[0]}, size: {components.shape[1]}x{components.shape[2]}")
        print(f"\n⚙️  Parameters:")
        print(f"   Window Size (nm): {llm_params.get('window_size_nm', 'auto')}")
        print(f"   NMF Components: {llm_params.get('n_components', 4)}")
        if llm_params.get('explanation'):
            print(f"\n🧠 Reasoning: {llm_params.get('explanation')}")
        print("-" * 70)
    
    def _get_user_input(self, prompt: str) -> str:
        """Get user input, handling different environments."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            return input().strip()
        except EOFError:
            return ""
    
    def _collect_feedback(self, state: dict) -> dict:
        """Collect user feedback."""
        llm_params = state.get("first_frame_results", {}).get("llm_params", {})
        
        print("\n👤 Options:")
        print("   [1] Accept (proceed to batch)")
        print("   [2] Modify parameters")
        print("   [c] Cancel")
        
        choice = self._get_user_input("\nChoice [1/2/c]: ").lower()
        
        if choice == '1' or choice == '':
            return {"action": "accept"}
        elif choice == 'c':
            return {"action": "cancel"}
        elif choice == '2':
            print("\nEnter new values (press Enter to keep current):")
            
            mods = {}
            ws = self._get_user_input(f"   Window size (nm) [{llm_params.get('window_size_nm', 'auto')}]: ")
            if ws:
                try:
                    mods['window_size_nm'] = float(ws)
                except ValueError:
                    print("   Invalid, keeping current")
            
            nc = self._get_user_input(f"   Components [{llm_params.get('n_components', 4)}]: ")
            if nc:
                try:
                    mods['n_components'] = int(nc)
                except ValueError:
                    print("   Invalid, keeping current")
            
            if mods:
                return {"action": "modify", "params": mods}
            return {"action": "accept"}
        else:
            print("Invalid choice, accepting current.")
            return {"action": "accept"}
    
    def _rerun_analysis(self, state: dict, new_params: dict) -> dict:
        """Re-run with updated parameters."""
        self.logger.info("🔄 Re-running with updated parameters...")
        
        llm_params = state.get("first_frame_results", {}).get("llm_params", {}).copy()
        llm_params.update(new_params)
        state["llm_params"] = llm_params
        
        RunFFTNMFController(self.logger, self.settings).execute(state)
        
        state["first_frame_results"] = {
            "components": state.get("fft_components"),
            "abundances": state.get("fft_abundances"),
            "llm_params": llm_params
        }
        return state
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        print("\n\n" + "=" * 70)
        print("👤 HUMAN FEEDBACK LOOP")
        print("=" * 70)
        
        for iteration in range(1, self.max_iterations + 1):
            self._display_results(state, iteration)
            
            if self.feedback_callback:
                feedback = self.feedback_callback(state)
            else:
                feedback = self._collect_feedback(state)
            
            if feedback.get("action") == "accept":
                self.logger.info("✅ User accepted results.")
                state["locked_params"] = state.get("first_frame_results", {}).get("llm_params", {})
                return state
            
            elif feedback.get("action") == "cancel":
                self.logger.info("❌ User cancelled.")
                state["batch_cancelled"] = True
                return state
            
            elif feedback.get("action") == "modify" and feedback.get("params"):
                state = self._rerun_analysis(state, feedback["params"])
        
        self.logger.warning(f"Max iterations reached, using current parameters.")
        state["locked_params"] = state.get("first_frame_results", {}).get("llm_params", {})
        return state


class SeriesBatchController:
    """[⚡ Batch Step] Process full series with locked parameters."""
    
    def __init__(self, logger, settings):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        self.logger.info("⚡ Processing full series...")
        
        locked_params = state.get("locked_params", {})
        series_data = state["series_data"]
        n_frames = state["n_frames"]
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        ws_nm = locked_params.get("window_size_nm", 10.0)
        n_components = locked_params.get("n_components", 4)
        
        ws_pixels = int(round(ws_nm / nm_per_pixel)) if nm_per_pixel > 0 else 64
        good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
        ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        step = max(1, ws_pixels // 4)
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels,
                window_size_y=ws_pixels,
                window_step_x=step,
                window_step_y=step,
                components=n_components
            )
            
            print(f"⏳ Processing {n_frames} frames...")
            components, abundances = analyzer.analyze(series_data, output_dir=None)
            
            state["series_components"] = components
            state["series_abundances"] = abundances
            state["batch_params"] = {
                "window_size_pixels": ws_pixels,
                "window_size_nm": ws_nm,
                "n_components": n_components,
                "n_frames": n_frames
            }
            
            output_dir = self.settings.get("output_dir", "analysis_output")
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "series_components.npy"), components)
            np.save(os.path.join(output_dir, "series_abundances.npy"), abundances)
            
            print(f"✅ Done! Components: {components.shape}, Abundances: {abundances.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Batch failed: {e}")
            state["error_dict"] = {"error": "Batch failed", "details": str(e)}
        
        return state


class SummaryScriptController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates and executes a custom Python script for trend analysis.
    Follows the same pattern as SAM's CustomAnalysisScriptController.
    """
    
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn, settings):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get("output_dir", "analysis_output"))
        self.max_retries = settings.get("max_script_retries", 3)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        self.logger.info("🧠 LLM generating custom analysis script...")
        
        # First, compute basic stats to give LLM context
        basic_stats = self._compute_basic_stats(state)
        state["basic_stats"] = basic_stats
        
        # Generate script via LLM
        script_result = self._generate_analysis_script(state, basic_stats)
        
        if not script_result or "script" not in script_result:
            self.logger.warning("LLM script generation failed, using fallback template")
            script = self._fallback_script(state)
        else:
            script = script_result["script"]
            approach = script_result.get("analysis_approach", "trend_analysis")
            metrics = script_result.get("key_metrics", [])
            self.logger.info(f"   📊 Analysis approach: {approach}")
            self.logger.info(f"   📈 Key metrics: {metrics}")
        
        # Execute with retry loop
        script_path = self.output_dir / "analyze_results.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        success, stdout, script = self._execute_with_retry(script, script_path, state, basic_stats)
        
        state["analysis_script_path"] = str(script_path)
        state["script_success"] = success
        state["script_output"] = stdout
        
        return state
    
    def _compute_basic_stats(self, state: dict) -> dict:
        """Compute basic statistics to give LLM context about the data."""
        components = state.get("series_components")
        abundances = state.get("series_abundances")
        
        if components is None or abundances is None:
            return {}
        
        n_frames, n_comps = abundances.shape[:2]
        mean_abundances = abundances.mean(axis=(2, 3))  # (n_frames, n_comps)
        
        stats = {
            "n_frames": n_frames,
            "n_components": n_comps,
            "component_shape": list(components.shape),
            "abundance_shape": list(abundances.shape),
            "components": []
        }
        
        for i in range(n_comps):
            ts = mean_abundances[:, i]
            slope = float(np.polyfit(range(len(ts)), ts, 1)[0])
            
            # Detect patterns
            if len(ts) > 2:
                fft_mag = np.abs(np.fft.fft(ts - ts.mean()))
                half_len = max(1, len(ts) // 2)
                dominant_freq_idx = np.argmax(fft_mag[1:half_len]) + 1 if half_len > 1 else 1
                period = n_frames / dominant_freq_idx if dominant_freq_idx > 0 else None
                has_periodicity = bool(fft_mag[dominant_freq_idx] > 2 * np.mean(fft_mag[1:half_len])) if half_len > 1 else False
            else:
                period = None
                has_periodicity = False
            
            stats["components"].append({
                "index": i + 1,
                "mean": float(np.mean(ts)),
                "std": float(np.std(ts)),
                "min": float(np.min(ts)),
                "max": float(np.max(ts)),
                "slope": slope,
                "trend": "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable",
                "has_periodicity": has_periodicity,
                "estimated_period": float(period) if period else None
            })
        
        # Cross-correlations
        if n_comps > 1:
            corr_matrix = np.corrcoef(mean_abundances.T)
            stats["correlations"] = []
            for i in range(n_comps):
                for j in range(i + 1, n_comps):
                    stats["correlations"].append({
                        "pair": [i + 1, j + 1],
                        "correlation": float(corr_matrix[i, j])
                    })
        
        return stats
    
    def _generate_analysis_script(self, state: dict, basic_stats: dict) -> Optional[dict]:
        """Generate custom analysis script using LLM. Returns dict with 'script' key."""
        
        output_dir_str = str(self.output_dir)
        
        prompt = f'''You are a scientific data analysis expert. Generate a Python script to analyze FFT/NMF decomposition results.

**DATA FILES** (in {output_dir_str}):
- series_components.npy: shape {basic_stats.get("component_shape", "unknown")} - NMF frequency components
- series_abundances.npy: shape {basic_stats.get("abundance_shape", "unknown")} - abundance maps (frames, components, grid_h, grid_w)

**PRE-COMPUTED STATISTICS:**
{json.dumps(basic_stats, indent=2)}

**REQUIREMENTS:**
1. Complete, runnable Python script
2. Libraries: numpy, matplotlib, scipy, json, pathlib only
3. Save figures as PNG to OUTPUT_DIR
4. Save trends.json with analysis results
5. Print summary to stdout

**ANALYSIS TO INCLUDE:**
- Plot NMF components as images
- Plot mean abundance timeseries per component  
- Compute trend directions and slopes
- If correlations > 0.5, plot correlation matrix
- Save findings to trends.json

Return a JSON object with:
{{
    "analysis_approach": "brief description of approach",
    "key_metrics": ["list", "of", "metrics"],
    "reasoning": "why this analysis is appropriate",
    "script": "complete Python script as a string"
}}

The script should start with imports and define OUTPUT_DIR = Path("{output_dir_str}")'''

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM script generation parse error: {error_dict}")
                # Try to salvage if we got partial results
                if result_json and "script" in result_json:
                    return result_json
                return None
            
            if not result_json or "script" not in result_json:
                self.logger.warning("LLM response missing 'script' key")
                return None
            
            self.logger.info("✅ LLM generated analysis script")
            return result_json
            
        except Exception as e:
            self.logger.error(f"LLM script generation failed: {e}")
            return None
    
    def _execute_with_retry(self, script: str, script_path: Path, state: dict, basic_stats: dict) -> tuple:
        """Execute script with retry on errors, asking LLM to fix."""
        import subprocess
        
        for attempt in range(self.max_retries):
            # Save script
            with open(script_path, 'w') as f:
                f.write(script)
            
            self.logger.info(f"📜 Executing script (attempt {attempt + 1}/{self.max_retries})...")
            
            try:
                result = subprocess.run(
                    ['python', str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(self.output_dir)
                )
                
                if result.returncode == 0:
                    self.logger.info("✅ Script executed successfully!")
                    print(result.stdout)
                    return True, result.stdout, script
                else:
                    error_msg = result.stderr
                    self.logger.warning(f"Script failed: {error_msg[:200]}")
                    
                    if attempt < self.max_retries - 1:
                        corrected = self._fix_script_with_llm(script, error_msg, basic_stats, attempt + 1)
                        if corrected:
                            script = corrected
                        else:
                            break
                    
            except subprocess.TimeoutExpired:
                self.logger.warning("Script timed out")
                break
            except Exception as e:
                self.logger.warning(f"Execution error: {e}")
                break
        
        # All retries failed, use fallback
        self.logger.warning("Using fallback script after retries exhausted")
        script = self._fallback_script(state)
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(
                ['python', str(script_path)],
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=str(self.output_dir)
            )
            print(result.stdout)
            return result.returncode == 0, result.stdout, script
        except Exception as e:
            self.logger.error(f"Fallback script also failed: {e}")
            return False, "", script
    
    def _fix_script_with_llm(self, original_script: str, error_msg: str, basic_stats: dict, attempt: int) -> Optional[str]:
        """Use LLM to correct a failed script."""
        self.logger.info(f"   🔧 Attempting script correction (attempt {attempt})...")
        
        if len(error_msg) > 1000:
            error_msg = error_msg[:500] + "\n...[truncated]...\n" + error_msg[-500:]
        
        prompt = f'''Fix this Python script that failed to execute.

**SCRIPT:**
```python
{original_script}
```

**ERROR:**
```
{error_msg}
```

**DATA CONTEXT:**
- Components shape: {basic_stats.get("component_shape")}
- Abundances shape: {basic_stats.get("abundance_shape")}
- Output directory: {self.output_dir}

Return a JSON object with:
{{
    "diagnosis": "what caused the error",
    "script": "corrected complete Python script"
}}'''

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict or not result_json:
                self.logger.warning("Failed to parse correction response")
                return None
            
            diagnosis = result_json.get("diagnosis", "N/A")
            self.logger.info(f"   📋 Diagnosis: {diagnosis}")
            
            return result_json.get("script")
            
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None
    
    def _fallback_script(self, state: dict) -> str:
        """Simple fallback script if LLM generation fails."""
        return f'''#!/usr/bin/env python3
"""Fallback analysis script for FFT/NMF results."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTPUT_DIR = Path("{self.output_dir}")

def main():
    # Load data
    components = np.load(OUTPUT_DIR / "series_components.npy")
    abundances = np.load(OUTPUT_DIR / "series_abundances.npy")
    print(f"Loaded: components {{components.shape}}, abundances {{abundances.shape}}")
    
    n_comps = components.shape[0]
    
    # Plot components
    fig, axes = plt.subplots(1, n_comps, figsize=(4*n_comps, 4))
    if n_comps == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(components[i], cmap='viridis')
        ax.set_title(f'Component {{i+1}}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "components.png", dpi=150)
    plt.close()
    print("Saved: components.png")
    
    # Plot abundance timeseries
    mean_ab = abundances.mean(axis=(2, 3))  # (n_frames, n_comps)
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
    print("Saved: abundance_timeseries.png")
    
    # Compute trends
    trends = {{}}
    for i in range(n_comps):
        data = mean_ab[:, i]
        slope = np.polyfit(range(len(data)), data, 1)[0]
        trends[f"component_{{i+1}}"] = {{
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "trend": "increasing" if slope > 0 else "decreasing",
            "slope": float(slope)
        }}
    
    # Save trends
    with open(OUTPUT_DIR / "trends.json", 'w') as f:
        json.dump(trends, f, indent=2)
    print("Saved: trends.json")
    
    # Print summary
    print("\\nTrend Analysis:")
    for k, v in trends.items():
        print(f"  {{k}}: {{v['trend']}} (slope={{v['slope']:.4f}})")

if __name__ == "__main__":
    main()
'''



class ReportGenerationController:
    """
    [📄 Report Step] Generates HTML report with LLM-based scientific analysis.
    
    This controller:
    1. Computes statistics from FFT/NMF results
    2. Loads generated visualizations
    3. Sends everything to LLM for scientific interpretation
    4. Generates HTML report from LLM analysis
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, 
                 safety_settings, parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.max_retries = settings.get('max_llm_retries', 2)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        self.logger.info("📄 Generating LLM-analyzed HTML report...")
        
        # 1. Compute statistics
        stats = self._compute_detailed_stats(
            state.get("series_components"),
            state.get("series_abundances")
        )
        
        # 2. Load visualizations
        visualizations = self._load_visualizations()
        
        # 3. Get LLM interpretation
        llm_analysis = self._get_llm_analysis(state, stats, visualizations)
        
        if llm_analysis is None:
            self.logger.warning("LLM analysis failed, using fallback")
            llm_analysis = self._generate_fallback_analysis(state, stats)
        
        # 4. Generate HTML report
        self._generate_report(state, stats, visualizations, llm_analysis)
        
        # Store analysis in state for downstream use
        state["llm_report_analysis"] = llm_analysis
        
        return state
    
    def _get_llm_analysis(self, state: dict, stats: dict, 
                          visualizations: list) -> Optional[dict]:
        """
        Send FFT/NMF results and visualizations to LLM for scientific interpretation.
        
        Returns dict with:
            - methodology_notes: str
            - scientific_interpretation: str
            - component_interpretations: list[dict] with keys: index, description, physical_meaning
            - temporal_interpretation: str
            - visualization_descriptions: list[dict] with keys: name, description
            - claims_with_questions: list[dict] with keys: claim, question, evidence
        """
        self.logger.info("🧠 LLM analyzing FFT/NMF results...")
        
        params = state.get("batch_params", {})
        n_frames = state.get("n_frames", 0)
        n_components = params.get("n_components", 4)
        
        # Build prompt with images and data
        prompt_parts = [SERIES_REPORT_ANALYSIS_INSTRUCTIONS]
        
        # Add analysis context
        context = {
            "n_frames": n_frames,
            "n_components": n_components,
            "window_size_nm": params.get("window_size_nm", "auto"),
            "window_size_pixels": params.get("window_size_pixels", "auto"),
            "component_statistics": stats.get("components", []),
            "correlations": stats.get("correlations", []),
            "system_info": state.get("system_info", {})
        }
        
        prompt_parts.append(f"\n\n## Analysis Context\n```json\n{json.dumps(context, indent=2)}\n```")
        
        # Add visualizations as images
        prompt_parts.append("\n\n## Generated Visualizations\n")
        prompt_parts.append("Analyze these visualizations and provide scientific interpretation:\n")
        
        for viz in visualizations:
            prompt_parts.append(f"\n### {viz['name']}\n")
            prompt_parts.append({
                "mime_type": "image/png",
                "data": viz["data"]
            })
        
        # Add the NMF components as images if available
        components = state.get("series_components")
        if components is not None:
            prompt_parts.append("\n\n## NMF Frequency Components\n")
            for i in range(min(components.shape[0], 6)):  # Limit to 6 components
                comp_bytes = self._array_to_png_bytes(components[i])
                if comp_bytes:
                    prompt_parts.append(f"\nComponent {i+1} (frequency pattern):\n")
                    prompt_parts.append({
                        "mime_type": "image/png",
                        "data": comp_bytes
                    })
        
        prompt_parts.append("\n\nProvide your analysis as a JSON object following the schema in the instructions.")
        
        # Call LLM with retry
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    contents=prompt_parts,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                
                result, error = self._parse_llm_response(response)
                
                if error:
                    self.logger.warning(f"LLM parse error (attempt {attempt+1}): {error}")
                    continue
                
                if result and self._validate_analysis(result):
                    self.logger.info("✅ LLM analysis complete")
                    return result
                else:
                    self.logger.warning(f"Invalid LLM response structure (attempt {attempt+1})")
                    
            except Exception as e:
                self.logger.error(f"LLM analysis error (attempt {attempt+1}): {e}")
        
        return None
    
    def _validate_analysis(self, analysis: dict) -> bool:
        """Check that LLM response has required fields."""
        required_fields = [
            "scientific_interpretation",
            "claims_with_questions"
        ]
        return all(field in analysis for field in required_fields)
    
    def _generate_fallback_analysis(self, state: dict, stats: dict) -> dict:
        """Generate template-based analysis if LLM fails."""
        params = state.get("batch_params", {})
        n_frames = state.get("n_frames", 0)
        n_components = params.get("n_components", 4)
        
        # Build component interpretations from stats
        component_interps = []
        for comp in stats.get("components", []):
            component_interps.append({
                "index": comp["index"],
                "description": f"Component {comp['index']} shows {comp['trend']} trend with {comp['pct_change']:+.1f}% change",
                "physical_meaning": "Requires expert interpretation of the frequency pattern"
            })
        
        # Build claims from stats
        claims = []
        claims.append({
            "claim": f"The FFT/NMF decomposition identifies {n_components} distinct frequency-domain components in the microscopy series.",
            "question": "What physical structures or processes correspond to each identified component?",
            "evidence": "NMF decomposition of sliding-window FFT spectra"
        })
        
        for comp in stats.get("components", []):
            if comp["trend"] != "stable":
                claims.append({
                    "claim": f"Component {comp['index']} exhibits a {comp['trend']} trend ({comp['pct_change']:+.1f}% change over {n_frames} frames).",
                    "question": f"What mechanism drives the {comp['trend']} abundance of Component {comp['index']}?",
                    "evidence": f"Linear fit slope: {comp['slope']:.4f}, R² from trend analysis"
                })
        
        return {
            "methodology_notes": "Sliding window FFT combined with NMF was used to decompose the image series into frequency-domain components.",
            "scientific_interpretation": f"Analysis of {n_frames} frames reveals {n_components} distinct structural signatures with varying temporal dynamics.",
            "component_interpretations": component_interps,
            "temporal_interpretation": "Temporal trends were computed from spatially-averaged abundance maps.",
            "visualization_descriptions": [],
            "claims_with_questions": claims
        }
    
    def _compute_detailed_stats(self, components: Optional[np.ndarray], 
                                 abundances: Optional[np.ndarray]) -> dict:
        """Compute detailed statistics for LLM context."""
        stats = {
            "components": [],
            "correlations": [],
            "has_data": components is not None and abundances is not None
        }
        
        if not stats["has_data"]:
            return stats
        
        n_frames = abundances.shape[0]
        n_comps = abundances.shape[1]
        mean_abundances = abundances.mean(axis=(2, 3))  # (n_frames, n_comps)
        
        for i in range(n_comps):
            ts = mean_abundances[:, i]
            
            # Linear fit
            slope, intercept = np.polyfit(range(len(ts)), ts, 1)
            y_pred = slope * np.arange(len(ts)) + intercept
            ss_res = np.sum((ts - y_pred) ** 2)
            ss_tot = np.sum((ts - np.mean(ts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Trend direction
            if slope > 0.001:
                trend_dir = "increasing"
            elif slope < -0.001:
                trend_dir = "decreasing"
            else:
                trend_dir = "stable"
            
            # Percent change
            pct_change = ((ts[-1] - ts[0]) / ts[0]) * 100 if ts[0] != 0 else 0
            
            # Periodicity detection
            period, has_periodicity = None, False
            if len(ts) > 4:
                fft_mag = np.abs(np.fft.fft(ts - ts.mean()))
                half_len = len(ts) // 2
                if half_len > 1:
                    dominant_idx = np.argmax(fft_mag[1:half_len]) + 1
                    period = n_frames / dominant_idx if dominant_idx > 0 else None
                    has_periodicity = bool(fft_mag[dominant_idx] > 2.5 * np.mean(fft_mag[1:half_len]))
            
            stats["components"].append({
                "index": i + 1,
                "mean": float(np.mean(ts)),
                "std": float(np.std(ts)),
                "min": float(np.min(ts)),
                "max": float(np.max(ts)),
                "slope": float(slope),
                "r_squared": float(r_squared),
                "trend": trend_dir,
                "pct_change": float(pct_change),
                "has_periodicity": has_periodicity,
                "period_frames": float(period) if period else None
            })
        
        # Correlations
        if n_comps > 1:
            corr_matrix = np.corrcoef(mean_abundances.T)
            for i in range(n_comps):
                for j in range(i + 1, n_comps):
                    stats["correlations"].append({
                        "components": [i + 1, j + 1],
                        "correlation": float(corr_matrix[i, j])
                    })
        
        return stats
    
    def _load_visualizations(self) -> list:
        """Load PNG visualizations from output directory."""
        visualizations = []
        
        png_files = sorted(self.output_dir.glob("*.png"))
        for png_path in png_files:
            # Skip review iterations
            if png_path.name.startswith("review_iteration"):
                continue
            
            try:
                with open(png_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    visualizations.append({
                        "name": png_path.stem,
                        "path": str(png_path),
                        "data": b64
                    })
            except Exception as e:
                self.logger.warning(f"Failed to load {png_path}: {e}")
        
        return visualizations
    
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
            self.logger.warning(f"Failed to convert array to PNG: {e}")
            return None
    
    def _generate_report(self, state: dict, stats: dict, 
                         visualizations: list, llm_analysis: dict) -> None:
        """Generate HTML report from LLM analysis."""
        params = state.get("batch_params", {})
        n_frames = state.get("n_frames", 0)
        n_components = params.get("n_components", 4)
        window_size_nm = params.get("window_size_nm", "auto")
        window_size_px = params.get("window_size_pixels", "auto")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Extract LLM analysis fields
        methodology = llm_analysis.get("methodology_notes", "")
        interpretation = llm_analysis.get("scientific_interpretation", "")
        component_interps = llm_analysis.get("component_interpretations", [])
        temporal_interp = llm_analysis.get("temporal_interpretation", "")
        viz_descriptions = {v.get("name", ""): v.get("description", "") 
                          for v in llm_analysis.get("visualization_descriptions", [])}
        claims_questions = llm_analysis.get("claims_with_questions", [])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FFT/NMF Analysis Report</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
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
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #9b59b6;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        h2 {{
            color: #8e44ad;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        h3 {{
            color: #5b2c6f;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        /* Scientific Analysis Section */
        .analysis-content {{
            background-color: #fafafa;
            padding: 25px 30px;
            border-radius: 8px;
            border: 1px solid #eee;
            font-size: 0.95em;
        }}
        .analysis-content p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        .analysis-subsection {{
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px dashed #ddd;
        }}
        .analysis-subsection:first-of-type {{
            margin-top: 0;
            padding-top: 0;
            border-top: none;
        }}
        .param-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        .param-table th, .param-table td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .param-table th {{
            background-color: #f5eef8;
            color: #5b2c6f;
            font-weight: 600;
        }}
        .component-interpretation {{
            background: linear-gradient(135deg, #f5eef8 0%, #fafafa 100%);
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 4px solid #9b59b6;
            margin-bottom: 15px;
        }}
        .component-interpretation h4 {{
            margin: 0 0 10px 0;
            color: #5b2c6f;
        }}
        .component-interpretation .physical-meaning {{
            font-style: italic;
            color: #666;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px dashed #ddd;
        }}
        .trend-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            margin-left: 10px;
        }}
        .trend-increasing {{
            background-color: #d5f5e3;
            color: #1e8449;
        }}
        .trend-decreasing {{
            background-color: #fadbd8;
            color: #922b21;
        }}
        .trend-stable {{
            background-color: #ebf5fb;
            color: #2471a3;
        }}
        
        /* Visualizations Section */
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .image-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .image-info {{
            padding: 15px 20px;
            border-top: 1px solid #eee;
        }}
        .image-label {{
            font-weight: 600;
            color: #2c3e50;
            font-size: 1em;
            margin-bottom: 8px;
        }}
        .image-description {{
            font-size: 0.9em;
            color: #666;
            line-height: 1.6;
        }}
        
        /* Claims & Questions Section */
        .claim-block {{
            margin-bottom: 25px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .claim-content {{
            background-color: #e8f6f3;
            padding: 20px 25px;
            border-left: 5px solid #1abc9c;
        }}
        .claim-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }}
        .claim-number {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            background: #1abc9c;
            color: white;
            border-radius: 50%;
            font-size: 0.85em;
            font-weight: bold;
            flex-shrink: 0;
        }}
        .claim-title {{
            font-weight: 600;
            color: #0e6655;
            font-size: 1em;
        }}
        .claim-text {{
            color: #1a5246;
            font-size: 0.95em;
            margin-left: 40px;
        }}
        .claim-evidence {{
            font-size: 0.85em;
            color: #148f77;
            margin-left: 40px;
            margin-top: 8px;
            font-style: italic;
        }}
        .question-content {{
            background-color: #fef9e7;
            padding: 15px 25px 15px 65px;
            border-left: 5px solid #f39c12;
            position: relative;
        }}
        .question-content::before {{
            content: "↳";
            position: absolute;
            left: 25px;
            top: 15px;
            color: #d4ac0d;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .question-label {{
            font-size: 0.8em;
            color: #9a7b0a;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        .question-text {{
            color: #7d6608;
            font-size: 0.95em;
        }}
        
        /* Footer */
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
        <header>
            <h1>🔬 FFT/NMF Series Analysis Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section>
            <h2>1. Scientific Analysis</h2>
            <div class="analysis-content">
                
                <div class="analysis-subsection">
                    <h3>1.1 Methodology</h3>
                    <p>{methodology if methodology else self._default_methodology()}</p>
                    <table class="param-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Frames Analyzed</td>
                            <td><strong>{n_frames}</strong></td>
                        </tr>
                        <tr>
                            <td>NMF Components</td>
                            <td><strong>{n_components}</strong></td>
                        </tr>
                        <tr>
                            <td>Window Size</td>
                            <td><strong>{window_size_nm} nm</strong> ({window_size_px} px)</td>
                        </tr>
                    </table>
                </div>
                
                <div class="analysis-subsection">
                    <h3>1.2 Key Findings</h3>
                    <p>{interpretation}</p>
                </div>
                
                <div class="analysis-subsection">
                    <h3>1.3 Component Interpretation</h3>
                    {self._render_component_interpretations(component_interps, stats)}
                </div>
                
                <div class="analysis-subsection">
                    <h3>1.4 Temporal Dynamics</h3>
                    <p>{temporal_interp if temporal_interp else "See component analysis above for temporal trends."}</p>
                </div>
                
            </div>
        </section>

        <section>
            <h2>2. Visualizations</h2>
            <div class="image-grid">
"""
        
        # Add visualizations with LLM descriptions
        for viz in visualizations:
            name = viz["name"]
            display_name = name.replace('_', ' ').replace('-', ' ').title()
            description = viz_descriptions.get(name, "Analysis visualization from FFT/NMF processing.")
            
            html += f"""                <div class="image-card">
                    <img src="data:image/png;base64,{viz['data']}" alt="{name}" loading="lazy">
                    <div class="image-info">
                        <div class="image-label">{display_name}</div>
                        <div class="image-description">{description}</div>
                    </div>
                </div>
"""
        
        if not visualizations:
            html += """                <p style="color: #7f8c8d; font-style: italic; padding: 20px;">No visualizations available.</p>
"""
        
        html += """            </div>
        </section>

        <section>
            <h2>3. Research Claims & Questions</h2>
"""
        
        # Add claims with questions
        for i, item in enumerate(claims_questions, 1):
            claim = item.get("claim", "")
            question = item.get("question", "")
            evidence = item.get("evidence", "")
            
            evidence_html = f'<p class="claim-evidence">Evidence: {evidence}</p>' if evidence else ""
            
            html += f"""            <div class="claim-block">
                <div class="claim-content">
                    <div class="claim-header">
                        <span class="claim-number">{i}</span>
                        <span class="claim-title">Scientific Claim</span>
                    </div>
                    <p class="claim-text">{claim}</p>
                    {evidence_html}
                </div>
                <div class="question-content">
                    <div class="question-label">Follow-up Research Question</div>
                    <p class="question-text">{question}</p>
                </div>
            </div>
"""
        
        html += """        </section>

        <div class="footer">
            Generated by FFT/NMF Series Analysis Agent (LLM-Analyzed)
        </div>
    </div>
</body>
</html>
"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"✅ Report saved: {report_path}")
        print(f"\n📊 Report: {report_path}")
    
    def _default_methodology(self) -> str:
        return ("This analysis employs Sliding Window Fast Fourier Transform (FFT) combined with "
                "Non-negative Matrix Factorization (NMF) to decompose microscopy image series into "
                "interpretable frequency-domain components. The sliding window approach captures local "
                "periodic structures, while NMF identifies recurring spectral patterns across the dataset.")
    
    def _render_component_interpretations(self, interps: list, stats: dict) -> str:
        """Render component interpretations with statistics."""
        if not interps:
            return "<p>No component interpretations available.</p>"
        
        # Create lookup for stats
        stats_lookup = {c["index"]: c for c in stats.get("components", [])}
        
        html = ""
        for interp in interps:
            idx = interp.get("index", 0)
            desc = interp.get("description", "")
            meaning = interp.get("physical_meaning", "")
            
            # Get trend from stats
            comp_stats = stats_lookup.get(idx, {})
            trend = comp_stats.get("trend", "stable")
            trend_class = f"trend-{trend}"
            
            html += f"""
                <div class="component-interpretation">
                    <h4>Component {idx} <span class="trend-badge {trend_class}">{trend.upper()}</span></h4>
                    <p>{desc}</p>
                    <p class="physical-meaning"><strong>Physical interpretation:</strong> {meaning}</p>
                </div>
"""
        
        return html
