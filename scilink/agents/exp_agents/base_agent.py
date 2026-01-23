import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Any
from pathlib import Path
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params


class BaseAnalysisAgent(ABC):
    """
    Base class for analysis agents
    """
    
    def __init__(self, 
                 api_key: str | None = None, 
                 model_name: str = "gemini-3-pro-preview", 
                 base_url: str = None,
                 output_dir: str = ".",
                 # Deprecated arguments
                 google_api_key: str | None = None,
                 local_model: str = None,
                 **kwargs):
        
        self.logger = logging.getLogger(__name__)

        # --- State Management Init ---
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_type = "base_analysis"  # Subclasses should override
        self.state: Dict[str, Any] = {}
        
        # Normalize parameters (api_key vs google_api_key, base_url vs local_model)
        self.api_key, self.base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source=self.__class__.__name__
        )
        
        self.model_name = model_name
        self.generation_config = None
        self.safety_settings = None
        
        # Initialize Model based on configuration
        if self.base_url:
            # A. GGUF / Local File Mode
            if 'gguf' in self.base_url:
                 logging.info(f"💻 Using local agent (GGUF path detected): {self.base_url}")
                 from ...wrappers.llama_wrapper import LocalLlamaModel
                 self.model = LocalLlamaModel(self.base_url)

            # B. Internal Proxy / Network Mode
            else:
                logging.info(f"🏛️ Using OpenAI-compatible agent: {self.base_url}")
                
                # Match planning_agents logic: Use specific proxy key
                if self.api_key is None:
                    self.api_key = get_internal_proxy_key()
                
                if not self.api_key:
                    raise ValueError(
                        "API key required for internal proxy.\n"
                        "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                    )
                
                self.model = OpenAIAsGenerativeModel(
                    model=model_name, 
                    api_key=self.api_key, 
                    base_url=self.base_url
                )
        else:
            # C. Public / LiteLLM Mode
            logging.info(f"☁️ Using LiteLLM agent: {model_name}")
            
            # LiteLLM looks for env vars automatically, so self.api_key can be None here
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=self.api_key
            )

        self._stored_analysis_images = []
        self._stored_analysis_metadata = {}

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Override in subclasses to add agent-specific state fields."""
        return {}
    
    def _init_state(self, **context) -> None:
        """Initialize state for a new session."""
        if self.state.get("session_id") is None:
            self.state = {
                "session_id": str(uuid.uuid4()),
                "start_time": datetime.now().isoformat(),
                "agent_type": self.agent_type,
                "action_history": [],
                "status": "initialized"
            }
            # Add agent-specific fields
            self.state.update(self._get_initial_state_fields())
        
        # Update with provided context
        for key, value in context.items():
            self.state[key] = value
        
        self.state["status"] = "active"
        self._save_state()

    def _log_action(self, 
                    action: str, 
                    input_ctx: Dict[str, Any], 
                    result: Dict[str, Any], 
                    rationale: Optional[str] = None, 
                    feedback: Optional[str] = None) -> None:
        """Record an atomic action to state history and save."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "input": input_ctx,
            "rationale": rationale,
            "result": self._normalize_result(result),
            "feedback": feedback
        }
        
        if "action_history" not in self.state:
            self.state["action_history"] = []
        
        self.state["action_history"].append(entry)
        self._save_state()

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """Normalize result for JSON serialization."""
        if isinstance(result, dict):
            # Create a summary copy to avoid dumping massive arrays if they exist
            summary = result.copy()
            # If result contains massive keys (like 'image_data'), summarize them
            return summary
        return {"raw_result": str(result)}
    
    def _get_state_filename(self) -> str:
        return f"{self.agent_type}_state.json"

    def _save_state(self) -> None:
        """Persist state to disk."""
        state_file = self.output_dir / self._get_state_filename()
        try:
            with open(state_file, 'w') as f:
                # Use default=str to handle non-serializable objects like numpy arrays
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save {self.agent_type} state: {e}")

    def load_state(self, state_path: str) -> bool:
        """Restore state from disk."""
        path = Path(state_path)
        if not path.exists():
            self.logger.warning(f"State file not found: {state_path}")
            return False
        
        try:
            with open(path, 'r') as f:
                self.state = json.load(f)
            
            if "action_history" not in self.state:
                self.state["action_history"] = []
            
            self.logger.info(f"Restored {self.agent_type} state: session {self.state.get('session_id')}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load {self.agent_type} state: {e}")
            return False

    @abstractmethod
    def analyze_for_claims(self, data_path: str, system_info: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Analyze experimental data to generate a detailed summary and scientific claims.
        This is the primary entry point for any analysis agent.
        """
        raise NotImplementedError
    
    def _get_stored_analysis_images(self) -> List[Dict[str, Any]]:
        """
        Retrieves visual evidence (processed images) generated during the analysis.
        """
        return self._stored_analysis_images.copy()

    def _store_analysis_images(self, images: list, metadata: dict = None):
        """Helper to store analysis images for retrieval by workflows."""
        self._stored_analysis_images = images.copy() if images else []
        self._stored_analysis_metadata = metadata or {}
        self.logger.debug(f"Stored {len(self._stored_analysis_images)} analysis images.")

    def _parse_llm_response(self, response: Any) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Parse LLM response to extract JSON, with multiple fallback strategies.
        
        Strategies (in order):
        1. Direct JSON parse of response text
        2. Extract JSON from markdown code blocks (```json ... ```)
        3. Extract JSON from anywhere in text using regex
        4. For script generation: extract Python code blocks
        
        Args:
            response: Response object from LLM API
            
        Returns:
            Tuple of (result_json, error_dict) - one will be None
        """
        try:
            # Get text from response object (handle different API formats)
            raw_text = self._extract_text_from_response(response)
            
            if not raw_text or not raw_text.strip():
                return None, {
                    "error": "Empty response from LLM",
                    "details": "Response text was empty"
                }
            
            # Strategy 1: Direct JSON parse (with markdown cleanup)
            result = self._try_direct_json_parse(raw_text)
            if result is not None:
                return result, None
            
            # Strategy 2: Find JSON in markdown code blocks
            result = self._try_extract_json_from_code_blocks(raw_text)
            if result is not None:
                return result, None
            
            # Strategy 3: Find any JSON object in text
            result = self._try_extract_json_from_text(raw_text)
            if result is not None:
                self.logger.warning("JSON extracted via regex fallback - may be incomplete")
                return result, None
            
            # Strategy 4: For script responses, try to extract Python code
            result = self._try_extract_python_script(raw_text)
            if result is not None:
                self.logger.warning("JSON parsing failed, but found Python code block")
                return result, None
            
            # All strategies failed
            snippet = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
            self.logger.error(f"All JSON parsing strategies failed. Response snippet: {snippet}")
            
            return None, {
                "error": "Failed to parse valid JSON from LLM response",
                "details": "No valid JSON found using any extraction strategy",
                "raw_response": raw_text[:2000]  # Limit size for logging
            }
            
        except Exception as e:
            self.logger.exception(f"Unexpected error parsing LLM response: {e}")
            return None, {
                "error": "Exception during response parsing",
                "details": str(e)
            }
    
    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text content from various LLM response formats."""
        # Try different attribute patterns
        if hasattr(response, 'text'):
            return response.text
        if hasattr(response, 'content'):
            return response.content
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            if hasattr(choice, 'text'):
                return choice.text
        # Fallback to string conversion
        return str(response)
    
    def _try_direct_json_parse(self, raw_text: str) -> Optional[dict]:
        """Attempt direct JSON parsing with markdown cleanup."""
        try:
            text_clean = raw_text.strip()
            
            # Remove markdown code block markers
            if text_clean.startswith('```json'):
                text_clean = text_clean[7:]
            elif text_clean.startswith('```'):
                text_clean = text_clean[3:]
            if text_clean.endswith('```'):
                text_clean = text_clean[:-3]
            
            text_clean = text_clean.strip()
            
            result = json.loads(text_clean)
            if isinstance(result, dict):
                return result
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _try_extract_json_from_code_blocks(self, raw_text: str) -> Optional[dict]:
        """Extract JSON from markdown code blocks."""
        # Match ```json ... ``` or ``` ... ```
        patterns = [
            r'```json\s*\n?([\s\S]*?)\n?```',
            r'```\s*\n?(\{[\s\S]*?\})\n?```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, raw_text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match.strip())
                    if isinstance(result, dict) and len(result) > 0:
                        return result
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _try_extract_json_from_text(self, raw_text: str) -> Optional[dict]:
        """Extract JSON objects from anywhere in the text."""
        # Find potential JSON objects by matching braces
        # This handles nested objects
        
        start_idx = None
        brace_count = 0
        
        for i, char in enumerate(raw_text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    # Found a complete JSON object
                    potential_json = raw_text[start_idx:i+1]
                    try:
                        result = json.loads(potential_json)
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
        
        return None
    
    def _try_extract_python_script(self, raw_text: str) -> Optional[dict]:
        """Extract Python code blocks and wrap in synthetic JSON structure."""
        python_block_pattern = r'```python\s*([\s\S]*?)\s*```'
        matches = re.findall(python_block_pattern, raw_text, re.DOTALL)
        
        if matches:
            # Return the first (usually longest/most complete) Python code block
            script = matches[0].strip()
            if script:
                return {
                    "script": script,
                    "analysis_approach": "extracted_from_code_block",
                    "key_metrics_to_track": [],
                    "_extraction_method": "python_code_block_fallback"
                }
        
        return None

    def _generate_json_from_text_parts(self, prompt_parts: list) -> tuple[dict | None, dict | None]:
        """
        Internal helper to generate JSON from a list of textual prompt parts.
        Shared implementation used by multiple agents.
        """
        try:
            self.logger.debug(f"Sending text-only prompt to LLM. Total parts: {len(prompt_parts)}")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return self._parse_llm_response(response)
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during text-based LLM call: {e}")
            return None, {"error": "An unexpected error occurred during text-based LLM call", "details": str(e)}

    def _handle_system_info(self, system_info: dict | str | None) -> dict:
        """
        Handle system_info input (can be dict, file path, or None).
        Converts to dict format for consistent processing.
        """
        if isinstance(system_info, str):
            try:
                with open(system_info, 'r') as f:
                    system_info = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.error(f"Error loading system_info from {system_info}: {e}")
                system_info = {} # Proceed without system info if loading fails
        elif system_info is None:
            system_info = {} # Ensure it's a dict for easier access later
        
        return system_info

    def _find_spatial_info(self, data: dict) -> dict | None:
        """
        Recursively search for spatial_info in a nested dictionary structure.
        
        Args:
            data: Dictionary to search through
            
        Returns:
            The spatial_info dictionary if found, None otherwise
        """
        if not isinstance(data, dict):
            return None
        
        # Check if spatial_info exists at current level
        if 'spatial_info' in data and isinstance(data['spatial_info'], dict):
            return data['spatial_info']
        
        # Recursively search through all nested dictionaries
        for key, value in data.items():
            if isinstance(value, dict):
                result = self._find_spatial_info(value)
                if result is not None:
                    return result
        
        return None

    def _calculate_spatial_scale(
        self, 
        system_info: dict, 
        image_shape: tuple
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate spatial scale from system metadata.
        
        Supports multiple common key formats:
        - spatial_info.nm_per_pixel (nested dict)
        - pixel_size_nm
        - nm_per_pixel
        - scale_nm_per_pixel
        - pixel_size (assumes nm)
        - pixel_size with pixel_size_unit
        
        Args:
            system_info: System metadata dictionary
            image_shape: Shape of the image (h, w) or (h, w, c)
            
        Returns:
            Tuple of (nm_per_pixel, field_of_view_nm) or (None, None)
        """
        if not system_info:
            self.logger.info("No system_info provided. Physical scale not applied.")
            return None, None
        
        pixel_size = None
        
        # Strategy 1: Check nested spatial_info dict
        if 'spatial_info' in system_info:
            spatial = system_info['spatial_info']
            if isinstance(spatial, dict):
                pixel_size = spatial.get('nm_per_pixel') or spatial.get('pixel_size_nm')
        
        # Strategy 2: Check top-level keys (various naming conventions)
        if pixel_size is None:
            key_candidates = [
                'pixel_size_nm', 
                'nm_per_pixel', 
                'scale_nm_per_pixel',
                'pixel_size',
                'pixelSize',
                'pixel_scale',
                'scale',
                'resolution_nm'
            ]
            for key in key_candidates:
                if key in system_info and system_info[key] is not None:
                    pixel_size = system_info[key]
                    break
        
        # Strategy 3: Check for scale with units specified separately
        if pixel_size is None and 'pixel_size' in system_info:
            pixel_size = system_info['pixel_size']
        
        if pixel_size is None:
            self.logger.info("No spatial calibration found in system metadata. Physical scale not applied.")
            return None, None
        
        # Convert to float
        try:
            pixel_size = float(pixel_size)
        except (TypeError, ValueError):
            self.logger.warning(f"Invalid pixel_size value: {pixel_size}. Physical scale not applied.")
            return None, None
        
        if pixel_size <= 0:
            self.logger.warning(f"Invalid pixel_size value: {pixel_size}. Must be positive.")
            return None, None
        
        # Handle unit conversion if unit is specified
        unit = system_info.get('pixel_size_unit', 'nm').lower()
        if unit in ['um', 'µm', 'micron', 'microns', 'micrometer']:
            pixel_size *= 1000  # Convert um to nm
        elif unit in ['pm', 'picometer']:
            pixel_size /= 1000  # Convert pm to nm
        elif unit in ['a', 'angstrom', 'å']:
            pixel_size /= 10    # Convert Angstrom to nm
        # 'nm' or unrecognized units are assumed to be nm
        
        # Calculate field of view
        h, w = image_shape[:2]
        fov = max(h, w) * pixel_size
        
        self.logger.info(f"Spatial calibration: {pixel_size:.3f} nm/pixel, FOV: {fov:.1f} nm")
        
        return pixel_size, fov

    def _validate_scientific_claims(self, scientific_claims: list) -> list:
        """
        Validate scientific claims structure and content.
        Shared validation logic used by all agents that generate claims.
        """
        valid_claims = []

        if not isinstance(scientific_claims, list):
            self.logger.warning(f"'scientific_claims' from LLM was not a list: {scientific_claims}")
            return valid_claims

        for claim in scientific_claims:
            if isinstance(claim, dict) and all(k in claim for k in ["claim", "scientific_impact", "has_anyone_question", "keywords"]):
                if isinstance(claim.get("keywords"), list):
                    valid_claims.append(claim)
                else:
                    self.logger.warning(f"Claim skipped due to 'keywords' not being a list: {claim}")
            else:
                self.logger.warning(f"Claim skipped due to missing keys or incorrect dict format: {claim}")
        
        return valid_claims

    def _validate_structure_recommendations(self, recommendations: list) -> list:
        """
        Validate and sort structure recommendations.
        Shared validation logic used by all agents that generate recommendations.
        """
        valid_recommendations = []
        
        if not isinstance(recommendations, list):
            self.logger.warning(f"'structure_recommendations' from LLM was not a list: {recommendations}")
            return valid_recommendations

        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in ["description", "scientific_interest", "priority"]):
                if isinstance(rec.get("priority"), int):
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Recommendation skipped due to invalid priority type (expected int): {rec.get('priority')}. Recommendation: {rec}")
            else:
                self.logger.warning(f"Recommendation skipped due to missing keys or incorrect dict format: {rec}")
        
        # Sort by priority (1 = highest priority)
        sorted_recommendations = sorted(valid_recommendations, key=lambda x: x.get("priority", 99))
        return sorted_recommendations

    def _build_system_info_prompt_section(self, system_info: dict) -> str:
        """
        Build the system information section for LLM prompts.
        """
        if not system_info:
            return ""
            
        system_info_text = "\n\nAdditional System Information (Metadata):\n"
        if isinstance(system_info, dict):
            system_info_text += json.dumps(system_info, indent=2)
        else:
            system_info_text += str(system_info)
        
        return system_info_text
    
    def _store_analysis_images(self, images: list, metadata: dict = None):
        """Store analysis images for potential reuse in refinement."""
        self._stored_analysis_images = images.copy() if images else []
        self._stored_analysis_metadata = metadata or {}
        self.logger.debug(f"Stored {len(self._stored_analysis_images)} analysis images for potential refinement")
    
    def _get_stored_analysis_images(self) -> list:
        """Retrieve stored analysis images."""
        return self._stored_analysis_images.copy()
    
    def _clear_stored_images(self):
        """Clear stored images to free memory."""
        self._stored_analysis_images = []
        self._stored_analysis_metadata = {}

    def _refine_analysis_with_feedback(self, original_analysis: str, 
                                     original_claims: list, 
                                     user_feedback: str,
                                     instruction_prompt: str,
                                     stored_images: list = None,
                                     system_info: dict = None) -> dict:
        """Use LLM to refine analysis and claims based on user feedback."""
        try:
            # Create refinement prompt
            refinement_prompt = f"""
    {instruction_prompt}

    ## REFINEMENT TASK
    You previously generated this analysis and claims:

    ORIGINAL DETAILED ANALYSIS:
    {original_analysis}

    ORIGINAL CLAIMS:
    {json.dumps(original_claims, indent=2)}

    A human expert provided this feedback:
    "{user_feedback}"

    Use this feedback *thoughtfully* to refine both the detailed analysis and scientific claims. 
    Maintain the same JSON output format with "detailed_analysis" and "scientific_claims" keys.
    """
            
            prompt_parts = [refinement_prompt]
            
            # Add stored images
            if stored_images:
                for img_data in stored_images:
                    if isinstance(img_data, dict) and 'label' in img_data and 'data' in img_data:
                        prompt_parts.append(f"\n{img_data['label']}:")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": img_data['data']})
                    elif isinstance(img_data, dict) and img_data.get("mime_type") == "image/jpeg":
                        prompt_parts.append("\nAnalysis image:")
                        prompt_parts.append(img_data)
            
            # Add system info
            if system_info:
                system_info_section = self._build_system_info_prompt_section(system_info)
                if system_info_section:
                    prompt_parts.append(system_info_section)
            
            prompt_parts.append("\nProvide the refined analysis in JSON format.")
            
            # Query LLM for refinement
            self.logger.info("🔄 Refining analysis using stored images...")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                return {"error": "Refinement failed", "details": error_dict}
            
            # Validate refined claims
            refined_claims = result_json.get("scientific_claims", [])
            validated_claims = self._validate_scientific_claims(refined_claims)
            
            self.logger.info(f"Refinement complete: {len(validated_claims)} validated claims")
            
            return {
                "detailed_analysis": result_json.get("detailed_analysis", original_analysis),
                "scientific_claims": validated_claims
            }
            
        except Exception as e:
            self.logger.error(f"Analysis refinement failed: {e}")
            return {"error": "Refinement failed", "details": str(e)}
        
    def _validate_measurement_recommendations(self, recommendations: list) -> list:
        """Simple validation of measurement recommendations."""
        valid_recommendations = []
        
        if not isinstance(recommendations, list):
            return valid_recommendations
        
        required_keys = ["description", "scientific_justification", "priority"]
        
        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in required_keys):
                # Simple validation
                if isinstance(rec.get("priority"), int) and 1 <= rec.get("priority") <= 5:
                    valid_recommendations.append(rec)
        
        # Sort by priority
        return sorted(valid_recommendations, key=lambda x: x.get("priority", 5))
    
    def generate_measurement_recommendations(self, analysis_result: dict, 
                                           system_info: dict = None,
                                           novelty_context: str = None) -> dict:
        """
        Generate measurement recommendations using existing analysis results and stored images.
        """
        if "error" in analysis_result:
            return {"error": "Cannot generate recommendations from failed analysis"}
        
        try:
            # Get agent-specific prompt
            instruction_prompt = self._get_measurement_recommendations_prompt()
            
            # Build simple prompt using existing data
            prompt_parts = [instruction_prompt]
            
            # Add analysis results (what's already available)
            prompt_parts.append("\n\n--- Analysis Results ---")
            
            if "detailed_analysis" in analysis_result:
                prompt_parts.append(f"Detailed Analysis:\n{analysis_result['detailed_analysis']}")
            
            if "scientific_claims" in analysis_result:
                prompt_parts.append(f"\nScientific Claims:")
                for i, claim in enumerate(analysis_result["scientific_claims"], 1):
                    prompt_parts.append(f"{i}. {claim.get('claim', 'N/A')}")
            
            # Add stored images (what's already available from the analysis)
            stored_images = self._get_stored_analysis_images()
            if stored_images:
                prompt_parts.append(f"\n\nAnalysis Images:")
                for img_data in stored_images[:3]:  # Limit to 3 images
                    if isinstance(img_data, dict) and 'label' in img_data and 'data' in img_data:
                        prompt_parts.append(f"\n{img_data['label']}:")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": img_data['data']})
            
            # Add optional context
            if novelty_context:
                prompt_parts.append(f"\n\nNovelty Context: {novelty_context}")
            
            if system_info:
                system_info_section = self._build_system_info_prompt_section(system_info)
                if system_info_section:
                    prompt_parts.append(system_info_section)
            
            prompt_parts.append("\n\nProvide measurement recommendations in JSON format.")
            
            # Query LLM
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                return {"error": "Recommendation generation failed", "details": error_dict}
            
            # Validate recommendations
            recommendations = result_json.get("measurement_recommendations", [])
            valid_recommendations = self._validate_measurement_recommendations(recommendations)
            
            return {
                "analysis_integration": result_json.get("analysis_integration", ""),
                "measurement_recommendations": valid_recommendations,
                "total_recommendations": len(valid_recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return {"error": "Recommendation generation failed", "details": str(e)}

    def _get_measurement_recommendations_prompt(self) -> str:
        """Must be implemented by each agent."""
        raise NotImplementedError("Each agent must implement this method")