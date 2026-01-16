import json
import logging
from typing import Dict, Any, List, Optional
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

    def _parse_llm_response(self, response) -> tuple[dict | None, dict | None]:
        """
        Parses the LLM response, expecting JSON.
        """
        result_json = None
        error_dict = None
        raw_text = None
        json_string = None

        try:
            # Handle LiteLLM/OpenAI wrapper response object
            if hasattr(response, 'text'):
                raw_text = response.text
            elif hasattr(response, 'choices'): # Fallback for raw OpenAI objects
                 raw_text = response.choices[0].message.content
            else:
                 raw_text = str(response)

            first_brace_index = raw_text.find('{')
            last_brace_index = raw_text.rfind('}')
            if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                json_string = raw_text[first_brace_index : last_brace_index + 1]
                result_json = json.loads(json_string)
            else:
                raise ValueError("Could not find valid JSON object delimiters '{' and '}' in the response text.")

        except (json.JSONDecodeError, AttributeError, IndexError, ValueError) as e:
            error_details = str(e)
            self.logger.error(f"Error parsing LLM JSON response: {e}")
            error_dict = {"error": "Failed to parse valid JSON from LLM response", "details": error_details, "raw_response": raw_text}
        
        except Exception as e:
            self.logger.exception(f"Unexpected error processing response: {e}")
            error_dict = {"error": "Unexpected error processing LLM response", "details": str(e)}
        
        return result_json, error_dict

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

    def _calculate_spatial_scale(self, system_info: dict, image_shape: tuple) -> tuple[float | None, float | None]:
        """
        Calculate nm/pixel and field of view from system metadata.
        
        Args:
            system_info: Dictionary containing spatial metadata (can be nested anywhere)
            image_shape: (height, width) of the loaded image
            
        Returns:
            tuple: (nm_per_pixel, fov_in_nm) or (None, None) if calculation fails
        """
        # Determine physical scale (nm/pixel) from metadata, mirroring spectroscopy agent's approach
        nm_per_pixel = None
        fov_in_nm = None
        
        # Search for spatial_info anywhere in the nested structure
        spatial = self._find_spatial_info(system_info)
        
        if spatial is not None:
            fov_x = spatial.get("field_of_view_x")
            units = spatial.get("field_of_view_units", "nm")  # Default to nm

            if fov_x is not None and isinstance(fov_x, (int, float)) and fov_x > 0:
                h, w = image_shape[:2]
                
                # Convert provided units to nm for consistent calculations
                scale_to_nm = 1.0
                if units.lower() in ['um', 'micrometer', 'micrometers']:
                    scale_to_nm = 1000.0
                elif units.lower() in ['a', 'angstrom', 'angstroms']:
                    scale_to_nm = 0.1
                
                fov_in_nm = fov_x * scale_to_nm
                
                if w > 0:
                    nm_per_pixel = fov_in_nm / w
                    self.logger.info(f"Found spatial_info and calculated nm/pixel: {fov_x} {units} / {w} px = {nm_per_pixel:.4f} nm/pixel")
                else:
                    self.logger.warning("Cannot calculate scale from FOV because image width is 0.")
            else:
                self.logger.warning(f"Invalid or missing 'field_of_view_x' in spatial_info: {fov_x}. Physical scale not applied.")
        else:
            self.logger.info("No spatial_info found in system metadata. Physical scale not applied.")
        
        return nm_per_pixel, fov_in_nm

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