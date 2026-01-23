import logging
import json

from .microscopy_agent import MicroscopyAnalysisAgent
from .sam_microscopy_agent import SAMMicroscopyAnalysisAgent
from .atomistic_microscopy_agent import AtomisticMicroscopyAnalysisAgent
from .hyperspectral_analysis_agent import HyperspectralAnalysisAgent
from .instruct import ORCHESTRATOR_INSTRUCTIONS

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params

# Mapping from integer ID to the corresponding agent class
AGENT_MAP = {
    0: MicroscopyAnalysisAgent,
    1: SAMMicroscopyAnalysisAgent,
    2: AtomisticMicroscopyAnalysisAgent,
    3: HyperspectralAnalysisAgent
}

class OrchestratorAgent:
    """
    An LLM-powered agent that selects the most appropriate experimental analysis agent.
    """
    def __init__(self, 
                 api_key: str | None = None, 
                 model_name: str = "gemini-3-flash-preview", 
                 base_url: str = None,
                 # Deprecated
                 google_api_key: str | None = None, 
                 local_model: str = None):
        
        self.logger = logging.getLogger(__name__)

        self.api_key, self.base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="OrchestratorAgent"
        )
        
        # Initialize Model
        if self.base_url:
            if 'gguf' in self.base_url:
                logging.info(f"💻 Using local agent (GGUF path detected): {self.base_url}")
                from ...wrappers.llama_wrapper import LocalLlamaModel
                self.model = LocalLlamaModel(self.base_url)
            else:
                logging.info(f"🏛️ Using OpenAI-compatible agent as orchestrator: {self.base_url}")
                if self.api_key is None:
                    self.api_key = get_internal_proxy_key()
                
                if not self.api_key:
                    raise ValueError("API key required for internal proxy.")

                self.model = OpenAIAsGenerativeModel(model=model_name, api_key=self.api_key, base_url=self.base_url)
        else:
            logging.info(f"☁️ Using LiteLLM agent as the orchestrator: {model_name}")
            # LiteLLM finds API keys in env vars automatically
            self.model = LiteLLMGenerativeModel(
                model=model_name, 
                api_key=self.api_key
            )

        self.generation_config = None

    def _parse_llm_response(self, response) -> tuple[dict | None, dict | None]:
        """Parses the LLM response, expecting JSON."""
        try:
            raw_text = response.text
            first_brace_index = raw_text.find('{')
            last_brace_index = raw_text.rfind('}')
            if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                json_string = raw_text[first_brace_index : last_brace_index + 1]
                return json.loads(json_string), None
            else:
                raise ValueError("Could not find valid JSON object in response.")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM JSON response: {e}")
            error_dict = {"error": "Failed to parse JSON from LLM response", "details": str(e), "raw_response": response.text}
            return None, error_dict

    def select_agent(self, data_type: str, system_info: dict | str | None = None, image_path: str | None = None) -> tuple[int, str | None]:
        """
        Selects the appropriate experimental agent by asking the LLM.

        Args:
            data_type: The primary type of data, e.g., 'microscopy' or 'spectroscopy'.
            system_info: Additional information about the system or analysis goal.
            image_path: Optional path to an image for visual context.

        Returns:
            A tuple containing the integer key for the selected agent and the reasoning string.
            Returns (-1, error_message) on failure.
        """
        self.logger.info(f"Orchestrator LLM selecting agent for data_type: '{data_type}'")

        # Require system_info to be provided for an informed decision.
        if not system_info or not str(system_info).strip():
            error_msg = "system_info must be provided for the orchestrator to select an agent."
            self.logger.error(error_msg)
            return -1, error_msg

        # If the data type is spectroscopy, the choice is clear.
        if data_type.lower() == 'spectroscopy':
            self.logger.info("Data type is 'spectroscopy'. Selecting Spectroscopy agent directly.")
            return 3, "Data type is 'spectroscopy'. Selecting Spectroscopy agent directly."

        prompt_parts = [ORCHESTRATOR_INSTRUCTIONS]
        
        # Add image analysis if available
        if image_path:
            try:
                from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes
                
                # Load and preprocess image (resize to a max dimension to save tokens)
                image = load_image(image_path)
                resized_image, _ = preprocess_image(image, max_dim=512) # Downsample for the orchestrator
                image_bytes = convert_numpy_to_jpeg_bytes(resized_image, quality=80)
                
                prompt_parts.append("\n--- Image for Context ---")
                prompt_parts.append("This is a downsampled version of the input image used to provide visual context. It may be lower resolution than the original.")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
                
            except Exception as e:
                self.logger.warning(f"Failed to load or analyze image for orchestrator: {e}")
                prompt_parts.append("\n(Image loading/analysis failed, proceeding without image data)")
        
        # Add system info and data type
        prompt_parts.append("\n--- User Request ---")
        prompt_parts.append(f"data_type: {data_type}")
        prompt_parts.append(f"system_info: {str(system_info)}")

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)

            if error_dict:
                self.logger.error(f"Orchestrator LLM call failed: {error_dict}")
                return -1, f"Orchestrator LLM call failed: {error_dict.get('details', 'Unknown error')}"

            agent_id = result_json.get('agent_id')
            reasoning = result_json.get('reasoning', 'No reasoning provided.')
            self.logger.info(f"\n\n🧠 Orchestrator Reasoning: {reasoning}\n")


            if isinstance(agent_id, int) and agent_id in AGENT_MAP:
                return agent_id, reasoning
            else:
                error_msg = f"LLM returned an invalid agent ID: {agent_id}."
                self.logger.warning(error_msg)
                return -1, error_msg

        except Exception as e:
            error_msg = f"An unexpected error occurred during agent selection: {e}"
            self.logger.exception(error_msg)
            return -1, error_msg