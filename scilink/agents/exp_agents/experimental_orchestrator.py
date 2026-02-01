"""
ExperimentalAnalysisOrchestrator - Interactive chat interface for experimental data analysis.

Follows the same architecture as PlanningOrchestratorAgent:
- LLM-powered chat with tool calling
- Dual backend support (OpenAI-compatible / LiteLLM)
- Tool registry with OpenAI and Gemini schema formats
- Session state and history management
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime

from .fft_microscopy_agent import FFTMicroscopyAnalysisAgent
from .sam_microscopy_agent import SAMMicroscopyAnalysisAgent
from .hyperspectral_analysis_agent import HyperspectralAnalysisAgent
from .curve_fitting_agent import CurveFittingAgent
from .experimental_orchestrator_tools import ExperimentalOrchestratorTools
from ._deprecation import normalize_params

# Try to import auth utilities
try:
    from ...auth import get_internal_proxy_key
except ImportError:
    def get_internal_proxy_key():
        return os.getenv("SCILINK_API_KEY")

# Try to import LLM wrappers
try:
    from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
    from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
except ImportError:
    OpenAIAsGenerativeModel = None
    LiteLLMGenerativeModel = None


# =============================================================================
# AGENT REGISTRY
# =============================================================================

class AgentType(Enum):
    """Available analysis agent types."""
    FFT_MICROSCOPY = "fft_microscopy"
    SAM_MICROSCOPY = "sam_microscopy"
    HYPERSPECTRAL = "hyperspectral"
    CURVE_FITTING = "curve_fitting"


AGENT_REGISTRY = {
    AgentType.FFT_MICROSCOPY: {
        "class": FFTMicroscopyAnalysisAgent,
        "description": "FFT/NMF-based microscopy analysis for microstructure, phases, and periodic patterns",
        "data_types": ["microscopy", "image"],
        "file_extensions": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    },
    AgentType.SAM_MICROSCOPY: {
        "class": SAMMicroscopyAnalysisAgent,
        "description": "Segment Anything Model for particle/object detection and morphological analysis",
        "data_types": ["microscopy", "image", "particle"],
        "file_extensions": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    },
    AgentType.HYPERSPECTRAL: {
        "class": HyperspectralAnalysisAgent,
        "description": "Hyperspectral/spectroscopic data analysis with NMF unmixing",
        "data_types": ["spectroscopy", "hyperspectral", "eels", "eds"],
        "file_extensions": [".npy"],
    },
    AgentType.CURVE_FITTING: {
        "class": CurveFittingAgent,
        "description": "1D curve fitting for spectra, diffractograms, and time series",
        "data_types": ["spectrum", "curve", "xrd", "pl", "raman", "absorption"],
        "file_extensions": [".csv", ".txt", ".npy", ".xlsx"],
    },
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are SciLink's Experimental Analysis Assistant - an expert system for analyzing scientific experimental data.

**RESPONSE GUIDELINES:**
- Be conversational and helpful
- After tool calls, summarize results concisely - do NOT repeat raw JSON output
- Suggest next steps or follow-up analyses when appropriate

**AVAILABLE TOOLS:**

**ANALYSIS TOOLS:**
1. `analyze_microscopy_fft` - FFT/NMF analysis for periodic structures, domains, phases, lattices
2. `analyze_microscopy_sam` - Particle/object detection and morphological measurements
3. `analyze_hyperspectral` - Spectroscopic unmixing and mapping (EELS, EDS, hyperspectral)
4. `analyze_curve` - 1D curve fitting (Raman, XRD, PL, IR, absorption spectra)

**SELECTION TOOLS:**
5. `select_microscopy_agent` - Visually examine an image to choose between FFT and SAM analysis

**UTILITY TOOLS:**
6. `read_file` - Read metadata files (JSON, text, YAML)
7. `list_directory` - List files in a directory
8. `convert_metadata` - Convert natural language description to structured metadata
9. `list_available_agents` - Show all available analysis agents

**WORKFLOW:**
1. User provides data file path
2. Get metadata (from file, user description, or ask for it)
3. For microscopy images, optionally use `select_microscopy_agent` to choose FFT vs SAM
4. Run appropriate analysis tool with data path and metadata JSON
5. Summarize findings and suggest follow-up

**METADATA REQUIREMENT:**
All analysis tools require metadata as a JSON string. Metadata should include:
- experiment_type: "Microscopy", "Spectroscopy", "Diffraction", etc.
- technique: "STEM", "TEM", "Raman", "XRD", "EELS", etc.
- sample material and description

If user provides a metadata file path, use `read_file` to load it first.
If user describes their experiment in natural language, use `convert_metadata` to structure it.

**EXAMPLE INTERACTIONS:**

User: "Analyze my TEM image at ./sample.tif"
→ Ask for metadata or use read_file if they mention a metadata file

User: "see metadata.json for the experiment info"  
→ Call read_file("metadata.json"), then proceed with analysis

User: "It's a STEM image of MoS2 monolayer"
→ Call convert_metadata with that description, then analyze
"""


# =============================================================================
# EXPERIMENTAL ANALYSIS ORCHESTRATOR
# =============================================================================

class ExperimentalAnalysisOrchestrator:
    """
    Interactive chat-based orchestrator for experimental data analysis.
    
    Follows the same architecture as PlanningOrchestratorAgent:
    - LLM-powered chat interface with tool calling
    - Dual backend support (OpenAI-compatible APIs and LiteLLM)
    - Tool registry with schemas for both providers
    - Session state and history management
    
    Args:
        api_key: API key for the LLM provider
        model_name: Model name for the chat LLM
        base_url: Base URL for OpenAI-compatible endpoint (internal proxy)
        output_dir: Base directory for analysis outputs
        enable_human_feedback: Enable human-in-the-loop feedback in sub-agents
        
    Example:
        orchestrator = ExperimentalAnalysisOrchestrator(api_key="...")
        
        # Start interactive chat session
        orchestrator.start_chat_session()
        
        # Or process a single message
        response = orchestrator.chat("Analyze my TEM image at sample.tif")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        output_dir: str = "./analysis_outputs",
        enable_human_feedback: bool = False,
        # Deprecated
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="ExperimentalAnalysisOrchestrator"
        )
        
        # Validate API key for internal proxy
        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
        
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_human_feedback = enable_human_feedback
        
        # Session state
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.message_count = 0
        self.current_metadata = None
        self.last_analysis = None
        self.analyses_run = []
        
        # Agent cache (lazy initialization)
        self._agent_cache: Dict[AgentType, Any] = {}
        
        # --- Initialize Tools Registry ---
        self.tools = ExperimentalOrchestratorTools(self)
        
        # --- LLM Initialization ---
        if base_url:
            logging.info(f"🔬 Analysis Orchestrator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
            self.use_openai = True
            self.tools_for_model = self.tools.openai_schemas
        else:
            logging.info(f"🔬 Analysis Orchestrator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key,
                system_instruction=SYSTEM_PROMPT,
                tools=self.tools.gemini_functions
            )
            self.use_openai = False
            self.tools_for_model = self.tools.gemini_functions
        
        # Store system prompt
        self._system_prompt = SYSTEM_PROMPT
        
        # --- Message History ---
        if self.use_openai:
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            self.chat_session = self.model.start_chat(
                history=[],
                enable_automatic_function_calling=True
            )
        
        logging.info(f"ExperimentalAnalysisOrchestrator initialized. Output: {self.output_dir}")
    
    def get_or_create_agent(self, agent_type: AgentType) -> Any:
        """Get or create an analysis agent instance (lazy initialization)."""
        if agent_type not in self._agent_cache:
            agent_info = AGENT_REGISTRY[agent_type]
            agent_output_dir = self.output_dir / agent_type.value
            agent_output_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"  🔧 Initializing {agent_type.value} agent...")
            
            self._agent_cache[agent_type] = agent_info["class"](
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                output_dir=str(agent_output_dir),
                enable_human_feedback=self.enable_human_feedback,
            )
        return self._agent_cache[agent_type]
    
    def chat(self, user_input: str) -> str:
        """
        Main chat interface with tool calling support.
        
        Args:
            user_input: User's message
            
        Returns:
            Assistant's response text
        """
        self.message_count += 1
        
        try:
            if self.use_openai:
                response_text = self._handle_openai_chat(user_input)
            else:
                response = self.chat_session.send_message(user_input)
                response_text = self._extract_response_text(response)
            
            return response_text
            
        except Exception as e:
            logging.error(f"Chat Error: {e}", exc_info=True)
            return f"❌ Error: {e}"
    
    def _handle_openai_chat(self, user_input: str) -> str:
        """Handle chat with OpenAI-compatible models with manual function calling loop."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url
        )
        
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # Trim history if too long
        if len(self.messages) > 100:
            logging.info("  ⚠️  Trimming conversation history...")
            system_msg = self.messages[0]
            recent_msgs = self.messages[-80:]
            self.messages = [system_msg] + recent_msgs
        
        max_iterations = 15
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = client.chat.completions.create(
                model=self.model.model,
                messages=self.messages,
                tools=self.tools_for_model,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Check if no tool calls - we have final response
            if not message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content or ""
            
            # Process tool calls
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in message.tool_calls
                ]
            })
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                print(f"  🔧 Calling tool: {func_name}")
                
                result = self.tools.execute_tool(func_name, **args)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        return "⚠️ Maximum tool iterations reached. Please simplify your request."
    
    def _extract_response_text(self, response) -> str:
        """Robustly extract text from different response formats."""
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            text_parts = [p.text for p in response.parts if hasattr(p, 'text')]
            return ' '.join(text_parts)
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def start_chat_session(self) -> None:
        """
        Start an interactive chat session in the terminal.
        
        The session continues until the user types 'exit', 'quit', or 'q'.
        """
        print("\n" + "="*60)
        print("🔬 SCILINK EXPERIMENTAL ANALYSIS")
        print("    Interactive Chat Session")
        print("="*60)
        print("\nI'm your experimental analysis assistant. I can help you analyze:")
        print("  • Microscopy images (TEM, STEM, SEM, AFM)")
        print("  • Hyperspectral/spectroscopic data (EELS, EDS)")
        print("  • 1D curves (Raman, XRD, PL, IR, absorption)")
        print("\nTo get started, tell me about your data file and experiment.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print(f"\n👋 Session ended. Results saved to: {self.output_dir}")
                    break
                
                print("\n🤔 Processing...\n")
                response = self.chat(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Session interrupted. Results saved to: {self.output_dir}")
                break
            except EOFError:
                print("\n\n👋 Session ended.")
                break
    
    def reset_session(self) -> None:
        """Reset the chat session (clear history and state)."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.message_count = 0
        self.current_metadata = None
        self.last_analysis = None
        self.analyses_run = []
        
        if self.use_openai:
            self.messages = [{"role": "system", "content": self._system_prompt}]
        else:
            self.chat_session = self.model.start_chat(
                history=[],
                enable_automatic_function_calling=True
            )
        
        logging.info("Session reset")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "analyses_run": len(self.analyses_run),
            "current_metadata": self.current_metadata,
            "output_directory": str(self.output_dir)
        }