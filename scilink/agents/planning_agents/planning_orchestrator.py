import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .planning_agent import PlanningAgent
from .scalarizer_agent import ScalarizerAgent
from .bo_agent import BOAgent
from .orchestrator_tools import OrchestratorTools
from ._deprecation import normalize_params


class AutonomyLevel(Enum):
    """
    Defines the level of autonomy for the orchestrator.
    
    CO_PILOT: AI assists human (default). Human reviews all plans/code.
    SUPERVISED: Human assists AI. AI proceeds unless human intervenes.
    AUTONOMOUS: Full autonomy. No human feedback requested.
    """
    CO_PILOT = "co_pilot"       # Human leads, AI assists (current default)
    SUPERVISED = "supervised"   # AI leads, human can intervene
    AUTONOMOUS = "autonomous"   # Full autonomy, no human feedback


# Mode-specific directives (inserted at the top)
_CO_PILOT_DIRECTIVE = """
**CRITICAL OPERATING MODE: CO-PILOT (Human Leads, AI Assists)**
- You are assisting the human researcher. They are in control.
- ALWAYS wait for human approval before proceeding to next steps.
- After generating plans or code, summarize and wait for feedback.
- Do NOT chain multiple tool calls without human confirmation.
- Ask clarifying questions when objectives are ambiguous.

**SINGLE-TOOL EXECUTION RULE:**
1. **EXECUTE ONE TOOL**: Call only ONE tool per response.
2. **OBSERVE OUTPUT**: meaningful "next steps" depend on what the tool *actually* returned.
"""

_SUPERVISED_DIRECTIVE = """
**CRITICAL OPERATING MODE: SUPERVISED (AI Leads, Human Supervises)**
- You lead the research workflow. Human supervises and can intervene.
- Proceed with reasonable next steps without asking for permission.
- Human will still review generated plans and code through the standard review interface.
- Do NOT ask clarifying questions unless truly ambiguous - make reasonable assumptions.
- If a tool returns error or unexpected results, pause and report to human.
- Periodically summarize progress (every 3-5 steps) but don't wait for response.
- Use your judgment but remain open to human corrections.
"""

_AUTONOMOUS_DIRECTIVE = """
**CRITICAL OPERATING MODE: FULLY AUTONOMOUS**
- Execute the complete research workflow independently.
- Chain tool calls as needed to achieve the objective.
- Only pause for human input if you encounter unrecoverable errors.
- Make decisions based on tool outputs and scientific reasoning.
- Save checkpoints regularly for human review later.
- Proceed through: plan → execute → analyze → optimize → iterate.
- Report final results and key decision points at the end.
- NOTE: Human still performs physical experiments in the lab
"""

_SYSTEM_PROMPT_BODY = """
You are the **Research Agent**. Your goal is to coordinate a scientific campaign.

**RESPONSE GUIDELINES (STRICT):**
- **NO REDUNDANCY**: Do NOT repeat the tool's output. Summarize insights only.
- **NO "WOULD YOU LIKE ME TO"**: Do NOT end your response with a generic menu of options. If the next step is obvious (e.g., "Run the experiment"), just say "Ready for results." or "I recommend saving a checkpoint."

**TOOLCHAIN & WORKFLOWS:**

**SETUP:**
0. `show_directory_guide`: Show recommended project structure. Use when user asks about setup/organization.

**STRATEGY & PLANNING TOOLS:**
1. `generate_initial_plan`: Use this when starting a NEW campaign or defining a new objective.
   - Extract knowledge_paths when user mentions papers/PDFs/documents
   - Extract primary_data_set when user mentions experimental data or results folders or files
   - additional_context: Lab constraints, equipment, reagents, budget
   - Previous TEA results automatically included
   - Example:
     * "Generate plan for Li recovery using info in ./papers/ and preliminary results in ./data/"
       → generate_initial_plan(specific_objective="Li recovery", 
                               knowledge_paths="./papers", 
                               primary_data_set="./data")
     
2. `run_economic_analysis`: Use this if the user asks about costs, viability, market fit, or TEA.
    - When primary_data_set is provided, ALL analysis and planning must be constrained to materials/conditions actually present in that data. Literature provides process knowledge, not feedstock assumptions.

    - Example:
        * "Use reports in ./papers/ and composition data in ./data/ to determine most profitable material"
        → run_economic_analysis(
            knowledge_paths="./papers",
            primary_data_set="./data"

3. `generate_implementation_code`: Add executable code to existing plan.
   - Maps experimental steps to APIs/automation code
   - Use AFTER generate_initial_plan() once strategy is approved
   - **TRIGGER CONDITIONS (both must be true):**
     a) User asks for "script", "protocol", "code", or mentions equipment (Opentrons, robot, automation)
     b) EITHER:
        - Code KB already loaded (you'll see "✅ Code KB loaded" at startup), OR
        - User specifies a code directory (e.g., "using ./opentrons_api", "from ./code folder")

4. `refine_plan_with_results`: Refine scientific strategy based on experimental results.
   - Use for: failures, pivots, qualitative observations, visual analysis
   - Accepts: text descriptions, file paths, or comma-separated files
   - Updates: Scientific plan only (no code changes)
   - Example:
     * "Refine based on ./run_005.csv and ./plot.png"
       → refine_plan_with_results(result_data="./run_005.csv,./plot.png")
   
5. `refine_implementation_code`: Update executable code for refined plan.
   - Use AFTER refine_plan_with_results() once strategy is approved
   - Maps refined experimental steps to code
   - Example:
     * After plan refinement is approved
       → refine_implementation_code()

6. `discard_plan`: Discard wrong plan (keeps in history for transparency).


**DATA TOOLS:**
7. `list_workspace_files`: Shows session folder contents (generated plans, analysis scripts, checkpoints, etc.)
8. `analyze_file`: Use this for RAW DATA files (CSV, XLSX, TXT) to calculate metrics via code.
    - First use: Generates analysis script automatically
    - Subsequent uses: Reuses script for consistency
    - force_regenerate=True: Use when analysis needs change
9. `reset_analysis_logic`: Use this if the analysis script is wrong.

**OPTIMIZATION TOOLS:**
10. `run_optimization`: Use this to get mathematical parameter suggestions (Bayesian Optimization).
    - Sequential: run_optimization() 
    - Parallel: run_optimization(parallel_capable=True, batch_size=N)
     * Infer N from context or ask user. Retry if "batch_size_required" returned.
11. `save_checkpoint`: Save campaign state. Use after every 3-5 experiments.

**FILE PATH RULES:**
Assume user runs agent from project directory. For example, when user says "file.csv in data", use "./data/file.csv"

**When user mentions a SPECIFIC filename:**
1. Extract the filename (with or without extension)
2. Pass it to the tool
3. Tool will automatically:
   - Try exact match
   - Try common extensions (.csv, .xlsx, .xls) if no extension provided
   - Search in ./experimental_results, ./data, ./results, ./
   - Suggest corrections for typos using fuzzy matching

**CRITICAL WORKFLOW RULES:**
**Use `run_optimization` (The Math Loop) IF:**
- You are optimizing a well-defined property for the current experimental setup.
- The experiments are running successfully (no failures), and you just need to tune parameters.
- **At least 3 data files have been successfully analyzed** (check by calling list_workspace_files).

**Use `iterate_with_results` (The Cognitive Loop) IF:**
- You need to propose a NEW strategy or experimental setup (e.g., "Change catalyst").
- The experiment FAILED (e.g., "Precipitate formed", "Equipment error").
- The result is qualitative (e.g., **Images**, visual observations, logs).
- There are NOT enough data points for numerical optimization yet.

**If user indicates the generated plan is wrong:**
- Common patterns: "That's not what I asked for", "Wrong material", "Focus on X instead"
- Actions:
  - Ask user to confirm: "I generated a plan for [X], but you mentioned [Y]. Should I correct this?" 
  - If user confirms it's wrong:
        a. Call discard_plan(reason="Specific explanation of what was wrong") 
        b. Call generate_initial_plan(...) again with corrected parameters 
- The discarded plan stays in history for transparency but won't appear in reports

**LONG CAMPAIGN MANAGEMENT:**
- Call `save_checkpoint` after every 3-5 experiments
- If conversation becomes very long (>50 messages), suggest user restart with checkpoint

**BEHAVIOR:**
- Extract ALL paths mentioned by user (papers, data, code, reports)
- Extract specific_objective from user's goal/intent
- Combine lab constraints into additional_context (equipment, reagents, pH, budget, etc.)
- Parse tool JSON responses before calling dependent tools
- If status="error", stop and report to user
- Save checkpoint periodically during long campaigns
"""


def get_system_prompt(autonomy_level: AutonomyLevel) -> str:
    """Returns the appropriate system prompt for the given autonomy level."""
    directives = {
        AutonomyLevel.CO_PILOT: _CO_PILOT_DIRECTIVE,
        AutonomyLevel.SUPERVISED: _SUPERVISED_DIRECTIVE,
        AutonomyLevel.AUTONOMOUS: _AUTONOMOUS_DIRECTIVE,
    }
    return directives[autonomy_level] + _SYSTEM_PROMPT_BODY



class PlanningOrchestratorAgent:
    """
    Orchestrator agent for coordinating multi-iteration research campaigns.
    
    Manages the full experimental loop with configurable autonomy:
    1. Hypothesis generation (PlanningAgent)
    2. Experiment execution (external)
    3. Result analysis (ScalarizerAgent)
    4. Parameter optimization (BOAgent)
    5. Iteration decisions
    
    Args:
        objective: Research objective description.
        base_dir: Base directory for campaign outputs.
        api_key: API key for the LLM provider.
        model_name: Model name.
        base_url: Base URL for internal proxy endpoint.
        embedding_model: Embedding model name.
        embedding_api_key: API key for the embedding LLM provider.
        futurehouse_api_key: Optional FutureHouse API key for literature search.
        restore_checkpoint: Whether to restore from previous checkpoint.
        autonomy_level: Level of autonomy (CO_PILOT, SUPERVISED, or AUTONOMOUS).
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        objective: str = "Undefined Research Goal",
        base_dir: str = "./campaign_outputs",
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        embedding_api_key: Optional[str] = None,
        futurehouse_api_key: Optional[str] = None,
        restore_checkpoint: bool = False,
        autonomy_level: AutonomyLevel = AutonomyLevel.CO_PILOT,
        data_dir: Optional[str] = None,
        knowledge_dir: Optional[str] = None,
        code_dir: Optional[str] = None,
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
            source="PlanningOrchestratorAgent"
        )
        
        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
            
            if embedding_api_key is not None:
                logging.warning(
                    "⚠️ embedding_api_key is ignored for internal proxy. "
                    "Using api_key for all requests."
                )
            
            embedding_api_key = api_key

        # Store autonomy level
        self.autonomy_level = autonomy_level
        self._enable_human_feedback = self._should_enable_human_feedback()
        logging.info(f"🎛️  Autonomy Level: {autonomy_level.value.upper()}")

        # Validate and store workspace directories
        if autonomy_level in (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS):
            if data_dir is None:
                raise ValueError(
                    f"data_dir is required for {autonomy_level.value} mode.\n"
                    f"Specify the directory containing experimental results."
                )
            if not Path(data_dir).exists():
                raise ValueError(f"data_dir does not exist: {data_dir}")

        self.data_dir = Path(data_dir) if data_dir else None
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else None
        self.code_dir = Path(code_dir) if code_dir else None

        if self.data_dir:
            logging.info(f"   Data directory: {self.data_dir}")

        self.objective = objective
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.analyzed_files_path = self.base_dir / "analyzed_files.json"
        self.analyzed_files = {}
        
        if self.analyzed_files_path.exists():
            try:
                with open(self.analyzed_files_path, 'r') as f:
                    self.analyzed_files = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load analyzed_files.json: {e}")
                self.analyzed_files = {}

        self.bo_data_path = self.base_dir / "optimization_data.csv"
        self.history_path = self.base_dir / "chat_history.json"
        self.checkpoint_path = self.base_dir / "checkpoint.json"
        
        self.active_scalarizer_script = None
        self.expected_input_columns = None
        self.expected_target_columns = []
        self.latest_tea_results = None
        
        self.message_count = 0
        self.last_checkpoint_message_count = 0
        
        if restore_checkpoint and self.checkpoint_path.exists():
            self._restore_checkpoint()
        
        # --- Init Sub-Agents ---
        print("🤖 Agent: Hiring sub-agents...")
        self.planner = PlanningAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            futurehouse_api_key=futurehouse_api_key,
            output_dir=str(self.base_dir)
        )
        self.scalarizer = ScalarizerAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            output_dir=str(self.base_dir / "scalarizer_outputs")
        )
        self.bo = BOAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            output_dir=str(self.base_dir / "bo_artifacts")
        )

        # --- Initialize Tools Registry ---
        self.tools = OrchestratorTools(self)
        
        # --- Get appropriate system prompt based on autonomy level ---
        system_prompt = get_system_prompt(self.autonomy_level)
        system_prompt += self._build_workspace_context()
        
        # --- LLM Initialization ---
        if base_url:
            logging.info(f"🏛️ Orchestrator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
            self.use_openai = True
            self.tools_for_model = self.tools.openai_schemas
        else:
            logging.info(f"🌐 Orchestrator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key,
                system_instruction=system_prompt,
                tools=self.tools.gemini_functions
            )
            self.use_openai = False
            self.tools_for_model = self.tools.gemini_functions
        
        # Store system prompt for OpenAI mode
        self._system_prompt = system_prompt
        
        # --- MEMORY INITIALIZATION ---
        history = self._load_history()
        
        if self.use_openai:
            self.messages = [{"role": "system", "content": system_prompt}]
            if history:
                recent_history = self._trim_history(history, max_messages=100)
                self.messages.extend(recent_history)
        else:
            self.chat_session = self.model.start_chat(
                history=history,
                enable_automatic_function_calling=True
            )

    def _build_workspace_context(self) -> str:
        """Build workspace context string for system prompt."""
        if self.autonomy_level == AutonomyLevel.CO_PILOT:
            return ""  # Not needed, human will guide
        
        context_parts = ["\n\n**WORKSPACE CONFIGURATION:**"]
        
        if self.data_dir:
            context_parts.append(f"- Data directory: {self.data_dir}")
        if self.knowledge_dir:
            context_parts.append(f"- Knowledge directory: {self.knowledge_dir}")
        if self.code_dir:
            context_parts.append(f"- Code directory: {self.code_dir}")
        
        context_parts.append("\nUse these paths directly without asking for confirmation.")
        
        return "\n".join(context_parts)
    
    def _should_enable_human_feedback(self) -> bool:
        """Determines if human feedback should be enabled based on autonomy level."""
        # CO_PILOT and SUPERVISED both keep human review of plans/code
        # Only AUTONOMOUS skips human feedback entirely
        return self.autonomy_level != AutonomyLevel.AUTONOMOUS

    def set_autonomy_level(self, level: AutonomyLevel) -> None:
        """
        Change the autonomy level at runtime.
        
        Args:
            level: New autonomy level to set.
        """
        old_level = self.autonomy_level
        self.autonomy_level = level
        self._enable_human_feedback = self._should_enable_human_feedback()
        
        # Update system prompt
        new_system_prompt = get_system_prompt(level)
        self._system_prompt = new_system_prompt
        
        if self.use_openai:
            # Update system message in OpenAI format
            if self.messages and self.messages[0]["role"] == "system":
                self.messages[0]["content"] = new_system_prompt
        
        logging.info(f"🔄 Autonomy level changed: {old_level.value} → {level.value}")
        logging.info(f"   Human feedback enabled: {self._enable_human_feedback}")

    def get_human_feedback_setting(self) -> bool:
        """Returns current human feedback setting for sub-agents."""
        return self._enable_human_feedback

    def _restore_checkpoint(self):
        """Restore campaign state from checkpoint."""
        print(f"  📂 Restoring checkpoint from: {self.checkpoint_path}")
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            self.active_scalarizer_script = state.get("active_scalarizer_script")
            self.expected_input_columns = state.get("expected_input_columns")

            if "expected_target_columns" in state:
                self.expected_target_columns = state.get("expected_target_columns")
            else:
                self.expected_target_columns = []

            self.latest_tea_results = state.get("latest_tea_results")
            
            # Restore autonomy level if saved
            if "autonomy_level" in state:
                try:
                    self.autonomy_level = AutonomyLevel(state["autonomy_level"])
                    self._enable_human_feedback = self._should_enable_human_feedback()
                except ValueError:
                    pass  # Keep default if invalid value

            if "data_dir" in state and state["data_dir"]:
                self.data_dir = Path(state["data_dir"])
            if "knowledge_dir" in state and state["knowledge_dir"]:
                self.knowledge_dir = Path(state["knowledge_dir"])
            if "code_dir" in state and state["code_dir"]:
                self.code_dir = Path(state["code_dir"])
            
            print(f"    ✅ Restored state:")
            print(f"       - Analysis script: {Path(self.active_scalarizer_script).name if self.active_scalarizer_script else 'None'}")
            print(f"       - Schema: {self.expected_input_columns} → {self.expected_target_columns}")
            print(f"       - Data points: {state.get('data_points_collected', 0)}")
            print(f"       - Autonomy level: {self.autonomy_level.value}")
            
        except Exception as e:
            logging.warning(f"Failed to restore checkpoint: {e}")

    def _trim_history(self, history: List[Dict], max_messages: int = 100) -> List[Dict]:
        """Keep only recent messages to avoid context window overflow."""
        if len(history) <= max_messages:
            return history
        
        print(f"  ⚠️  Trimming history: {len(history)} → {max_messages} messages")
        
        context_window = 10
        recent_window = max_messages - context_window
        
        trimmed = history[:context_window] + history[-recent_window:]
        
        summary_marker = {
            "role": "system",
            "content": f"[{len(history) - max_messages} messages omitted for context management]"
        }
        trimmed.insert(context_window, summary_marker)
        
        return trimmed

    def chat(self, user_input: str) -> str:
        """Main chat interface with robust function calling support."""
        self.message_count += 1
        
        # AUTO-CHECKPOINT: Every 10 messages
        if self.message_count - self.last_checkpoint_message_count >= 10:
            print("  💾 Auto-checkpoint triggered (every 10 messages)...")
            self._auto_checkpoint()
            self.last_checkpoint_message_count = self.message_count
        
        try:
            if self.use_openai:
                response_text = self._handle_openai_chat(user_input)
            else:
                response = self.chat_session.send_message(user_input)
                response_text = self._extract_response_text(response)
            
            print(f"🤖 Agent: {response_text}")
            self._save_history()
            
            if self.message_count > 80:
                warning = "\n\n⚠️ Note: Conversation is getting long. Consider calling save_checkpoint and restarting."
                response_text += warning
            
            return response_text
            
        except Exception as e:
            logging.error(f"Chat Error: {e}", exc_info=True)
            
            print("  💾 Error detected - saving emergency checkpoint...")
            self._auto_checkpoint()
            
            return f"❌ Error: {e}\n\n(Emergency checkpoint saved to {self.checkpoint_path})"

    def _auto_checkpoint(self):
        """Internal auto-checkpoint without LLM interaction."""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "objective": self.objective,
                "active_scalarizer_script": self.active_scalarizer_script,
                "expected_input_columns": self.expected_input_columns,
                "expected_target_columns": self.expected_target_columns,
                "data_points_collected": len(pd.read_csv(self.bo_data_path)) if self.bo_data_path.exists() else 0,
                "planner_state": self.planner.state,
                "message_count": self.message_count,
                "latest_tea_results": self.latest_tea_results,
                "autonomy_level": self.autonomy_level.value,
                "data_dir": str(self.data_dir) if self.data_dir else None,
                "knowledge_dir": str(self.knowledge_dir) if self.knowledge_dir else None,
                "code_dir": str(self.code_dir) if self.code_dir else None,
            }
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"    ✅ Auto-checkpoint saved")
            
        except Exception as e:
            logging.warning(f"Auto-checkpoint failed: {e}")

    def _handle_openai_chat(self, user_input: str) -> str:
        """Handle chat with OpenAI-compatible models with manual function calling loop."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url
        )
        
        self.messages.append({"role": "user", "content": user_input})
        
        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=100)
            self.messages = [system_msg] + recent_msgs
        
        max_iterations = 20
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
            
            if not message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content
            
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
                args = json.loads(tool_call.function.arguments)
                
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

    def _load_history(self) -> List[Dict]:
        """Load conversation history from disk."""
        if not self.history_path.exists(): 
            return []
        print("  🧠 Memory: Loading previous conversation...")
        try:
            with open(self.history_path, 'r') as f: 
                saved = json.load(f)
            
            if self.use_openai:
                return saved
            else:
                return [{"role": t["role"], "parts": [t["text"]]} for t in saved]
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")
            return []

    def _save_history(self):
        """Save conversation history to disk."""
        history_data = []
        
        try:
            if self.use_openai:
                history_data = [m for m in self.messages if m["role"] != "system"]
            else:
                if not hasattr(self.chat_session, 'history'):
                    return
                
                for content in self.chat_session.history:
                    role = content.role if hasattr(content, 'role') else content.get('role', 'user')
                    text_parts = []
                    if hasattr(content, 'parts'):
                        text_parts = [p.text for p in content.parts if hasattr(p, 'text') and p.text]
                    elif hasattr(content, 'content'):
                        text_parts = [content.content]
                    elif isinstance(content, dict) and 'content' in content:
                        text_parts = [content['content']]
                    
                    if text_parts: 
                        history_data.append({"role": role, "text": " ".join(text_parts)})
            
            with open(self.history_path, 'w') as f: 
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save history: {e}")

    @classmethod
    def restore_from_checkpoint(cls, base_dir: str, **kwargs):
        """Factory method to create an OrchestratorAgent from a checkpoint."""
        return cls(base_dir=base_dir, restore_checkpoint=True, **kwargs)
