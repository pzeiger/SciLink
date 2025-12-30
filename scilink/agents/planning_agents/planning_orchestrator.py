import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from .planning_agent import PlanningAgent
from .scalarizer_agent import ScalarizerAgent
from .bo_agent import BOAgent
from .orchestrator_tools import OrchestratorTools

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the **Autonomous Research Agent**. Your goal is to coordinate a scientific campaign.

**SETUP & ORGANIZATION:**
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


class PlanningOrchestratorAgent:
    def __init__(self, 
                 objective: str = "Undefined Research Goal",
                 base_dir: str = "./campaign_outputs",
                 google_api_key: str = None, 
                 futurehouse_api_key: str = None,
                 model_name: str = "gemini-3-pro-preview",
                 local_model: str = None,
                 restore_checkpoint: bool = False):
        
        self.objective = objective
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.analyzed_files_path = self.base_dir / "analyzed_files.json"
        self.analyzed_files = {}  # {file_path: row_count}
        
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
        
        # STATE: Tracks the currently approved analysis script AND expected schema
        self.active_scalarizer_script = None
        self.expected_input_columns = None
        self.expected_target_column = None

        # TEA results for auto-context
        self.latest_tea_results = None
        
        # CAMPAIGN METRICS
        self.message_count = 0
        self.last_checkpoint_message_count = 0
        
        # --- Restore from Checkpoint (if requested) ---
        if restore_checkpoint and self.checkpoint_path.exists():
            self._restore_checkpoint()
        
        # --- Init Sub-Agents ---
        print("🤖 Agent: Hiring sub-agents...")
        self.planner = PlanningAgent(
            google_api_key=google_api_key, 
            futurehouse_api_key=futurehouse_api_key,
            model_name=model_name, 
            local_model=local_model,
            output_dir=str(self.base_dir)
        )
        self.scalarizer = ScalarizerAgent(
            google_api_key=google_api_key, 
            model_name=model_name, 
            local_model=local_model,
            output_dir=str(self.base_dir / "scalarizer_outputs")
        )
        self.bo = BOAgent(
            google_api_key=google_api_key, 
            model_name=model_name, 
            local_model=local_model,
            output_dir=str(self.base_dir / "bo_artifacts")
        )

        # --- Initialize Tools Registry ---
        self.tools = OrchestratorTools(self)
        
        # --- Auth & Model Initialization ---
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            logging.info(f"🏛️  Orchestrator using OpenAI-compatible model: {model_name}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name, 
                api_key=google_api_key, 
                base_url=local_model
            )
            self.use_openai = True
            self.tools_for_model = self.tools.openai_schemas
        else:
            logging.info(f"☁️  Orchestrator using Google Gemini model: {model_name}")
            if google_api_key:
                genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                tools=self.tools.gemini_functions,
                system_instruction=ORCHESTRATOR_SYSTEM_PROMPT
            )
            self.use_openai = False
            self.tools_for_model = self.tools.gemini_functions
        
        # --- MEMORY INITIALIZATION ---
        history = self._load_history()
        
        # Start Chat Session
        if self.use_openai:
            self.messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]
            if history:
                # CONTEXT WINDOW MANAGEMENT: Keep only recent history
                recent_history = self._trim_history(history, max_messages=100)
                self.messages.extend(recent_history)
        else:
            self.chat_session = self.model.start_chat(
                history=history,
                enable_automatic_function_calling=True
            )

    def _restore_checkpoint(self):
        """Restore campaign state from checkpoint."""
        print(f"  📂 Restoring checkpoint from: {self.checkpoint_path}")
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            self.active_scalarizer_script = state.get("active_scalarizer_script")
            self.expected_input_columns = state.get("expected_input_columns")
            self.expected_target_column = state.get("expected_target_column")
            self.latest_tea_results = state.get("latest_tea_results")  # ← ADD THIS LINE
            
            print(f"    ✅ Restored state:")
            print(f"       - Analysis script: {Path(self.active_scalarizer_script).name if self.active_scalarizer_script else 'None'}")
            print(f"       - Schema: {self.expected_input_columns} → {self.expected_target_column}")
            print(f"       - Data points: {state.get('data_points_collected', 0)}")
            print(f"       - TEA results: {'Available' if self.latest_tea_results else 'None'}")  # ← ADD THIS LINE
            
        except Exception as e:
            logging.warning(f"Failed to restore checkpoint: {e}")

    def _trim_history(self, history: List[Dict], max_messages: int = 100) -> List[Dict]:
        """
        Keep only recent messages to avoid context window overflow.
        Uses a sliding window approach with summary preservation.
        """
        if len(history) <= max_messages:
            return history
        
        print(f"  ⚠️  Trimming history: {len(history)} → {max_messages} messages")
        
        # Strategy: Keep first 10 messages (context) + last N messages (recent)
        context_window = 10
        recent_window = max_messages - context_window
        
        trimmed = history[:context_window] + history[-recent_window:]
        
        # Insert summary marker
        summary_marker = {
            "role": "system",
            "content": f"[{len(history) - max_messages} messages omitted for context management]"
        }
        trimmed.insert(context_window, summary_marker)
        
        return trimmed

    def chat(self, user_input: str) -> str:
        """Main chat interface with robust function calling support."""
        #print(f"\n👤 User: {user_input}")
        
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
            
            # CONTEXT WARNING: If conversation is getting too long
            if self.message_count > 80:
                warning = "\n\n⚠️ Note: Conversation is getting long. Consider calling save_checkpoint and restarting to avoid context overflow."
                response_text += warning
            
            return response_text
            
        except Exception as e:
            logging.error(f"Chat Error: {e}", exc_info=True)
            
            # AUTO-SAVE on error
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
                "expected_target_column": self.expected_target_column,
                "data_points_collected": len(pd.read_csv(self.bo_data_path)) if self.bo_data_path.exists() else 0,
                "planner_state": self.planner.state,
                "message_count": self.message_count,
                "latest_tea_results": self.latest_tea_results
            }
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"    ✅ Auto-checkpoint saved")
            
        except Exception as e:
            logging.warning(f"Auto-checkpoint failed: {e}")

    def _handle_openai_chat(self, user_input: str) -> str:
        """
        Handle chat with OpenAI-compatible models with manual function calling loop.
        Includes context window management.
        """
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url
        )
        
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # CONTEXT WINDOW MANAGEMENT: Trim if getting too long
        if len(self.messages) > 120:  # System + 100 history + 20 current
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=100)
            self.messages = [system_msg] + recent_msgs
        
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call LLM
            response = client.chat.completions.create(
                model=self.model.model,
                messages=self.messages,
                tools=self.tools_for_model,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Check if tool calls are needed
            if not message.tool_calls:
                # No more tool calls, return final response
                self.messages.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content
            
            # Add assistant message with tool calls
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
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                print(f"  🔧 Calling tool: {func_name}")
                
                # Execute tool
                result = self.tools.execute_tool(func_name, **args)
                
                # Add tool result to messages
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

    # --- MEMORY MANAGEMENT ---
    def _load_history(self) -> List[Dict]:
        if not self.history_path.exists(): 
            return []
        print("  🧠 Memory: Loading previous conversation...")
        try:
            with open(self.history_path, 'r') as f: 
                saved = json.load(f)
            
            if self.use_openai:
                return saved  # OpenAI format
            else:
                # Gemini format
                return [{"role": t["role"], "parts": [t["text"]]} for t in saved]
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")
            return []

    def _save_history(self):
        """Save conversation history to disk."""
        history_data = []
        
        try:
            if self.use_openai:
                # Save OpenAI messages directly (excluding system message)
                history_data = [m for m in self.messages if m["role"] != "system"]
            else:
                # Gemini format
                if not hasattr(self.chat_session, 'history'):
                    return
                
                for content in self.chat_session.history:
                    role = content.role
                    text_parts = []
                    if hasattr(content, 'parts'):
                        text_parts = [p.text for p in content.parts if hasattr(p, 'text') and p.text]
                    elif hasattr(content, 'content'):
                         text_parts = [content.content]
                    
                    if text_parts: 
                        history_data.append({"role": role, "text": " ".join(text_parts)})
            
            with open(self.history_path, 'w') as f: 
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save history: {e}")

    @classmethod
    def restore_from_checkpoint(cls, base_dir: str, **kwargs):
        """
        Factory method to create an OrchestratorAgent from a checkpoint.
        
        Usage:
            agent = OrchestratorAgent.restore_from_checkpoint(
                base_dir="./campaign_outputs",
                google_api_key="..."
            )
        """
        return cls(base_dir=base_dir, restore_checkpoint=True, **kwargs)