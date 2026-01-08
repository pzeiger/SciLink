#!/usr/bin/env python3
"""
scilink plan - Interactive Planning Orchestrator
Experimental design, data analysis, and Bayesian optimization
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def main():
    """Main entry point for 'scilink plan' command"""
    
    parser = argparse.ArgumentParser(
        prog='scilink plan',
        description='SciLink Planning Orchestrator - Interactive AI Research Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings (co-pilot mode, Gemini model)
  scilink plan
  
  # Use supervised mode with explicit directories
  scilink plan --autonomy supervised --data-dir ./experimental_results
  
  # Full autonomous mode with all directories specified
  scilink plan --autonomy autonomous --data-dir ./data --knowledge-dir ./papers --code-dir ./code
  
  # Use a different model
  scilink plan --model gemini-2.0-flash
  
  # Use Claude
  scilink plan --model claude-sonnet-4-20250514
  
  # Use OpenAI
  scilink plan --model gpt-4o
  
  # Use internal proxy with custom endpoint
  scilink plan --base-url https://my-proxy.example.com/v1 --model my-model

Autonomy Levels:
  co-pilot     Human leads, AI assists. Reviews every step. (default)
  supervised   AI leads, human supervises. Human reviews plans/code only.
  autonomous   Full autonomy. No human review, AI chains all tools.

Environment Variables:
  SCILINK_API_KEY          API key for internal proxy
  GEMINI_API_KEY           Google Gemini API key
  GOOGLE_API_KEY           Google API key (alias for GEMINI_API_KEY)
  OPENAI_API_KEY           OpenAI API key
  ANTHROPIC_API_KEY        Anthropic API key
  CLAUDE_API_KEY           Anthropic API key (alias)
  FUTUREHOUSE_API_KEY      FutureHouse API key for literature search (optional)

Supported Models:
  Google:    gemini-3-pro-preview, gemini-2.0-flash, gemini-1.5-pro, etc.
  OpenAI:    gpt-4o, gpt-4-turbo, o1-preview, etc.
  Anthropic: claude-sonnet-4-20250514, claude-opus-4-20250514, etc.
        """
    )
    
    # Model and API arguments
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-3-pro-preview',
        help='Model name (default: gemini-3-pro-preview)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        dest='base_url',
        help='Base URL for OpenAI-compatible endpoint'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        dest='embedding_model',
        default='gemini-embedding-001',
        help='Embedding model name (default: gemini-embedding-001)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        dest='api_key',
        help='API key for LLM provider (overrides environment variables)'
    )
    
    # Autonomy arguments
    parser.add_argument(
        '--autonomy',
        type=str,
        choices=['co-pilot', 'supervised', 'autonomous'],
        default=None,  # None means "ask interactively"
        help='Autonomy level (default: interactive selection)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        dest='data_dir',
        help='Path to experimental data directory (required for supervised/autonomous)'
    )
    
    parser.add_argument(
        '--knowledge-dir',
        type=str,
        dest='knowledge_dir',
        help='Path to papers/literature directory (optional)'
    )
    
    parser.add_argument(
        '--code-dir',
        type=str,
        dest='code_dir',
        help='Path to code/API documentation directory (optional)'
    )
    
    # Deprecated arguments (hidden but functional)
    parser.add_argument(
        '--local-model',
        type=str,
        dest='local_model',
        help=argparse.SUPPRESS
    )
    
    parser.add_argument(
        '--google-api-key',
        type=str,
        dest='google_api_key',
        help=argparse.SUPPRESS
    )
    
    args = parser.parse_args()
    
    # Handle deprecated --local-model -> --base-url
    base_url = args.base_url
    if args.local_model:
        import warnings
        warnings.warn(
            "'--local-model' is deprecated. Use '--base-url' instead.",
            DeprecationWarning
        )
        print("⚠️  Warning: '--local-model' is deprecated. Use '--base-url' instead.")
        if not base_url:
            base_url = args.local_model
    
    # Handle deprecated --google-api-key -> --api-key
    api_key = args.api_key
    if args.google_api_key:
        import warnings
        warnings.warn(
            "'--google-api-key' is deprecated. Use '--api-key' instead.",
            DeprecationWarning
        )
        print("⚠️  Warning: '--google-api-key' is deprecated. Use '--api-key' instead.")
        if not api_key:
            api_key = args.google_api_key
    
    # Build config dict
    config = {
        'model_name': args.model,
        'base_url': base_url,
        'embedding_model': args.embedding_model,
        'api_key': api_key,
        'autonomy_level': args.autonomy,  # None if not specified
        'data_dir': args.data_dir,
        'knowledge_dir': args.knowledge_dir,
        'code_dir': args.code_dir,
    }
    
    # Run the interactive orchestrator
    try:
        playground = OrchestratorPlayground(config)
        playground.run()
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ==============================================================================
# Interactive Orchestrator
# ==============================================================================

class OrchestratorPlayground:
    """Interactive session manager for the Planning Orchestrator Agent."""
    
    def __init__(self, config: dict = None):
        self.agent = None
        self.session_dir = None
        self.config = config or {}
        
        # Will be set during setup
        self.data_dir = None
        self.knowledge_dir = None
        self.code_dir = None
        
    def _infer_provider(self, model_name: str) -> tuple:
        """
        Infer provider info from model name.
        
        Returns:
            (provider_name, env_var_hint, env_vars_to_check)
        """
        model_lower = model_name.lower()
        
        if 'claude' in model_lower:
            return (
                "Anthropic",
                "ANTHROPIC_API_KEY or CLAUDE_API_KEY",
                ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]
            )
        elif model_lower.startswith(('gpt-', 'o1-', 'o3-', 'text-embedding')):
            return (
                "OpenAI",
                "OPENAI_API_KEY",
                ["OPENAI_API_KEY"]
            )
        else:
            return (
                "Google Gemini",
                "GEMINI_API_KEY or GOOGLE_API_KEY",
                ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
            )
    
    def _get_api_key_from_env(self, env_vars: list) -> str:
        """Get API key from list of environment variables."""
        for var in env_vars:
            key = os.getenv(var)
            if key:
                return key
        return None
    
    def _auto_detect_directory(self, candidates: list, purpose: str) -> str:
        """
        Try to auto-detect a directory from a list of candidates.
        
        Returns:
            Path string if found and confirmed, None otherwise
        """
        for candidate in candidates:
            if Path(candidate).exists():
                confirm = input(f"   Found {purpose}: {candidate}. Use this? [Y/n]: ").strip().lower()
                if confirm != 'n':
                    return candidate
        return None
    
    def _prompt_for_directory(self, purpose: str, required: bool = False) -> str:
        """
        Prompt user for a directory path.
        
        Returns:
            Valid path string, or None if skipped/invalid
        """
        prompt = f"   Path to {purpose}"
        if required:
            prompt += " (required): "
        else:
            prompt += " (optional, Enter to skip): "
        
        path = input(prompt).strip()
        
        if not path:
            if required:
                return None  # Caller handles the error
            return None
        
        if not Path(path).exists():
            print(f"   ⚠️  Directory does not exist: {path}")
            if required:
                return None
            return None
        
        return path
    
    def _select_autonomy_level(self):
        """
        Interactive autonomy level selection.
        
        Returns:
            AutonomyLevel enum value
        """
        from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
        
        print("\n" + "="*60)
        print("🎛️  AUTONOMY LEVEL SELECTION")
        print("="*60)
        print("""
  1. Co-Pilot (default)
     Human leads, AI assists.
     - Reviews every plan and code before proceeding
     - One tool at a time, waits for your approval
     - Best for: learning, exploration, sensitive work

  2. Supervised
     AI leads, human supervises.
     - You still review generated plans and code
     - AI chains tools without asking permission
     - Pauses only on errors
     - Best for: routine analysis, batch processing
     - Requires: organized data directory

  3. Autonomous
     Full autonomy, no human review.
     - AI chains all tools independently
     - Only stops on unrecoverable errors
     - Best for: overnight runs, well-defined workflows
     - Requires: organized data directory
""")
        print("="*60)
        
        choice = input("\nSelect autonomy level [1/2/3] (default: 1): ").strip()
        
        if choice == '2':
            return AutonomyLevel.SUPERVISED
        elif choice == '3':
            return AutonomyLevel.AUTONOMOUS
        else:
            return AutonomyLevel.CO_PILOT
    
    def _configure_workspace_directories(self, autonomy_level) -> bool:
        """
        Configure workspace directories for higher autonomy modes.
        
        Returns:
            True if configuration successful, False if should fall back to CO_PILOT
        """
        from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
        
        print(f"\n" + "="*60)
        print(f"📂 WORKSPACE CONFIGURATION ({autonomy_level.value} mode)")
        print("="*60)
        print("Higher autonomy modes require organized directories so the")
        print("agent can find files without asking questions.\n")
        
        # Data directory (required for SUPERVISED/AUTONOMOUS)
        if not self.data_dir:
            print("📊 Data Directory (required)")
            self.data_dir = self._auto_detect_directory(
                ['./experimental_results', './data', './results'],
                'data directory'
            )
            
            if not self.data_dir:
                self.data_dir = self._prompt_for_directory(
                    'experimental data directory',
                    required=True
                )
            
            if not self.data_dir:
                print("\n❌ Data directory is required for higher autonomy modes.")
                print("   Falling back to co-pilot mode.\n")
                return False
        
        print(f"   ✅ Data directory: {self.data_dir}")
        
        # Knowledge directory (optional)
        if not self.knowledge_dir:
            print("\n📚 Knowledge Directory (optional)")
            self.knowledge_dir = self._auto_detect_directory(
                ['./papers', './docs', './literature', './reports'],
                'knowledge directory'
            )
            
            if not self.knowledge_dir:
                self.knowledge_dir = self._prompt_for_directory(
                    'papers/literature directory',
                    required=False
                )
        
        if self.knowledge_dir:
            print(f"   ✅ Knowledge directory: {self.knowledge_dir}")
        else:
            print("   ℹ️  Knowledge directory: not configured")
        
        # Code directory (optional)
        if not self.code_dir:
            print("\n💻 Code Directory (optional)")
            self.code_dir = self._auto_detect_directory(
                ['./code', './scripts', './opentrons_api', './automation'],
                'code directory'
            )
            
            if not self.code_dir:
                self.code_dir = self._prompt_for_directory(
                    'code/API documentation directory',
                    required=False
                )
        
        if self.code_dir:
            print(f"   ✅ Code directory: {self.code_dir}")
        else:
            print("   ℹ️  Code directory: not configured")
        
        print()
        return True
        
    def setup(self):
        """Setup the agent with user configuration."""
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent as OrchestratorAgent,
            AutonomyLevel
        )
        
        # Read from config (passed from CLI)
        model_name = self.config.get('model_name', 'gemini-3-pro-preview')
        base_url = self.config.get('base_url')
        embedding_model = self.config.get('embedding_model', 'gemini-embedding-001')
        api_key = self.config.get('api_key')
        autonomy_level_str = self.config.get('autonomy_level')  # None if not specified
        self.data_dir = self.config.get('data_dir')
        self.knowledge_dir = self.config.get('knowledge_dir')
        self.code_dir = self.config.get('code_dir')
        
        # Show directory guide
        print("\n" + "="*60)
        print("📁 RECOMMENDED DIRECTORY STRUCTURE")
        print("="*60)
        print("""
    Run orchestrator from your project directory:

    📁 my_project/
    ├── 📚 papers/               ← PDFs, scientific literature
    ├── 📊 experimental_results/ ← CSV/XLSX data files  
    └── 💻 code/                 ← (Optional) Scripts, API docs

    Then use natural language:
    "Generate a plan using ./papers/"
    "Analyze ./experimental_results/batch_001.csv"
    "Run optimization"
""")
        print("="*60)
        
        # === AUTONOMY LEVEL SELECTION ===
        if autonomy_level_str:
            # Specified via CLI
            autonomy_map = {
                'co-pilot': AutonomyLevel.CO_PILOT,
                'supervised': AutonomyLevel.SUPERVISED,
                'autonomous': AutonomyLevel.AUTONOMOUS,
            }
            autonomy_level = autonomy_map.get(autonomy_level_str, AutonomyLevel.CO_PILOT)
        else:
            # Interactive selection
            autonomy_level = self._select_autonomy_level()
        
        # === DIRECTORY CONFIGURATION FOR HIGHER AUTONOMY ===
        if autonomy_level in (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS):
            success = self._configure_workspace_directories(autonomy_level)
            if not success:
                autonomy_level = AutonomyLevel.CO_PILOT
                self.data_dir = None
                self.knowledge_dir = None
                self.code_dir = None
        
        # === API KEY RESOLUTION ===
        if not api_key:
            if base_url:
                # Internal proxy
                api_key = os.getenv("SCILINK_API_KEY")
                if not api_key:
                    print(f"\n⚠️  No SCILINK_API_KEY found in environment.")
                    print(f"   When using --base-url, set SCILINK_API_KEY for authentication.")
                    api_key = input(f"Enter your proxy API key (SCILINK_API_KEY): ").strip()
                    if not api_key:
                        print("❌ Cannot proceed without API key for internal proxy.")
                        sys.exit(1)
            else:
                # Public deployment - check provider-specific keys
                provider_name, env_var_hint, env_vars = self._infer_provider(model_name)
                api_key = self._get_api_key_from_env(env_vars)
                
                if not api_key:
                    print(f"\n⚠️  No {env_var_hint} found in environment.")
                    print(f"   LiteLLM will attempt to auto-detect credentials.")
                    user_key = input(f"Enter your {provider_name} API key (or Enter to auto-detect): ").strip()
                    if user_key:
                        api_key = user_key

        # === FUTUREHOUSE API KEY (Optional) ===
        futurehouse_key = os.getenv("FUTUREHOUSE_API_KEY")
        if not futurehouse_key:
            print("\n📚 FutureHouse API Key (Optional - for literature search)")
            print("   Enables scientific literature queries if you have a key.")
            futurehouse_key = input("   FutureHouse API key (or Enter to skip): ").strip()
            if futurehouse_key:
                print("   ✅ Literature search enabled")
            else:
                futurehouse_key = None
                print("   ℹ️  Literature search disabled")
        
        # === RESEARCH OBJECTIVE ===
        print("\n📋 What's your research objective?")
        print("Examples:")
        print("  - Optimize reaction yield")
        print("  - Screen drug candidates")
        print("  - Find optimal fermentation conditions")
        
        objective = input("\nYour objective: ").strip()
        if not objective:
            objective = "Optimize experimental conditions"
            print(f"   Using default: {objective}")
        
        # === SESSION DIRECTORY ===
        default_dir = f"./campaign_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n📁 Where should I save session data?")
        session_dir = input(f"   Path (default: {default_dir}): ").strip()
        if not session_dir:
            session_dir = default_dir
        
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # === INITIALIZE AGENT ===
        print("\n🔧 Initializing agent...")
        try:
            self.agent = OrchestratorAgent(
                objective=objective,
                base_dir=str(self.session_dir),
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                embedding_model=embedding_model,
                futurehouse_api_key=futurehouse_key,
                autonomy_level=autonomy_level,
                data_dir=self.data_dir,
                knowledge_dir=self.knowledge_dir,
                code_dir=self.code_dir,
            )
            print("✅ Agent ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 Troubleshooting:")
            print("   1. Check that planning_agents package is installed")
            print("   2. Verify all dependencies are installed")
            print("   3. Check your API key is valid")
            print("   4. For supervised/autonomous: ensure data directory exists")
            sys.exit(1)
        
        # === SHOW SESSION INFO ===
        print("\n" + "="*60)
        print("SESSION INFO")
        print("="*60)
        print(f"Objective: {objective}")
        print(f"Session Directory: {self.session_dir}")
        print(f"Autonomy Level: {autonomy_level.value}")
        print(f"Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
        
        # Model info
        provider_name, _, _ = self._infer_provider(model_name)
        if base_url:
            print(f"Model: {model_name}")
            print(f"Endpoint: {base_url}")
        else:
            print(f"Model: {model_name} ({provider_name})")
        
        print(f"Embedding Model: {embedding_model}")
        print(f"Literature Search: {'Enabled' if futurehouse_key else 'Disabled'}")
        
        # Directory info for higher autonomy
        if autonomy_level in (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS):
            print(f"\nWorkspace Directories:")
            print(f"  Data: {self.data_dir}")
            print(f"  Knowledge: {self.knowledge_dir or 'not configured'}")
            print(f"  Code: {self.code_dir or 'not configured'}")
        
        # Tools info
        print(f"\nAvailable Tools: {len(self.agent.tools.functions_map)}")
        tool_names = list(self.agent.tools.functions_map.keys())
        if len(tool_names) > 5:
            print(f"  {', '.join(tool_names[:5])}...")
        else:
            print(f"  {', '.join(tool_names)}")
        
    def print_help(self):
        """Print available commands."""
        print("\n" + "="*60)
        print("AVAILABLE COMMANDS")
        print("="*60)
        print("  /help              Show this help message")
        print("  /tools             List available tools")
        print("  /files             List files in workspace")
        print("  /state             Show agent state")
        print("  /autonomy [level]  Show or change autonomy level")
        print("  /checkpoint        Save checkpoint")
        print("  /clear             Clear screen")
        print("  /quit or /exit     Exit playground")
        print("\nOr just chat naturally with the agent!")
        print("="*60)
    
    def handle_command(self, user_input: str) -> bool:
        """
        Handle special commands.
        
        Returns:
            True if command was handled
            'QUIT' to exit
            False if not a command
        """
        cmd = user_input.lower().strip()
        
        if cmd == "/help":
            self.print_help()
            return True
        
        elif cmd == "/tools":
            print("\n📦 Available Tools:")
            for i, tool_name in enumerate(self.agent.tools.functions_map.keys(), 1):
                print(f"  {i}. {tool_name}")
            return True
        
        elif cmd == "/files":
            print("\n📁 Workspace Files:")
            files = list(self.session_dir.iterdir())
            if files:
                for f in sorted(files):
                    size = f.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    print(f"  - {f.name} ({size_str})")
            else:
                print("  (empty)")
            return True
        
        elif cmd == "/state":
            print("\n🔍 Agent State:")
            print(f"  Objective: {self.agent.objective}")
            print(f"  Autonomy Level: {self.agent.autonomy_level.value}")
            print(f"  Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Active Script: {Path(self.agent.active_scalarizer_script).name if self.agent.active_scalarizer_script else 'None'}")
            print(f"  Input Columns: {self.agent.expected_input_columns}")
            print(f"  Target Columns: {self.agent.expected_target_columns}")
            
            # Workspace directories
            if self.data_dir:
                print(f"  Data Directory: {self.data_dir}")
            if self.knowledge_dir:
                print(f"  Knowledge Directory: {self.knowledge_dir}")
            if self.code_dir:
                print(f"  Code Directory: {self.code_dir}")
            
            # Check data points
            if self.agent.bo_data_path.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(self.agent.bo_data_path)
                    print(f"  Data Points: {len(df)}")
                except Exception:
                    print(f"  Data Points: Error reading file")
            else:
                print(f"  Data Points: 0")
            return True
        
        elif cmd.startswith("/autonomy"):
            from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
            
            parts = cmd.split()
            if len(parts) == 1:
                # Show current level
                print(f"\n🎛️  Current Autonomy Level: {self.agent.autonomy_level.value}")
                print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                print("\n   To change: /autonomy <co-pilot|supervised|autonomous>")
            else:
                # Change level
                level_map = {
                    'co-pilot': AutonomyLevel.CO_PILOT,
                    'copilot': AutonomyLevel.CO_PILOT,
                    'supervised': AutonomyLevel.SUPERVISED,
                    'autonomous': AutonomyLevel.AUTONOMOUS,
                }
                new_level = level_map.get(parts[1].lower())
                
                if new_level:
                    # Warn if switching to higher autonomy without directories
                    if new_level in (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS):
                        if not self.data_dir:
                            print(f"\n   ⚠️  Warning: No data directory configured.")
                            print(f"   Higher autonomy modes work best with organized directories.")
                            print(f"   Agent may need to ask for file locations.")
                    
                    self.agent.set_autonomy_level(new_level)
                    print(f"\n   ✅ Autonomy level changed to: {new_level.value}")
                    print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                else:
                    print(f"\n   ❌ Unknown level: {parts[1]}")
                    print("   Valid options: co-pilot, supervised, autonomous")
            return True
        
        elif cmd == "/checkpoint":
            print("\n💾 Saving checkpoint...")
            response = self.agent.chat("Save checkpoint")
            print(response)
            return True
        
        elif cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🤖 Orchestrator Agent - Session Resumed\n")
            return True
        
        elif cmd in ["/quit", "/exit", "/q"]:
            return "QUIT"
        
        return False
    
    def run(self):
        """Main interactive loop."""
        self.setup()
        self.print_help()
        
        print("\n" + "="*60)
        print("💬 CHAT SESSION STARTED")
        print("="*60)
        print("Type /help for commands, or just chat naturally!\n")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                result = self.handle_command(user_input)
                if result == "QUIT":
                    print("\n👋 Goodbye! Your session is saved at:")
                    print(f"   {self.session_dir}")
                    break
                elif result:
                    continue
                
                # Send to agent
                response = self.agent.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type /quit to exit properly.")
            except EOFError:
                print("\n\n👋 Session ended.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print("   Type /help for commands or /quit to exit")


if __name__ == '__main__':
    sys.exit(main())