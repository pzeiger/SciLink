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
  # Start with default Gemini model (uses $GEMINI_API_KEY or $GOOGLE_API_KEY)
  scilink plan
  
  # Use a different model
  scilink plan --model gemini-2.0-flash
  
  # Use Claude
  scilink plan --model claude-sonnet-4-20250514
  
  # Use OpenAI
  scilink plan --model gpt-4o
  
  # Use internal proxy with custom endpoint
  scilink plan --base-url https://my-proxy.example.com/v1 --model my-model

  # Mix providers (Claude for LLM, Gemini for embeddings)
  scilink plan --model claude-sonnet-4-20250514 --embedding-model gemini-embedding-001

Environment Variables:
SCILINK_API_KEY          API key for internal proxy
  GEMINI_API_KEY         Google Gemini API key
  GOOGLE_API_KEY         Google API key (alias for GEMINI_API_KEY)
  OPENAI_API_KEY         OpenAI API key
  ANTHROPIC_API_KEY      Anthropic API key
  CLAUDE_API_KEY         Anthropic API key (alias for ANTHROPIC_API_KEY)
  FUTUREHOUSE_API_KEY    FutureHouse API key for literature search (optional)

Supported Models:
  Google:    gemini-3-pro-preview, gemini-2.0-flash, gemini-1.5-pro, etc.
  OpenAI:    gpt-4o, gpt-4-turbo, o1-preview, etc.
  Anthropic: claude-sonnet-4-20250514, claude-opus-4-20250514, etc.

Note: Model prefixes (gemini/, openai/, anthropic/) are added automatically.
        """
    )
    
    # Current arguments
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-3-pro-preview',
        help='Model name (default: gemini-3-pro-preview). Prefix auto-detected.'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        dest='base_url',
        help='Base URL for OpenAI-compatible endpoint (e.g., https://my-proxy.com/v1)'
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
    
    # Deprecated arguments (hidden but functional for backward compatibility)
    parser.add_argument(
        '--local-model',
        type=str,
        dest='local_model',
        help=argparse.SUPPRESS  # Hidden from help
    )
    
    parser.add_argument(
        '--google-api-key',
        type=str,
        dest='google_api_key',
        help=argparse.SUPPRESS  # Hidden from help
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
# Interactive Orchestrator (Integrated)
# ==============================================================================

class OrchestratorPlayground:
    """Interactive session manager for the Planning Orchestrator Agent."""
    
    def __init__(self, config: dict = None):
        self.agent = None
        self.session_dir = None
        self.config = config or {}
        
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
            # Default to Gemini
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
        
    def setup(self):
        """Setup the agent with user configuration."""
        
        # Read from config (passed from CLI)
        model_name = self.config.get('model_name', 'gemini-3-pro-preview')
        base_url = self.config.get('base_url')
        embedding_model = self.config.get('embedding_model', 'gemini-embedding-001')
        api_key = self.config.get('api_key')
        
        # Show directory guide
        print("\n" + "="*60)
        print("📁 RECOMMENDED DIRECTORY STRUCTURE")
        print("="*60)
        print("""
    Run orchestrator from your project directory with:

    📁 my_project/
    ├── 📚 papers/              ← PDFs, scientific literature
    ├── 📊 experimental_results/ ← CSV/XLSX data files  
    └── 💻 code/                ← (Optional) Scripts, API docs

    Then use natural language:
    "Generate a plan using ./papers/"
    "Analyze ./experimental_results/batch_001.csv"
    "Run optimization"
    """)
        print("="*60)
        
        # Resolve API key if not provided via CLI
        if not api_key:
            # If using internal proxy (base_url), check for SCILINK_API_KEY first
            if base_url:
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
                    # For LiteLLM without base_url, check if env vars are set
                    # LiteLLM will auto-detect, but we should warn if nothing is set
                    print(f"\n⚠️  No {env_var_hint} found in environment.")
                    print(f"   LiteLLM will attempt to auto-detect credentials.")
                    user_key = input(f"Enter your {provider_name} API key (or press Enter to let LiteLLM auto-detect): ").strip()
                    if user_key:
                        api_key = user_key

        # Get FutureHouse API key (optional)
        futurehouse_key = os.getenv("FUTUREHOUSE_API_KEY")
        if not futurehouse_key:
            print("\n📚 FutureHouse API Key (Optional - for literature search)")
            print("   If you have one, enter it to enable scientific literature queries.")
            futurehouse_key = input("   FutureHouse API key (or press Enter to skip): ").strip()
            if futurehouse_key:
                print("   ✅ Literature search will be enabled")
            else:
                futurehouse_key = None
                print("   ℹ️  Literature search will be skipped")
        
        # Get objective
        print("\n📋 What's your research objective?")
        print("Examples:")
        print("  - Optimize reaction yield")
        print("  - Screen drug candidates")
        print("  - Find optimal fermentation conditions")
        
        objective = input("\nYour objective: ").strip()
        if not objective:
            objective = "Optimize experimental conditions"
            print(f"   Using default: {objective}")
        
        # Get session directory
        default_dir = f"./campaign_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n📁 Where should I save session data?")
        session_dir = input(f"   Path (default: {default_dir}): ").strip()
        if not session_dir:
            session_dir = default_dir
        
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agent
        print("\n🔧 Initializing agent...")
        try:
            from scilink.agents.planning_agents.planning_orchestrator import (
                PlanningOrchestratorAgent as OrchestratorAgent
            )
            
            self.agent = OrchestratorAgent(
                objective=objective,
                base_dir=str(self.session_dir),
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                embedding_model=embedding_model,
                futurehouse_api_key=futurehouse_key,
            )
            print("✅ Agent ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 Troubleshooting:")
            print("   1. Make sure you're in the correct directory")
            print("   2. Check that planning_agents package is installed/accessible")
            print("   3. Verify all dependencies are installed")
            print("   4. Check your API key is valid")
            sys.exit(1)
        
        # Show session info
        print("\n" + "="*60)
        print("SESSION INFO")
        print("="*60)
        print(f"Objective: {objective}")
        print(f"Session Directory: {self.session_dir}")
        
        # Show model configuration
        provider_name, _, _ = self._infer_provider(model_name)
        if base_url:
            print(f"Model: {model_name}")
            print(f"Endpoint: {base_url}")
            print(f"Auth: SCILINK_API_KEY")
        else:
            print(f"Model: {model_name} ({provider_name})")
        
        print(f"Embedding Model: {embedding_model}")
        print(f"Literature Search: {'Enabled' if futurehouse_key else 'Disabled'}")
        print(f"Available Tools: {len(self.agent.tools.functions_map)}")
        tool_names = list(self.agent.tools.functions_map.keys())
        if len(tool_names) > 5:
            print(f"  - {', '.join(tool_names[:5])}...")
        else:
            print(f"  - {', '.join(tool_names)}")
        
    def print_help(self):
        """Print available commands."""
        print("\n" + "="*60)
        print("AVAILABLE COMMANDS")
        print("="*60)
        print("  /help              Show this help message")
        print("  /tools             List available tools")
        print("  /files             List files in workspace")
        print("  /state             Show agent state")
        print("  /checkpoint        Save checkpoint")
        print("  /clear             Clear screen")
        print("  /quit or /exit     Exit playground")
        print("\nOr just chat naturally with the agent!")
        print("="*60)
    
    def handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled, 'QUIT' to exit."""
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
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Active Script: {Path(self.agent.active_scalarizer_script).name if self.agent.active_scalarizer_script else 'None'}")
            print(f"  Input Columns: {self.agent.expected_input_columns}")
            print(f"  Target Columns: {self.agent.expected_target_columns}")
            
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
