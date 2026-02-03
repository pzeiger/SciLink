#!/usr/bin/env python3
"""
scilink analyze - Interactive Analysis Orchestrator
Experimental data analysis with automatic agent selection
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def main():
    """Main entry point for 'scilink analyze' command"""
    
    parser = argparse.ArgumentParser(
        prog='scilink analyze',
        description='SciLink Analysis Orchestrator - Interactive AI Analysis Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings (co-pilot mode)
  scilink analyze
  
  # Start with data file
  scilink analyze --data ./sample.tif --metadata ./metadata.json
  
  # Use supervised mode (AI leads, human approves)
  scilink analyze --mode supervised --data ./data/
  
  # Full autonomous mode
  scilink analyze --mode autonomous --data ./sample.npy --metadata ./description.txt
  
  # Use a different model
  scilink analyze --model gemini-2.0-flash
  
  # Use Claude
  scilink analyze --model claude-sonnet-4-20250514
  
  # Use OpenAI
  scilink analyze --model gpt-4o

Analysis Modes (matching 'scilink plan' for consistent UX):
  co-pilot (default)   Human leads, AI assists. Reviews all agent selections.
  supervised           AI leads, human approves. AI proceeds with reasonable defaults.
  autonomous           Full autonomy. AI selects agents and runs without confirmation.

Environment Variables:
  SCILINK_API_KEY          API key for internal proxy
  GEMINI_API_KEY           Google Gemini API key
  GOOGLE_API_KEY           Google API key (alias for GEMINI_API_KEY)
  OPENAI_API_KEY           OpenAI API key
  ANTHROPIC_API_KEY        Anthropic API key
  CLAUDE_API_KEY           Anthropic API key (alias)

Supported Data Types:
  Microscopy:    .tif, .tiff, .png, .jpg, .jpeg, .bmp
  Spectroscopy:  .npy (3D hyperspectral)
  Curves:        .npy (1D/2D), .csv, .txt

Metadata Options:
  --metadata file.json     Load structured JSON metadata
  --metadata file.txt      Convert natural language to metadata
  (or provide metadata interactively during session)
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
        '--api-key',
        type=str,
        dest='api_key',
        help='API key for LLM provider (overrides environment variables)'
    )
    
    # Mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['co-pilot', 'supervised', 'autonomous'],
        default='co-pilot',
        help='Analysis mode (default: co-pilot)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        dest='data_path',
        help='Path to data file or directory'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        dest='metadata_path',
        help='Path to metadata file (.json or .txt)'
    )
    
    # Session arguments
    parser.add_argument(
        '--session-dir',
        type=str,
        dest='session_dir',
        help='Session directory for outputs (default: auto-generated)'
    )
    
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore from previous checkpoint in session directory'
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
    
    # Handle deprecated arguments
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
    
    # Validate data path if provided
    if args.data_path and not Path(args.data_path).exists():
        parser.error(f"--data path does not exist: {args.data_path}")
    
    # Validate metadata path if provided
    if args.metadata_path and not Path(args.metadata_path).exists():
        parser.error(f"--metadata path does not exist: {args.metadata_path}")
    
    # Build config dict
    config = {
        'model_name': args.model,
        'base_url': base_url,
        'api_key': api_key,
        'analysis_mode': args.mode,
        'data_path': args.data_path,
        'metadata_path': args.metadata_path,
        'session_dir': args.session_dir,
        'restore': args.restore,
    }
    
    # Run the interactive orchestrator
    try:
        playground = AnalysisPlayground(config)
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

class AnalysisPlayground:
    """Interactive session manager for the Analysis Orchestrator Agent."""
    
    def __init__(self, config: dict = None):
        self.agent = None
        self.session_dir = None
        self.config = config or {}
        
    def _infer_provider(self, model_name: str) -> tuple:
        """Infer provider info from model name."""
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
        
    def setup(self):
        """Setup the agent with user configuration."""
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent,
            AnalysisMode
        )
        
        # Read from config
        model_name = self.config.get('model_name', 'gemini-3-pro-preview')
        base_url = self.config.get('base_url')
        api_key = self.config.get('api_key')
        analysis_mode_str = self.config.get('analysis_mode', 'interactive')
        data_path = self.config.get('data_path')
        metadata_path = self.config.get('metadata_path')
        session_dir = self.config.get('session_dir')
        restore = self.config.get('restore', False)
        
        # Convert mode string to enum
        mode_map = {
            'co-pilot': AnalysisMode.CO_PILOT,
            'supervised': AnalysisMode.SUPERVISED,
            'autonomous': AnalysisMode.AUTONOMOUS,
        }
        analysis_mode = mode_map.get(analysis_mode_str, AnalysisMode.CO_PILOT)
        
        # Show welcome message
        print("\n" + "="*60)
        print("🔬 SCILINK ANALYSIS ORCHESTRATOR")
        print("="*60)
        print("""
This agent helps you analyze experimental data by:
1. Examining your data to determine the appropriate analysis
2. Managing metadata (loading or converting from text)
3. Selecting the best analysis agent for your data
4. Running analysis and generating scientific insights

Supported data types:
  • Microscopy images (.tif, .png, .jpg)
  • Spectroscopy data (.npy - hyperspectral)
  • 1D curves/spectra (.npy, .csv, .txt)
""")
        print("="*60)
        
        # === API KEY RESOLUTION ===
        if not api_key:
            if base_url:
                api_key = os.getenv("SCILINK_API_KEY")
                if not api_key:
                    print(f"\n⚠️  No SCILINK_API_KEY found in environment.")
                    api_key = input(f"Enter your proxy API key (SCILINK_API_KEY): ").strip()
                    if not api_key:
                        print("❌ Cannot proceed without API key for internal proxy.")
                        sys.exit(1)
            else:
                provider_name, env_var_hint, env_vars = self._infer_provider(model_name)
                api_key = self._get_api_key_from_env(env_vars)
                
                if not api_key:
                    print(f"\n⚠️  No {env_var_hint} found in environment.")
                    user_key = input(f"Enter your {provider_name} API key (or Enter to auto-detect): ").strip()
                    if user_key:
                        api_key = user_key
        
        # === SESSION DIRECTORY ===
        if session_dir:
            self.session_dir = Path(session_dir)
        elif restore:
            # Look for existing session
            default_pattern = "./analysis_session_*"
            import glob
            sessions = sorted(glob.glob(default_pattern), reverse=True)
            if sessions:
                self.session_dir = Path(sessions[0])
                print(f"\n📂 Found session to restore: {self.session_dir}")
            else:
                print("\n⚠️  No existing sessions found. Creating new session.")
                self.session_dir = Path(f"./analysis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            default_dir = f"./analysis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n📁 Where should I save session data?")
            session_input = input(f"   Path (default: {default_dir}): ").strip()
            self.session_dir = Path(session_input) if session_input else Path(default_dir)
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # === INITIALIZE AGENT ===
        print("\n🔧 Initializing agent...")
        try:
            self.agent = AnalysisOrchestratorAgent(
                base_dir=str(self.session_dir),
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                analysis_mode=analysis_mode,
                restore_checkpoint=restore,
            )
            print("✅ Agent ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # === SHOW SESSION INFO ===
        print("\n" + "="*60)
        print("SESSION INFO")
        print("="*60)
        print(f"Session Directory: {self.session_dir}")
        print(f"Analysis Mode: {analysis_mode.value}")
        print(f"Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
        
        provider_name, _, _ = self._infer_provider(model_name)
        if base_url:
            print(f"Model: {model_name}")
            print(f"Endpoint: {base_url}")
        else:
            print(f"Model: {model_name} ({provider_name})")
        
        print(f"\nAvailable Tools: {len(self.agent.tools.functions_map)}")
        tool_names = list(self.agent.tools.functions_map.keys())
        print(f"  {', '.join(tool_names[:5])}...")
        
        # === HANDLE INITIAL DATA/METADATA ===
        if data_path:
            print(f"\n📊 Initial data: {data_path}")
            self.agent.current_data_path = str(Path(data_path).absolute())
        
        if metadata_path:
            print(f"📋 Initial metadata: {metadata_path}")
            # Will be processed during first chat interaction
            self._initial_metadata_path = metadata_path
        else:
            self._initial_metadata_path = None
        
    def print_help(self):
        """Print available commands."""
        print("\n" + "="*60)
        print("AVAILABLE COMMANDS")
        print("="*60)
        print("  /help              Show this help message")
        print("  /tools             List available tools")
        print("  /agents            List available analysis agents")
        print("  /status            Show current session state")
        print("  /mode [level]      Show or change analysis mode")
        print("  /checkpoint        Save checkpoint")
        print("  /schema            Show metadata schema")
        print("  /clear             Clear screen")
        print("  /quit or /exit     Exit")
        print("\nOr just chat naturally with the agent!")
        print("="*60)
    
    def handle_command(self, user_input: str) -> bool:
        """Handle special commands."""
        cmd = user_input.lower().strip()
        
        if cmd == "/help":
            self.print_help()
            return True
        
        elif cmd == "/tools":
            print("\n📦 Available Tools:")
            for i, tool_name in enumerate(self.agent.tools.functions_map.keys(), 1):
                print(f"  {i}. {tool_name}")
            return True
        
        elif cmd == "/agents":
            print("\n🤖 Available Analysis Agents:")
            for agent_id, name in self.agent.tools.AGENT_NAMES.items():
                desc = self.agent.tools.AGENT_DESCRIPTIONS[agent_id]
                selected = " ← selected" if agent_id == self.agent.selected_agent_id else ""
                print(f"  {agent_id}: {name}")
                print(f"     {desc}{selected}")
            return True
        
        elif cmd == "/status":
            print("\n🔍 Session State:")
            print(f"  Session Directory: {self.session_dir}")
            print(f"  Analysis Mode: {self.agent.analysis_mode.value}")
            print(f"  Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Current Data: {self.agent.current_data_path or 'None'}")
            print(f"  Current Data Type: {self.agent.current_data_type or 'None'}")
            print(f"  Selected Agent: {self.agent.selected_agent_id}")
            print(f"  Has Metadata: {'Yes' if self.agent.current_metadata else 'No'}")
            print(f"  Analyses Completed: {len(self.agent.analysis_results)}")
            return True
        
        elif cmd.startswith("/mode"):
            from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
            
            parts = cmd.split()
            if len(parts) == 1:
                print(f"\n🎛️  Current Analysis Mode: {self.agent.analysis_mode.value}")
                print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                print("\n   To change: /mode <co-pilot|supervised|autonomous>")
            else:
                mode_map = {
                    'co-pilot': AnalysisMode.CO_PILOT,
                    'supervised': AnalysisMode.SUPERVISED,
                    'autonomous': AnalysisMode.AUTONOMOUS,
                }
                new_mode = mode_map.get(parts[1].lower())
                
                if new_mode:
                    self.agent.set_analysis_mode(new_mode)
                    print(f"\n   ✅ Analysis mode changed to: {new_mode.value}")
                    print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                else:
                    print(f"\n   ❌ Unknown mode: {parts[1]}")
                    print("   Valid options: co-pilot, supervised, autonomous")
            return True
        
        elif cmd == "/checkpoint":
            print("\n💾 Saving checkpoint...")
            response = self.agent.chat("Save checkpoint")
            print(response)
            return True
        
        elif cmd == "/schema":
            from scilink.agents.exp_agents.metadata_converter import METADATA_SCHEMA_DICT
            import json
            print("\n📋 Metadata Schema:")
            print(json.dumps(METADATA_SCHEMA_DICT, indent=2))
            return True
        
        elif cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🔬 Analysis Orchestrator - Session Resumed\n")
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
        
        # Handle initial metadata if provided
        if hasattr(self, '_initial_metadata_path') and self._initial_metadata_path:
            metadata_path = self._initial_metadata_path
            ext = Path(metadata_path).suffix.lower()
            
            if ext == '.json':
                print(f"📋 Loading metadata from {metadata_path}...")
                response = self.agent.chat(f"Load the metadata from {metadata_path}")
            else:
                print(f"📋 Converting metadata from {metadata_path}...")
                response = self.agent.chat(f"Convert the metadata from {metadata_path}")
        
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