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
  # Start with default Gemini model (uses $GOOGLE_API_KEY)
  scilink plan
  
  # Use a different Gemini model
  scilink plan --model gemini-2.0-flash-exp
  
  # Use local/OpenAI-compatible model
  scilink plan --local-model http://localhost:8000/v1 --model llama-3

Environment Variables:
  GOOGLE_API_KEY         Google Gemini API key (required for Gemini models)
  FUTUREHOUSE_API_KEY    FutureHouse API key for literature search (optional)

Recommended Directory Structure:
  📁 my_project/
  ├── 📚 papers/              ← Scientific literature (PDFs)
  ├── 📊 experimental_results/ ← Data files (CSV/XLSX)
  └── 💻 code/                ← API docs, scripts (optional)
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-3-pro-preview',
        help='Model name to use (default: gemini-3-pro-preview)'
    )
    
    parser.add_argument(
        '--local-model',
        type=str,
        help='Base URL for OpenAI-compatible local model (e.g., http://localhost:8000/v1)'
    )
    
    args = parser.parse_args()
    
    # Set model configuration in environment
    if args.model:
        os.environ['SCILINK_MODEL_NAME'] = args.model
    if args.local_model:
        os.environ['SCILINK_LOCAL_MODEL'] = args.local_model
    
    # Run the interactive orchestrator
    try:
        playground = OrchestratorPlayground()
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
    
    def __init__(self):
        self.agent = None
        self.session_dir = None
        
    def setup(self):
        """Setup the agent with user configuration."""
        
        # Read model configuration from environment (set by CLI)
        model_name = os.getenv('SCILINK_MODEL_NAME', 'gemini-3-pro-preview')
        local_model = os.getenv('SCILINK_LOCAL_MODEL', None)
        
        # Logo already shown by main CLI, just show directory guide
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
        
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("\n⚠️  No GOOGLE_API_KEY found in environment.")
            api_key = input("Enter your Google API key (or press Enter to skip): ").strip()
            if not api_key:
                print("❌ Cannot proceed without API key.")
                sys.exit(1)

        # Get FutureHouse API key (optional)
        futurehouse_key = os.getenv("FUTUREHOUSE_API_KEY")
        if not futurehouse_key:
            print("\n📚 FutureHouse API Key (Optional - for literature search)")
            print("   If you have one, enter it to enable scientific literature queries.")
            futurehouse_key = input("   FutureHouse API key (or press Enter to skip): ").strip()
            if futurehouse_key:
                print("   ✅ Literature search will be enabled")
            else:
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
                google_api_key=api_key,
                futurehouse_api_key=futurehouse_key if futurehouse_key else None,
                model_name=model_name,
                local_model=local_model
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
            sys.exit(1)
        
        # Show session info
        print("\n" + "="*60)
        print("SESSION INFO")
        print("="*60)
        print(f"Objective: {objective}")
        print(f"Session Directory: {self.session_dir}")
        
        # Show model configuration
        if local_model:
            print(f"Model: {model_name} (Local: {local_model})")
        else:
            print(f"Model: {model_name} (Gemini)")
        
        print(f"Available Tools: {len(self.agent.tools.functions_map)}")
        print(f"  - {', '.join(list(self.agent.tools.functions_map.keys())[:5])}...")
        
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
        """Handle special commands. Returns True if command was handled."""
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
                for f in files:
                    print(f"  - {f.name}")
            else:
                print("  (empty)")
            return True
        
        elif cmd == "/state":
            print("\n🔍 Agent State:")
            print(f"  Objective: {self.agent.objective}")
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Active Script: {Path(self.agent.active_scalarizer_script).name if self.agent.active_scalarizer_script else 'None'}")
            print(f"  Input Columns: {self.agent.expected_input_columns}")
            print(f"  Target Column: {self.agent.expected_target_column}")
            
            # Check data points
            if self.agent.bo_data_path.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(self.agent.bo_data_path)
                    print(f"  Data Points: {len(df)}")
                except:
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
        
        elif cmd in ["/quit", "/exit"]:
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