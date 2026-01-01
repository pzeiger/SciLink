# interactive_orchestrator.py
"""
Interactive playground for the OrchestratorAgent.
Lets you chat with the agent and see what it does.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent as OrchestratorAgent


class OrchestratorPlayground:
    """Interactive session manager for the OrchestratorAgent."""
    
    def __init__(self):
        self.agent = None
        self.session_dir = None
        
    def setup(self):
        """Setup the agent with user configuration."""
        #print("\n" + "="*60)
        # Call the gradient printer
        self.print_gradient_logo()
        #print("="*60)
        
        # ALWAYS show directory guide first
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
        default_dir = f"./playground_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n📁 Where should I save session data?")
        session_dir = input(f"   Path (default: {default_dir}): ").strip()
        if not session_dir:
            session_dir = default_dir
        
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agent
        print("\n🔧 Initializing agent...")
        try:            
            self.agent = OrchestratorAgent(
                objective=objective,
                base_dir=str(self.session_dir),
                google_api_key=api_key
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
        """
        Handle special commands. Returns True if command was handled.
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
    

    def print_gradient_logo(self):
        """Prints the ASCII logo centered with a Blue-to-Green gradient."""
        # The raw logo lines (max width is approx 34 chars)
        logo_text = [
            "  ____       _ _     _       _    ",
            " / ___|  ___(_) |   (_)_ __ | | __",
            " \\___ \\ / __| | |   | | '_ \\| |/ /",
            "  ___) | (__| | |___| | | | |   < ",
            " |____/ \\___|_|_____|_|_| |_|_|\\_\\"
        ]

        # SciLink Colors: Blue (#4285F4) to Green (#34A853)
        start_rgb = (66, 133, 244)
        end_rgb = (52, 168, 83)

        # Center the logo within 60 characters
        # Logo width is ~34, so (60 - 34) / 2 = 13 spaces left padding
        term_width = 60
        logo_width = max(len(line) for line in logo_text)
        padding = " " * ((term_width - logo_width) // 2)

        for line in logo_text:
            colored_line = padding  # Start with plain padding
            length = len(line)
            
            for i, char in enumerate(line):
                # Calculate gradient ratio (0.0 to 1.0)
                # We use the character index 'i' to shift color across the word
                ratio = i / max(length - 1, 1)
                
                r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
                g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
                b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
                
                # Append colored character
                colored_line += f"\033[38;2;{r};{g};{b}m{char}"
            
            # Print the line and reset color
            print(colored_line + "\033[0m")
    
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
                # Get user input
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
                #print("\n🤖 Agent: ", end="", flush=True)
                response = self.agent.chat(user_input)
                #print(response)
                
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


def create_sample_data_files(session_dir: Path):
    """Helper to create sample data files for testing."""
    import pandas as pd
    
    data_dir = session_dir / "sample_data"
    data_dir.mkdir(exist_ok=True)
    
    print("\n📊 Creating sample data files...")
    
    # Create 3 sample CSV files
    for i in range(1, 4):
        data = {
            'time': [0, 1, 2, 3, 4, 5],
            'signal': [10, 20 + i*5, 40 + i*10, 30, 20, 10],
            'temperature': [50 + i*5] * 6,
            'pH': [6.0 + i*0.2] * 6
        }
        df = pd.DataFrame(data)
        
        filepath = data_dir / f"experiment_{i:03d}.csv"
        df.to_csv(filepath, index=False)
        print(f"  ✅ Created: {filepath.name}")
    
    print(f"\n💡 Try: 'Analyze the file {data_dir}/experiment_001.csv'")
    return data_dir


def quick_demo():
    """Run a quick guided demo."""
    print("\n" + "="*60)
    print("🎯 QUICK DEMO MODE")
    print("="*60)
    
    playground = OrchestratorPlayground()
    playground.setup()
    
    # Create sample data
    data_dir = create_sample_data_files(playground.session_dir)
    
    print("\n" + "="*60)
    print("📚 SUGGESTED COMMANDS TO TRY")
    print("="*60)
    print("\n1️⃣  List tools:")
    print("   /tools")
    
    print("\n2️⃣  Generate a plan:")
    print("   Generate an initial experimental plan")
    
    print("\n3️⃣  Analyze data:")
    print(f"   Analyze {data_dir}/experiment_001.csv and extract peak signal")
    
    print("\n4️⃣  Check status:")
    print("   /state")
    
    print("\n5️⃣  Run optimization (after 3 files):")
    print("   Run optimization")
    
    print("\n" + "="*60)
    print("Ready to start! Type /help for more commands.\n")
    
    # Start interactive session
    playground.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Orchestrator Agent Playground")
    parser.add_argument("--demo", action="store_true", help="Run quick demo with sample data")
    parser.add_argument("--create-data", action="store_true", help="Just create sample data files")
    
    args = parser.parse_args()
    
    if args.create_data:
        session_dir = Path("./sample_data_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
        session_dir.mkdir(exist_ok=True)
        create_sample_data_files(session_dir)
        print(f"\n✅ Sample data created in: {session_dir}")
        
    elif args.demo:
        quick_demo()
        
    else:
        playground = OrchestratorPlayground()
        playground.run()