#!/usr/bin/env python3
"""
scilink meta - Interactive Meta-Agent Orchestrator

The meta-agent is SciLink's default entry point (bare `scilink`): a single
conversational agent that auto-routes each request to the right specialist —
the analysis orchestrator (experimental data) or the planning orchestrator
(campaign design / Bayesian optimization) — and bridges findings between them.

Each delegation runs as a nested child sub-session under the meta session
directory. Simulation delegation is not available in this build.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# ==============================================================================
# CLI entry point
# ==============================================================================

def main():
    """Main entry point for the 'scilink meta' command (and bare 'scilink')."""

    parser = argparse.ArgumentParser(
        prog='scilink meta',
        description='SciLink Meta-Agent - one chat surface that auto-routes '
                    'to the analyze and plan specialists',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch the meta-agent (identical to running bare `scilink`)
  scilink meta

  # Seed the first turn with your research goal
  scilink meta --message "Analyze grains.tif then plan a follow-up campaign"

  # Autopilot mode (meta delegates on its own judgement; reports as it goes)
  scilink meta --mode autopilot

  # Use a different model / an internal proxy
  scilink meta --model gemini-2.0-flash
  scilink meta --base-url https://my-proxy.example.com/v1 --model my-model

Modes — the meta has two levels (a delegation runs a specialist through its
one-shot run_task, so the specialists' step-by-step co-pilot mode does not
apply here):
  autopilot (default)  Specialists pause at decision points for you to
                        approve / edit plans and outputs.
  autonomous            Specialists run end to end without pausing.

Environment Variables:
  SCILINK_API_KEY          API key for internal proxy
  GEMINI_API_KEY           Google Gemini API key
  GOOGLE_API_KEY           Google API key (alias for GEMINI_API_KEY)
  OPENAI_API_KEY           OpenAI API key
  ANTHROPIC_API_KEY        Anthropic API key
  CLAUDE_API_KEY           Anthropic API key (alias)
  FUTUREHOUSE_API_KEY      FutureHouse API key (enables literature search)
        """
    )

    # Model / API
    parser.add_argument('--model', type=str, default='claude-opus-4-6',
                        help='Model name (default: claude-opus-4-6)')
    parser.add_argument('--base-url', type=str, dest='base_url',
                        help='Base URL for OpenAI-compatible endpoint')
    parser.add_argument('--api-key', type=str, dest='api_key',
                        help='API key for LLM provider (overrides environment variables)')
    parser.add_argument('--embedding-model', type=str, dest='embedding_model',
                        default='gemini-embedding-001',
                        help='Embedding model for the child orchestrators '
                             '(default: gemini-embedding-001)')
    parser.add_argument('--embedding-api-key', type=str, dest='embedding_api_key',
                        help='API key for the embedding provider')
    parser.add_argument('--futurehouse-api-key', type=str, dest='futurehouse_api_key',
                        help='FutureHouse API key (or set FUTUREHOUSE_API_KEY env var). '
                             'Enables literature search in delegated analysis.')

    # Mode
    parser.add_argument('--mode', type=str, dest='mode',
                        choices=['autopilot', 'autonomous'],
                        default='autopilot',
                        help='Autonomy mode (default: autopilot). The meta '
                             'has two levels, not the modes’ three.')

    # Initial message — seeds the first chat turn
    parser.add_argument('--message', type=str, dest='initial_message',
                        help='Optional initial message to seed the session. '
                             'If omitted, starts at an empty chat prompt.')

    # Session
    parser.add_argument('--session-dir', type=str, dest='session_dir',
                        help='Session directory for outputs (default: auto-generated)')
    parser.add_argument('--restore', action='store_true',
                        help='Restore from previous checkpoint in session directory')

    args = parser.parse_args()

    config = {
        'model_name': args.model,
        'base_url': args.base_url,
        'api_key': args.api_key,
        'embedding_model': args.embedding_model,
        'embedding_api_key': args.embedding_api_key,
        'futurehouse_api_key': args.futurehouse_api_key,
        'meta_mode': args.mode,
        'initial_message': args.initial_message,
        'session_dir': args.session_dir,
        'restore': args.restore,
    }

    try:
        playground = MetaPlayground(config)
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

class MetaPlayground:
    """Interactive session manager for the Meta-Agent Orchestrator."""

    def __init__(self, config: dict = None):
        self.agent = None
        self.session_dir = None
        self.config = config or {}

    def _infer_provider(self, model_name: str) -> tuple:
        """Infer provider info from model name."""
        m = model_name.lower()
        if 'claude' in m:
            return ("Anthropic", "ANTHROPIC_API_KEY or CLAUDE_API_KEY",
                    ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"])
        if m.startswith(('gpt-', 'o1-', 'o3-', 'text-embedding')):
            return ("OpenAI", "OPENAI_API_KEY", ["OPENAI_API_KEY"])
        return ("Google Gemini", "GEMINI_API_KEY or GOOGLE_API_KEY",
                ["GEMINI_API_KEY", "GOOGLE_API_KEY"])

    def _api_key_from_env(self, env_vars: list) -> str:
        for var in env_vars:
            v = os.getenv(var)
            if v:
                return v
        return None

    def setup(self):
        """Resolve config and build the meta-agent."""
        from scilink.agents.meta_agent.meta_orchestrator import (
            MetaOrchestratorAgent, MetaMode,
        )

        model_name = self.config.get('model_name', 'claude-opus-4-6')
        base_url = self.config.get('base_url')
        api_key = self.config.get('api_key')
        embedding_model = self.config.get('embedding_model', 'gemini-embedding-001')
        embedding_api_key = self.config.get('embedding_api_key')
        futurehouse_api_key = self.config.get('futurehouse_api_key')
        mode_str = self.config.get('meta_mode', 'autopilot')
        session_dir = self.config.get('session_dir')
        restore = self.config.get('restore', False)

        self._initial_message = self.config.get('initial_message')

        mode_map = {
            'autopilot': MetaMode.AUTOPILOT,
            'autonomous': MetaMode.AUTONOMOUS,
        }
        meta_mode = mode_map.get(mode_str, MetaMode.AUTOPILOT)

        print("\n" + "=" * 60)
        print("🧭  SCILINK META-AGENT")
        print("=" * 60)
        print("""
One chat surface that coordinates SciLink's specialists for you:
  • Analysis  — interpret experimental data (microscopy, spectroscopy, ...)
  • Planning  — design experimental campaigns / Bayesian optimization

Describe your research goal and the meta-agent routes the work to the
right specialist, bridging findings between them. Each delegation runs as
a nested child sub-session under this meta session.
""")
        print("=" * 60)

        # ── API key resolution ───────────────────────────────────────
        if not api_key:
            if base_url:
                api_key = os.getenv("SCILINK_API_KEY")
                if not api_key:
                    print("\n⚠️  No SCILINK_API_KEY found in environment.")
                    api_key = input("Enter your proxy API key (SCILINK_API_KEY): ").strip()
                    if not api_key:
                        print("❌ Cannot proceed without API key for internal proxy.")
                        sys.exit(1)
            else:
                provider_name, env_var_hint, env_vars = self._infer_provider(model_name)
                api_key = self._api_key_from_env(env_vars)
                if not api_key:
                    print(f"\n⚠️  No {env_var_hint} found in environment.")
                    user_key = input(f"Enter your {provider_name} API key (or Enter to skip): ").strip()
                    if user_key:
                        api_key = user_key

        if not futurehouse_api_key:
            futurehouse_api_key = os.getenv("FUTUREHOUSE_API_KEY")

        # ── Session directory ────────────────────────────────────────
        if session_dir:
            self.session_dir = Path(session_dir)
        elif restore:
            import glob
            sessions = sorted(glob.glob("./meta_session_*"), reverse=True)
            if sessions:
                self.session_dir = Path(sessions[0])
                print(f"\n📂 Found session to restore: {self.session_dir}")
            else:
                print("\n⚠️  No existing meta sessions found. Creating new session.")
                self.session_dir = Path(f"./meta_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            default_dir = f"./meta_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n📁 Where should I save session data?")
            session_input = input(f"   Path (default: {default_dir}): ").strip()
            self.session_dir = Path(session_input) if session_input else Path(default_dir)

        self.session_dir.mkdir(parents=True, exist_ok=True)

        # ── Initialize agent ────────────────────────────────────────
        print("\n🔧 Initializing meta-agent...")
        try:
            self.agent = MetaOrchestratorAgent(
                base_dir=str(self.session_dir),
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                futurehouse_api_key=futurehouse_api_key,
                restore_checkpoint=restore,
                meta_mode=meta_mode,
            )
            print("✅ Meta-agent ready!")
        except Exception as e:
            print(f"❌ Failed to initialize meta-agent: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # ── Session info ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SESSION INFO")
        print("=" * 60)
        print(f"Session Directory: {self.session_dir}")
        print(f"Meta Mode: {meta_mode.value}")
        print(f"Human Feedback: {'Enabled' if self.agent.get_human_feedback_setting() else 'Disabled'}")
        print(f"Literature search (in delegated analysis): "
              f"{'Enabled' if futurehouse_api_key else 'Disabled'}")

        provider_name, _, _ = self._infer_provider(model_name)
        if base_url:
            print(f"Model: {model_name}")
            print(f"Endpoint: {base_url}")
        else:
            print(f"Model: {model_name} ({provider_name})")

        print(f"\nDelegation Tools: {', '.join(self.agent.tools.functions_map.keys())}")

    def print_help(self):
        print("\n" + "=" * 60)
        print("AVAILABLE COMMANDS")
        print("=" * 60)
        print("  /help              Show this help message")
        print("  /tools             List the meta-agent's delegation tools")
        print("  /children          List delegations made this session")
        print("  /status            Show current session state")
        print("  /mode [level]      Show or change meta autonomy mode")
        print("  /clear             Clear screen")
        print("  /quit or /exit     Exit")
        print("\nOr just describe your research goal and let the meta-agent route it!")
        print("=" * 60)

    def handle_command(self, user_input: str):
        """Returns True if handled, 'QUIT' to exit, False otherwise."""
        cmd = user_input.lower().strip()

        if cmd == "/help":
            self.print_help()
            return True

        if cmd == "/tools":
            print("\n📦 Delegation Tools:")
            for i, tool_name in enumerate(self.agent.tools.functions_map.keys(), 1):
                print(f"  {i}. {tool_name}")
            return True

        if cmd == "/children":
            ledger = self.agent._delegation_ledger or []
            print(f"\n🧩 Delegations this session: {len(ledger)}")
            for entry in ledger:
                task = (entry.get('task') or '')[:70]
                summary = (entry.get('summary') or '')[:70]
                print(f"  [{entry.get('index')}] {entry.get('mode')} "
                      f"— status: {entry.get('status')}")
                print(f"      task:    {task}")
                if summary:
                    print(f"      result:  {summary}")
            return True

        if cmd == "/status":
            children = self.agent._children or {}
            print("\n🔍 Session State:")
            print(f"  Session Directory: {self.session_dir}")
            print(f"  Meta Mode: {self.agent.meta_mode.value}")
            print(f"  Human Feedback: {'Enabled' if self.agent.get_human_feedback_setting() else 'Disabled'}")
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Specialists active: {', '.join(sorted(children.keys())) or 'none'}")
            print(f"  Delegations: {len(self.agent._delegation_ledger or [])}")
            return True

        if cmd.startswith("/mode"):
            from scilink.agents.meta_agent.meta_orchestrator import MetaMode
            parts = cmd.split()
            if len(parts) == 1:
                print(f"\n🎛️  Current Meta Mode: {self.agent.meta_mode.value}")
                print("\n   To change: /mode <autopilot|autonomous>")
            else:
                mode_map = {
                    'autopilot': MetaMode.AUTOPILOT,
                    'autonomous': MetaMode.AUTONOMOUS,
                }
                new_mode = mode_map.get(parts[1].lower())
                if new_mode:
                    self.agent.set_meta_mode(new_mode)
                    print(f"\n   ✅ Meta mode changed to: {new_mode.value}")
                else:
                    print(f"\n   ❌ Unknown mode: {parts[1]}")
                    print("   Valid options: autopilot, autonomous")
            return True

        if cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🧭  Meta-Agent - Session Resumed\n")
            return True

        if cmd in ("/quit", "/exit", "/q", "quit", "exit"):
            return "QUIT"

        return False

    def run(self):
        """Main interactive loop."""
        self.setup()
        self.print_help()

        print("\n" + "=" * 60)
        print("💬 CHAT SESSION STARTED")
        print("=" * 60)
        print("Type /help for commands, or just describe your research goal!\n")

        # Process initial message if provided
        if self._initial_message:
            print(f"📝 Initial message: {self._initial_message}\n")
            try:
                response = self.agent.chat(self._initial_message)
                print(f"\n🤖 Meta-Agent: {response}\n")
            except Exception as e:
                print(f"❌ Error processing initial message: {e}")

        # Main chat loop
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

            if not user_input:
                continue

            handled = self.handle_command(user_input)
            if handled == "QUIT":
                print("\n👋 Goodbye!")
                break
            if handled is True:
                continue

            try:
                response = self.agent.chat(user_input)
                print(f"\n🤖 Meta-Agent: {response}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    sys.exit(main())
