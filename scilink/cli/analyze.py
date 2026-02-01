#!/usr/bin/env python3
"""
scilink analyze - Interactive Experimental Data Analysis CLI

Simple unified interface for analyzing experimental data (microscopy, spectroscopy, curves).
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime


def _infer_provider(model_name: str) -> tuple:
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


def _get_api_key_from_env(env_vars: list) -> str:
    """Get API key from list of environment variables."""
    for var in env_vars:
        key = os.getenv(var)
        if key:
            return key
    return None


def main():
    """Main entry point for 'scilink analyze' command."""
    
    parser = argparse.ArgumentParser(
        prog='scilink analyze',
        description='SciLink Experimental Analysis - Unified Data Analysis Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with JSON metadata
  scilink analyze image.tif --metadata metadata.json
  
  # Analyze with text metadata (will be converted)
  scilink analyze spectrum.csv --metadata-text "Raman spectrum of TiO2, 532nm excitation"
  
  # Analyze with explicit agent selection
  scilink analyze particles.png --metadata meta.json --agent sam_microscopy
  
  # Interactive mode (prompts for metadata)
  scilink analyze data.npy --interactive
  
  # Specify analysis goal
  scilink analyze sample.tif --metadata meta.json --goal "Count nanoparticles and measure size distribution"

Available Agents:
  fft_microscopy   FFT/NMF-based microscopy analysis
  sam_microscopy   Segment Anything for particle/object detection
  hyperspectral    Hyperspectral/spectroscopic unmixing
  curve_fitting    1D curve fitting (Raman, XRD, PL, etc.)

Metadata Requirements:
  Metadata can be provided as:
  - JSON file (--metadata path/to/metadata.json)
  - Text file (--metadata path/to/description.txt) - will be converted
  - Inline text (--metadata-text "description...")
  - Interactive input (--interactive)

Environment Variables:
  SCILINK_API_KEY     API key for internal proxy
  GEMINI_API_KEY      Google Gemini API key
  OPENAI_API_KEY      OpenAI API key
  ANTHROPIC_API_KEY   Anthropic API key
        """
    )
    
    # Data path - optional (can be provided in interactive chat)
    parser.add_argument(
        'data_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to the data file to analyze (optional - can provide in chat)'
    )
    
    # Metadata options (mutually exclusive group)
    metadata_group = parser.add_mutually_exclusive_group()
    metadata_group.add_argument(
        '--metadata', '-m',
        type=str,
        dest='metadata_path',
        help='Path to metadata file (JSON or text)'
    )
    metadata_group.add_argument(
        '--metadata-text', '-t',
        type=str,
        dest='metadata_text',
        help='Inline metadata description (natural language)'
    )
    metadata_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode - prompt for metadata'
    )
    
    # Optional arguments
    parser.add_argument(
        '--agent', '-a',
        type=str,
        choices=['fft_microscopy', 'sam_microscopy', 'hyperspectral', 'curve_fitting'],
        help='Force specific agent (skip auto-selection)'
    )
    
    parser.add_argument(
        '--goal', '-g',
        type=str,
        help='Specific analysis goal'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./analysis_outputs',
        help='Output directory (default: ./analysis_outputs)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-3-pro-preview',
        help='LLM model name (default: gemini-3-pro-preview)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        help='Base URL for OpenAI-compatible endpoint'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (overrides environment variables)'
    )
    
    parser.add_argument(
        '--no-llm-selection',
        action='store_true',
        help='Use heuristics only for agent selection (no LLM)'
    )
    
    parser.add_argument(
        '--human-feedback',
        action='store_true',
        help='Enable human-in-the-loop feedback'
    )
    
    parser.add_argument(
        '--list-agents',
        action='store_true',
        help='List available agents and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import orchestrator
    try:
        from scilink.agents.exp_agents.experimental_orchestrator import (
            ExperimentalAnalysisOrchestrator,
            AGENT_REGISTRY
        )
    except ImportError:
        # Try relative import for development
        try:
            from experimental_orchestrator import (
                ExperimentalAnalysisOrchestrator,
                AGENT_REGISTRY
            )
        except ImportError:
            print("❌ Error: Could not import ExperimentalAnalysisOrchestrator")
            print("   Make sure scilink is installed or run from the correct directory")
            return 1
    
    # Handle --list-agents
    if args.list_agents:
        print("\n" + "="*60)
        print("AVAILABLE ANALYSIS AGENTS")
        print("="*60)
        for agent_type, info in AGENT_REGISTRY.items():
            print(f"\n📊 {agent_type.value}")
            print(f"   {info['description']}")
            print(f"   Data types: {', '.join(info['data_types'])}")
            print(f"   Extensions: {', '.join(info['file_extensions'])}")
        print()
        return 0
    
    # Validate data path if provided
    if args.data_path and not Path(args.data_path).exists():
        print(f"❌ Error: Data file not found: {args.data_path}")
        return 1
    
    # Handle metadata
    metadata_path = args.metadata_path
    metadata_text = args.metadata_text
    
    # If --interactive flag is set, we'll let the orchestrator handle it
    # by not providing any metadata (it will prompt)
    if args.interactive:
        metadata_path = None
        metadata_text = None
    
    # === API KEY RESOLUTION ===
    api_key = args.api_key
    model_name = args.model
    base_url = args.base_url
    
    if not api_key:
        if base_url:
            # Internal proxy - check for SCILINK_API_KEY
            api_key = os.getenv("SCILINK_API_KEY")
            if not api_key:
                print(f"\n⚠️  No SCILINK_API_KEY found in environment.")
                print(f"   When using --base-url, set SCILINK_API_KEY for authentication.")
                api_key = input(f"Enter your proxy API key (SCILINK_API_KEY): ").strip()
                if not api_key:
                    print("❌ Cannot proceed without API key for internal proxy.")
                    return 1
        else:
            # Public deployment - check provider-specific keys
            provider_name, env_var_hint, env_vars = _infer_provider(model_name)
            api_key = _get_api_key_from_env(env_vars)
            
            if not api_key:
                print(f"\n⚠️  No {env_var_hint} found in environment.")
                print(f"   LiteLLM will attempt to auto-detect credentials.")
                user_key = input(f"Enter your {provider_name} API key (or Enter to auto-detect): ").strip()
                if user_key:
                    api_key = user_key
    
    # Create orchestrator
    print("\n" + "="*60)
    print("🔬 SCILINK EXPERIMENTAL ANALYSIS")
    print("="*60)
    if args.data_path:
        print(f"\nData: {args.data_path}")
    else:
        print(f"\nData: (provide in chat)")
    print(f"Output: {args.output_dir}")
    
    # Show model info
    provider_name, _, _ = _infer_provider(model_name)
    if base_url:
        print(f"Model: {model_name}")
        print(f"Endpoint: {base_url}")
    else:
        print(f"Model: {model_name} ({provider_name})")
    
    if args.agent:
        print(f"Agent: {args.agent} (user-specified)")
    print()
    
    try:
        orchestrator = ExperimentalAnalysisOrchestrator(
            api_key=api_key,
            model_name=args.model,
            base_url=args.base_url,
            output_dir=args.output_dir,
            enable_human_feedback=args.human_feedback,
        )
    except Exception as e:
        print(f"❌ Error initializing orchestrator: {e}")
        return 1
    
    # Run analysis
    try:
        # Enter interactive mode if:
        # - No data_path provided, OR
        # - --interactive flag set, OR  
        # - No metadata provided (need to get it in chat)
        use_interactive = (
            args.data_path is None or 
            args.interactive or 
            (not metadata_path and not metadata_text)
        )
        
        if use_interactive:
            # Interactive chat mode
            print("\n📝 Starting interactive chat session...")
            print("   (Provide your data file and experiment description in the chat)\n")
            
            # If data path was provided, seed the conversation
            if args.data_path:
                initial_message = f"I have a data file at: {args.data_path}"
                if args.goal:
                    initial_message += f". My analysis goal is: {args.goal}"
                print(f"You: {initial_message}\n")
                print("🤔 Processing...\n")
                response = orchestrator.chat(initial_message)
                print(f"Assistant: {response}\n")
            
            # Continue with interactive session
            orchestrator.start_chat_session()
            result = {"status": "session_ended", "output_directory": str(orchestrator.output_dir)}
        else:
            # Single-shot analysis mode (data_path AND metadata provided)
            result = orchestrator.analyze(
                data_path=args.data_path,
                metadata_path=metadata_path,
                metadata_text=metadata_text,
                analysis_goal=args.goal,
                agent_type=args.agent,
            )
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
    
    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    status = result.get("status", "unknown")
    if status == "session_ended":
        print(f"\n✅ Interactive session completed")
        print(f"\n📁 Output Directory: {result.get('output_directory', 'N/A')}")
        
    elif status == "success" or status == "completed":
        print(f"\n✅ Status: SUCCESS")
        
        # Show key results
        if "detailed_analysis" in result:
            print(f"\n📋 Analysis Summary:")
            analysis = result["detailed_analysis"]
            if len(analysis) > 500:
                print(f"   {analysis[:500]}...")
            else:
                print(f"   {analysis}")
        
        if "scientific_claims" in result:
            claims = result["scientific_claims"]
            print(f"\n🎯 Scientific Claims ({len(claims)} generated):")
            for i, claim in enumerate(claims[:3], 1):
                print(f"   {i}. {claim.get('claim', 'N/A')[:100]}...")
        
        if "orchestrator_info" in result:
            info = result["orchestrator_info"]
            print(f"\n🤖 Agent Used: {info.get('agent_type', 'unknown')}")
            print(f"   Selection: {info.get('selection_info', {}).get('reasoning', 'N/A')[:80]}...")
        
        print(f"\n📁 Output Directory: {result.get('output_directory', 'N/A')}")
        
    elif status == "error":
        print(f"\n❌ Status: ERROR")
        error = result.get("error", {})
        print(f"   Error: {error.get('error', 'Unknown error')}")
        if "details" in error:
            print(f"   Details: {error['details']}")
    else:
        print(f"\n⚠️  Status: {status}")
    
    # Save full results
    results_path = Path(args.output_dir) / "analysis_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n💾 Full results saved to: {results_path}")
    
    print("\n" + "="*60 + "\n")
    
    return 0 if status == "success" else 1


def collect_interactive_metadata(data_path: str) -> str:
    """Collect metadata interactively from user."""
    print("\n" + "="*60)
    print("📝 METADATA COLLECTION")
    print("="*60)
    print(f"\nData file: {data_path}")
    print("\nPlease provide information about your experiment.")
    print("(Press Enter to skip optional fields)\n")
    
    try:
        # Experiment type
        print("Experiment types: Microscopy, Spectroscopy, Diffraction, Curve Analysis")
        exp_type = input("Experiment type: ").strip()
        if not exp_type:
            exp_type = "Unknown"
        
        # Technique
        print("\nExamples: STEM, TEM, SEM, AFM, Raman, XRD, PL, EELS")
        technique = input("Technique: ").strip()
        
        # Material
        material = input("\nMaterial/Sample (e.g., TiO2, MoS2): ").strip()
        if not material:
            material = "Unknown"
        
        # Additional description
        description = input("\nAdditional description (optional): ").strip()
        
        # Build metadata text
        parts = [f"Experiment type: {exp_type}"]
        if technique:
            parts.append(f"Technique: {technique}")
        parts.append(f"Material: {material}")
        if description:
            parts.append(f"Description: {description}")
        
        metadata_text = ". ".join(parts)
        
        print(f"\n✅ Metadata collected:")
        print(f"   {metadata_text}")
        
        confirm = input("\nProceed with this metadata? [Y/n]: ").strip().lower()
        if confirm in ['n', 'no']:
            return None
        
        return metadata_text
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled")
        return None
    except EOFError:
        print("\n\n⚠️  Input ended")
        return None


if __name__ == '__main__':
    sys.exit(main())