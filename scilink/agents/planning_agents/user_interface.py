from typing import Dict, Any, Optional, List
import re


def format_caveats(findings: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Concise, advisory caveat lines from critic findings.

    Assumes ``findings`` is already ordered critical-first (the agent sorts it).
    Minor findings get a ``Minor:`` prefix; each line is ``[dimension] issue``.
    Returns ``[]`` when there are no findings — the single source rendered by
    both the console summary and ``run_task`` warnings.
    """
    lines: List[str] = []
    for f in findings or []:
        dim = f.get("dimension", "")
        issue = (f.get("issue") or "").strip()
        if not issue:
            continue
        prefix = "Minor: " if f.get("severity") == "minor" else ""
        lines.append(f"{prefix}[{dim}] {issue}")
    return lines


def display_plan_summary(result: Dict[str, Any]) -> None:
    """
    Parses the agent's results and prints a structured, pretty-printed 
    summary to the console for human review.
    """
    # 1. Error Handling
    if result.get("error"):
        print(f"\n❌ Agent finished with an error: {result['error']}\n")
        return

    # 2. Structure Validation
    experiments = result.get("proposed_experiments")
    if not experiments or not isinstance(experiments, list):
        print("\n⚠️  The agent returned a result, but no experiments were found.")
        # Optional: Print raw if debugging needed
        # print(json.dumps(result, indent=2))
        return

    # 3. Header
    print("\n" + "="*80)
    print("✅ PROPOSED EXPERIMENTAL PLAN")
    print("="*80)

    # 4. Loop through Experiments
    for i, exp in enumerate(experiments, 1):
        
        # --- Name & Hypothesis ---
        print(f"\n🔬 EXPERIMENT {i}: {exp.get('experiment_name', 'Unnamed Experiment')}")
        print("-" * 80)
        print(f"\n> 🎯 Hypothesis:\n> {exp.get('hypothesis', 'N/A')}")

        # --- Experimental Steps (Numbered) ---
        print("\n--- 🧪 Experimental Steps ---")
        steps = exp.get('experimental_steps', [])
        if steps:
            for j, step in enumerate(steps, 1):
                # Remove leading numbers/bullets provided by LLM
                # Regex removes "1.", "1 -", "1)", etc.
                clean_step = re.sub(r'^[\d\-\.\)\s]+', '', str(step)).strip()
                print(f" {j}. {clean_step}")
        else:
            print("  (No steps provided)")
        
        # --- Equipment ---
        print("\n--- 🛠️  Required Equipment ---")
        equipment = exp.get('required_equipment', [])
        if equipment:
            # Print as a clean comma-separated list if short, or bullets if long
            if len(equipment) > 5:
                for item in equipment: print(f"  * {item}")
            else:
                print(f"  {', '.join(equipment)}")
        else:
            print("  (No equipment specified)")

        # --- Outcome & Justification (Critical for Review) ---
        print("\n--- 📈 Expected Outcome ---")
        print(f"  {exp.get('expected_outcome', 'N/A')}")

        print("\n--- 💡 Justification ---")
        print(f"  {exp.get('justification', 'N/A')}")
        
        # --- Source Documents ---
        print("\n--- 📄 Source Documents ---")
        sources = exp.get('source_documents', [])
        if sources:
            for src in sources:
                print(f"  - {src}")
        else:
            print("  (No sources listed)")

        # --- Code Indicator (If generated) ---
        if "implementation_code" in exp:
            print("\n--- 💻 Implementation Code ---")
            print("  ℹ️  Plan includes implementation script.")

    # --- Plan-level caveats (advisory; from the critic — the plan is unchanged) ---
    caveats = format_caveats(result.get("critic_findings"))
    if caveats:
        print("\n" + "-"*80)
        print("⚠️  Caveats & Potential Limitations")
        for c in caveats:
            print(f"  • {c}")

    print("\n" + "="*80)


def get_user_feedback() -> Optional[str]:
    """
    Pauses execution to get user input via the CLI. 
    Returns None if the user just presses ENTER (indicating approval).
    """
    print("\n" + "-"*60)
    
    print("📝 REQUESTING FEEDBACK")
    print("-" * 60)
    print("Review the plan and any caveats above.")
    print("• To APPROVE as-is: Press [ENTER] directly.")
    print("• To REVISE (e.g. address the caveats, or your own changes): "
          "Type instructions and press [ENTER].")
    
    feedback = input("\n> Instruction: ").strip()
    
    if not feedback:
        return None # User accepted the plan
        
    return feedback


def get_dataset_description(filename: str) -> str:
    """
    Interactive prompt when metadata is missing.
    """
    print("\n" + "!"*60)
    print(f"⚠️  MISSING METADATA FOR: {filename}")
    print("!"*60)
    print("The agent needs context to understand columns/units in this file.")
    print("• Option 1: Press [ENTER] to skip (Agent will guess based on headers).")
    print("• Option 2: Type a brief description (e.g., 'Yield results from Suzuki coupling').")
    
    desc = input("\n> Context: ").strip()
    return desc