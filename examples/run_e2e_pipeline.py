"""End-to-end post-run pipeline for the VASP agentic workflow.

Takes a benchmark output directory (produced by run_benchmark_suite.py)
where each subdirectory holds the input files + the captured outputs
of one VASP run, and ties the agents together:

  for each case dir:
      classify success / failure via post_run_analysis
      if FAILURE:
          VaspUpdater.refine_inputs -> propose a corrected INCAR
          write INCAR.fixed + submit_fix.sh (resubmit-ready)
          prompt: graduate this observation to a skill?
      if SUCCESS:
          VaspQualityAgent.run_quality_check -> structured assessment
          prompt: graduate any noted issue to a skill?
      print the per-case summary

Skips the cluster-side submit / poll / pull-back step on purpose --
that's covered by run_benchmark_suite.py + sbatch. This script is the
"once results are back, what does the agent system do with them" piece.

Usage:

    python examples/run_e2e_pipeline.py \\
        --output benchmark_suite_20260510_135123/ \\
        --pseudo-dir /share/apps/vasp/potpaw_PBE.54 \\
        --modules "module purge\\nmodule load openmpi/5.0.7"

Flags for non-interactive runs:
    --no-graduate       skip all graduation prompts
    --auto-graduate     graduate every observation without asking
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════
# Result types
# ══════════════════════════════════════════════════════════════

@dataclasses.dataclass
class CaseResult:
    label: str
    case_dir: Path
    classification: str  # "success" | "failure" | "unknown"
    summary: str = ""
    updater_method: Optional[str] = None
    updater_diagnoses: List[str] = dataclasses.field(default_factory=list)
    updater_fix_keys: List[str] = dataclasses.field(default_factory=list)
    quality_status: Optional[str] = None
    quality_issues: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    quality_summary: Optional[str] = None
    knowledge_id: Optional[str] = None
    skill_path: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# Classification
# ══════════════════════════════════════════════════════════════

def classify_case(case_dir: Path, facts: Dict[str, Any]) -> str:
    """Decide whether a case run succeeded, failed, or is incomplete.

    Convergence flags from vasprun.xml are the primary signal. When
    the file isn't present (e.g. VASP died very early), fall back to
    looking for a SLURM stdout / vasp.out tail."""
    if facts.get("converged") is True:
        return "success"
    if facts.get("converged_electronic") is False:
        return "failure"
    if facts.get("converged_ionic") is False:
        return "failure"
    if facts.get("error_hints") or facts.get("classified_errors"):
        return "failure"
    # No vasprun.xml at all -> either VASP didn't run, or the run is
    # mid-flight. Look for any *.out / vasp.out as a hint.
    if list(case_dir.glob("*.out")) or (case_dir / "vasp.out").exists():
        return "failure"  # something ran but didn't converge
    return "unknown"


# ══════════════════════════════════════════════════════════════
# Failure handling: updater
# ══════════════════════════════════════════════════════════════

def _find_failure_log(case_dir: Path) -> Optional[Path]:
    """Pick a stdout / log file to feed to VaspUpdater."""
    candidates = (
        sorted(case_dir.glob("*.out"))
        + [case_dir / "vasp.out"]
        + [case_dir / "OUTCAR"]
    )
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def run_updater_on_failure(
    case: CaseResult,
    *,
    updater,
    pseudo_dir: str,
    modules: str,
    vasp_command: str,
) -> None:
    """Call VaspUpdater on the case's failure log; record proposed fix
    + a resubmit-ready submit_fix.sh in the case dir."""
    from scilink.agents.sim_agents.skill_graduation import (
        load_graduated_skills,
    )  # noqa: F401  (loaded inside refine_inputs anyway)

    case_dir = case.case_dir
    log_path = _find_failure_log(case_dir)
    if log_path is None:
        case.summary = "Failure detected but no log file to feed updater."
        return

    poscar = case_dir / "POSCAR"
    incar = case_dir / "INCAR"
    kpoints = case_dir / "KPOINTS"
    if not all(p.exists() for p in (poscar, incar, kpoints)):
        case.summary = "Failure detected but missing POSCAR / INCAR / KPOINTS."
        return

    try:
        result = updater.refine_inputs(
            poscar_path=str(poscar),
            incar_path=str(incar),
            kpoints_path=str(kpoints),
            vasp_log=log_path.read_text(),
            original_request=f"e2e pipeline case '{case.label}'",
        )
    except Exception as exc:
        case.summary = f"VaspUpdater raised: {exc}"
        return

    explanation = result.get("explanation", {}) or {}
    case.updater_method = result.get("method")
    case.updater_diagnoses = (
        explanation.get("diagnoses")
        or explanation.get("deterministic_diagnoses")
        or []
    )
    det_fixes = (
        explanation.get("fixes_applied")
        or explanation.get("deterministic_fixes")
        or {}
    )
    case.updater_fix_keys = sorted(det_fixes.keys())
    case.summary = (
        f"updater: {case.updater_method}, "
        f"deterministic fix keys = {case.updater_fix_keys}"
    )

    # Stage corrected INCAR + a resubmit-ready submit_fix.sh.
    if "suggested_incar" in result:
        (case_dir / "INCAR.fixed").write_text(result["suggested_incar"])
        fix_script = case_dir / "submit_fix.sh"
        fix_script.write_text(
            _build_submit_fix_script(
                case.label,
                pseudo_dir=pseudo_dir,
                modules=modules,
                vasp_command=vasp_command,
            )
        )
        fix_script.chmod(0o755)


def _build_submit_fix_script(
    case_label: str,
    *,
    pseudo_dir: str,
    modules: str,
    vasp_command: str,
    walltime: str = "00:15:00",
) -> str:
    """Mirror of run_breakage_benchmark.build_submit_fix_script -- kept
    inline here so this script is usable on its own without importing
    a sibling example file."""
    job_name = f"scilink_e2e_fix_{case_label}"
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

# ---- Module environment ----
{modules}

# ---- Move original INCAR aside and use the fix ----
[ -f INCAR.original ] || cp INCAR INCAR.original
cp INCAR.fixed INCAR

# ---- Assemble POTCAR from pseudo dir ----
PSEUDO_DIR="{pseudo_dir}"

ELEMENTS=($(sed -n '6p' POSCAR))
if [ ${{#ELEMENTS[@]}} -eq 0 ]; then
    echo "No elements parsed from POSCAR; aborting." >&2
    exit 1
fi
: > POTCAR
for e in "${{ELEMENTS[@]}}"; do
    if [ ! -f "$PSEUDO_DIR/$e/POTCAR" ]; then
        echo "Missing POTCAR for element $e at $PSEUDO_DIR/$e/POTCAR" >&2
        exit 1
    fi
    cat "$PSEUDO_DIR/$e/POTCAR" >> POTCAR
done

# ---- Run VASP with the corrected INCAR ----
{vasp_command}
"""


# ══════════════════════════════════════════════════════════════
# Success handling: quality agent
# ══════════════════════════════════════════════════════════════

def run_quality_on_success(
    case: CaseResult,
    *,
    quality_agent,
    research_goal: str,
) -> None:
    """Call VaspQualityAgent and record the assessment fields."""
    try:
        result = quality_agent.run_quality_check(
            output_dir=str(case.case_dir),
            research_goal=research_goal,
        )
    except Exception as exc:
        case.summary = f"VaspQualityAgent raised: {exc}"
        return

    case.quality_status = result.get("status")
    case.quality_issues = result.get("issues", []) or []
    case.quality_summary = result.get("assessment_summary", "")
    case.summary = f"quality: {case.quality_status}"
    # Persist for later inspection.
    (case.case_dir / "quality_assessment.json").write_text(
        json.dumps(result, indent=2, default=str)
    )


# ══════════════════════════════════════════════════════════════
# Graduation prompts
# ══════════════════════════════════════════════════════════════

def maybe_graduate_failure(
    case: CaseResult,
    *,
    updater,
    mode: str,
) -> None:
    """Offer to graduate the failure observation into a learned-fixes skill.

    Only meaningful when the updater proposed something."""
    if mode == "skip" or not case.updater_fix_keys:
        return
    obs = {
        "summary": (
            f"VASP run for case '{case.label}' failed; "
            f"updater proposed fix on keys {case.updater_fix_keys} "
            f"via {case.updater_method}."
        ),
        "case_label": case.label,
        "diagnoses": case.updater_diagnoses,
        "fix_keys": case.updater_fix_keys,
    }
    if mode == "auto" or _confirm(f"Graduate observation for failure '{case.label}'?"):
        kid = updater.record_knowledge(obs)
        case.knowledge_id = kid
        result = updater.graduate_to_skill(kid)
        case.skill_path = result.get("skill_path")
        print(f"    → graduated to skill: {case.skill_path}")
        if result.get("warning"):
            print(f"    ⚠ {result['warning']}")


def maybe_graduate_quality(
    case: CaseResult,
    *,
    quality_agent,
    mode: str,
) -> None:
    """Offer to graduate any non-healthy quality finding into a heuristics skill."""
    if mode == "skip":
        return
    if case.quality_status == "healthy" or not case.quality_issues:
        # Nothing useful to graduate from a clean run.
        return
    obs = {
        "summary": (
            f"VASP run for case '{case.label}' converged but quality assessment "
            f"flagged status='{case.quality_status}'."
        ),
        "case_label": case.label,
        "issues": case.quality_issues,
        "assessment_summary": case.quality_summary,
    }
    if mode == "auto" or _confirm(
        f"Graduate quality observation for '{case.label}' (status: {case.quality_status})?"
    ):
        kid = quality_agent.record_knowledge(obs)
        case.knowledge_id = kid
        result = quality_agent.graduate_to_skill(kid)
        case.skill_path = result.get("skill_path")
        print(f"    → graduated to skill: {case.skill_path}")
        if result.get("warning"):
            print(f"    ⚠ {result['warning']}")


def _confirm(prompt: str) -> bool:
    """Interactive yes/no. Default no (a typo doesn't accidentally graduate)."""
    if not sys.stdin.isatty():
        # Running non-interactively without --auto-graduate or --no-graduate:
        # default to skip rather than block on input().
        return False
    try:
        ans = input(f"  {prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


# ══════════════════════════════════════════════════════════════
# Main driver
# ══════════════════════════════════════════════════════════════

def find_case_dirs(parent: Path) -> List[Path]:
    """Subdirectories of `parent` that look like VASP cases (have a POSCAR)."""
    if not parent.is_dir():
        return []
    return sorted(
        d for d in parent.iterdir()
        if d.is_dir() and (d / "POSCAR").exists()
    )


def load_research_goal_for_case(case_dir: Path, *, manifest: Dict[str, Any]) -> str:
    """Pull the per-case description from the manifest (run_benchmark_suite
    writes one). Falls back to a generic goal if the manifest is missing
    or lacks the case."""
    label = case_dir.name
    for record in manifest.get("cases", []):
        if record.get("label") == label:
            return record.get("description", f"VASP run for case '{label}'")
    return f"VASP run for case '{label}'"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Benchmark output dir from run_benchmark_suite.py.",
    )
    parser.add_argument(
        "--pseudo-dir",
        default="/path/to/potpaw_PBE.54",
        help="PSEUDO_DIR baked into any submit_fix.sh emitted on failures.",
    )
    parser.add_argument(
        "--modules",
        default="# module load <vasp>",
        help='Module-load block for submit_fix.sh. Same "\\n -> newline" '
             "translation as run_benchmark_suite.py.",
    )
    parser.add_argument(
        "--vasp-cmd",
        default="mpirun vasp_std",
        help="VASP launch command in submit_fix.sh.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="LLM model for VaspUpdater + VaspQualityAgent.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit LLM API key. If unset, the agents auto-discover.",
    )
    grad_group = parser.add_mutually_exclusive_group()
    grad_group.add_argument(
        "--no-graduate",
        dest="grad_mode",
        action="store_const",
        const="skip",
        help="Skip all graduation prompts.",
    )
    grad_group.add_argument(
        "--auto-graduate",
        dest="grad_mode",
        action="store_const",
        const="auto",
        help="Graduate every flagged observation without asking.",
    )
    parser.set_defaults(grad_mode="prompt")

    args = parser.parse_args()
    if args.modules:
        args.modules = args.modules.replace("\\n", "\n")

    parent = Path(args.output).resolve()
    if not parent.is_dir():
        print(f"ERROR: --output {parent} is not a directory", flush=True)
        return 1

    case_dirs = find_case_dirs(parent)
    if not case_dirs:
        print(f"No case directories found under {parent}", flush=True)
        return 1

    # Optional manifest from run_benchmark_suite -- gives per-case
    # research goals to feed the quality agent.
    manifest_path = parent / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"cases": []}
    )

    # Lazy imports so --help works without scilink's heavier deps.
    from scilink.agents.sim_agents.post_run_analysis import analyze_run_directory
    from scilink.agents.sim_agents.vasp_updater import VaspUpdater
    from scilink.agents.sim_agents.vasp_quality import VaspQualityAgent

    updater = VaspUpdater(api_key=args.api_key, model_name=args.model)
    quality_agent = VaspQualityAgent(api_key=args.api_key, model_name=args.model)

    print(f"Pipeline target: {parent}")
    print(f"Cases found:     {[d.name for d in case_dirs]}")
    print(f"Graduation mode: {args.grad_mode}")
    print()

    results: List[CaseResult] = []
    for case_dir in case_dirs:
        case = CaseResult(label=case_dir.name, case_dir=case_dir, classification="unknown")
        facts = analyze_run_directory(str(case_dir))
        case.classification = classify_case(case_dir, facts)

        print(f"━━━ {case.label}  [{case.classification}] ━━━")

        if case.classification == "failure":
            run_updater_on_failure(
                case,
                updater=updater,
                pseudo_dir=args.pseudo_dir,
                modules=args.modules,
                vasp_command=args.vasp_cmd,
            )
            print(f"  {case.summary}")
            maybe_graduate_failure(case, updater=updater, mode=args.grad_mode)
        elif case.classification == "success":
            research_goal = load_research_goal_for_case(case_dir, manifest=manifest)
            run_quality_on_success(
                case, quality_agent=quality_agent, research_goal=research_goal
            )
            print(f"  {case.summary}")
            if case.quality_summary:
                # Print the first 200 chars so the user can scan it.
                snippet = case.quality_summary.replace("\n", " ")
                print(f"  → {snippet[:200]}{'…' if len(snippet) > 200 else ''}")
            maybe_graduate_quality(
                case, quality_agent=quality_agent, mode=args.grad_mode
            )
        else:
            print(f"  No vasprun.xml or recognizable output; skipping.")

        results.append(case)
        print()

    # Final summary.
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Pipeline summary:")
    for c in results:
        bits = [c.classification]
        if c.updater_method:
            bits.append(f"updater={c.updater_method}")
        if c.quality_status:
            bits.append(f"quality={c.quality_status}")
        if c.skill_path:
            bits.append(f"graduated={Path(c.skill_path).name}")
        print(f"  {c.label:20s}  {'  '.join(bits)}")

    # Persist a JSON of the run for easy review later.
    out_path = parent / f"e2e_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            [dataclasses.asdict(r) | {"case_dir": str(r.case_dir)} for r in results],
            indent=2,
            default=str,
        )
    )
    print()
    print(f"Wrote run details to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
