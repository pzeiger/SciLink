"""Engineered-breakage harness for VaspUpdater.

Takes a known-good Si bulk case (or any baseline you point at), mutates
the INCAR in known ways to force specific VASP error classes, then
(after the failures are captured on the cluster) runs
`VaspUpdater.refine_inputs` to verify the proposed fix matches what
the deterministic-fix layer should produce.

Two stages:

  PREP -- generate broken inputs + per-case submit scripts:
    python examples/run_breakage_benchmark.py prep \\
      --baseline ~/scilink_baseline_si \\
      --pseudo-dir /share/apps/vasp/potpaw_PBE.54

  ANALYZE -- after submitting on the cluster and the jobs fail, run
  the updater on each captured failure log:
    python examples/run_breakage_benchmark.py analyze \\
      --output breakage_<timestamp>/

PREP writes:
    breakage_<timestamp>/
      low_nbands/   POSCAR INCAR (broken) KPOINTS submit.sh
      zbrent/       POSCAR INCAR (broken) KPOINTS submit.sh
      low_nelm/     POSCAR INCAR (broken) KPOINTS submit.sh
      submit_all.sh
      breakage_manifest.json

ANALYZE writes (per case):
    breakage_<timestamp>/<case>/
      INCAR.fixed       proposed corrected INCAR
      submit_fix.sh     SLURM script that runs the corrected INCAR
      analysis.json     {expected_fix_keys, actual_fix_keys, diagnoses, match}

Each breakage targets a single deterministic-fix pattern in
scilink/agents/sim_agents/vasp_updater.py::KNOWN_FIXES; the analyze
stage's `match` field tells you per-case whether the updater
proposed exactly the fix keys we expected.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ══════════════════════════════════════════════════════════════
# Breakage case registry
# ══════════════════════════════════════════════════════════════

@dataclasses.dataclass
class BreakageCase:
    label: str
    targets: str               # which KNOWN_FIXES pattern this should trigger
    overrides: Dict[str, str]  # INCAR keys to set / overwrite
    remove_keys: List[str]     # INCAR keys to remove entirely
    expected_fix_keys: Set[str]  # fix-key set the updater should propose
    walltime: str = "00:15:00"
    nodes: int = 1
    ntasks_per_node: int = 16


BREAKAGE_CASES: List[BreakageCase] = [
    BreakageCase(
        label="low_nbands",
        targets="'Your highest band is occupied'",
        overrides={"NBANDS": "4"},
        remove_keys=[],
        expected_fix_keys={"NBANDS"},
    ),
    BreakageCase(
        label="zbrent",
        targets="'ZBRENT: fatal error' (bracketing failure)",
        overrides={"IBRION": "2", "POTIM": "5.0", "NSW": "20"},
        remove_keys=[],
        expected_fix_keys={"POTIM", "IBRION"},
    ),
    BreakageCase(
        label="low_nelm",
        targets="'electronic SC steps reached NELM'",
        overrides={"NELM": "3"},
        remove_keys=[],
        # KNOWN_FIXES for SCF-NELM proposes {"NELM": "200", "ALGO": "All"}.
        expected_fix_keys={"NELM", "ALGO"},
    ),
]


# ══════════════════════════════════════════════════════════════
# INCAR mutation
# ══════════════════════════════════════════════════════════════

# Match a non-comment "KEY = value" line. Values can include letters,
# digits, dots, signs, exponents, dashes, dots, and embedded spaces
# (for things like "1 1 1" magmom values), so we just take everything
# until the first '#'/'!' or end-of-line.
_INCAR_LINE_RE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*([^#!\n]*)", re.IGNORECASE)


def mutate_incar(
    incar_text: str,
    overrides: Dict[str, str],
    remove_keys: List[str],
) -> str:
    """Apply overrides + removals to an INCAR. Preserves comments,
    blank lines, and the order of unaffected entries."""
    overrides_norm = {k.upper(): v for k, v in overrides.items()}
    remove_norm = {k.upper() for k in remove_keys}

    out_lines: List[str] = []
    seen_keys: Set[str] = set()

    for line in incar_text.splitlines():
        m = _INCAR_LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        key = m.group(1).upper()
        if key in remove_norm:
            # Drop the line entirely.
            continue
        if key in overrides_norm:
            out_lines.append(f"  {key} = {overrides_norm[key]}")
            seen_keys.add(key)
            continue
        out_lines.append(line)

    # Append overrides for keys that weren't already present.
    missing = set(overrides_norm) - seen_keys
    if missing:
        out_lines.append("")
        out_lines.append("# --- breakage overrides ---")
        for key in sorted(missing):
            out_lines.append(f"  {key} = {overrides_norm[key]}")

    return "\n".join(out_lines) + ("\n" if not out_lines or out_lines[-1] else "")


# ══════════════════════════════════════════════════════════════
# SLURM script generation (single-file, anchored at column 0)
# ══════════════════════════════════════════════════════════════

SLURM_DEFAULT_PSEUDO_DIR = "/path/to/potpaw_PBE.54"
SLURM_DEFAULT_MODULES = (
    "# module purge\n"
    "# module load vasp/6.4.2 intel-mpi/2021"
)
SLURM_DEFAULT_VASP_CMD = "mpirun vasp_std"


def build_submit_script(
    case: BreakageCase,
    *,
    pseudo_dir: str,
    modules: str,
    vasp_command: str,
    job_prefix: str = "scilink_breakage",
) -> str:
    job_name = f"{job_prefix}_{case.label}"
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={case.walltime}
#SBATCH --nodes={case.nodes}
#SBATCH --ntasks-per-node={case.ntasks_per_node}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err

set +e   # do NOT abort on VASP failure -- the failure IS the test
cd "$SLURM_SUBMIT_DIR"

# ---- Module environment (edit to match your cluster) ----
{modules}

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

# ---- Run VASP (expected to fail in a known way) ----
{vasp_command}

EXIT_CODE=$?
echo "VASP exit code: $EXIT_CODE"
exit 0  # always succeed at the SLURM level so the failure log lands intact
"""


def build_submit_all_script(case_labels: List[str]) -> str:
    sbatch_lines = "\n".join(
        f'( cd "$BASE/{label}" && sbatch submit.sh )' for label in case_labels
    )
    return f"""#!/bin/bash
# Submit every engineered-breakage case in this dir.
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
echo "Submitting all breakage cases under $BASE"

{sbatch_lines}

echo "All breakage jobs submitted. Each is expected to fail in a"
echo "specific known way; check 'squeue -u $USER' and then re-run"
echo "this script with 'analyze' once they're done."
"""


def build_submit_fix_script(case_label: str, *, walltime: str = "00:15:00") -> str:
    """SLURM script to run the corrected INCAR. Identical structure to
    submit.sh but renames so the original failure log isn't overwritten."""
    job_name = f"scilink_breakage_fix_{case_label}"
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

# Same module / POTCAR setup as submit.sh -- edit there if needed.
PSEUDO_DIR="$(grep '^PSEUDO_DIR' submit.sh | head -1 | cut -d'"' -f2)"
ELEMENTS=($(sed -n '6p' POSCAR))
: > POTCAR
for e in "${{ELEMENTS[@]}}"; do
    cat "$PSEUDO_DIR/$e/POTCAR" >> POTCAR
done

# Move the original INCAR out of the way and use the fix.
[ -f INCAR.original ] || cp INCAR INCAR.original
cp INCAR.fixed INCAR

mpirun vasp_std
"""


# ══════════════════════════════════════════════════════════════
# Stage 1: PREP
# ══════════════════════════════════════════════════════════════

def cmd_prep(args: argparse.Namespace) -> int:
    baseline = Path(args.baseline).resolve()
    if not baseline.is_dir():
        print(f"ERROR: --baseline {baseline} is not a directory", flush=True)
        return 1

    needed = ("POSCAR", "INCAR", "KPOINTS")
    missing = [f for f in needed if not (baseline / f).exists()]
    if missing:
        print(
            f"ERROR: baseline {baseline} is missing required files: {missing}",
            flush=True,
        )
        return 1

    parent = Path(
        args.output
        or f"breakage_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    print(f"Baseline: {baseline}")
    print(f"Output:   {parent}")
    print(f"Cases:    {[c.label for c in BREAKAGE_CASES]}")
    print()

    incar_text = (baseline / "INCAR").read_text()

    manifest: Dict[str, Any] = {
        "stage": "prep",
        "started_at": datetime.now().isoformat(),
        "baseline": str(baseline),
        "cases": [],
    }

    for case in BREAKAGE_CASES:
        case_dir = parent / case.label
        case_dir.mkdir(parents=True, exist_ok=True)

        # Copy POSCAR / KPOINTS verbatim.
        for f in ("POSCAR", "KPOINTS"):
            shutil.copy2(baseline / f, case_dir / f)

        # Mutated INCAR.
        broken = mutate_incar(incar_text, case.overrides, case.remove_keys)
        (case_dir / "INCAR").write_text(broken)

        # SLURM submit script.
        submit_path = case_dir / "submit.sh"
        submit_path.write_text(build_submit_script(
            case,
            pseudo_dir=args.pseudo_dir,
            modules=args.modules or SLURM_DEFAULT_MODULES,
            vasp_command=args.vasp_cmd,
        ))
        submit_path.chmod(0o755)

        manifest["cases"].append({
            "label": case.label,
            "targets": case.targets,
            "overrides": case.overrides,
            "remove_keys": case.remove_keys,
            "expected_fix_keys": sorted(case.expected_fix_keys),
            "case_dir": str(case_dir),
        })

        print(f"  ✓ {case.label:14s}  → targets {case.targets}")

    # Top-level submit_all + manifest.
    submit_all = parent / "submit_all.sh"
    submit_all.write_text(
        build_submit_all_script([c.label for c in BREAKAGE_CASES])
    )
    submit_all.chmod(0o755)
    (parent / "breakage_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print()
    print("Next steps:")
    print(f"  rsync -av {parent}/ alle927@deception.pnl.gov:/people/alle927/{parent.name}/")
    print(f"  ssh alle927@deception.pnl.gov 'cd /people/alle927/{parent.name} && bash submit_all.sh'")
    print(f"  # wait for failures, then:")
    print(f"  rsync -av alle927@deception.pnl.gov:/people/alle927/{parent.name}/ {parent}/")
    print(f"  python examples/run_breakage_benchmark.py analyze --output {parent}")

    return 0


# ══════════════════════════════════════════════════════════════
# Stage 2: ANALYZE
# ══════════════════════════════════════════════════════════════

def _parse_incar_keys(incar_text: str) -> Dict[str, str]:
    """Extract KEY → value dict from an INCAR. Reuses _INCAR_LINE_RE."""
    keys: Dict[str, str] = {}
    for line in incar_text.splitlines():
        m = _INCAR_LINE_RE.match(line)
        if m:
            keys[m.group(1).upper()] = m.group(2).strip()
    return keys


def _diff_incar_keys(original: str, suggested: str) -> Set[str]:
    """Set of INCAR keys whose values differ between original and
    suggested (added, removed, or changed)."""
    a = _parse_incar_keys(original)
    b = _parse_incar_keys(suggested)
    return {k for k in (set(a) | set(b)) if a.get(k) != b.get(k)}


def _find_failure_log(case_dir: Path) -> Optional[Path]:
    """Locate the captured VASP failure log. Prefer the SLURM .out file
    (which contains stdout from VASP), fall back to vasp.out if present."""
    candidates = sorted(case_dir.glob("scilink_breakage_*_*.out"))
    if candidates:
        # Take the most recent if multiple resubmissions happened.
        return max(candidates, key=lambda p: p.stat().st_mtime)
    fallback = case_dir / "vasp.out"
    if fallback.exists():
        return fallback
    return None


def cmd_analyze(args: argparse.Namespace) -> int:
    parent = Path(args.output).resolve()
    if not parent.is_dir():
        print(f"ERROR: --output {parent} is not a directory", flush=True)
        return 1

    manifest_path = parent / "breakage_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Did you run --prep first?", flush=True)
        return 1
    manifest = json.loads(manifest_path.read_text())

    # Lazy import: we need credentials for the LLM fallback path, but
    # for our specific patterns the deterministic layer fires.
    from scilink.agents.sim_agents.vasp_updater import VaspUpdater
    updater = VaspUpdater(api_key=args.api_key, model_name=args.model)

    summary: List[Dict[str, Any]] = []
    for record in manifest["cases"]:
        case_dir = Path(record["case_dir"])
        if not case_dir.is_dir():
            print(f"  SKIP {record['label']:14s}  ({case_dir} missing)")
            summary.append({"label": record["label"], "status": "skip",
                            "reason": "case dir missing"})
            continue

        log_path = _find_failure_log(case_dir)
        if log_path is None:
            print(f"  SKIP {record['label']:14s}  (no failure log found in {case_dir})")
            summary.append({"label": record["label"], "status": "skip",
                            "reason": "no failure log"})
            continue

        try:
            result = updater.refine_inputs(
                poscar_path=str(case_dir / "POSCAR"),
                incar_path=str(case_dir / "INCAR"),
                kpoints_path=str(case_dir / "KPOINTS"),
                vasp_log=log_path.read_text(),
                original_request=(
                    f"benchmark breakage case '{record['label']}': "
                    f"engineered to trigger {record['targets']}."
                ),
            )
        except Exception as exc:
            print(f"  ERR  {record['label']:14s}  refine_inputs raised: {exc}")
            summary.append({"label": record["label"], "status": "error",
                            "reason": str(exc)})
            continue

        explanation = result.get("explanation", {}) or {}
        # The result's explanation key shape varies by method:
        #   "deterministic"        -> {diagnoses, fixes_applied}
        #   "deterministic_only"   -> {diagnoses, fixes_applied, llm_error, ...}
        #   "deterministic+llm"    -> {deterministic_diagnoses, deterministic_fixes, llm_explanation}
        #   "llm"                  -> same as deterministic+llm but det dicts are empty
        # Pull diagnoses from whichever key is populated.
        diagnoses = (
            explanation.get("diagnoses")
            or explanation.get("deterministic_diagnoses")
            or []
        )
        det_fixes = (
            explanation.get("fixes_applied")
            or explanation.get("deterministic_fixes")
            or {}
        )

        # Most robust measure of "what the updater changed" -- diff the
        # original INCAR vs the suggested one. Captures both the
        # deterministic fixes and any LLM-side additions in one set.
        original_incar = (case_dir / "INCAR").read_text()
        suggested_incar = result.get("suggested_incar", "")
        actual_fix_keys = _diff_incar_keys(original_incar, suggested_incar)

        expected = set(record["expected_fix_keys"])
        # Superset match: the agent might add MORE keys than the minimal
        # fix (e.g. the LLM tightens an unrelated parameter); we only
        # require that the expected keys are all present.
        match = expected.issubset(actual_fix_keys)

        analysis = {
            "label": record["label"],
            "targets": record["targets"],
            "log": str(log_path),
            "method": result.get("method"),
            "diagnoses": diagnoses,
            "deterministic_fix_keys": sorted(det_fixes.keys()),
            "expected_fix_keys": sorted(expected),
            "actual_fix_keys": sorted(actual_fix_keys),
            "match": match,
        }
        (case_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))

        # Save the corrected INCAR + a fix-resubmit script.
        if "suggested_incar" in result:
            (case_dir / "INCAR.fixed").write_text(result["suggested_incar"])
            fix_script = case_dir / "submit_fix.sh"
            fix_script.write_text(build_submit_fix_script(record["label"]))
            fix_script.chmod(0o755)

        marker = "✓" if match else ("⚠" if actual_fix_keys else "✗")
        print(
            f"  {marker} {record['label']:14s}  "
            f"method={result.get('method', '—'):20s}  "
            f"expected={sorted(expected)}  actual={sorted(actual_fix_keys)}"
        )
        summary.append(analysis)

    (parent / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote analysis_summary.json to {parent}")
    matches = sum(1 for r in summary if r.get("match"))
    print(f"Updater fixes matched expected for {matches}/{len(summary)} cases")
    print()
    print("To re-run each case with the proposed fix:")
    print(f"  rsync -av {parent}/ alle927@deception.pnl.gov:/people/alle927/{parent.name}/")
    print(f"  for d in {parent.name}/*/; do (cd $d && sbatch submit_fix.sh); done")

    return 0


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep", help="Generate broken inputs + submit scripts.")
    p_prep.add_argument(
        "--baseline",
        required=True,
        help="Directory with verified-good POSCAR / INCAR / KPOINTS.",
    )
    p_prep.add_argument(
        "--output",
        default=None,
        help="Output dir (default: breakage_benchmark_<timestamp>/)",
    )
    p_prep.add_argument(
        "--pseudo-dir",
        default=SLURM_DEFAULT_PSEUDO_DIR,
        help="Path to potpaw_PBE on the cluster.",
    )
    p_prep.add_argument(
        "--vasp-cmd",
        default=SLURM_DEFAULT_VASP_CMD,
        help="VASP launch command in the SLURM script.",
    )
    p_prep.add_argument(
        "--modules",
        default=None,
        help='Module-load block to bake into submit.sh. Pass either with '
             'real newlines (e.g. via $\'module purge\\nmodule load X\') or '
             'with literal backslash-n sequences -- both forms are accepted '
             'and converted to real newlines in the generated script.',
    )
    p_prep.set_defaults(func=cmd_prep)

    p_an = sub.add_parser("analyze", help="Run VaspUpdater on captured failure logs.")
    p_an.add_argument("--output", required=True, help="Existing breakage dir from prep.")
    p_an.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="LLM model for VaspUpdater's LLM fallback (deterministic layer "
             "doesn't need it for our patterns).",
    )
    p_an.add_argument("--api-key", default=None, help="Explicit LLM API key.")
    p_an.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    # Bash's double-quoted "\n" is a literal backslash-n, not a newline.
    # Accept the obvious-looking shell invocation and translate before use.
    if getattr(args, "modules", None):
        args.modules = args.modules.replace("\\n", "\n")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
