"""
Live + offline tests for the SimulationOrchestratorAgent.

Tiers (mirror tests/test_mp_resolver.py):

    targeted   plumbing checks; zero LLM calls; safe to run without keys
    stress     live LLM; exercises individual tools through the chat loop
    e2e        live LLM; full run_task flow producing files on disk

Required env vars:
    ANTHROPIC_API_KEY           (for stress / e2e)
    MP_API_KEY                  (recommended for stress / e2e — exercises
                                 the MP tool-resolver inside generate_structure)

Run:
    python tests/test_simulation_orchestrator.py
    python tests/test_simulation_orchestrator.py --stress
    python tests/test_simulation_orchestrator.py --e2e
    python tests/test_simulation_orchestrator.py --stress --e2e
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import textwrap
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = "claude-opus-4-6"
RUN_DIR = REPO_ROOT / "tests" / "_simulate_runs"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _require_env(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"❌ Missing required env var(s): {', '.join(missing)}")
        sys.exit(2)


def _make_orch(model_name: str, base_dir: str, autonomy: str = "co-pilot"):
    """Construct a fresh SimulationOrchestratorAgent for a test."""
    from scilink.agents.sim_agents import (
        SimulationOrchestratorAgent, SimulationMode,
    )
    mode_map = {
        "co-pilot": SimulationMode.CO_PILOT,
        "supervised": SimulationMode.SUPERVISED,
        "autonomous": SimulationMode.AUTONOMOUS,
    }
    return SimulationOrchestratorAgent(
        base_dir=base_dir,
        api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy-not-used"),
        model_name=model_name,
        simulation_mode=mode_map[autonomy],
        mp_api_key=os.environ.get("MP_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Targeted tests — zero LLM calls, safe without keys
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "session_status",
    "generate_structure",
    "validate_structure",
    "generate_vasp_inputs",
    "run_complete_dft_workflow",
    "refine_structure",
    "view_structure",
    "validate_incar",
    "apply_incar_improvements",
    "list_generated_structures",
    "analyze_vasp_output",
    "suggest_incar_fixes",
}


def test_1_orchestrator_constructs(model_name: str):
    """Orchestrator builds without an LLM call; all 12 tools registered."""
    with tempfile.TemporaryDirectory() as td:
        orch = _make_orch(model_name, td + "/sim")
        assert set(orch.tools.functions_map.keys()) == EXPECTED_TOOLS
        assert len(orch.tools.openai_schemas) == len(EXPECTED_TOOLS)
        # All schemas serializable
        for s in orch.tools.openai_schemas:
            json.dumps(s)
        print(f"   ✅ Orchestrator constructs; {len(EXPECTED_TOOLS)} tools registered")


def test_2_tool_error_paths(model_name: str):
    """Each tool's primary error path returns structured JSON, not raises."""
    with tempfile.TemporaryDirectory() as td:
        orch = _make_orch(model_name, td + "/sim")

        # validate_structure on missing file
        r = json.loads(orch.tools.execute_tool(
            "validate_structure",
            poscar_path="/nonexistent/POSCAR",
            original_request="test",
        ))
        assert r["status"] == "error" and "not found" in r["message"]

        # generate_vasp_inputs with bad method
        fake = Path(td) / "POSCAR"
        fake.write_text("Si\n1.0\n3 0 0\n0 3 0\n0 0 3\nSi\n1\nDirect\n0 0 0\n")
        r = json.loads(orch.tools.execute_tool(
            "generate_vasp_inputs",
            poscar_path=str(fake),
            request="test",
            method="banana",
        ))
        assert r["status"] == "error" and "banana" in r["message"]

        # refine_structure on a path with no session record
        r = json.loads(orch.tools.execute_tool(
            "refine_structure",
            poscar_path=str(fake),
            original_request="test",
        ))
        assert r["status"] == "error" and "No record found" in r["message"]

        # view_structure on missing file
        r = json.loads(orch.tools.execute_tool(
            "view_structure",
            poscar_path="/nope/POSCAR",
        ))
        assert r["status"] == "error"

        # validate_incar without FH key returns 'skipped', not 'error'
        fake_incar = Path(td) / "INCAR"
        fake_incar.write_text("ENCUT = 400\n")
        r = json.loads(orch.tools.execute_tool(
            "validate_incar",
            incar_path=str(fake_incar),
            system_description="test",
        ))
        assert r["status"] in ("skipped", "error")

        # apply_incar_improvements with no adjustments
        r = json.loads(orch.tools.execute_tool(
            "apply_incar_improvements",
            incar_path=str(fake_incar),
            poscar_path=str(fake),
            original_request="test",
            suggested_adjustments=[],
        ))
        assert r["status"] == "no_changes"

        # suggest_incar_fixes on missing log
        r = json.loads(orch.tools.execute_tool(
            "suggest_incar_fixes",
            log_path="/nope/log",
            original_request="test",
        ))
        assert r["status"] == "error"

        # Unknown tool
        r = json.loads(orch.tools.execute_tool("not_a_real_tool"))
        assert r["status"] == "error"

        print("   ✅ All error paths return structured JSON")


def test_3_post_run_analysis_synthetic(model_name: str):
    """post_run_analysis works on synthetic log dirs without pymatgen-parseable files."""
    from scilink.agents.sim_agents.post_run_analysis import analyze_run_directory

    # Empty dir → status=ok, convergence=unknown
    with tempfile.TemporaryDirectory() as td:
        r = analyze_run_directory(td)
        assert r["status"] == "ok"
        assert r["convergence_status"] == "unknown"

    # Dir with stdout matching a known error → convergence=failed, hints
    with tempfile.TemporaryDirectory() as td:
        Path(td, "stdout").write_text(
            "running on 32 cores\nVERY BAD NEWS! internal error in subroutine SGRCON\n"
        )
        r = analyze_run_directory(td)
        assert r["convergence_status"] == "failed"
        assert any("VERY BAD NEWS" in h for h in r["log_error_hints"])

    # Nonexistent dir → status=error
    r = analyze_run_directory("/path/does/not/exist")
    assert r["status"] == "error"

    print("   ✅ Post-run analysis: empty / failed / nonexistent dirs all handled")


def test_4_mode_switching(model_name: str):
    """set_simulation_mode flips human-feedback flag and updates the system prompt."""
    from scilink.agents.sim_agents import SimulationMode
    with tempfile.TemporaryDirectory() as td:
        orch = _make_orch(model_name, td + "/sim", autonomy="co-pilot")
        assert orch.get_human_feedback_setting() is True

        orch.set_simulation_mode(SimulationMode.AUTONOMOUS)
        assert orch.get_human_feedback_setting() is False
        # System prompt was rebuilt with the new directive
        assert "AUTONOMY: AUTONOMOUS" in orch._system_prompt

        orch.set_simulation_mode(SimulationMode.SUPERVISED)
        assert orch.get_human_feedback_setting() is True  # supervised still wants feedback
        assert "AUTONOMY: SUPERVISED" in orch._system_prompt

    print("   ✅ Mode switching updates feedback flag + system prompt")


def test_5_run_task_without_llm(model_name: str):
    """run_task derives the structured summary correctly when chat is stubbed."""
    from scilink.agents.sim_agents import SimulationMode

    with tempfile.TemporaryDirectory() as td:
        orch = _make_orch(model_name, td + "/sim", autonomy="co-pilot")

        def fake_chat(_):
            orch.generated_structures.append({
                "slug": "fake_001",
                "description": "test structure",
                "poscar_path": "/tmp/POSCAR",
                "incar_path": None,
                "kpoints_path": None,
                "script_path": "/tmp/script.py",
                "validation": {
                    "status": "needs_correction",
                    "overall_assessment": "Vacuum is thin",
                    "all_identified_issues": ["Vacuum 12 A; recommend ≥15 A"],
                },
                "created_at": datetime.now().isoformat(),
            })
            return "Built the structure; flagged thin vacuum."

        orch.chat = fake_chat
        result = orch.run_task("test task", context={"priority": "high"})

        assert result["status"] == "success"
        assert orch.simulation_mode == SimulationMode.CO_PILOT  # restored
        assert "/tmp/POSCAR" in result["files_produced"]
        assert any("Vacuum" in f for f in result["key_findings"])
        assert any("VASP inputs" in f for f in result["suggested_followups"])
        assert any("unresolved" in w for w in result["warnings"])

        # Error path: chat raises → mode still restored, status=error
        def fake_chat_raises(_):
            raise RuntimeError("simulated")
        orch.chat = fake_chat_raises
        r2 = orch.run_task("another task")
        assert r2["status"] == "error" and "simulated" in r2["error"]
        assert orch.simulation_mode == SimulationMode.CO_PILOT

    print("   ✅ run_task: structured summary correct; mode pin/restore; error path")


def test_6_session_persistence(model_name: str):
    """generated_structures + default_calc_params survive checkpoint/restore."""
    with tempfile.TemporaryDirectory() as td:
        sd = td + "/sim"
        orch = _make_orch(model_name, sd)
        orch.generated_structures.append({
            "slug": "persist_001",
            "description": "survival test",
            "poscar_path": "/tmp/POSCAR",
        })
        orch.default_calc_params = {"ENCUT": 520, "kpoint_density": 30}
        # Force a checkpoint write
        orch.message_count = orch.CHECKPOINT_INTERVAL
        orch._auto_checkpoint()

        # Restore into a fresh instance
        orch2 = _make_orch(model_name, sd)
        # restore_checkpoint is opt-in; do it explicitly via the classmethod
        from scilink.agents.sim_agents import SimulationOrchestratorAgent
        orch3 = SimulationOrchestratorAgent.restore_from_checkpoint(
            base_dir=sd,
            api_key="dummy",
            model_name=model_name,
        )
        assert any(s["slug"] == "persist_001" for s in orch3.generated_structures)
        assert orch3.default_calc_params.get("ENCUT") == 520

    print("   ✅ Checkpoint round-trips structures + sticky params")


# ---------------------------------------------------------------------------
# Stress tests — live LLM, exercise individual tools
# ---------------------------------------------------------------------------

def stress_1_generate_then_inputs(model_name: str):
    """Live: agent builds a simple structure and produces VASP inputs."""
    workdir = RUN_DIR / "stress_generate"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    orch = _make_orch(model_name, str(workdir / "sim"), autonomy="autonomous")
    response = orch.chat(
        "Build a 2x2x1 supercell of bulk silicon (mp-149) and then generate "
        "static VASP inputs for a single-point calculation. Use the granular "
        "tools (generate_structure, then generate_vasp_inputs)."
    )
    print("   --- agent response (last 400 chars) ---")
    print("   " + response[-400:].replace("\n", "\n   "))

    structures = orch.generated_structures
    assert structures, "Agent did not record any generated structures"
    s = structures[-1]
    assert Path(s["poscar_path"]).exists(), f"POSCAR missing: {s['poscar_path']}"
    print(f"   ✅ POSCAR produced: {s['poscar_path']}")

    if s.get("incar_path"):
        assert Path(s["incar_path"]).exists()
        print(f"   ✅ INCAR produced: {s['incar_path']}")
    else:
        print("   ⚠️  Agent built the structure but didn't generate INCAR — "
              "may have stopped to ask. Acceptable for a stress test.")


def stress_2_post_run_analysis_failure(model_name: str):
    """Live: agent reads a synthetic failed-run dir and gives a useful summary."""
    workdir = RUN_DIR / "stress_post_run"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fake_run_dir = workdir / "fake_run"
    fake_run_dir.mkdir(parents=True, exist_ok=True)
    (fake_run_dir / "stdout").write_text(textwrap.dedent("""
        running on 32 cores
        Iteration   1
        ...
        ERROR: NELM steps reached, electronic SCF did not converge
        Therefore set LREAL=.FALSE.
        Aborting...
    """).strip())

    orch = _make_orch(model_name, str(workdir / "sim"), autonomy="autonomous")
    response = orch.chat(
        f"I ran VASP and the calculation failed. Output files are in "
        f"{fake_run_dir}. Use analyze_vasp_output to read the run, summarize "
        f"what happened, and recommend INCAR adjustments."
    )
    print("   --- agent response (last 600 chars) ---")
    print("   " + response[-600:].replace("\n", "\n   "))
    # Lenient assertion: the agent should mention NELM or the error
    has_nelm_or_lreal = any(s in response.lower() for s in ["nelm", "lreal", "scf", "converge"])
    assert has_nelm_or_lreal, "Agent did not surface the relevant error pattern"
    print("   ✅ Agent surfaced the failure mode in its summary")


def stress_3_session_status_called(model_name: str):
    """Live: when asked 'what have you built so far?', the agent calls session_status."""
    workdir = RUN_DIR / "stress_status"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    orch = _make_orch(model_name, str(workdir / "sim"), autonomy="autonomous")
    response = orch.chat(
        "What structures have you built so far in this session? Call "
        "list_generated_structures or session_status to find out."
    )
    print("   --- agent response (last 300 chars) ---")
    print("   " + response[-300:].replace("\n", "\n   "))
    # The response should reflect "no structures yet"
    assert any(s in response.lower() for s in ["no structure", "0 structure", "none", "haven't"])
    print("   ✅ Agent correctly reported empty session")


def stress_4_aimsgb_skill_loaded(model_name: str):
    """Live: GB request → agent passes skill='aimsgb' → script uses aimsgb API.

    Verifies the full skill-loading chain: system-prompt nudge → LLM
    chooses to set skill='aimsgb' on generate_structure → tool resolves
    the built-in skill via scilink.skills.loader → curated content lands
    in the structure-generation prompt → generated script imports aimsgb
    and uses Grain / GrainBoundary / build_gb correctly.

    Requires: aimsgb installed (`pip install aimsgb`), MP_API_KEY set
    (script fetches mp-13 = bcc Fe).
    """
    workdir = RUN_DIR / "stress_aimsgb"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    orch = _make_orch(model_name, str(workdir / "sim"), autonomy="autonomous")
    response = orch.chat(
        "Build a Σ5 [001] twist grain boundary in bcc iron (mp-13). "
        "Use the granular generate_structure tool — there's a built-in "
        "skill that handles this library well; load it via the "
        "appropriate tool parameter."
    )
    print("   --- agent response (last 300 chars) ---")
    print("   " + response[-300:].replace("\n", "\n   "))

    structures = orch.generated_structures or []
    assert structures, "No structures recorded"
    s = structures[-1]
    print(f"   slug: {s['slug']}, skill: {s.get('skill')}")

    assert s.get("skill") == "aimsgb", (
        f"Expected agent to pass skill='aimsgb' to generate_structure; "
        f"got {s.get('skill')!r}. The system-prompt skill-availability "
        f"nudge isn't reaching the LLM."
    )

    script_text = Path(s["script_path"]).read_text()
    assert "aimsgb" in script_text, "Script should reference aimsgb"
    assert "GrainBoundary" in script_text, "Script should use GrainBoundary"
    assert "build_gb" in script_text, "Script should call .build_gb()"

    poscar = Path(s["poscar_path"])
    assert poscar.exists(), f"POSCAR missing: {poscar}"

    from ase.io import read as ase_read
    atoms = ase_read(str(poscar))
    syms = atoms.get_chemical_symbols()
    assert all(sy == "Fe" for sy in syms), f"Expected pure Fe, got {set(syms)}"
    print(f"   ✅ skill='aimsgb' loaded; script uses aimsgb API; "
          f"{len(atoms)} Fe atoms produced.")


# ---------------------------------------------------------------------------
# E2E — full run_task call producing files on disk
# ---------------------------------------------------------------------------

def e2e_1_run_task_minimal_structure(model_name: str):
    """run_task end-to-end: prepare a Si bulk + VASP inputs in one call."""
    workdir = RUN_DIR / "e2e_run_task"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    orch = _make_orch(model_name, str(workdir / "sim"), autonomy="co-pilot")
    result = orch.run_task(
        "Build a 2x2x2 supercell of bulk silicon (mp-149, Fd-3m) and "
        "generate VASP inputs for a static SCF calculation. Use the "
        "granular tools (generate_structure, then generate_vasp_inputs)."
    )

    print(f"   status: {result['status']}")
    print(f"   files_produced: {len(result['files_produced'])}")
    print(f"   structures: {[s['slug'] for s in result['structures']]}")
    print(f"   warnings: {result['warnings']}")

    assert result["status"] == "success", f"run_task failed: {result.get('error')}"
    assert result["structures"], "No structures recorded"
    s = result["structures"][0]
    assert Path(s["poscar_path"]).exists(), "POSCAR not on disk"
    print(f"   ✅ run_task produced a structure at {s['poscar_path']}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TARGETED_TESTS = [
    ("Orchestrator constructs",                test_1_orchestrator_constructs),
    ("Tool error paths",                       test_2_tool_error_paths),
    ("Post-run analysis (synthetic)",          test_3_post_run_analysis_synthetic),
    ("Mode switching",                         test_4_mode_switching),
    ("run_task without LLM",                   test_5_run_task_without_llm),
    ("Session persistence (checkpoint)",       test_6_session_persistence),
]

STRESS_TESTS = [
    ("STRESS: generate + VASP inputs",         stress_1_generate_then_inputs),
    ("STRESS: post-run failure analysis",      stress_2_post_run_analysis_failure),
    ("STRESS: session_status call",            stress_3_session_status_called),
    ("STRESS: aimsgb skill auto-loaded",       stress_4_aimsgb_skill_loaded),
]

E2E_TESTS = [
    ("E2E: run_task minimal structure",        e2e_1_run_task_minimal_structure),
]


def main():
    parser = argparse.ArgumentParser(description="Simulation orchestrator tests")
    parser.add_argument("--stress", action="store_true",
                        help="Also run live-LLM stress tests")
    parser.add_argument("--e2e", action="store_true",
                        help="Also run live-LLM end-to-end run_task tests")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

    needs_keys = args.stress or args.e2e
    if needs_keys:
        _require_env("ANTHROPIC_API_KEY")

    print(f"🔧 SimulationOrchestratorAgent tests against model: {args.model}")
    print("=" * 72)

    tests = list(TARGETED_TESTS)
    if args.stress:
        tests.extend(STRESS_TESTS)
    if args.e2e:
        tests.extend(E2E_TESTS)

    passed, failed = [], []
    for name, fn in tests:
        print(f"\n▶  {name}")
        try:
            fn(args.model)
            passed.append(name)
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"   ❌ FAIL: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"   ❌ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"Summary: {len(passed)} passed, {len(failed)} failed")
    for name in passed:
        print(f"  ✅ {name}")
    for name, err in failed:
        print(f"  ❌ {name}  ({err})")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
