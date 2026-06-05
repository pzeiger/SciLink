"""Unit tests for the engine-neutral refinement loop.

Drives ``run_refinement`` with a fake executor and a scripted fake critic —
no LLM, no simulation engine, no API keys. Verifies the loop's control flow
(cycle counting, multi-phase chaining, fix application) and the three
autonomy policies' decisions.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.refinement import (  # noqa: E402
    AutonomousPolicy,
    AutopilotPolicy,
    CoPilotPolicy,
    CycleDecision,
    Executor,
    Phase,
    RefinementContext,
    run_refinement,
    policy_for,
)


# ──────────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────────

class FakeExecutor(Executor):
    """Records each run and reports completion without running anything."""

    def __init__(self):
        self.calls = []

    def run(self, input_files, run_command, run_dir):
        self.calls.append({
            "input_files": dict(input_files),
            "run_command": run_command,
            "run_dir": run_dir,
        })
        return {"status": "completed", "output_dir": run_dir, "returncode": 0}


class ScriptedCritic:
    """Returns a pre-set sequence of verdicts, one per assess() call."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0

    def assess(self, output_dir, research_goal, skill=None, domain=None,
               fixes_mode="auto"):
        self.calls += 1
        # Repeat the last verdict if the loop asks for more than scripted.
        idx = min(self.calls - 1, len(self._verdicts) - 1)
        return dict(self._verdicts[idx])


def _good():
    return {"status": "success", "run_status": "succeeded", "verdict": "good",
            "suggested_fixes": None}


def _needs_fixes(files=None):
    return {"status": "success", "run_status": "failed", "verdict": "needs_fixes",
            "suggested_fixes": files or {"in.sim": "patched contents"}}


def _poor(files=None):
    return {"status": "success", "run_status": "succeeded", "verdict": "poor",
            "suggested_fixes": files or {"in.sim": "patched"}}


def _warning(files=None):
    # files=None → benign warning (critic offers no fix); files set → fixable.
    return {"status": "success", "run_status": "succeeded", "verdict": "warning",
            "suggested_fixes": files}


def _phase(name="production", run_dir="/tmp/rf_test_phase"):
    return Phase(name=name, input_files={"in.sim": "original"},
                 run_command="true", run_dir=run_dir)


def _ctx(autonomy="autonomous", max_cycles=3, interact=None):
    return RefinementContext(
        research_goal="compute something", scale="molecular_dynamics",
        engine="lammps", skill="lammps", domain="molecular_dynamics",
        autonomy=autonomy, max_cycles=max_cycles, interact=interact,
    )


# ──────────────────────────────────────────────────────────────────────────
# Core loop behavior (autonomous)
# ──────────────────────────────────────────────────────────────────────────

class TestLoopFlow:
    def test_converges_first_try(self):
        ex = FakeExecutor()
        critic = ScriptedCritic([_good()])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert len(ex.calls) == 1
        assert critic.calls == 1
        assert result["phases"][0]["verdict"] == "good"

    def test_fail_fix_succeed(self):
        ex = FakeExecutor()
        critic = ScriptedCritic([_needs_fixes({"in.sim": "fixed"}), _good()])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        # Two executions: original, then the fixed inputs.
        assert len(ex.calls) == 2
        assert ex.calls[0]["input_files"] == {"in.sim": "original"}
        assert ex.calls[1]["input_files"] == {"in.sim": "fixed"}

    def test_stall_stops_when_not_improving(self):
        # A verdict that never improves stops on the stall check (after one
        # non-improving cycle), not by burning the whole cycle budget.
        ex = FakeExecutor()
        critic = ScriptedCritic([_needs_fixes()])  # never improves
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(),
                                _ctx(max_cycles=5))
        assert result["status"] == "failed"
        assert len(ex.calls) == 2  # cycle 0 refines, cycle 1 stalls → stop
        assert result["phases"][0]["status"] == "stopped"

    def test_fixable_warning_is_refined(self):
        # A warning the critic offers a fix for is pursued toward good, not
        # accepted as-is.
        ex = FakeExecutor()
        critic = ScriptedCritic([_warning({"in.sim": "smaller timestep"}), _good()])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert len(ex.calls) == 2  # warning → fix → good
        assert ex.calls[1]["input_files"] == {"in.sim": "smaller timestep"}

    def test_benign_warning_stops_as_success(self):
        # A warning with no proposed fix is benign — stop immediately, and it
        # counts as success (acceptable terminal state).
        ex = FakeExecutor()
        critic = ScriptedCritic([_warning(None)])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert len(ex.calls) == 1
        assert result["phases"][0]["status"] == "success"
        assert result["phases"][0]["verdict"] == "warning"

    def test_no_fixes_stops_even_if_not_good(self):
        # A poor verdict with no actionable fixes should stop, not spin.
        ex = FakeExecutor()
        critic = ScriptedCritic([{"verdict": "poor", "run_status": "succeeded",
                                  "suggested_fixes": None}])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert len(ex.calls) == 1
        assert result["phases"][0]["status"] == "stopped"

    def test_multi_phase_chains(self):
        ex = FakeExecutor()
        # phase 1 good first try; phase 2 needs a fix then good.
        critic = ScriptedCritic([_good(), _needs_fixes({"in.sim": "f"}), _good()])
        phases = [_phase("equil", "/tmp/rf_eq"), _phase("prod", "/tmp/rf_prod")]
        result = run_refinement(phases, ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert [p["phase"] for p in result["phases"]] == ["equil", "prod"]
        # equil: 1 run; prod: 2 runs.
        assert len(ex.calls) == 3
        assert {c["run_dir"] for c in ex.calls} == {"/tmp/rf_eq", "/tmp/rf_prod"}

    def test_cycle_resets_per_phase(self):
        ex = FakeExecutor()
        critic = ScriptedCritic([_good(), _good()])
        ctx = _ctx()
        run_refinement([_phase("a", "/tmp/a"), _phase("b", "/tmp/b")], ex,
                       critic, AutonomousPolicy(), ctx)
        # History has one entry per phase, each at cycle 0.
        assert [h["cycle"] for h in ctx.history] == [0, 0]
        assert [h["phase"] for h in ctx.history] == ["a", "b"]

    def test_empty_phases_fails_cleanly(self):
        result = run_refinement([], FakeExecutor(), ScriptedCritic([_good()]),
                                AutonomousPolicy(), _ctx())
        assert result["status"] == "failed"


# ──────────────────────────────────────────────────────────────────────────
# Policies
# ──────────────────────────────────────────────────────────────────────────

class TestPolicies:
    def test_copilot_abort_via_interact(self):
        # Human says "no" at the pre-run gate → aborted, nothing runs.
        answers = iter(["no"])
        ex = FakeExecutor()
        ctx = _ctx(autonomy="co-pilot",
                   interact=lambda prompt, payload: next(answers))
        result = run_refinement([_phase()], ex, ScriptedCritic([_good()]),
                                CoPilotPolicy(), ctx)
        assert result["status"] == "aborted"
        assert len(ex.calls) == 0

    def test_copilot_approves_then_runs(self):
        answers = iter(["yes", "yes"])  # approve pre-run, then approve a fix
        ex = FakeExecutor()
        ctx = _ctx(autonomy="co-pilot",
                   interact=lambda prompt, payload: next(answers, "yes"))
        critic = ScriptedCritic([_needs_fixes({"in.sim": "f"}), _good()])
        result = run_refinement([_phase()], ex, critic, CoPilotPolicy(), ctx)
        assert result["status"] == "success"
        assert len(ex.calls) == 2

    def test_copilot_headless_degrades(self):
        # No interact handle → co-pilot must not block; behaves autonomously.
        ex = FakeExecutor()
        critic = ScriptedCritic([_needs_fixes({"in.sim": "f"}), _good()])
        result = run_refinement([_phase()], ex, critic, CoPilotPolicy(),
                                _ctx(autonomy="co-pilot"))
        assert result["status"] == "success"
        assert len(ex.calls) == 2

    def test_autopilot_stops_when_stalled(self):
        # Two non-improving cycles → autopilot stops before exhausting budget.
        ex = FakeExecutor()
        critic = ScriptedCritic([_poor()])  # always poor, never improves
        result = run_refinement([_phase()], ex, critic, AutopilotPolicy(),
                                _ctx(autonomy="autopilot", max_cycles=5))
        # Runs cycle 0 (poor), cycle 1 (poor → stalled) then stops.
        assert len(ex.calls) == 2
        assert result["phases"][0]["status"] == "stopped"

    def test_autopilot_surfaces_failing_prerun(self):
        # A "fails" pre-run verdict with an interact handle prompts the human.
        seen = {}
        def interact(prompt, payload):
            seen["asked"] = True
            return "no"
        ex = FakeExecutor()
        ctx = _ctx(autonomy="autopilot", interact=interact)
        result = run_refinement(
            [_phase()], ex, ScriptedCritic([_good()]), AutopilotPolicy(), ctx,
            pre_run_verdict={"validation_status": "fails"},
        )
        assert seen.get("asked") is True
        assert result["status"] == "aborted"
        assert len(ex.calls) == 0

    def test_policy_for_mapping(self):
        assert isinstance(policy_for("co-pilot"), CoPilotPolicy)
        assert isinstance(policy_for("autopilot"), AutopilotPolicy)
        assert isinstance(policy_for("autonomous"), AutonomousPolicy)
        # Unknown label defaults to autonomous (safe for headless callers).
        assert isinstance(policy_for("nonsense"), AutonomousPolicy)
        assert isinstance(policy_for(""), AutonomousPolicy)


class TestCollectPhases:
    """The pipeline's generation-result → Phase normalization (engine-neutral)."""

    def test_single_phase_from_entry_file(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        gr = {"input_files": {"run.lammps": "units metal\n"}, "entry_file": "run.lammps"}
        phases = _collect_phases(gr, "/tmp/runX", "lmp -in {script}")
        assert len(phases) == 1
        assert phases[0].run_command == "lmp -in run.lammps"
        assert phases[0].run_dir == "/tmp/runX"

    def test_entry_synthesized_from_lone_input_file(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        p = _collect_phases({"input_files": {"in.lmp": "x"}}, "/tmp/y",
                            "lmp -in {script}")[0]
        assert p.run_command == "lmp -in in.lmp"

    def test_explicit_multiphase_passthrough(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        gr = {"phases": [
            {"name": "equil", "input_files": {"eq.lmp": "a"}, "entry_file": "eq.lmp"},
            {"name": "prod", "input_files": {"pr.lmp": "b"}, "entry_file": "pr.lmp"},
        ]}
        ps = _collect_phases(gr, "/tmp/z", "lmp -in {script}")
        assert [p.name for p in ps] == ["equil", "prod"]
        assert ps[1].run_command == "lmp -in pr.lmp"

    def test_template_without_placeholder_used_verbatim(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        gr = {"input_files": {"run.lammps": "x"}, "entry_file": "run.lammps"}
        assert _collect_phases(gr, "/tmp/y", "run_md.sh")[0].run_command == "run_md.sh"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
