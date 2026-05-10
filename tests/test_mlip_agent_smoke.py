"""
Local smoke tests for MLIPAgent assembly + LAMMPS input generation.

These cover the relocation/wiring contract without making any LLM
calls and without invoking real MACE / NequIP / DeePMD backends.

What they exercise:
  - MLIPAgent constructs and auto-loads the `general` skill from the
    machine_learning_potentials bundle directory.
  - _load_backend_skill('mace') merges the backend bundle into
    skill_sections (each section now contains a `--- MACE SPECIFIC ---`
    block).
  - mlip_tools.generate_lammps_input emits a syntactically plausible
    LAMMPS input file for the MACE pair_style, in `metal` units.
"""

import os
import tempfile
from pathlib import Path


def _agent_kwargs():
    return dict(api_key="sk-smoke-not-real", model_name="gpt-4o-mini")


def test_mlip_agent_assembly():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, **_agent_kwargs())

        assert agent.skill_name == "general", (
            f"Expected `general` skill to auto-load, got {agent.skill_name!r}. "
            f"If this fails, the skill loader's domain mapping has drifted."
        )
        assert agent.skill_sections is not None
        for section in ("planning", "validation", "implementation"):
            content = agent.skill_sections.get(section, "")
            assert content, (
                f"`general` skill is missing the {section!r} section -- "
                f"the agent will hand empty context to the LLM."
            )

        ctx = agent._get_skill_context(section="validation")
        assert "MLIP" in ctx or "Energy MAE" in ctx, (
            "validation context did not include the MLIP rules"
        )


def test_mlip_backend_skill_merge():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, skill="mace", **_agent_kwargs())

        impl = agent.skill_sections.get("implementation", "")
        assert "MACE SPECIFIC" in impl, (
            "Backend skill merge marker missing. _load_backend_skill should "
            "append a `--- MACE SPECIFIC ---` block onto each section."
        )
        assert "pair_style" in impl and "mace" in impl, (
            "MACE LAMMPS pair_style guidance missing from merged context"
        )

        planning = agent.skill_sections.get("planning", "")
        assert "mace-mp-0" in planning or "mace-off23" in planning, (
            "MACE foundation-model selection guidance not merged into planning"
        )


def test_mlip_lammps_input_generation():
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        path = mlip_tools.generate_lammps_input(
            backend="mace",
            model_file="/path/to/mace-mp-0.model",
            elements=["Cu"],
            working_dir=td,
            timestep=0.5,
            temperature=300.0,
            pressure=None,
        )

        assert Path(path).exists()
        content = Path(path).read_text()

        for required in (
            "units          metal",
            "atom_style     atomic",
            "pair_style     mace no_domain_decomposition",
            "pair_coeff     * * /path/to/mace-mp-0.model Cu",
            "fix 1 all nvt",
            "thermo_style",
            "dump",
        ):
            assert required in content, (
                f"Generated LAMMPS input missing required line: {required!r}\n"
                f"--- content ---\n{content}"
            )

        assert "npt" not in content, (
            "NVT ensemble requested (pressure=None) but NPT slipped in"
        )

        path_npt = mlip_tools.generate_lammps_input(
            backend="mace",
            model_file="/path/to/mace-mp-0.model",
            elements=["Cu"],
            working_dir=td,
            timestep=0.5,
            temperature=300.0,
            pressure=1.0,
        )
        npt_content = Path(path_npt).read_text()
        assert "fix 1 all npt" in npt_content, (
            "pressure=1.0 should produce an NPT fix line"
        )


def test_mlip_unknown_backend_raises():
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        try:
            mlip_tools.generate_lammps_input(
                backend="not_a_real_backend",
                model_file="/tmp/x.model",
                elements=["Cu"],
                working_dir=td,
            )
        except ValueError as exc:
            assert "Unknown backend" in str(exc)
        else:
            raise AssertionError("Unknown backend should raise ValueError")


if __name__ == "__main__":
    print("=== smoke 1: MLIPAgent assembly + skill load ===")
    test_mlip_agent_assembly()
    print("  OK")
    print()
    print("=== smoke 2: MACE backend skill merge ===")
    test_mlip_backend_skill_merge()
    print("  OK")
    print()
    print("=== smoke 3: MACE LAMMPS input generation ===")
    test_mlip_lammps_input_generation()
    print("  OK")
    print()
    print("=== smoke 4: unknown backend raises ===")
    test_mlip_unknown_backend_raises()
    print("  OK")
    print()
    print("All smokes passed.")
