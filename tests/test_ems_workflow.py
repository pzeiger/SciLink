"""
Hermetic tests for the EMS wizard's task dispatch (_run_generation).

No Streamlit rendering and no LLM: a fake orchestrator captures the task
string handed to run_task() so we can assert the wizard chains the right
tools for each structure source.

Guards the 2026-06 change that added a third structure source ("Describe
the system"): when the user describes a system instead of supplying a file,
the wizard must instruct the orchestrator to build the structure FIRST
(generate_structure) and only THEN simulate it (generate_ems_simulation),
mirroring the decoupled hand-off the VASP workflow uses.
"""

import pytest

pytest.importorskip("streamlit")  # the wizard module is Streamlit-only


class _FakeOrchestrator:
    """Records the task; simulates the orchestrator emitting an EMS record."""

    def __init__(self):
        self.generated_structures = []
        self.tasks = []

    def run_task(self, task):
        self.tasks.append(task)
        self.generated_structures.append(
            {"type": "ems", "script_path": "/tmp/run_abtem.py"}
        )
        return {}


def _patch_agent(monkeypatch):
    from scilink.ui.components import ems_workflow as w
    agent = _FakeOrchestrator()
    monkeypatch.setattr(w, "_get_or_create_agent", lambda: agent)
    return w, agent


def test_describe_path_builds_structure_before_simulating(monkeypatch):
    w, agent = _patch_agent(monkeypatch)

    rec = w._run_generation(
        structure_file=None,
        system_description="monolayer MoS2",
        research_goal="HAADF at 80 keV",
        structure_description="",
        output_format="npz",
    )

    assert rec is not None and rec["type"] == "ems"
    task = agent.tasks[-1]
    assert "monolayer MoS2" in task
    # Both tools appear, and the build step precedes the simulation step.
    assert "generate_structure" in task
    assert "generate_ems_simulation" in task
    assert task.index("generate_structure") < task.index("generate_ems_simulation")
    # The research goal must be forwarded verbatim (carries instrument params).
    assert "VERBATIM" in task


def test_file_path_skips_structure_build(monkeypatch):
    w, agent = _patch_agent(monkeypatch)

    rec = w._run_generation(
        structure_file="/data/Si.vasp",
        system_description="",
        research_goal="HAADF at 200 keV",
        structure_description="3x3x1 supercell",
        output_format="zarr",
    )

    assert rec is not None and rec["type"] == "ems"
    task = agent.tasks[-1]
    # An existing file goes straight to the EMS tool — no structure build.
    assert "generate_structure" not in task
    assert "/data/Si.vasp" in task
    assert "generate_ems_simulation" in task
    # Prep directive and output format are threaded into the tool call.
    assert "structure_description='3x3x1 supercell'" in task
    assert "output_format='zarr'" in task


def test_no_ems_record_returns_none(monkeypatch):
    """If the orchestrator produces no EMS record, the wizard reports nothing."""
    from scilink.ui.components import ems_workflow as w

    class _Empty:
        generated_structures: list = []

        def run_task(self, task):
            return {}

    monkeypatch.setattr(w, "_get_or_create_agent", lambda: _Empty())
    rec = w._run_generation(
        structure_file="/data/Si.vasp",
        system_description="",
        research_goal="HAADF",
        structure_description="",
        output_format="npz",
    )
    assert rec is None
