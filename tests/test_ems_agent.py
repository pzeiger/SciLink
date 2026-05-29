"""
Hermetic tests for EMSAgent (electron microscopy simulation, abTEM engine).

No live LLM calls. Two surfaces are exercised:

  1. Parameter precedence in plan_simulation — caller-supplied instrument
     settings (beam energy, semiangle) must OVERRIDE the planning LLM's own
     choices, while genuinely-unspecified values let the LLM decide. This is
     the regression guard for the bug where setdefault() silently dropped the
     UI-specified energy/semiangle. The planning LLM is mocked.

  2. The deterministic geometry validator (_validate_geometry) — pure physics,
     no LLM at all. Antialiasing (sampling vs. max scattering angle), detector
     angle ordering, frozen-phonon count, and the reported diagnostics.

  3. Assembly smoke — construction, skill auto-loading, supported_software(),
     and the ASE-backed analyze_system() fast path. Gated on ase.

The agent is constructed with a fake API key (construction-only; no network
call is made), mirroring tests/test_md_agent_smoke.py.
"""

import tempfile
from pathlib import Path

import pytest


# A small orthogonal Si cell (VASP POSCAR), enough for analyze_system.
SI_POSCAR = """\
Si test cell
1.0
5.43 0.00 0.00
0.00 5.43 0.00
0.00 0.00 5.43
Si
8
Direct
0.00 0.00 0.00
0.50 0.50 0.00
0.50 0.00 0.50
0.00 0.50 0.50
0.25 0.25 0.25
0.75 0.75 0.25
0.75 0.25 0.75
0.25 0.75 0.75
"""

# Minimal system_info accepted by plan_simulation / _validate_geometry.
SYSTEM_INFO = {
    "element_counts": {"Si": 8},
    "atom_count": 8,
    "elements": ["Si"],
    "lateral_extent_a": 10.86,
    "lateral_extent_b": 10.86,
    "thickness": 21.72,
    "is_orthogonal": True,
}


def _make_agent(workdir, **kwargs):
    """Construct an EMSAgent with a fake key (no LLM call at construction)."""
    from scilink.agents.sim_agents.ems_agent import EMSAgent
    return EMSAgent(
        working_dir=str(workdir),
        api_key="sk-smoke-not-real",
        model_name="gpt-4o-mini",
        **kwargs,
    )


# ─── 1. Parameter precedence (the regression guard) ────────────────


def test_caller_params_override_llm_choice(tmp_path):
    """beam_energy_kev / semiangle_mrad passed by the caller win over the LLM."""
    agent = _make_agent(tmp_path)
    # Planning LLM proposes its own (different) instrument settings — exactly
    # the failure mode: it returns 200 keV / 21 mrad regardless of the request.
    agent._generate_json = lambda prompt: {
        "technique": "multislice",
        "beam_energy_ev": 200000,
        "semiangle_mrad": 21.0,
        "sampling_angstrom": 0.05,
    }

    plan = agent.plan_simulation(
        "HAADF at 60 keV", SYSTEM_INFO,
        beam_energy_kev=60.0, semiangle_mrad=33.5,
    )

    assert plan["beam_energy_ev"] == 60000.0, (
        f"caller beam energy not honored: {plan['beam_energy_ev']}"
    )
    assert plan["semiangle_mrad"] == 33.5, (
        f"caller semiangle not honored: {plan['semiangle_mrad']}"
    )


def test_llm_choice_kept_when_caller_omits(tmp_path):
    """With no caller-supplied instrument settings, the LLM's choice stands."""
    agent = _make_agent(tmp_path)
    agent._generate_json = lambda prompt: {
        "beam_energy_ev": 80000,
        "semiangle_mrad": 25.0,
    }

    plan = agent.plan_simulation("some goal", SYSTEM_INFO)

    assert plan["beam_energy_ev"] == 80000
    assert plan["semiangle_mrad"] == 25.0


def test_fixed_instrument_settings_injected_into_prompt(tmp_path):
    """Caller-pinned settings are surfaced to the LLM so it plans the dependent
    parameters (sampling, detector angles) consistently around them."""
    agent = _make_agent(tmp_path)
    captured = {}

    def _capture(prompt):
        captured["prompt"] = prompt
        return {"beam_energy_ev": 200000, "semiangle_mrad": 21.0}

    agent._generate_json = _capture
    agent.plan_simulation(
        "goal", SYSTEM_INFO, beam_energy_kev=60.0, semiangle_mrad=33.5,
    )

    prompt = captured["prompt"]
    assert "60000 eV" in prompt
    assert "33.5 mrad" in prompt
    assert "FIXED" in prompt


def test_plan_is_complete_even_when_llm_returns_nothing(tmp_path):
    """A failed/empty LLM response still yields a complete, runnable plan."""
    agent = _make_agent(tmp_path)
    agent._generate_json = lambda prompt: {}

    plan = agent.plan_simulation("goal", SYSTEM_INFO)

    for key in (
        "technique", "beam_energy_ev", "semiangle_mrad", "sampling_angstrom",
        "slice_thickness_angstrom", "detector_type", "detector_inner_mrad",
        "detector_outer_mrad", "frozen_phonon_configs", "use_prism",
        "output_format",
    ):
        assert key in plan, f"plan missing fallback key {key!r}"


# ─── 2. Deterministic geometry validator (no LLM) ──────────────────


def _good_plan(**overrides):
    plan = {
        "beam_energy_ev": 200000,
        "sampling_angstrom": 0.04,
        "semiangle_mrad": 20,
        "detector_inner_mrad": 50,
        "detector_outer_mrad": 150,
        "slice_thickness_angstrom": 2.0,
        "frozen_phonon_configs": 8,
    }
    plan.update(overrides)
    return plan


def test_geometry_validator_passes_for_good_plan(tmp_path):
    agent = _make_agent(tmp_path)
    result = agent._validate_geometry(SYSTEM_INFO, _good_plan())
    assert result["valid"] is True
    assert result["errors"] == []


def test_geometry_validator_flags_undersampling(tmp_path):
    """Coarse sampling violates antialiasing and suggests a finer value."""
    agent = _make_agent(tmp_path)
    result = agent._validate_geometry(
        SYSTEM_INFO, _good_plan(sampling_angstrom=0.5),
    )
    assert result["valid"] is False
    assert any("sampling" in e.lower() for e in result["errors"])
    adjustments = result["suggested_adjustments"]
    assert any(a["parameter"] == "sampling_angstrom" for a in adjustments)
    # The suggested sampling must be finer (smaller) than the offending one.
    adj = next(a for a in adjustments if a["parameter"] == "sampling_angstrom")
    assert adj["suggested_value"] < 0.5


def test_geometry_validator_rejects_inverted_detector(tmp_path):
    agent = _make_agent(tmp_path)
    result = agent._validate_geometry(
        SYSTEM_INFO,
        _good_plan(detector_inner_mrad=150, detector_outer_mrad=50),
    )
    assert result["valid"] is False
    assert any(
        "inner" in e.lower() and "outer" in e.lower()
        for e in result["errors"]
    )


def test_geometry_validator_flags_zero_frozen_phonons(tmp_path):
    agent = _make_agent(tmp_path)
    result = agent._validate_geometry(
        SYSTEM_INFO, _good_plan(frozen_phonon_configs=0),
    )
    assert result["valid"] is False
    assert any("frozen_phonon" in e.lower() for e in result["errors"])


def test_geometry_validator_warns_on_low_frozen_phonons(tmp_path):
    """1–3 configs is allowed but should warn (not error)."""
    agent = _make_agent(tmp_path)
    result = agent._validate_geometry(
        SYSTEM_INFO, _good_plan(frozen_phonon_configs=2),
    )
    assert result["valid"] is True
    assert any("frozen-phonon" in w.lower() or "frozen phonon" in w.lower()
               for w in result["warnings"])


def test_geometry_validator_reports_wavelength_diagnostic(tmp_path):
    """200 keV electrons have a relativistic wavelength of ~0.0251 Å."""
    agent = _make_agent(tmp_path)
    result = agent._validate_geometry(SYSTEM_INFO, _good_plan())
    diag = result["diagnostics"]
    assert "wavelength_angstrom" in diag
    assert 0.024 < diag["wavelength_angstrom"] < 0.026
    assert "max_representable_angle_mrad" in diag


def test_geometry_validator_warns_on_non_orthogonal_cell(tmp_path):
    agent = _make_agent(tmp_path)
    si = dict(SYSTEM_INFO, is_orthogonal=False)
    result = agent._validate_geometry(si, _good_plan())
    assert any("orthogonal" in w.lower() for w in result["warnings"])


# ─── 3. Assembly smoke (ASE-backed analyze_system) ─────────────────


def test_supported_software_lists_abtem():
    from scilink.agents.sim_agents.ems_agent import EMSAgent
    assert EMSAgent.supported_software() == ["abtem"]


def test_ems_agent_assembly(tmp_path):
    pytest.importorskip("ase")
    from scilink.agents.sim_agents import EMSAgent

    agent = _make_agent(tmp_path, skill="abtem")

    assert agent.SKILL_DOMAIN == "electron_microscopy_simulation"
    assert agent.skill_name == "abtem"
    for section in ("overview", "planning", "implementation", "validation"):
        assert agent._get_skill_context(section=section), (
            f"skill section {section!r} should be non-empty"
        )

    poscar = tmp_path / "Si.vasp"
    poscar.write_text(SI_POSCAR)
    info = agent.analyze_system(str(poscar))

    assert info["atom_count"] == 8, f"expected 8 atoms, got {info}"
    assert info["elements"] == ["Si"]
    assert info["is_orthogonal"] is True
    assert abs(info["thickness"] - 5.43) < 1e-3


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        print("=== precedence: caller overrides LLM ===")
        test_caller_params_override_llm_choice(tp)
        test_llm_choice_kept_when_caller_omits(tp)
        test_fixed_instrument_settings_injected_into_prompt(tp)
        test_plan_is_complete_even_when_llm_returns_nothing(tp)
        print("  OK")
        print("=== geometry validator ===")
        test_geometry_validator_passes_for_good_plan(tp)
        test_geometry_validator_flags_undersampling(tp)
        test_geometry_validator_rejects_inverted_detector(tp)
        test_geometry_validator_flags_zero_frozen_phonons(tp)
        test_geometry_validator_warns_on_low_frozen_phonons(tp)
        test_geometry_validator_reports_wavelength_diagnostic(tp)
        test_geometry_validator_warns_on_non_orthogonal_cell(tp)
        print("  OK")
        print("=== supported_software ===")
        test_supported_software_lists_abtem()
        print("  OK")
        print("All non-ASE EMS tests passed.")
