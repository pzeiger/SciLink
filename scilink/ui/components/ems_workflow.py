"""EMS (Electron Microscopy Simulation) wizard — Configure → Review.

Drives SimulationOrchestratorAgent.run_task() via the generate_ems_simulation
tool to produce a runnable abTEM script from an uploaded structure file.
Mirrors the vasp_workflow.py shape: a two-phase Streamlit state machine
(configure → review) with a shared agent instance cached in session state.

Submission is lighter than VASP: the output is a self-contained Python
script (run_abtem.py) that the user runs locally or on a GPU node with
`python run_abtem.py`. No binary path assembly or POTCAR is required.
For HPC GPU submission the user can use the generic Submit tab in the
main Simulations view.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import streamlit as st

from scilink.ui.components.wizard_state import apply_ems_defaults, save_ems


# ══════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════

def render_agent_workflow() -> None:
    """Main entry point — dispatches to the current workflow phase."""
    phase = st.session_state.get("hpc_workflow_phase", "configure")

    if phase == "configure":
        _render_configure()
    elif phase == "review":
        _render_review()
    else:
        st.session_state.hpc_workflow_phase = "configure"
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Phase 1: Configure
# ══════════════════════════════════════════════════════════════

def _render_configure() -> None:
    apply_ems_defaults()

    st.subheader("🔬 Electron Microscopy Simulation (abTEM)")

    cfg = st.session_state.get("agent_config", {})
    _has_key = bool(
        cfg.get("api_key")
        or os.environ.get("SCILINK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not _has_key:
        st.warning(
            "No LLM API key detected. Launch a simulate session from the "
            "sidebar with a key set, or export SCILINK_API_KEY before starting."
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Structure input
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("**Structure**")

    structure_source = st.radio(
        "Source",
        ["Upload file", "Use session structure"],
        horizontal=True,
        key="ems_structure_source",
    )

    structure_file_path: Optional[str] = None

    if structure_source == "Upload file":
        uploaded = st.file_uploader(
            "Structure file (CIF, VASP POSCAR, XYZ, extXYZ)",
            type=["cif", "vasp", "xyz", "extxyz", "poscar", "cfg"],
            key="ems_structure_upload",
        )
        if uploaded is not None:
            save_dir = Path(
                st.session_state.get("session_dir", "./simulate_session")
            ) / "ems_uploads"
            save_dir.mkdir(parents=True, exist_ok=True)
            dest = save_dir / uploaded.name
            dest.write_bytes(uploaded.getvalue())
            structure_file_path = str(dest)
            st.caption(f"Saved to `{dest}`")
    else:
        # Let the user pick from structures generated earlier in this session
        structures = [
            s for s in (st.session_state.get("hpc_sim_structures") or [])
            if s.get("poscar_path") and Path(s["poscar_path"]).exists()
        ]
        if not structures:
            st.info(
                "No structures in the current session yet. "
                "Generate one in the VASP or LAMMPS workflow, "
                "or upload a file instead."
            )
        else:
            options = {s["slug"]: s["poscar_path"] for s in structures}
            chosen_slug = st.selectbox(
                "Session structure",
                list(options.keys()),
                key="ems_session_slug",
            )
            structure_file_path = options[chosen_slug]
            st.caption(f"`{structure_file_path}`")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Research goal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("**Research goal**")
    research_goal = st.text_area(
        "Describe the simulation",
        height=100,
        key="ems_research_goal",
        placeholder=(
            "e.g. HAADF STEM image of Si along [001] at 200 keV\n"
            "e.g. 4D-STEM datacube of ZnO for ptychographic reconstruction\n"
            "e.g. CBED diffraction pattern of MgO along [110]"
        ),
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EM parameters
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("**Simulation parameters** (the agent sets these from the goal; override here if needed)")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        beam_energy_kev = st.number_input(
            "Beam energy (keV)",
            min_value=20.0,
            max_value=1000.0,
            value=200.0,
            step=10.0,
            key="ems_beam_energy_kev",
            help="Accelerating voltage. Typical: 60–80 keV (beam-sensitive), 200 keV (general), 300 keV (hard materials).",
        )
    with pc2:
        semiangle_mrad = st.number_input(
            "Probe semiangle (mrad)",
            min_value=1.0,
            max_value=60.0,
            value=20.0,
            step=1.0,
            key="ems_semiangle_mrad",
            help="Convergence semi-angle of the objective aperture. Typical: 10–15 mrad (uncorrected), 20–30 mrad (aberration-corrected).",
        )
    with pc3:
        output_format = st.selectbox(
            "Output format",
            ["npz", "zarr"],
            key="ems_output_format",
            help="npz: NumPy archive, compact. zarr: chunked, lazy, preferred for 4D-STEM / TACAW.",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Generate button
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    can_proceed = bool(
        structure_file_path
        and research_goal.strip()
        and _has_key
    )
    if not can_proceed:
        missing: list[str] = []
        if not structure_file_path:
            missing.append("structure file")
        if not research_goal.strip():
            missing.append("research goal")
        if not _has_key:
            missing.append("LLM API key")
        st.info(f"Still needed: {', '.join(missing)}")

    err_slot = st.empty()

    if st.button(
        "🔬 Generate EMS script",
        type="primary",
        disabled=not can_proceed,
        key="ems_generate_btn",
    ):
        save_ems()
        try:
            with st.spinner(
                "Asking SimulationOrchestratorAgent to generate abTEM script…"
            ):
                result = _run_generation(
                    structure_file=structure_file_path,
                    research_goal=research_goal.strip(),
                    beam_energy_kev=float(beam_energy_kev),
                    semiangle_mrad=float(semiangle_mrad),
                    output_format=output_format,
                )
        except Exception as exc:
            err_slot.error(f"Generation failed: {exc}")
            return

        if result is None:
            err_slot.error(
                "Agent returned no EMS result. Check the orchestrator logs. "
                "The agent may have asked a clarifying question."
            )
            return

        script_path = result.get("script_path")
        prepped_path = result.get("prepped_structure_path")

        try:
            script_text = Path(script_path).read_text() if script_path else ""
            prepped_text = Path(prepped_path).read_text() if prepped_path and Path(prepped_path).exists() else ""
        except Exception as exc:
            err_slot.error(f"Could not read generated files: {exc}")
            return

        st.session_state.hpc_gen_ems_result = result
        st.session_state.hpc_gen_ems_script = script_text
        st.session_state.hpc_gen_ems_prepped = prepped_text
        st.session_state.hpc_workflow_phase = "review"
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Agent construction + task dispatch
# ══════════════════════════════════════════════════════════════

def _get_or_create_agent():
    """Lazily build SimulationOrchestratorAgent, cached in session state."""
    if (
        "hpc_ems_agent" in st.session_state
        and st.session_state.hpc_ems_agent is not None
    ):
        return st.session_state.hpc_ems_agent

    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent,
        SimulationMode,
    )

    cfg = st.session_state.get("agent_config", {})
    agent = SimulationOrchestratorAgent(
        base_dir=st.session_state.get("session_dir", "./simulate_session"),
        api_key=cfg.get("api_key") or None,
        model_name=cfg.get("model") or "claude-opus-4-6",
        base_url=cfg.get("base_url") or None,
        simulation_mode=SimulationMode.CO_PILOT,
        hpc_connection=st.session_state.get("hpc_connection"),
        hpc_scheduler=st.session_state.get("hpc_scheduler"),
    )
    st.session_state.hpc_ems_agent = agent
    return agent


def _run_generation(
    *,
    structure_file: str,
    research_goal: str,
    beam_energy_kev: float,
    semiangle_mrad: float,
    output_format: str,
) -> Optional[dict]:
    """Call agent.run_task() with an EMS-focused prompt; return the EMS record."""
    agent = _get_or_create_agent()

    task = textwrap.dedent(f"""\
        Generate an abTEM electron microscopy simulation script for the
        following objective.

        Structure file: {structure_file}
        Research goal: {research_goal}
        Beam energy: {beam_energy_kev} keV
        Probe semiangle: {semiangle_mrad} mrad
        Output format: {output_format}

        Steps:
        1. Call generate_ems_simulation with the structure_file path above,
           the research_goal, beam_energy_kev={beam_energy_kev},
           semiangle_mrad={semiangle_mrad}, output_format={output_format!r}.
        2. Return when run_abtem.py is written. Do NOT attempt to execute
           the script — the user will run it locally or on a GPU node.
    """)

    n_before = len(agent.generated_structures)
    agent.run_task(task)

    new = [
        s for s in agent.generated_structures[n_before:]
        if s.get("type") == "ems"
    ]
    return new[-1] if new else None


# ══════════════════════════════════════════════════════════════
# Phase 2: Review
# ══════════════════════════════════════════════════════════════

def _render_review() -> None:
    st.subheader("📝 Review EMS Script")
    st.caption(
        "Inspect and edit the generated script before running. "
        "Verify all abTEM API calls against the installed version."
    )

    result = st.session_state.get("hpc_gen_ems_result", {})
    script_text = st.session_state.get("hpc_gen_ems_script", "")
    prepped_text = st.session_state.get("hpc_gen_ems_prepped", "")

    # Geometry validation summary
    geo = result.get("geometry_validation") or {}
    if geo.get("errors"):
        st.error("**Geometry validation errors:** " + " · ".join(geo["errors"]))
    if geo.get("warnings"):
        st.warning("**Geometry warnings:** " + " · ".join(geo["warnings"]))
    if geo.get("valid") and not geo.get("warnings"):
        st.success("Geometry validation passed.")

    diag = (geo.get("diagnostics") or {})
    if diag:
        dc1, dc2 = st.columns(2)
        with dc1:
            st.metric("Wavelength (Å)", diag.get("wavelength_angstrom", "—"))
        with dc2:
            st.metric("Max representable angle (mrad)", diag.get("max_representable_angle_mrad", "—"))

    params = result.get("simulation_parameters") or {}
    if params:
        with st.expander("Simulation parameters chosen by agent"):
            import json
            st.json(params)

    st.markdown("---")

    tab_script, tab_structure = st.tabs(["run_abtem.py", "structure_prepped.vasp"])

    with tab_script:
        script_edit = st.text_area(
            "run_abtem.py",
            value=script_text,
            height=500,
            key="ems_edit_script",
        )

    with tab_structure:
        if prepped_text:
            st.text_area(
                "structure_prepped.vasp",
                value=prepped_text,
                height=300,
                key="ems_edit_prepped",
                disabled=True,
            )
        else:
            st.caption("Prepared structure file not available.")

    st.markdown("---")

    # Download buttons
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇ Download run_abtem.py",
            data=script_edit,
            file_name="run_abtem.py",
            mime="text/x-python",
            key="ems_dl_script",
        )
    with dl2:
        if prepped_text:
            st.download_button(
                "⬇ Download structure_prepped.vasp",
                data=prepped_text,
                file_name="structure_prepped.vasp",
                mime="text/plain",
                key="ems_dl_structure",
            )
    with dl3:
        run_instructions = textwrap.dedent(f"""\
            # How to run this simulation
            # Requirements: pip install scilink[ems]  (installs abtem + ase)

            # 1. Copy run_abtem.py and structure_prepped.vasp to the same directory.
            # 2. Run:
            python run_abtem.py

            # 3. Output will be written to:
            #    {result.get('output_path', 'measurement.npz')}

            # GPU acceleration (optional — requires CuPy):
            #   ABTEM_DEVICE=gpu python run_abtem.py

            # For HPC GPU submission, use the Submit tab in the Simulations view
            # and adapt the GPU template to run: python run_abtem.py
        """)
        st.download_button(
            "⬇ Download run_instructions.txt",
            data=run_instructions,
            file_name="run_instructions.txt",
            mime="text/plain",
            key="ems_dl_instructions",
        )

    st.info(
        f"**Output will be written to:** `{result.get('output_path', 'measurement.npz')}`  \n"
        "Run with `python run_abtem.py` (CPU) or `ABTEM_DEVICE=gpu python run_abtem.py` (GPU).  \n"
        "For HPC: use the **Submit Job** tab in the Simulations view and adapt the GPU template."
    )

    st.markdown("---")
    c_back, c_save = st.columns(2)
    with c_back:
        if st.button("← Back to configure", key="ems_review_back", use_container_width=True):
            st.session_state.hpc_workflow_phase = "configure"
            st.rerun()
    with c_save:
        if st.button(
            "💾 Save edited script",
            key="ems_save_script",
            use_container_width=True,
            type="primary",
        ):
            script_path = result.get("script_path")
            if script_path:
                try:
                    Path(script_path).write_text(script_edit)
                    st.success(f"Saved to `{script_path}`")
                except Exception as exc:
                    st.error(f"Could not save: {exc}")
            else:
                st.warning("No script_path in result — cannot save.")
