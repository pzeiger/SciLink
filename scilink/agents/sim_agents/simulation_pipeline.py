"""Scale-agnostic deterministic simulation pipeline.

A single one-shot pipeline that turns a natural-language request into a
validated structure plus ready-to-run inputs, for any simulation scale.
One scale-agnostic entry point (``run_complete_workflow``) serves every
engine; the scale selects the foundation agent and the engine selects the
skill bundle.

The pipeline is deterministic — it runs a fixed step sequence rather than
letting an orchestration LLM choose steps — which is what makes its output
reproducible for benchmarking. The chat / LLM-driven path lives on
``SimulationOrchestratorAgent`` (``chat`` / ``run_task``); this is the
headless sequence both that orchestrator and analyze-mode call.

Steps:
    1. Structure   — StructurePipeline (scale-agnostic) builds and
                     validates the atomic structure.
    2. Inputs      — the routed scale's foundation agent generates inputs,
                     returning a normalized ``input_files`` map (engine
                     selected by ``software``; an optional named ``method``
                     selects a deterministic generation backend registered
                     in the engine's skill bundle).
    3. Validation  — InputValidator reviews the generated inputs (skill
                     guidance + deterministic syntax check + literature
                     grounding when a FutureHouse key is present).

Adding a new scale (e.g. molecular DFT) is a new foundation agent plus a
skill bundle and one dispatch branch in ``_generate_inputs`` — no new
orchestrator class, and no hardcoded engine filenames anywhere.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Default engine per scale, used when the caller does not name one. Each
# scale's foundation agent resolves the engine to a skill bundle.
_DEFAULT_ENGINE = {
    "periodic_dft": "vasp",
    "molecular_dynamics": "lammps",
}


def _generate_inputs(
    *,
    scale: str,
    software: str,
    method: str,
    structure_file: str,
    request: str,
    output_dir: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
) -> Dict[str, Any]:
    """Generate inputs for ``scale``, returning a normalized result.

    Every branch returns a result dict carrying an ``input_files`` mapping
    (filename → contents), so downstream steps never guess engine
    filenames. When ``method`` names a deterministic backend (anything
    other than ``"llm"``), inputs come from a ``generate_inputs_<method>``
    tool in the engine's skill bundle; otherwise the routed scale's
    foundation agent produces them with its default LLM path.

    Args:
        scale: Simulation scale (e.g. ``"periodic_dft"``).
        software: Engine name within the scale (e.g. ``"vasp"``).
        method: ``"llm"`` for the agent's baseline generation, or a named
            backend resolved from the skill bundle.
        structure_file: Path to the built structure.
        request: The scientific request driving parameter choices.
        output_dir: Where inputs should be written.
        api_key, base_url, model_name: LLM credentials forwarded to the
            foundation agent.

    Returns:
        A dict with at least ``status`` and, on success, an ``input_files``
        mapping (filename → contents).

    Raises:
        ValueError: If the scale is not supported by the pipeline.
    """
    # Named deterministic backend: a skill-bundle generation tool. The tool
    # is responsible for returning a normalized input_files map.
    if method and method != "llm":
        from ...skills._shared._registry import get_tool_function
        gen = get_tool_function(f"generate_inputs_{method}", active_skills=[software])
        return gen(structure_file=structure_file, request=request,
                   output_dir=output_dir)

    if scale == "periodic_dft":
        from .periodic_dft_agent import PeriodicDFTAgent
        agent = PeriodicDFTAgent(
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        result = agent.generate_inputs(
            structure_file=structure_file, request=request, software=software,
        )
        # PeriodicDFTAgent already returns input_files as {filename: contents}.
        if result.get("status") == "success":
            agent.save_inputs(result, output_dir)
        return result

    if scale == "molecular_dynamics":
        from .md_simulation_agent import MDSimulationAgent
        agent = MDSimulationAgent(
            working_dir=output_dir,
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        result = agent.generate_simulation(
            structure_file=structure_file, research_goal=request, runner=software,
        )
        # Normalize the MD agent's single script_path into the common
        # input_files map so the pipeline stays engine-neutral downstream.
        if "input_files" not in result:
            script_path = result.get("script_path")
            if script_path and Path(script_path).exists():
                result["input_files"] = {
                    Path(script_path).name: Path(script_path).read_text()
                }
        result.setdefault("status", "success")
        return result

    raise ValueError(
        f"Unsupported simulation scale: {scale!r}. "
        f"Supported: {sorted(_DEFAULT_ENGINE)}. Adding a scale means a new "
        "foundation agent + skill bundle and one branch here."
    )


def run_complete_workflow(
    user_request: str,
    *,
    scale: str = "periodic_dft",
    software: Optional[str] = None,
    method: str = "llm",
    structure_class: str = "crystal",
    output_dir: str = "simulation_workflow_output",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "claude-opus-4-6",
    futurehouse_api_key: Optional[str] = None,
    mp_api_key: Optional[str] = None,
    max_refinement_cycles: int = 4,
    script_timeout: int = 300,
    validate: bool = True,
) -> Dict[str, Any]:
    """Run the full structure → inputs → validation pipeline for any scale.

    Args:
        user_request: Natural-language description of the calculation.
        scale: Simulation scale (``"periodic_dft"``, ``"molecular_dynamics"``,
            …). Selects the foundation agent.
        software: Engine within the scale (e.g. ``"vasp"``, ``"lammps"``).
            Defaults to the scale's conventional engine.
        method: Input-generation backend. ``"llm"`` (default) uses the
            foundation agent's generation; a named backend (e.g.
            ``"atomate2"``) resolves to a skill-bundle generation tool.
        structure_class: Structure-class hint forwarded to structure
            generation.
        output_dir: Directory for all generated files.
        api_key, base_url, model_name: LLM credentials.
        futurehouse_api_key: Optional FutureHouse key enabling
            literature-grounded validation.
        mp_api_key: Optional Materials Project key for structure lookups.
        max_refinement_cycles: Structure validator-guided refinement cap.
        script_timeout: Timeout for executing generated structure scripts.
        validate: When True, run the pre-run InputValidator on the generated
            inputs (skipped for non-LLM methods, which are expert-defined).

    Returns:
        A workflow-result dict with ``final_status``, ``scale``, ``engine``,
        ``steps_completed``, ``output_directory``, and the per-step results
        (``structure_generation``, ``input_generation``, ``input_validation``).
    """
    software = software or _DEFAULT_ENGINE.get(scale)
    os.makedirs(output_dir, exist_ok=True)
    result: Dict[str, Any] = {
        "user_request": user_request,
        "scale": scale,
        "engine": software,
        "steps_completed": [],
        "final_status": "started",
        "output_directory": output_dir,
    }

    # ── Step 1: structure generation + validation (scale-agnostic) ──
    from .structure_pipeline import StructurePipeline
    structure = StructurePipeline(
        api_key=api_key, base_url=base_url, mp_api_key=mp_api_key,
        generator_model=model_name, validator_model=model_name,
        output_dir=output_dir, max_refinement_cycles=max_refinement_cycles,
        script_timeout=script_timeout,
    )
    # Reuse the structure orchestrator's resolved credentials downstream.
    api_key = structure.api_key
    base_url = structure.base_url

    structure_result = structure.generate_and_validate(
        user_request, structure_class=structure_class,
    )
    result["structure_generation"] = structure_result
    if structure_result.get("status") != "success":
        result["final_status"] = "failed_structure_generation"
        return result
    result["steps_completed"].append("structure_generation")
    structure_path = structure_result["final_structure_path"]

    # ── Step 2: input generation (routed to the scale's foundation agent) ──
    try:
        gen_result = _generate_inputs(
            scale=scale, software=software, method=method,
            structure_file=structure_path, request=user_request,
            output_dir=output_dir, api_key=api_key, base_url=base_url,
            model_name=model_name,
        )
    except Exception as e:
        result["final_status"] = "failed_input_generation"
        result["input_generation"] = {"status": "error", "message": str(e)}
        return result
    result["input_generation"] = gen_result
    if gen_result.get("status") not in (None, "success"):
        result["final_status"] = "failed_input_generation"
        return result
    result["steps_completed"].append("input_generation")

    # ── Step 3: pre-run input validation (engine-neutral critic) ──
    # Skipped for named (deterministic, expert-defined) backends and when
    # the caller opts out.
    if validate and method == "llm":
        input_files = _collect_input_files(gen_result)
        if input_files:
            from .critics import InputValidator
            validator = InputValidator(
                api_key=api_key, base_url=base_url, model_name=model_name,
                futurehouse_api_key=futurehouse_api_key,
            )
            result["input_validation"] = validator.validate(
                input_files=input_files, system_description=user_request,
                skill=software, domain=scale,
            )
            result["steps_completed"].append("input_validation")
    else:
        reason = ("non-LLM method uses expert-defined inputs"
                  if method != "llm" else "validation disabled by caller")
        result["input_validation"] = {"status": "skipped", "message": reason}

    result["final_status"] = "success"
    return result


def _collect_input_files(gen_result: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{filename: contents}`` from a generation result.

    Reads the normalized ``input_files`` map every ``_generate_inputs``
    branch produces. Values may be inlined contents or paths; paths are
    read so the InputValidator always receives contents. No engine-specific
    filenames are assumed.
    """
    contents: Dict[str, str] = {}
    files = gen_result.get("input_files")
    if not isinstance(files, dict):
        return contents
    for name, val in files.items():
        if not isinstance(val, str):
            continue
        try:
            p = Path(val)
            if p.exists():
                contents[name] = p.read_text()
                continue
        except (OSError, ValueError):
            pass
        contents[name] = val
    return contents
