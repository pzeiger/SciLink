"""
Tool registry for the SimulationOrchestratorAgent.

Mirrors the shape of AnalysisOrchestratorTools — each tool is a closure
registered via _register_tool with an OpenAI-format JSONSchema. Tools are
dispatched from the chat loop's manual tool-call handler.

Each tool wraps a piece of the existing sim_agents stack
(StructureGenerator, StructureValidatorAgent, VaspInputAgent, etc.) and
records a structure-centric session record in
`orch.generated_structures` so subsequent tools can find prior work.

Tools are constructed fresh per call (StructureGenerator's per-call
`generated_script_dir` makes caching awkward, and construction is fast).
"""

import glob
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class SimulationOrchestratorTools:
    """Tool registry + dispatch for SimulationOrchestratorAgent.

    Each tool is registered as a closure so it can capture a reference
    to the parent orchestrator (and therefore its session state).
    """

    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: Reference to the parent
                SimulationOrchestratorAgent.
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)

        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []

        self._register_all_tools()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_all_tools(self) -> None:
        """Register all tools with OpenAI format. Called once from __init__."""

        # =====================================================================
        # 0. SESSION STATUS  (low-cost diagnostic)
        # =====================================================================
        def session_status() -> str:
            structures = self.orch.generated_structures or []
            params = self.orch.default_calc_params or {}
            return json.dumps({
                "status": "ok",
                "session_dir": str(self.orch.base_dir),
                "structures_generated": len(structures),
                "structures": [
                    {
                        "slug": s.get("slug"),
                        "description": s.get("description"),
                        "poscar_path": s.get("poscar_path"),
                        "incar_path": s.get("incar_path"),
                    } for s in structures
                ],
                "default_calc_params": params,
                "simulation_mode": self.orch.simulation_mode.value,
            })

        self._register_tool(
            func=session_status,
            name="session_status",
            description=(
                "Report the current simulation session state — structures "
                "generated so far, sticky calculation parameters, output "
                "directory. Free to call; useful when you need to remember "
                "what's already been built before deciding the next step."
            ),
            parameters={},
            required=[],
        )

        # =====================================================================
        # 1. GENERATE STRUCTURE
        # =====================================================================
        def generate_structure(description: str) -> str:
            from .structure_agent import StructureGenerator

            slug = self._make_slug(description)
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            try:
                sg = StructureGenerator(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                    generated_script_dir=str(workdir),
                    mp_api_key=self.orch.mp_api_key,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct StructureGenerator: {e}",
                })

            # Append POSCAR-format request so downstream VASP tools can read it.
            request = description
            if "poscar" not in request.lower():
                request = request + ". Save the structure in POSCAR format."

            result = sg.generate_script(
                original_user_request=request,
                attempt_number_overall=1,
                is_refinement_from_validation=False,
            )

            if result.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or result.get("last_error") or "Unknown failure",
                    "last_attempted_script_path": result.get("last_attempted_script_path"),
                })

            poscar_path = result["output_file"]
            script_path = result["final_script_path"]
            script_content = result["final_script_content"]
            n_atoms = self._count_atoms(poscar_path)

            record = {
                "slug": slug,
                "description": description,
                "structure_dir": str(workdir),
                "poscar_path": poscar_path,
                "script_path": script_path,
                "script_content": script_content,
                "incar_path": None,
                "kpoints_path": None,
                "vasp_summary": None,
                "validation": None,
                "created_at": datetime.now().isoformat(),
            }
            self.orch.generated_structures.append(record)

            return json.dumps({
                "status": "success",
                "slug": slug,
                "structure_dir": str(workdir),
                "poscar_path": poscar_path,
                "script_path": script_path,
                "n_atoms": n_atoms,
                "next_steps": (
                    "Optionally call validate_structure(poscar_path=..., "
                    "original_request=...) to review the geometry, or "
                    "generate_vasp_inputs(...) to produce INCAR + KPOINTS."
                ),
            })

        self._register_tool(
            func=generate_structure,
            name="generate_structure",
            description=(
                "Build a single atomic structure from a natural-language "
                "description (e.g., 'rutile TiO2 with one O vacancy', "
                "'graphene/MoS2 heterostructure'). Generates the structure "
                "file (POSCAR) only — does not produce VASP inputs. Use "
                "this when iterating on geometry; pair with "
                "`validate_structure` and `refine_structure` for review/"
                "refinement, and with `generate_vasp_inputs` afterward. "
                "For the one-shot pipeline (structure + VASP inputs "
                "together) use `run_complete_dft_workflow` instead."
            ),
            parameters={
                "description": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the structure to "
                        "build. Be specific about polymorph (e.g., "
                        "'rutile TiO2', 'wurtzite GaN'), supercell size, "
                        "defects, and other modifications. Materials Project "
                        "lookup is automatic when MP_API_KEY is configured."
                    ),
                },
            },
            required=["description"],
        )

        # =====================================================================
        # 2. VALIDATE STRUCTURE
        # =====================================================================
        def validate_structure(poscar_path: str, original_request: str) -> str:
            from .val_agent import StructureValidatorAgent

            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })

            script_content = self._find_script_content(poscar_path)
            if not script_content:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "Could not locate the generating script next to the "
                        "POSCAR. Validation requires the original script for "
                        "context. Re-run generate_structure if needed."
                    ),
                })

            try:
                validator = StructureValidatorAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct StructureValidatorAgent: {e}",
                })

            val_result = validator.validate_structure_and_script(
                structure_file_path=poscar_path,
                generating_script_content=script_content,
                original_request=original_request,
            )

            # Attach to the matching session record (if any)
            record = self._find_structure_record(poscar_path)
            if record is not None:
                record["validation"] = val_result

            return json.dumps({
                "status": val_result.get("status", "unknown"),
                "overall_assessment": val_result.get("overall_assessment", ""),
                "all_identified_issues": val_result.get("all_identified_issues", []),
                "script_modification_hints": val_result.get("script_modification_hints", []),
                "poscar_path": poscar_path,
            })

        self._register_tool(
            func=validate_structure,
            name="validate_structure",
            description=(
                "Run a multimodal review of a previously generated structure "
                "(POSCAR + generating script + axis-view images). Returns "
                "overall_assessment, identified issues, and script-modification "
                "hints. Status is 'success' when no issues remain, "
                "'needs_correction' when refinement is warranted. Use after "
                "generate_structure to verify the geometry before producing "
                "VASP inputs."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR file to validate.",
                },
                "original_request": {
                    "type": "string",
                    "description": (
                        "The original natural-language request the structure "
                        "was built for — used to check that the result "
                        "matches what was asked for."
                    ),
                },
            },
            required=["poscar_path", "original_request"],
        )

        # =====================================================================
        # 5. GENERATE VASP INPUTS
        # =====================================================================
        def generate_vasp_inputs(poscar_path: str, request: str,
                                 method: str = "llm") -> str:
            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })

            structure_dir = Path(poscar_path).parent

            try:
                if method == "llm":
                    from .vasp_agent import VaspInputAgent
                    agent = VaspInputAgent(
                        api_key=self.orch.api_key,
                        base_url=self.orch.base_url,
                        model_name=self.orch.model_name,
                    )
                    vasp_result = agent.generate_vasp_inputs(
                        poscar_path=poscar_path,
                        original_request=request,
                    )
                    if vasp_result.get("status") != "success":
                        return json.dumps({
                            "status": "error",
                            "message": vasp_result.get("message") or "VASP input generation failed",
                        })
                    saved = agent.save_inputs(vasp_result, str(structure_dir))
                    if "error" in saved:
                        return json.dumps({"status": "error", "message": saved["error"]})
                    summary = vasp_result.get("summary", "")

                elif method == "atomate2":
                    try:
                        from .atomate2_utils import Atomate2Input
                    except ImportError as e:
                        return json.dumps({
                            "status": "error",
                            "message": (
                                "method='atomate2' requires the [sim] extras "
                                "(pymatgen, atomate2). Install with: "
                                "pip install 'scilink[sim]'. "
                                f"Original error: {e}"
                            ),
                        })
                    from ase.io import read as ase_read
                    structure_obj = ase_read(poscar_path)
                    Atomate2Input().generate(
                        structure=structure_obj,
                        output_dir=str(structure_dir),
                    )
                    summary = "Standard relaxation set from atomate2/pymatgen"

                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"Invalid method '{method}'. Choose 'llm' or 'atomate2'.",
                    })

            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"VASP input generation failed: {e}",
                })

            incar_path = structure_dir / "INCAR"
            kpoints_path = structure_dir / "KPOINTS"

            record = self._find_structure_record(poscar_path)
            if record is not None:
                record["incar_path"] = str(incar_path)
                record["kpoints_path"] = str(kpoints_path)
                record["vasp_summary"] = summary

            return json.dumps({
                "status": "success",
                "incar_path": str(incar_path),
                "kpoints_path": str(kpoints_path),
                "summary": summary,
                "method": method,
                "structure_dir": str(structure_dir),
            })

        self._register_tool(
            func=generate_vasp_inputs,
            name="generate_vasp_inputs",
            description=(
                "Generate VASP INCAR and KPOINTS files for a given structure "
                "tailored to the scientific objective in `request`. Saves "
                "INCAR + KPOINTS alongside the POSCAR. method='llm' (default) "
                "uses an LLM to derive parameters; method='atomate2' uses "
                "pymatgen/atomate2's MPRelaxSet (deterministic, requires the "
                "[sim] extras)."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR the inputs should match.",
                },
                "request": {
                    "type": "string",
                    "description": (
                        "Scientific objective / calculation type description "
                        "(e.g., 'static SCF for band structure', "
                        "'relaxation with vdW corrections for an interface'). "
                        "Drives INCAR parameter choices."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": ["llm", "atomate2"],
                    "description": (
                        "'llm' (default): AI-driven, more flexible. "
                        "'atomate2': rule-based, deterministic, requires the "
                        "[sim] extras."
                    ),
                },
            },
            required=["poscar_path", "request"],
        )

        # =====================================================================
        # 10. RUN COMPLETE DFT WORKFLOW (one-shot shortcut)
        # =====================================================================
        def run_complete_dft_workflow(description: str,
                                      max_refinement_cycles: int = 4,
                                      vasp_generator_method: str = "llm") -> str:
            from .dft_orchestrator import DFTOrchestrator

            slug = self._make_slug(description)
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            try:
                wf = DFTOrchestrator(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    futurehouse_api_key=self.orch.futurehouse_api_key,
                    mp_api_key=self.orch.mp_api_key,
                    generator_model=self.orch.model_name,
                    validator_model=self.orch.model_name,
                    output_dir=str(workdir),
                    max_refinement_cycles=max_refinement_cycles,
                    vasp_generator_method=vasp_generator_method,
                )
                result = wf.run_complete_workflow(description)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"DFT workflow failed: {e}",
                })

            final_status = result.get("final_status")
            structure_gen = result.get("structure_generation", {}) or {}
            structure_warning = structure_gen.get("warning")
            cycles_used = structure_gen.get("cycles_used")
            val_result = structure_gen.get("validation_result", {}) or {}
            outstanding_issues = val_result.get("all_identified_issues", []) or []

            poscar_path = workdir / "POSCAR"
            incar_path = workdir / "INCAR"
            kpoints_path = workdir / "KPOINTS"

            # Record in session state (only if structure exists)
            if poscar_path.exists():
                record = {
                    "slug": slug,
                    "description": description,
                    "structure_dir": str(workdir),
                    "poscar_path": str(poscar_path),
                    "script_path": structure_gen.get("final_script_path"),
                    "script_content": None,  # not surfaced by run_complete_workflow
                    "incar_path": str(incar_path) if incar_path.exists() else None,
                    "kpoints_path": str(kpoints_path) if kpoints_path.exists() else None,
                    "vasp_summary": result.get("vasp_generation", {}).get("summary"),
                    "validation": val_result,
                    "created_at": datetime.now().isoformat(),
                }
                self.orch.generated_structures.append(record)

            return json.dumps({
                "status": final_status if final_status else "error",
                "ready_for_vasp": final_status == "success",
                "output_directory": str(workdir),
                "manifest_path": str(workdir / "final_files_manifest.json"),
                "structure_warning": structure_warning,
                "structure_refinement_cycles": cycles_used,
                "structure_outstanding_issues_count": len(outstanding_issues),
                "structure_outstanding_issues": outstanding_issues[:10],
            })

        self._register_tool(
            func=run_complete_dft_workflow,
            name="run_complete_dft_workflow",
            description=(
                "Run the full DFT input pipeline as a one-shot: structure "
                "generation → validation → refinement → VASP inputs (with "
                "optional literature validation when FUTUREHOUSE_API_KEY is "
                "set). Use when the user just wants 'a complete DFT setup' "
                "without iterating on each step. For iterative work "
                "(build → check → refine → inputs), use the granular tools "
                "(generate_structure, validate_structure, refine_structure, "
                "generate_vasp_inputs) instead."
            ),
            parameters={
                "description": {
                    "type": "string",
                    "description": "Natural-language description of the structure to build and prep.",
                },
                "max_refinement_cycles": {
                    "type": "integer",
                    "description": "Maximum validator-guided refinement cycles (default: 4).",
                },
                "vasp_generator_method": {
                    "type": "string",
                    "enum": ["llm", "atomate2"],
                    "description": (
                        "How to produce INCAR/KPOINTS. 'llm' (default) is "
                        "more flexible; 'atomate2' is rule-based and faster."
                    ),
                },
            },
            required=["description"],
        )

        # =====================================================================
        # 3. REFINE STRUCTURE
        # =====================================================================
        def refine_structure(poscar_path: str, original_request: str) -> str:
            from .structure_agent import StructureGenerator

            record = self._find_structure_record(poscar_path)
            if record is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "Refinement requires a structure that was generated "
                        "in this session (so the validator feedback and "
                        "prior script are available). No record found for: "
                        f"{poscar_path}. Generate the structure first via "
                        "generate_structure, then validate, then refine."
                    ),
                })

            validator_feedback = record.get("validation")
            if not validator_feedback or validator_feedback.get("status") == "success":
                return json.dumps({
                    "status": "no_changes_needed",
                    "message": (
                        "No refinement-worthy validator feedback on record. "
                        "Run validate_structure first; if it returns "
                        "'success', the structure is already a fine starting "
                        "point and no refinement is needed."
                    ),
                })

            prior_script = record.get("script_content") or self._find_script_content(poscar_path)
            if not prior_script:
                return json.dumps({
                    "status": "error",
                    "message": "Could not locate the prior script for refinement.",
                })

            workdir = Path(record["structure_dir"])
            try:
                sg = StructureGenerator(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                    generated_script_dir=str(workdir),
                    mp_api_key=self.orch.mp_api_key,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct StructureGenerator: {e}",
                })

            request = original_request
            if "poscar" not in request.lower():
                request = request + ". Save the structure in POSCAR format."

            result = sg.generate_script(
                original_user_request=request,
                attempt_number_overall=2,  # refinement cycle
                is_refinement_from_validation=True,
                previous_script_content=prior_script,
                validator_feedback=validator_feedback,
            )

            if result.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or result.get("last_error") or "Refinement failed",
                })

            new_poscar = result["output_file"]
            new_script_path = result["final_script_path"]
            new_script_content = result["final_script_content"]
            n_atoms = self._count_atoms(new_poscar)

            # Update the record in place rather than appending — refinement
            # produces a successor of the same logical structure.
            record["poscar_path"] = new_poscar
            record["script_path"] = new_script_path
            record["script_content"] = new_script_content
            record["validation"] = None  # invalidate prior validation
            record["incar_path"] = None  # invalidate prior inputs (geometry changed)
            record["kpoints_path"] = None
            record["vasp_summary"] = None

            return json.dumps({
                "status": "success",
                "slug": record["slug"],
                "poscar_path": new_poscar,
                "script_path": new_script_path,
                "n_atoms": n_atoms,
                "next_steps": (
                    "Optionally call validate_structure again to confirm the "
                    "refinement addressed the prior issues; then proceed to "
                    "generate_vasp_inputs."
                ),
            })

        self._register_tool(
            func=refine_structure,
            name="refine_structure",
            description=(
                "Re-generate a structure that this session already built, "
                "incorporating feedback from a prior validate_structure call. "
                "Updates the structure in place — the slug, directory, and "
                "session record are preserved; the POSCAR + script are "
                "replaced. Prior INCAR/KPOINTS (if any) are invalidated since "
                "the geometry changed. Requires the structure to have been "
                "validated in this session — run validate_structure first."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR to refine.",
                },
                "original_request": {
                    "type": "string",
                    "description": (
                        "The original natural-language request the structure "
                        "was built for. Refinement uses this together with "
                        "the validator's feedback."
                    ),
                },
            },
            required=["poscar_path", "original_request"],
        )

        # =====================================================================
        # 4. VIEW STRUCTURE
        # =====================================================================
        def view_structure(poscar_path: str) -> str:
            from .utils import generate_structure_views

            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })

            try:
                image_paths = generate_structure_views(poscar_path)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to render structure views: {e}",
                })

            if not image_paths:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "Structure rendering produced no output (ASE may be "
                        "missing or the file may be unparseable)."
                    ),
                })

            return json.dumps({
                "status": "success",
                "image_paths": image_paths,
                "note": (
                    "PNG renders along the X, Y, and Z axes have been written "
                    "next to the POSCAR. The user can open them; the model "
                    "cannot view image bytes through this text-only tool "
                    "interface."
                ),
            })

        self._register_tool(
            func=view_structure,
            name="view_structure",
            description=(
                "Render axis-view PNG images (along X, Y, Z) of a structure "
                "for visual inspection. Saves images alongside the POSCAR. "
                "Useful when a user wants to eyeball the geometry before "
                "running calculations; the images are surfaced to the user, "
                "not to the model itself."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR to render.",
                },
            },
            required=["poscar_path"],
        )

        # =====================================================================
        # 6. VALIDATE INCAR (literature-grounded)
        # =====================================================================
        def validate_incar(incar_path: str, system_description: str) -> str:
            if not Path(incar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"INCAR not found: {incar_path}",
                })

            if not self.orch.futurehouse_api_key:
                return json.dumps({
                    "status": "skipped",
                    "message": (
                        "Literature validation requires a FutureHouse API "
                        "key. Set FUTUREHOUSE_API_KEY in the environment "
                        "or pass futurehouse_api_key when constructing the "
                        "orchestrator."
                    ),
                })

            try:
                from .val_agent import IncarValidatorAgent
                validator = IncarValidatorAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                    futurehouse_api_key=self.orch.futurehouse_api_key,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct IncarValidatorAgent: {e}",
                })

            incar_content = Path(incar_path).read_text()
            result = validator.validate_and_improve_incar(
                incar_content=incar_content,
                system_description=system_description,
            )

            if result.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or "Literature validation failed",
                })

            return json.dumps({
                "status": "success",
                "validation_status": result.get("validation_status"),
                "overall_assessment": result.get("overall_assessment"),
                "suggested_adjustments": result.get("suggested_adjustments", []),
                "literature_review": result.get("literature_review", "")[:2000],
                "incar_path": incar_path,
            })

        self._register_tool(
            func=validate_incar,
            name="validate_incar",
            description=(
                "Run a literature-grounded review of an INCAR file: pulls "
                "papers via FutureHouse, asks an LLM whether the chosen "
                "parameters are consistent with established practice for "
                "the system in question, returns suggested adjustments. "
                "Returns 'skipped' status when no FutureHouse API key is "
                "configured. Pair with apply_incar_improvements to write "
                "the suggested INCAR to disk."
            ),
            parameters={
                "incar_path": {
                    "type": "string",
                    "description": "Absolute path to the INCAR to validate.",
                },
                "system_description": {
                    "type": "string",
                    "description": (
                        "What system the INCAR is for and what the calculation "
                        "is supposed to compute (used as context for the "
                        "literature review)."
                    ),
                },
            },
            required=["incar_path", "system_description"],
        )

        # =====================================================================
        # 7. APPLY INCAR IMPROVEMENTS
        # =====================================================================
        def apply_incar_improvements(incar_path: str, poscar_path: str,
                                     original_request: str,
                                     suggested_adjustments: list,
                                     overall_assessment: str = "") -> str:
            if not Path(incar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"INCAR not found: {incar_path}",
                })
            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })
            if not suggested_adjustments:
                return json.dumps({
                    "status": "no_changes",
                    "message": "No adjustments provided — nothing to apply.",
                })

            try:
                from .vasp_agent import VaspInputAgent
                agent = VaspInputAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct VaspInputAgent: {e}",
                })

            original_incar = Path(incar_path).read_text()
            output_dir = str(Path(incar_path).parent)

            result = agent.apply_improvements(
                original_incar=original_incar,
                validation_result={
                    "validation_status": "needs_adjustment",
                    "suggested_adjustments": suggested_adjustments,
                    "overall_assessment": overall_assessment,
                },
                poscar_path=poscar_path,
                original_request=original_request,
                output_dir=output_dir,
            )

            if result.get("status") not in ("success", "no_changes"):
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or "Apply-improvements failed",
                })

            return json.dumps({
                "status": result.get("status"),
                "improvements_applied": result.get("improvements_applied", False),
                "adjustments_count": result.get("adjustments_count", 0),
                "improved_incar_path": result.get("improved_incar_path"),
            })

        self._register_tool(
            func=apply_incar_improvements,
            name="apply_incar_improvements",
            description=(
                "Apply a list of literature-validated INCAR adjustments to an "
                "existing INCAR, writing the result as INCAR_improved next to "
                "the original. Pair with validate_incar — pass its "
                "suggested_adjustments through directly."
            ),
            parameters={
                "incar_path": {
                    "type": "string",
                    "description": "Absolute path to the original INCAR.",
                },
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR (provides system context).",
                },
                "original_request": {
                    "type": "string",
                    "description": "The original calculation-type request.",
                },
                "suggested_adjustments": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "List of adjustment dicts in the shape returned by "
                        "validate_incar (each with parameter/current_value/"
                        "suggested_value/reason)."
                    ),
                },
                "overall_assessment": {
                    "type": "string",
                    "description": "Brief literature-review summary (passed verbatim).",
                },
            },
            required=["incar_path", "poscar_path", "original_request", "suggested_adjustments"],
        )

        # =====================================================================
        # 11. LIST GENERATED STRUCTURES
        # =====================================================================
        def list_generated_structures() -> str:
            structures = self.orch.generated_structures or []
            return json.dumps({
                "status": "ok",
                "count": len(structures),
                "structures": [
                    {
                        "slug": s.get("slug"),
                        "description": s.get("description"),
                        "structure_dir": s.get("structure_dir"),
                        "poscar_path": s.get("poscar_path"),
                        "incar_path": s.get("incar_path"),
                        "kpoints_path": s.get("kpoints_path"),
                        "has_validation": s.get("validation") is not None,
                        "created_at": s.get("created_at"),
                    } for s in structures
                ],
            })

        self._register_tool(
            func=list_generated_structures,
            name="list_generated_structures",
            description=(
                "List all structures generated in this session with their "
                "paths and current state (whether VASP inputs exist, whether "
                "validation has been run). Use to remember what's been built "
                "before deciding next steps."
            ),
            parameters={},
            required=[],
        )

        # ↓↓↓ Post-run analysis (8, 9) lands in step 4.

    # ------------------------------------------------------------------
    # Helpers used by tool closures
    # ------------------------------------------------------------------

    def _make_slug(self, description: str) -> str:
        """Build a unique short slug from a description for use as a
        directory name. Always increments the orchestrator's structure
        counter so concurrent calls with the same description don't
        collide."""
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", description)[:40].strip("_") or "structure"
        self.orch._structure_counter += 1
        return f"{safe}_{self.orch._structure_counter:03d}"

    @staticmethod
    def _count_atoms(poscar_path: str) -> Optional[int]:
        """Best-effort atom count via ASE; returns None on parse failure."""
        try:
            from ase.io import read as ase_read
            atoms = ase_read(poscar_path)
            return len(atoms)
        except Exception:
            return None

    def _find_script_content(self, poscar_path: str) -> Optional[str]:
        """Find the generating script for a POSCAR.

        First check the orchestrator's session records (cheap, exact). If
        not found, fall back to globbing `script_*.py` in the POSCAR's
        directory and reading the most recent one — this lets validation
        work even when the LLM passes around paths without going through
        the session record path.
        """
        record = self._find_structure_record(poscar_path)
        if record and record.get("script_content"):
            return record["script_content"]

        poscar_dir = Path(poscar_path).parent
        candidates = sorted(
            poscar_dir.glob("script_*.py"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        try:
            return candidates[0].read_text(encoding="utf-8")
        except Exception:
            return None

    def _find_structure_record(self, poscar_path: str) -> Optional[Dict[str, Any]]:
        """Find the session record matching a POSCAR path, by string match."""
        target = str(Path(poscar_path).resolve())
        for record in self.orch.generated_structures or []:
            try:
                if str(Path(record.get("poscar_path", "")).resolve()) == target:
                    return record
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Registration + dispatch primitives (mirror analyze-mode shapes)
    # ------------------------------------------------------------------

    def _register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: list = None,
    ) -> None:
        """Register a tool in OpenAI format."""
        self.functions_map[name] = func
        self.openai_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        })

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given arguments. Always returns a
        JSON string the chat loop can hand back to the LLM."""
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found",
            })
        try:
            return self.functions_map[tool_name](**kwargs)
        except Exception as e:
            self.logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name,
            })
