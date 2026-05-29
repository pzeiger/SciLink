# agents/sim_agents/ems_agent.py
"""
Electron Microscopy Simulation (EMS) agent.

Foundation agent for forward EM image and diffraction simulations across
engines (abTEM, DrProbe, PySlice, ...) selected at runtime via skill bundles.

The modality axis is "electron_microscopy_simulation". Adding a new engine
is done by dropping a new <engine>.md bundle under
scilink/skills/electron_microscopy_simulation/<engine>/ — no code changes
required. abTEM is the first supported engine.

Key additions over a plain MDSimulationAgent-style agent:
  - Structure prep: orient to zone axis, tile laterally, orthogonalize (ASE).
  - Deterministic geometric validator: sampling vs. max scattering angle,
    cell orthogonality, slice thickness, frozen-phonon count.
  - Output contract: runnable Python script that reads a prepped structure from
    a known path and writes NPZ or Zarr to a known output path.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_agent import SimulationAgent

_TOOL_REGISTRY: Dict[str, Any] = {}


class EMSAgent(SimulationAgent):
    """
    Foundation agent for electron microscopy image and diffraction simulations.

    Supported technique: STEM multislice, 4D-STEM / CBED, PRISM (SMatrix).
    First engine: abTEM (abtem.md skill bundle).

    Engine selection is deferred to the loaded skill bundle; additional engines
    become available automatically once their skill bundle is present.

    Inherits from SimulationAgent (free):
      - skill loading / auto-selection / _get_skill_context
      - _generate_json / _generate_text / _clean_output
      - _validate / _llm_validate / _attempt_fix
      - refine(feedback) / fix_error(error_message)

    New pieces (no DFT / MD analog):
      - _prep_structure()     orient, tile, orthogonalize via ASE
      - _validate_geometry()  deterministic sampling/angle/thickness checks
    """

    SKILL_DOMAIN = "electron_microscopy_simulation"
    EXTENSION_MAP = {
        "abtem": [".cif", ".xyz", ".vasp", ".extxyz", ".poscar", ".cfg"],
    }
    TOOL_REGISTRY = _TOOL_REGISTRY

    def __init__(self, working_dir: str, **kwargs):
        super().__init__(working_dir=working_dir, **kwargs)
        self._last_plan: Optional[Dict[str, Any]] = None
        self._last_system_info: Optional[Dict[str, Any]] = None
        self._prepped_structure_path: Optional[str] = None

    @classmethod
    def supported_software(cls) -> List[str]:
        """Engines discovered from skill bundles in this domain."""
        try:
            from ...skills.loader import list_skills
            return list_skills(domain=cls.SKILL_DOMAIN)
        except Exception:
            return []

    # ================================================================
    # SYSTEM ANALYSIS
    # ================================================================

    def analyze_system(self, structure_file: str) -> Dict[str, Any]:
        """Parse structure with ASE; characterise cell geometry for EM."""
        self.logger.info(f"Analyzing structure: {structure_file}")
        try:
            import ase.io
            atoms = ase.io.read(structure_file)
            cell = atoms.get_cell()
            cell_lengths = cell.lengths()   # a, b, c in Å
            cell_angles = cell.angles()     # α, β, γ in degrees
            is_orthogonal = all(abs(a - 90.0) < 0.5 for a in cell_angles)
            ec: Dict[str, int] = {}
            for s in atoms.get_chemical_symbols():
                ec[s] = ec.get(s, 0) + 1
            return {
                "atom_count": len(atoms),
                "elements": sorted(ec.keys()),
                "element_counts": ec,
                "cell_lengths": cell_lengths.tolist(),
                "cell_angles": cell_angles.tolist(),
                "is_orthogonal": is_orthogonal,
                "lateral_extent_a": float(cell_lengths[0]),
                "lateral_extent_b": float(cell_lengths[1]),
                "thickness": float(cell_lengths[2]),
                "volume": float(atoms.get_volume()),
            }
        except ImportError:
            self.logger.warning("ASE not available; falling back to LLM analysis")
            return self._llm_analyze_structure(structure_file)
        except Exception as exc:
            self.logger.warning(f"ASE parse failed ({exc}); falling back to LLM analysis")
            return self._llm_analyze_structure(structure_file)

    def _llm_analyze_structure(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            header = f.read(4000)
        ctx = self._get_skill_context(section="overview")
        prompt = (
            "Analyze this atomic structure file for electron microscopy simulation.\n\n"
            f"{ctx}\n\n"
            "FILE (first 4000 chars):\n"
            f"{header}\n\n"
            "Return JSON:\n"
            '{"atom_count": int, "elements": [...], "element_counts": {}, '
            '"cell_lengths": [a,b,c], "cell_angles": [alpha,beta,gamma], '
            '"is_orthogonal": bool, "lateral_extent_a": float, '
            '"lateral_extent_b": float, "thickness": float, "volume": float}'
        )
        try:
            return self._generate_json(prompt)
        except Exception:
            return {
                "atom_count": 0, "elements": [], "element_counts": {},
                "cell_lengths": [], "cell_angles": [], "is_orthogonal": False,
                "lateral_extent_a": 0.0, "lateral_extent_b": 0.0,
                "thickness": 0.0, "volume": 0.0,
            }

    # ================================================================
    # STRUCTURE PREPARATION
    # ================================================================

    def _prep_structure(
        self,
        structure_file: str,
        zone_axis: Optional[List[int]] = None,
        tile: Optional[List[int]] = None,
        min_lateral_extent_angstrom: float = 10.0,
    ) -> str:
        """
        Prepare structure for EM simulation: tile laterally and enforce an
        orthogonal cell. Returns path to the prepared file (VASP POSCAR format).

        zone_axis:  Miller indices of the beam direction, e.g. [0, 0, 1].
                    Zone-axis rotation is not yet implemented — pass a
                    pre-oriented structure if a non-[001] axis is needed.
        tile:       [nx, ny, nz] repeat factors. Auto-computed from
                    min_lateral_extent_angstrom when not provided.
        min_lateral_extent_angstrom:
                    Minimum cell extent in x and y after tiling. Default 10 Å.
        """
        try:
            import ase.io
            atoms = ase.io.read(structure_file)
        except ImportError:
            self.logger.warning("ASE not available; skipping structure prep")
            return structure_file
        except Exception as exc:
            self.logger.warning(f"Structure prep — read failed ({exc}); using original")
            return structure_file

        try:
            cell = atoms.get_cell()
            cell_lengths = cell.lengths()

            # Determine tiling so that lateral extents meet the minimum.
            if tile is None:
                nx = max(1, int(np.ceil(min_lateral_extent_angstrom / cell_lengths[0])))
                ny = max(1, int(np.ceil(min_lateral_extent_angstrom / cell_lengths[1])))
                tile = [nx, ny, 1]

            if any(t > 1 for t in tile):
                atoms = atoms.repeat(tile)
                self.logger.info(f"Tiled structure: {tile}")

            # Orthogonalize if needed (abTEM requires a rectangular cell).
            angles = atoms.get_cell().angles()
            if any(abs(a - 90.0) > 0.5 for a in angles):
                self.logger.warning(
                    "Cell is not orthogonal — converting to orthorhombic. "
                    "Verify the prepared structure visually before running."
                )
                from ase.geometry import cell_to_cellpar, cellpar_to_cell
                cellpar = cell_to_cellpar(atoms.get_cell())
                cellpar[3:] = 90.0   # force α=β=γ=90°
                atoms.set_cell(cellpar_to_cell(cellpar), scale_atoms=True)

            if zone_axis is not None and zone_axis != [0, 0, 1]:
                self.logger.warning(
                    f"Zone-axis rotation to {zone_axis} is not yet automated. "
                    "Please supply a pre-oriented structure with the beam direction "
                    "along the c-axis ([0,0,1])."
                )

            out_path = str(self.working_dir / "structure_prepped.vasp")
            import ase.io
            ase.io.write(out_path, atoms, format="vasp")
            self.logger.info(f"Prepared structure: {out_path}")
            return out_path

        except Exception as exc:
            self.logger.warning(f"Structure prep failed ({exc}); using original file")
            return structure_file

    # ================================================================
    # PLANNING
    # ================================================================

    def plan_simulation(
        self,
        research_goal: str,
        system_info: Dict[str, Any],
        beam_energy_kev: Optional[float] = None,
        semiangle_mrad: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Choose EM simulation parameters from goal and system info.

        ``beam_energy_kev`` and ``semiangle_mrad`` are instrument settings: when
        the caller supplies them they are AUTHORITATIVE and override whatever the
        planning LLM proposes. The LLM still chooses the dependent parameters
        (sampling, detector geometry, frozen phonons) — but around those fixed
        values. When left as ``None`` the LLM is free to pick them from the goal.
        """
        self.logger.info(f"Planning simulation: {research_goal[:80]}")
        elements_str = ", ".join(
            f"{e}: {c}" for e, c in system_info.get("element_counts", {}).items()
        )
        planning = self._get_skill_context(section="planning")

        # Instrument settings the caller pinned. Inject them into the prompt so
        # the LLM plans the dependent parameters (sampling/antialiasing, detector
        # angles) to be consistent with them rather than choosing its own.
        fixed_lines = []
        if beam_energy_kev is not None:
            fixed_lines.append(
                f"- Beam energy: {beam_energy_kev * 1000.0:.0f} eV "
                "(FIXED — use this exact value, do not change it)"
            )
        if semiangle_mrad is not None:
            fixed_lines.append(
                f"- Probe semiangle: {semiangle_mrad} mrad "
                "(FIXED — use this exact value, do not change it)"
            )
        fixed_block = (
            "FIXED INSTRUMENT SETTINGS (plan sampling and detector geometry to be "
            "consistent with these; do not override them):\n"
            + "\n".join(fixed_lines) + "\n\n"
            if fixed_lines else ""
        )

        prompt = (
            "Recommend electron microscopy simulation parameters for this goal.\n\n"
            f'GOAL: "{research_goal}"\n\n'
            "SYSTEM:\n"
            f"- Elements: {elements_str}\n"
            f"- Atoms: {system_info.get('atom_count', 0)}\n"
            f"- Cell (Å): a={system_info.get('lateral_extent_a', 0):.2f}, "
            f"b={system_info.get('lateral_extent_b', 0):.2f}, "
            f"c={system_info.get('thickness', 0):.2f}\n"
            f"- Orthogonal: {system_info.get('is_orthogonal', False)}\n\n"
            f"{fixed_block}"
            f"{planning}\n\n"
            "Return JSON with these exact keys:\n"
            "{\n"
            '  "technique": "multislice",\n'
            '  "beam_energy_ev": 200000,\n'
            '  "semiangle_mrad": 20.0,\n'
            '  "sampling_angstrom": 0.05,\n'
            '  "slice_thickness_angstrom": 2.0,\n'
            '  "detector_type": "annular",\n'
            '  "detector_inner_mrad": 50,\n'
            '  "detector_outer_mrad": 150,\n'
            '  "frozen_phonon_configs": 8,\n'
            '  "use_prism": false,\n'
            '  "output_format": "npz",\n'
            '  "methodology_description": "brief explanation"\n'
            "}"
        )
        try:
            params = self._generate_json(prompt)
        except Exception as exc:
            self.logger.error(f"Planning LLM call failed: {exc}")
            params = {}

        # Fallbacks so callers always get a complete plan. These apply only
        # when neither the LLM nor the caller supplied a value.
        params.setdefault("technique", "multislice")
        params.setdefault("beam_energy_ev", 200000.0)
        params.setdefault("semiangle_mrad", 20.0)
        params.setdefault("sampling_angstrom", 0.05)
        params.setdefault("slice_thickness_angstrom", 2.0)
        params.setdefault("detector_type", "annular")
        params.setdefault("detector_inner_mrad", 50)
        params.setdefault("detector_outer_mrad", 150)
        params.setdefault("frozen_phonon_configs", 8)
        params.setdefault("use_prism", False)
        params.setdefault("output_format", "npz")

        # Caller-supplied instrument settings are authoritative. setdefault above
        # cannot enforce this because the planning LLM always populates these keys
        # (they're in the prompt schema), so override them explicitly here.
        if beam_energy_kev is not None:
            params["beam_energy_ev"] = beam_energy_kev * 1000.0
        if semiangle_mrad is not None:
            params["semiangle_mrad"] = semiangle_mrad

        for k, v in kwargs.items():
            params[k] = v

        return params

    # ================================================================
    # DETERMINISTIC GEOMETRY VALIDATOR
    # ================================================================

    def _validate_geometry(
        self, system_info: Dict[str, Any], plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deterministic physical checks — no LLM involved.

        Returns the standard validation contract:
            {valid, errors, warnings, suggested_adjustments, ...diagnostics}
        """
        errors: List[str] = []
        warnings: List[str] = []
        adjustments: List[Dict[str, Any]] = []

        # Relativistic electron wavelength (Å).
        energy_ev = float(plan.get("beam_energy_ev", 200000))
        m0c2_ev = 510999.0
        wavelength_angstrom = 12.264 / np.sqrt(
            energy_ev * (1.0 + energy_ev / (2.0 * m0c2_ev))
        )

        sampling = float(plan.get("sampling_angstrom", 0.05))
        max_angle_mrad = (wavelength_angstrom / (2.0 * sampling)) * 1000.0

        outer = float(plan.get("detector_outer_mrad", 150))
        semiangle = float(plan.get("semiangle_mrad", 20))

        # 1. Antialiasing: max representable angle vs. 1.5× probe/detector limit.
        required_angle = max(outer, semiangle) * 1.5
        if max_angle_mrad < required_angle:
            corrected_sampling = round(
                (wavelength_angstrom * 1000.0) / (2.0 * required_angle) * 0.9, 3
            )
            errors.append(
                f"Real-space sampling {sampling:.3f} Å/px is too coarse: "
                f"max representable angle {max_angle_mrad:.0f} mrad < "
                f"{required_angle:.0f} mrad (1.5 × antialiasing margin). "
                f"Reduce sampling to ≤ {corrected_sampling:.3f} Å/px."
            )
            adjustments.append({
                "parameter": "sampling_angstrom",
                "current_value": sampling,
                "suggested_value": corrected_sampling,
                "reason": "antialiasing: θ_max < 1.5 × max(outer_angle, semiangle)",
            })

        # 2. Cell orthogonality.
        if not system_info.get("is_orthogonal", True):
            warnings.append(
                "Cell is not orthogonal. abTEM requires a rectangular cell. "
                "Structure prep will convert it — verify the result visually."
            )

        # 3. Slice thickness.
        dz = float(plan.get("slice_thickness_angstrom", 2.0))
        if dz < 0.5 or dz > 5.0:
            warnings.append(
                f"Slice thickness {dz:.2f} Å is outside the typical range "
                "[0.5, 5.0] Å. Values < 1 Å are slow; values > 3 Å may "
                "under-sample rapidly varying projected potentials."
            )

        # 4. Frozen phonon configuration count.
        fp = int(plan.get("frozen_phonon_configs", 0))
        if fp < 1:
            errors.append(
                "frozen_phonon_configs must be ≥ 1 (set to 1 for a "
                "static-potential approximation)."
            )
        elif fp < 4:
            warnings.append(
                f"Only {fp} frozen-phonon config(s): ADF intensities may not "
                "be converged for thermal diffuse scattering. Recommend ≥ 8 "
                "for qualitative work, ≥ 20 for quantitative comparisons."
            )

        # 5. Sample thickness sanity.
        thickness = float(system_info.get("thickness", 0.0))
        if thickness <= 0:
            errors.append(
                "Structure has zero or negative thickness along the beam axis."
            )
        elif thickness > 500:
            warnings.append(
                f"Sample thickness {thickness:.1f} Å is unusually large. "
                "Multislice is still valid but will be slow."
            )

        # 6. Detector angle ordering.
        inner = float(plan.get("detector_inner_mrad", 50))
        if inner >= outer:
            errors.append(
                f"detector_inner_mrad ({inner}) must be < detector_outer_mrad ({outer})."
            )

        # 7. Detector outer angle within representable range.
        if outer >= max_angle_mrad:
            warnings.append(
                f"Detector outer angle {outer:.0f} mrad is outside the max "
                f"representable angle {max_angle_mrad:.0f} mrad for the "
                "current sampling. Reduce detector outer angle or sampling."
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggested_adjustments": adjustments,
            "diagnostics": {
                "wavelength_angstrom": round(float(wavelength_angstrom), 5),
                "max_representable_angle_mrad": round(float(max_angle_mrad), 1),
            },
        }

    # ================================================================
    # SCRIPT GENERATION
    # ================================================================

    def _generate_em_script(
        self,
        structure_file: str,
        output_path: str,
        research_goal: str,
        system_info: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> str:
        """LLM-driven generation of a runnable EM simulation script."""
        implementation = self._get_skill_context(section="implementation")
        planning_ctx = self._get_skill_context(section="planning")
        validation_ctx = self._get_skill_context(section="validation")
        has_skill = bool(implementation)

        elements_str = ", ".join(system_info.get("elements", []))
        energy_ev = plan.get("beam_energy_ev", 200000)
        sampling = plan.get("sampling_angstrom", 0.05)
        dz = plan.get("slice_thickness_angstrom", 2.0)
        semiangle = plan.get("semiangle_mrad", 20.0)
        fp_configs = plan.get("frozen_phonon_configs", 8)
        det_type = plan.get("detector_type", "annular")
        det_inner = plan.get("detector_inner_mrad", 50)
        det_outer = plan.get("detector_outer_mrad", 150)
        use_prism = plan.get("use_prism", False)
        out_format = plan.get("output_format", "npz")

        if has_skill:
            prompt = (
                "You are an expert electron microscopy simulation engineer.\n"
                "Generate a complete, runnable Python script for the simulation below.\n\n"
                "## Research goal\n"
                f"{research_goal}\n\n"
                "## System\n"
                f"- Elements: {elements_str}\n"
                f"- Atoms: {system_info.get('atom_count', 0)}\n"
                f"- Cell (Å): a={system_info.get('lateral_extent_a', 0):.2f}, "
                f"b={system_info.get('lateral_extent_b', 0):.2f}, "
                f"c={system_info.get('thickness', 0):.2f}\n\n"
                "## Simulation parameters\n"
                f"- Technique: {plan.get('technique', 'multislice')}\n"
                f"- Beam energy: {energy_ev:.0f} eV\n"
                f"- Probe semiangle: {semiangle} mrad\n"
                f"- Real-space sampling: {sampling} Å/px\n"
                f"- Slice thickness: {dz} Å\n"
                f"- Detector: {det_type} ({det_inner}–{det_outer} mrad)\n"
                f"- Frozen phonon configs: {fp_configs}\n"
                f"- Use PRISM (SMatrix): {use_prism}\n"
                f"- Output format: {out_format}\n\n"
                "## I/O contract (use these exact paths)\n"
                f"- STRUCTURE_PATH = {structure_file!r}\n"
                f"- OUTPUT_PATH = {output_path!r}\n\n"
                "## Implementation reference\n"
                f"{implementation}\n\n"
                "## Planning reference\n"
                f"{planning_ctx}\n\n"
                "## Validation rules (the generated script must satisfy these)\n"
                f"{validation_ctx}\n\n"
                "RULES:\n"
                "1. Use the literal parameter values above — no unresolved template variables.\n"
                "2. The script must call `.compute()` to trigger evaluation.\n"
                "3. Save output to OUTPUT_PATH in the specified format.\n"
                "4. Include a short comment block at the top describing what the simulation does.\n"
                "5. Verify every abTEM API call against the installed version; prefer keyword args.\n"
                "\nReturn ONLY the Python script. No markdown fences."
            )
        else:
            prompt = (
                "Generate a runnable abTEM STEM multislice Python script.\n"
                f"No engine-specific skill is loaded — use conservative defaults.\n\n"
                f"GOAL: {research_goal}\n"
                f"STRUCTURE: {structure_file!r}\n"
                f"OUTPUT: {output_path!r}\n"
                f"PARAMS: energy={energy_ev}eV, semiangle={semiangle}mrad, "
                f"sampling={sampling}Å, dz={dz}Å, "
                f"detector={det_type}({det_inner}–{det_outer}mrad), "
                f"frozen_phonons={fp_configs}, prism={use_prism}\n\n"
                "# GENERATED WITHOUT SKILL BUNDLE — VERIFY BEFORE RUNNING\n"
                "Return ONLY the Python script. No markdown."
            )

        return self._generate_text(prompt)

    # ================================================================
    # FULL PIPELINE
    # ================================================================

    def generate_simulation(
        self,
        structure_file: str,
        research_goal: str,
        beam_energy_kev: Optional[float] = None,
        semiangle_mrad: Optional[float] = None,
        zone_axis: Optional[List[int]] = None,
        tile: Optional[List[int]] = None,
        output_format: str = "npz",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full pipeline: analyze → prep structure → plan → validate geometry →
        generate script → validate script → save.

        Parameters
        ----------
        structure_file:     Path to the input structure (CIF, VASP, XYZ, …).
        research_goal:      Natural-language description of the simulation goal.
        beam_energy_kev:    Accelerating voltage in keV. When None, the planning
                            LLM picks it from the goal; when set, the value is
                            authoritative (overrides the LLM). Default None.
        semiangle_mrad:     Probe convergence semi-angle in mrad. Same precedence
                            as beam_energy_kev. Default None.
        zone_axis:          Miller indices of beam direction (e.g. [0,0,1]).
                            Structure prep warns but does not yet auto-rotate.
        tile:               [nx, ny, nz] tiling override (auto-computed otherwise).
        output_format:      "npz" or "zarr".

        Returns a dict with:
            script_path, prepped_structure_path, output_path,
            system_info, simulation_parameters,
            geometry_validation, script_validation, skill_used.
        """
        self._auto_select_skill(structure_file)

        # 1. Analyze original structure.
        system_info = self.analyze_system(structure_file)
        self._last_system_info = system_info

        # 2. Prepare structure (tile, orthogonalize).
        prepped_path = self._prep_structure(
            structure_file, zone_axis=zone_axis, tile=tile
        )
        self._prepped_structure_path = prepped_path

        # Re-analyse the prepped structure if it changed.
        if prepped_path != structure_file:
            system_info = self.analyze_system(prepped_path)
            self._last_system_info = system_info

        # 3. Plan simulation parameters.
        plan = self.plan_simulation(
            research_goal=research_goal,
            system_info=system_info,
            beam_energy_kev=beam_energy_kev,
            semiangle_mrad=semiangle_mrad,
            output_format=output_format,
            **kwargs,
        )
        self._last_plan = plan

        # 4. Deterministic geometry validation — adjust plan if needed.
        geo_validation = self._validate_geometry(system_info, plan)
        if not geo_validation["valid"]:
            self.logger.warning("Geometry validation issues — applying adjustments.")
            for adj in geo_validation.get("suggested_adjustments", []):
                old = plan.get(adj["parameter"])
                plan[adj["parameter"]] = adj["suggested_value"]
                self.logger.info(
                    f"  {adj['parameter']}: {old} → {adj['suggested_value']} "
                    f"({adj['reason']})"
                )
            geo_validation = self._validate_geometry(system_info, plan)

        # 5. Generate script.
        out_ext = "zarr" if output_format == "zarr" else "npz"
        output_path = str(self.working_dir / f"measurement.{out_ext}")
        script = self._generate_em_script(
            prepped_path, output_path, research_goal, system_info, plan
        )
        script = self._clean_output(script)

        script_path = self.working_dir / "run_abtem.py"
        script_path.write_text(script)
        self.logger.info(f"Script written: {script_path}")

        # 6. LLM / tool validation of the generated script.
        script_validation = self._validate(str(script_path), system_info, plan)
        if not script_validation.get("valid", True) and script_validation.get("errors"):
            self.logger.warning("Script validation errors — attempting fix.")
            script = self._attempt_fix(script, script_validation["errors"], plan)
            script_path.write_text(script)
            script_validation = self._validate(str(script_path), system_info, plan)

        return {
            "script_path": str(script_path),
            "prepped_structure_path": prepped_path,
            "output_path": output_path,
            "system_info": system_info,
            "simulation_parameters": plan,
            "geometry_validation": geo_validation,
            "script_validation": script_validation,
            "skill_used": self.skill_name,
        }
