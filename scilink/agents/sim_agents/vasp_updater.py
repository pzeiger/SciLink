# scilink/agents/sim_agents/vasp_updater.py

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .vasp_agent import VaspInputAgent
from ._deprecation import normalize_params


# Known error patterns with deterministic fixes.
# Checked before any LLM call. Each entry:
#   pattern:    regex to match against VASP log
#   diagnosis:  human-readable explanation
#   incar_fixes: dict of INCAR params to set (or None if not an INCAR fix)
#   add_to_allowed: if True, these keys bypass the allowed-keys guard
KNOWN_FIXES = [
    {
        "pattern": r"Looking for PP for potpaw/|No pseudopotential for",
        "diagnosis": "Missing GGA tag — ASE searching wrong pseudopotential directory",
        "incar_fixes": {"GGA": "PE"},
        "add_to_allowed": True,
    },
    {
        "pattern": r"ZBRENT: fatal error",
        "diagnosis": "Ionic step too large, reducing POTIM and switching to RMM-DIIS",
        "incar_fixes": {"POTIM": "0.1", "IBRION": "1"},
    },
    {
        "pattern": r"Sub-Space-Matrix is not hermitian",
        "diagnosis": "Electronic minimization instability, switching to Normal algorithm",
        "incar_fixes": {"ALGO": "Normal"},
    },
    {
        "pattern": r"BRMIX: very serious problems",
        "diagnosis": "Charge density mixing failure — reducing mixing parameters and increasing NELM",
        "incar_fixes": {
            "AMIX": "0.1", "BMIX": "0.01",
            "AMIX_MAG": "0.2", "BMIX_MAG": "0.01",
            "NELM": "200",
        },
        "add_to_allowed": True,
    },
    {
        "pattern": r"ERROR RSPHER",
        "diagnosis": "Real-space projection sphere overlap",
        "incar_fixes": {"LREAL": ".FALSE."},
    },
    {
        "pattern": r"EDDDAV.*did not converge|WARNING.*EDDRMM",
        "diagnosis": "Electronic minimization not converging",
        "incar_fixes": {"ALGO": "All", "NELM": "200"},
    },
    {
        "pattern": r"Your highest band is occupied|you have no unoccupied bands",
        "diagnosis": "Not enough empty bands",
        "incar_fixes": None,  # Handled by _fix_nbands
        "special_handler": "_fix_nbands",
        "add_to_allowed": True,
    },
    {
        "pattern": r"electronic self-consistency was not achieved|number of electronic SC.*reached",
        "diagnosis": "Electronic SCF did not converge within NELM steps — increasing NELM and switching algorithm",
        "incar_fixes": {"NELM": "200", "ALGO": "All"},
        "add_to_allowed": True,
    },
    {
        "pattern": r"ZBRENT: fatal error in bracketing",
        "diagnosis": "Ionic step bracketing failed — restarting from CONTCAR with reduced step size",
        "incar_fixes": {"POTIM": "0.1", "IBRION": "1"},
        "special_handler": "_fix_zbrent",
    },
]


class VaspUpdater:
    """
    Inline updater: analyzes VASP error output and proposes INCAR/KPOINTS fixes.

    Uses a two-layer approach:
    1. Deterministic pattern matching for known errors (fast, reliable)
    2. LLM-based analysis for unknown errors (flexible, uses VaspInputAgent)

    If all errors are resolved deterministically, no LLM call is made.
    """

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 # Legacy params
                 local_model: str = None,
                 google_api_key: str = None):

        self.logger = logging.getLogger(__name__)

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="VaspUpdater"
        )

        self.vasp_agent = VaspInputAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url
        )

    def _extract_errors(self, log: str) -> str:
        """Extract error and warning lines from VASP output."""
        patterns = [
            r"Fatal error.*",
            r"ERROR.*",
            r"ZBRENT.*",
            r"BRMIX.*",
            r"EDDDAV.*",
            r"EDDRMM.*",
            r"RSPHER.*",
            r"Sub-Space-Matrix.*",
            r"highest band is occupied.*",
            r"no unoccupied bands.*",
            r"No pseudopotential.*",
            r"Looking for PP.*",
            r"KPAR.*",
            r"too many k-points.*",
            r"VERY BAD NEWS.*",
        ]
        errs = []
        for pat in patterns:
            errs += [m.strip() for m in re.findall(pat, log, flags=re.IGNORECASE)]
        return "\n".join(errs) or "\n".join(log.splitlines()[:20])

    def _try_deterministic_fixes(self, vasp_log: str,
                                  incar_txt: str) -> Dict[str, Any]:
        """
        Match known error patterns and return deterministic fixes.

        Returns
        -------
        dict with:
            fixes: dict of INCAR param -> value
            diagnoses: list of human-readable diagnoses
            add_to_allowed: set of param keys that should bypass the allowed-keys guard
            remaining_errors: error lines not matched by any known pattern
        """
        fixes = {}
        diagnoses = []
        add_to_allowed = set()
        matched_patterns = set()

        for known in KNOWN_FIXES:
            if re.search(known["pattern"], vasp_log, re.IGNORECASE):
                diagnoses.append(known["diagnosis"])
                matched_patterns.add(known["pattern"])

                if known.get("incar_fixes"):
                    fixes.update(known["incar_fixes"])

                if known.get("add_to_allowed"):
                    if known.get("incar_fixes"):
                        add_to_allowed.update(known["incar_fixes"].keys())

                # Special handlers for fixes that need context
                handler_name = known.get("special_handler")
                if handler_name:
                    handler = getattr(self, handler_name, None)
                    if handler:
                        special_fixes = handler(vasp_log, incar_txt)
                        fixes.update(special_fixes)
                        if known.get("add_to_allowed"):
                            add_to_allowed.update(special_fixes.keys())

        # Identify errors not matched by any known pattern
        error_text = self._extract_errors(vasp_log)
        remaining = []
        for line in error_text.split("\n"):
            if not line.strip():
                continue
            is_matched = any(
                re.search(pat, line, re.IGNORECASE)
                for pat in matched_patterns
            )
            if not is_matched:
                remaining.append(line)

        return {
            "fixes": fixes,
            "diagnoses": diagnoses,
            "add_to_allowed": add_to_allowed,
            "remaining_errors": remaining,
        }

    def _fix_nbands(self, vasp_log: str, incar_txt: str) -> Dict[str, str]:
        """Increase NBANDS by 50% based on current value from log."""
        match = re.search(r"NBANDS\s*=\s*(\d+)", vasp_log)
        if match:
            current = int(match.group(1))
            return {"NBANDS": str(int(current * 1.5))}
        return {"NBANDS": "128"}  # Safe fallback

    def _apply_fixes_to_incar(self, incar_txt: str,
                               fixes: Dict[str, str]) -> str:
        """Apply parameter fixes to existing INCAR text."""
        lines = incar_txt.split("\n")
        fixed_keys = set()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith(("#", "!")):
                key = stripped.split("=")[0].strip().upper()
                if key in {k.upper() for k in fixes}:
                    fix_key = next(k for k in fixes if k.upper() == key)
                    new_lines.append(f"  {key} = {fixes[fix_key]}")
                    fixed_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Add fixes for keys not already in the INCAR
        for key, val in fixes.items():
            if key.upper() not in fixed_keys:
                new_lines.append(f"  {key.upper()} = {val}")

        return "\n".join(new_lines)

    def _fix_zbrent(self, vasp_log: str, incar_txt: str) -> Dict[str, str]:
        """
        Handle ZBRENT bracketing failure.

        The best fix is usually to restart from CONTCAR. This handler
        doesn't modify the INCAR (that's done by incar_fixes) but
        copies CONTCAR to POSCAR if both exist in the same directory.

        Returns empty dict since INCAR fixes are handled by incar_fixes.
        Logs the CONTCAR copy action.
        """
        return {}

    def _copy_contcar_to_poscar(self, calc_dir: str) -> bool:
        """
        Copy CONTCAR to POSCAR for restarting a calculation.
        Returns True if successful.
        """
        import shutil
        contcar = Path(calc_dir) / "CONTCAR"
        poscar = Path(calc_dir) / "POSCAR"
        poscar_backup = Path(calc_dir) / "POSCAR.orig"

        if not contcar.exists():
            self.logger.warning("No CONTCAR found — cannot restart from last geometry")
            return False

        # Check CONTCAR is not empty
        if contcar.stat().st_size < 10:
            self.logger.warning("CONTCAR is empty or too small — cannot use for restart")
            return False

        # Backup original POSCAR
        if poscar.exists() and not poscar_backup.exists():
            shutil.copy2(str(poscar), str(poscar_backup))
            self.logger.info(f"Backed up POSCAR → POSCAR.orig")

        shutil.copy2(str(contcar), str(poscar))
        self.logger.info("Copied CONTCAR → POSCAR for restart")
        return True

    def refine_inputs(
        self,
        poscar_path: str,
        incar_path: str,
        kpoints_path: str,
        vasp_log: str,
        original_request: str,
        skill: Optional[str] = "vasp_input_generation",
        config: Optional["VASPProjectConfig"] = None,
    ) -> Dict[str, Any]:
        """
        Refine VASP inputs based on error output.

        Tries deterministic fixes first. Only calls the LLM for
        errors not matched by known patterns.

        Parameters
        ----------
        poscar_path : str
            Path to POSCAR file.
        incar_path : str
            Path to current INCAR file.
        kpoints_path : str
            Path to current KPOINTS file.
        vasp_log : str
            VASP stdout/stderr log content. Can combine multiple log files.
        original_request : str
            Original description of the calculation.
        skill : str, optional
            Domain skill passed through to VaspInputAgent.
        config : VASPProjectConfig, optional
            Project config passed through to VaspInputAgent.
        """
        incar_txt = Path(incar_path).read_text()
        kpoints_txt = Path(kpoints_path).read_text()

        # Layer 1: Deterministic fixes
        det = self._try_deterministic_fixes(vasp_log, incar_txt)

        # Handle CONTCAR restart for ZBRENT
        contcar_copied = False
        if re.search(r"ZBRENT: fatal error", vasp_log, re.IGNORECASE):
            calc_dir = str(Path(poscar_path).parent)
            contcar_copied = self._copy_contcar_to_poscar(calc_dir)

        if det["fixes"] and not det["remaining_errors"]:
            self.logger.info(
                f"All errors resolved deterministically: {det['diagnoses']}"
            )
            corrected_incar = self._apply_fixes_to_incar(incar_txt, det["fixes"])

            if config:
                config_result = config.apply_to_incar(corrected_incar)
                corrected_incar = config_result["incar"]

            result = {
                "status": "success",
                "method": "deterministic",
                "suggested_incar": corrected_incar,
                "suggested_kpoints": kpoints_txt,
                "explanation": {
                    "diagnoses": det["diagnoses"],
                    "fixes_applied": det["fixes"],
                },
            }

            if contcar_copied:
                result["explanation"]["contcar_restart"] = (
                    "Copied CONTCAR → POSCAR to restart from last converged geometry"
                )

            return result

        # Apply deterministic fixes before sending to LLM
        working_incar = incar_txt
        if det["fixes"]:
            working_incar = self._apply_fixes_to_incar(incar_txt, det["fixes"])
            self.logger.info(
                f"Applied {len(det['fixes'])} deterministic fixes, "
                f"{len(det['remaining_errors'])} errors remain for LLM"
            )

        # Layer 2: LLM for remaining errors
        allowed_keys = [
            line.split('=')[0].strip()
            for line in working_incar.splitlines()
            if '=' in line and not line.strip().startswith('#')
        ]
        for key in det["add_to_allowed"]:
            if key not in allowed_keys:
                allowed_keys.append(key)

        allowed_line = (
            "You may only modify these INCAR parameters: "
            + ", ".join(allowed_keys)
            + ". Do not add any keys not in this list.\n\n"
        )

        snippet = "\n".join(det["remaining_errors"])

        advice_match = re.search(
            r"please specify ([A-Za-z0-9_,\s]+?) in the INCAR file",
            snippet,
            flags=re.IGNORECASE
        )
        advice_line = ""
        if advice_match:
            advice_params = advice_match.group(1).strip()
            advice_line = (
                f"Based on the VASP log, please include these INCAR parameters: "
                f"{advice_params}.\n\n"
            )

        already_fixed = ""
        if det["fixes"]:
            fixes_desc = "\n".join(
                f"  {k} = {v} ({d})"
                for (k, v), d in zip(det["fixes"].items(), det["diagnoses"])
            )
            already_fixed = (
                "NOTE: The following issues were already fixed deterministically. "
                "Do NOT revert these changes:\n"
                f"{fixes_desc}\n\n"
            )

        prompt = (
            allowed_line
            + already_fixed
            + advice_line
            + f'The VASP run for "{original_request}" failed with:\n\n{snippet}\n\n'
            + f"Current INCAR (with deterministic fixes already applied):\n{working_incar}\n\n"
            + f"Current KPOINTS:\n{kpoints_txt}\n\n"
            + "Please reply with a JSON object with keys:\n"
            '  "suggested_incar": full revised INCAR,\n'
            '  "suggested_kpoints": full revised KPOINTS,\n'
            '  "explanation": rationale for each change.\n'
        )

        vasp_res = self.vasp_agent.generate_vasp_inputs(
            poscar_path=poscar_path,
            original_request=prompt,
            skill=skill,
            config=config,
        )

        if vasp_res.get("status") != "success":
            if det["fixes"]:
                corrected_incar = working_incar
                if config:
                    config_result = config.apply_to_incar(corrected_incar)
                    corrected_incar = config_result["incar"]

                result = {
                    "status": "partial",
                    "method": "deterministic_only",
                    "suggested_incar": corrected_incar,
                    "suggested_kpoints": kpoints_txt,
                    "explanation": {
                        "diagnoses": det["diagnoses"],
                        "fixes_applied": det["fixes"],
                        "llm_error": vasp_res.get("message", ""),
                        "remaining_errors": det["remaining_errors"],
                    },
                }
                if contcar_copied:
                    result["explanation"]["contcar_restart"] = (
                        "Copied CONTCAR → POSCAR to restart from last converged geometry"
                    )
                return result

            return {"status": "error", "message": vasp_res.get("message", "")}

        result = {
            "status": "success",
            "method": "deterministic+llm" if det["fixes"] else "llm",
            "suggested_incar": vasp_res["incar"],
            "suggested_kpoints": vasp_res["kpoints"],
            "explanation": {
                "deterministic_diagnoses": det["diagnoses"],
                "deterministic_fixes": det["fixes"],
                "llm_explanation": vasp_res.get("explanation", ""),
            },
        }
        if contcar_copied:
            result["explanation"]["contcar_restart"] = (
                "Copied CONTCAR → POSCAR to restart from last converged geometry"
            )
        return result
