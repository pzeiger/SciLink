"""``simulate_xrd_pattern`` tool — kinematic XRD pattern from a crystal structure.

Thin wrapper around :class:`pymatgen.analysis.diffraction.xrd.XRDCalculator`.
The wrapper exists so the LLM-generated script imports a single, named tool
(via skill-registry registration) instead of constructing the calculator
directly — the skill bundle owns the convention.
"""

from __future__ import annotations

import logging
from typing import Any, Union

from ..._shared._spec import ToolSpec

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_XRD_AVAILABLE = True
except ImportError:
    PYMATGEN_XRD_AVAILABLE = False
    Structure = None  # type: ignore
    XRDCalculator = None  # type: ignore

_logger = logging.getLogger(__name__)


TOOL_SPEC = ToolSpec(
    name="simulate_xrd_pattern",
    description=(
        "Compute the kinematic XRD pattern from a crystal structure using "
        "pymatgen's XRDCalculator. Returns 2-theta peak positions, relative "
        "intensities (max = 100), Miller indices, and d-spacings."
    ),
    import_line="from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern",
    signature=(
        "simulate_xrd_pattern(structure_path: str, wavelength: str | float = "
        "'CuKa', two_theta_range: tuple = (10, 90)) -> dict"
    ),
    parameters={
        "structure_path": {
            "type": "str",
            "description": "Path to a CIF file (typically returned by search_structures).",
        },
        "wavelength": {
            "type": "str | float",
            "description": (
                "X-ray source name ('CuKa', 'CuKa1', 'MoKa', 'CoKa', 'FeKa', "
                "'AgKa', 'CrKa') or wavelength in angstroms."
            ),
        },
        "two_theta_range": {
            "type": "tuple",
            "description": "(min, max) in degrees. Default (10, 90).",
        },
    },
    required=["structure_path"],
    returns=(
        "dict with 'two_theta' (list[float]), 'intensities' (list[float]; "
        "normalized so max = 100), 'hkls' (list[list[int]]; primary "
        "reflection per peak), 'd_spacings' (list[float]), 'wavelength' "
        "(Å), 'structure_path' (echo)."
    ),
    when_to_use=(
        "After search_structures, to generate a simulated pattern from each "
        "candidate structure. Chain with score_xrd_match to quantify the "
        "match against experimental data."
    ),
)


def simulate_xrd_pattern(
    structure_path: str,
    wavelength: Union[str, float] = "CuKa",
    two_theta_range: tuple = (10, 90),
) -> dict[str, Any]:
    """Compute kinematic XRD pattern. See ``TOOL_SPEC`` for full contract."""
    if not PYMATGEN_XRD_AVAILABLE:
        raise RuntimeError(
            "simulate_xrd_pattern requires pymatgen with XRDCalculator support; "
            "install pymatgen >= 2022.x"
        )

    structure = Structure.from_file(structure_path)
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=tuple(two_theta_range))

    hkls = [_primary_hkl(entry) for entry in pattern.hkls]

    return {
        "two_theta": [float(x) for x in pattern.x],
        "intensities": [float(y) for y in pattern.y],
        "hkls": hkls,
        "d_spacings": [float(d) for d in pattern.d_hkls],
        "wavelength": float(calc.wavelength),
        "structure_path": structure_path,
    }


def _primary_hkl(entry: Any) -> list[int]:
    """Extract the first (h, k, l) triplet from pymatgen's per-peak hkl payload.

    ``pattern.hkls`` is a list[list[dict]]; each inner dict has key 'hkl'
    pointing to a length-3 tuple. Symmetry-equivalent reflections share a
    peak — we return the first for brevity.
    """
    if not entry:
        return [0, 0, 0]
    first = entry[0]
    if isinstance(first, dict):
        hkl = first.get("hkl") or first.get("HKL")
    else:
        hkl = first
    if hkl is None:
        return [0, 0, 0]
    return [int(c) for c in hkl]
