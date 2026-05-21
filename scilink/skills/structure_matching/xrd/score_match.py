"""``score_xrd_match`` tool — score a simulated pattern against an experimental one.

Broadens the simulated peak list with a Lorentzian profile, resamples on
the experimental 2-theta grid, normalizes both to peak intensity 1, and
reports an R-factor, weighted profile R-factor (Rwp), and cosine
similarity. The verdict is derived from R-factor thresholds.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


# Default verdict thresholds on the R-factor (lower is better).
_R_ACCEPT_MAX = 0.10
_R_MARGINAL_MAX = 0.20


TOOL_SPEC = ToolSpec(
    name="score_xrd_match",
    description=(
        "Score a simulated XRD pattern against an experimental one. Broadens "
        "the simulated peak list with a Lorentzian profile, resamples on the "
        "experimental 2-theta grid, normalizes both, and returns R-factor, "
        "Rwp, cosine similarity, and a coarse verdict."
    ),
    import_line="from scilink.skills.structure_matching.xrd.score_match import score_xrd_match",
    signature=(
        "score_xrd_match(exp_two_theta, exp_intensity, sim_two_theta, "
        "sim_intensity, fwhm: float = 0.15, background: str = 'subtract_min') "
        "-> dict"
    ),
    parameters={
        "exp_two_theta": {
            "type": "list[float]",
            "description": "Experimental 2-theta grid (degrees), monotonically increasing.",
        },
        "exp_intensity": {
            "type": "list[float]",
            "description": "Experimental intensity at each 2-theta. Same length as exp_two_theta.",
        },
        "sim_two_theta": {
            "type": "list[float]",
            "description": "Simulated peak positions (degrees) from simulate_xrd_pattern.",
        },
        "sim_intensity": {
            "type": "list[float]",
            "description": "Simulated peak intensities (relative) from simulate_xrd_pattern.",
        },
        "fwhm": {
            "type": "float",
            "description": "Lorentzian FWHM (degrees) for broadening simulated peaks. Default 0.15.",
        },
        "background": {
            "type": "str",
            "description": "Background handling: 'subtract_min' (default) subtracts the minimum experimental intensity; 'none' uses raw intensities.",
        },
    },
    required=["exp_two_theta", "exp_intensity", "sim_two_theta", "sim_intensity"],
    returns=(
        "dict with 'r_factor' (float, lower is better), 'rwp' (float, "
        "weighted profile R-factor), 'cosine_similarity' (float in [-1, 1]), "
        "'verdict' ('accept' | 'marginal' | 'reject'), 'fwhm_used' (echo)."
    ),
    when_to_use=(
        "After simulate_xrd_pattern, to rank candidate structures by how "
        "well their simulated patterns match the experiment. The verdict "
        "feeds the standard per-item verification loop."
    ),
)


def score_xrd_match(
    exp_two_theta: Sequence[float],
    exp_intensity: Sequence[float],
    sim_two_theta: Sequence[float],
    sim_intensity: Sequence[float],
    fwhm: float = 0.15,
    background: str = "subtract_min",
) -> dict[str, Any]:
    """Score sim vs exp. See ``TOOL_SPEC`` for full contract."""
    exp_x = np.asarray(exp_two_theta, dtype=float)
    exp_y = np.asarray(exp_intensity, dtype=float)
    sim_x = np.asarray(sim_two_theta, dtype=float)
    sim_y = np.asarray(sim_intensity, dtype=float)

    if exp_x.shape != exp_y.shape:
        raise ValueError("exp_two_theta and exp_intensity must have the same length")
    if sim_x.shape != sim_y.shape:
        raise ValueError("sim_two_theta and sim_intensity must have the same length")
    if exp_x.size < 2:
        raise ValueError("exp_two_theta must contain at least 2 points")
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")

    if background == "subtract_min":
        exp_y = exp_y - np.min(exp_y)
    elif background != "none":
        raise ValueError(f"Unknown background option: {background!r}")

    sim_broadened = _broaden_peaks(exp_x, sim_x, sim_y, fwhm)

    exp_max = float(np.max(exp_y)) if np.max(exp_y) > 0 else 1.0
    sim_max = float(np.max(sim_broadened)) if np.max(sim_broadened) > 0 else 1.0
    exp_n = exp_y / exp_max
    sim_n = sim_broadened / sim_max

    r_factor = _r_factor(exp_n, sim_n)
    rwp = _rwp(exp_n, sim_n)
    cosine = _cosine_similarity(exp_n, sim_n)

    if r_factor <= _R_ACCEPT_MAX:
        verdict = "accept"
    elif r_factor <= _R_MARGINAL_MAX:
        verdict = "marginal"
    else:
        verdict = "reject"

    return {
        "r_factor": float(r_factor),
        "rwp": float(rwp),
        "cosine_similarity": float(cosine),
        "verdict": verdict,
        "fwhm_used": float(fwhm),
    }


# --- numerical helpers --------------------------------------------------------

def _broaden_peaks(
    grid: np.ndarray,
    peak_x: np.ndarray,
    peak_y: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    """Place Lorentzian peaks of given FWHM on the experimental grid."""
    if peak_x.size == 0:
        return np.zeros_like(grid)
    gamma = fwhm / 2.0
    out = np.zeros_like(grid)
    for x0, amp in zip(peak_x, peak_y):
        if amp <= 0:
            continue
        out += amp * (gamma ** 2) / ((grid - x0) ** 2 + gamma ** 2)
    return out


def _r_factor(exp: np.ndarray, sim: np.ndarray) -> float:
    """Profile R-factor: sum(|I_exp - I_sim|) / sum(I_exp)."""
    denom = float(np.sum(np.abs(exp)))
    if denom <= 0:
        return float("inf")
    return float(np.sum(np.abs(exp - sim)) / denom)


def _rwp(exp: np.ndarray, sim: np.ndarray) -> float:
    """Weighted profile R-factor with w = 1/I_exp (clipped to avoid div-by-zero)."""
    w = 1.0 / np.clip(exp, 1e-6, None)
    num = float(np.sum(w * (exp - sim) ** 2))
    den = float(np.sum(w * exp ** 2))
    if den <= 0:
        return float("inf")
    return float(np.sqrt(num / den))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
