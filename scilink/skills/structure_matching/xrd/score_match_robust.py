"""``score_xrd_match_robust`` tool — peak-list-based scoring with two algorithms.

The robust tier of the structure-matching workflow. Operates on peak
lists (extracted from continuous patterns via :func:`extract_peaks`)
rather than continuous data, so it factors out background, scale, and
preferred-orientation effects that profile-level metrics fold into the
residual.

Two algorithms:

- ``"hanawalt"`` — classical search-match. For each top-N experimental
  peak, find the closest simulated peak within a tolerance window;
  score on coverage + position residual + (lightly) intensity residual.
  Pure numpy / scipy. Default algorithm; matches the long-standing
  Hanawalt-Fink approach implemented in commercial XRD packages.

- ``"mip"`` — mixed-integer linear programming via PuLP. Jointly fits
  peak assignment + zero-shift + lattice scale by gridding scale and
  solving one MILP per scale. Provably optimal under the formulation;
  natural to extend to multi-phase later (one assignment matrix per
  candidate phase, phase-fraction continuous variables). Requires
  ``pulp`` (declared in ``scilink[structure-matching]``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec
from .extract_peaks import extract_peaks

try:
    import pulp  # type: ignore
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    pulp = None  # type: ignore

_logger = logging.getLogger(__name__)


# Hanawalt verdict thresholds on the figure-of-merit (0..1, higher is better)
_HANAWALT_ACCEPT_MIN = 0.70
_HANAWALT_MARGINAL_MIN = 0.40

# MIP verdict thresholds — cost is normalized to tolerance, so a "match" with
# 0 residual contributes 0 and an unmatched peak contributes 1.
_MIP_ACCEPT_MAX = 0.25
_MIP_MARGINAL_MAX = 0.55


TOOL_SPEC = ToolSpec(
    name="score_xrd_match_robust",
    description=(
        "Peak-list-based scoring of a simulated XRD pattern against an "
        "experimental one, with selectable algorithm. 'hanawalt' (default) "
        "is classical figure-of-merit search-match; 'mip' is mixed-integer "
        "linear programming with joint peak-assignment + zero-shift + "
        "lattice-scale optimization. Use after the fast tier identifies a "
        "short list of candidates; this tier is for confident "
        "identification on real-lab patterns."
    ),
    import_line="from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_robust",
    signature=(
        "score_xrd_match_robust(exp_two_theta=None, exp_intensity=None, "
        "exp_peaks=None, sim_two_theta, sim_intensity, algorithm='hanawalt', "
        "tol_deg=0.3, max_exp_peaks=20, max_sim_peaks=30, "
        "scale_search=(0.99, 1.01, 0.0025), shift_search=(-0.4, 0.4)) -> dict"
    ),
    parameters={
        "exp_two_theta": {
            "type": "list[float]",
            "description": "Experimental 2-theta grid. Required unless exp_peaks is provided.",
        },
        "exp_intensity": {
            "type": "list[float]",
            "description": "Experimental intensity. Required unless exp_peaks is provided.",
        },
        "exp_peaks": {
            "type": "dict",
            "description": "Pre-extracted experimental peak list from extract_peaks. If provided, skips internal peak extraction.",
        },
        "sim_two_theta": {
            "type": "list[float]",
            "description": "Simulated peak positions (degrees) from simulate_xrd_pattern.",
        },
        "sim_intensity": {
            "type": "list[float]",
            "description": "Simulated peak intensities (relative).",
        },
        "algorithm": {
            "type": "str",
            "description": "'hanawalt' (default, fast, pure-Python) or 'mip' (slower, provably optimal under the formulation; requires pulp).",
        },
        "tol_deg": {
            "type": "float",
            "description": "Position tolerance for matching peaks (degrees). Default 0.3.",
        },
        "max_exp_peaks": {
            "type": "int",
            "description": "Cap on experimental peaks considered (strongest kept). Default 20.",
        },
        "max_sim_peaks": {
            "type": "int",
            "description": "Cap on simulated peaks considered (strongest kept). Default 30.",
        },
        "scale_search": {
            "type": "tuple",
            "description": "(min, max, step) lattice-scale grid for MIP. Default (0.99, 1.01, 0.0025). Ignored by hanawalt.",
        },
        "shift_search": {
            "type": "tuple",
            "description": "(min, max) zero-shift bounds for MIP (degrees). Default (-0.4, 0.4). Ignored by hanawalt.",
        },
    },
    required=["sim_two_theta", "sim_intensity"],
    returns=(
        "dict with 'algorithm', 'figure_of_merit' (or 'cost' for MIP), "
        "'verdict' ('accept' | 'marginal' | 'reject'), 'matched_peaks' "
        "(list of {exp_idx, sim_idx, exp_pos, sim_pos, residual_deg}), "
        "'unmatched_exp' (list of exp peak indices), 'fitted_shift' "
        "(degrees; 0 for hanawalt), 'fitted_scale' (1.0 for hanawalt), "
        "'n_exp_peaks', 'n_sim_peaks'."
    ),
    when_to_use=(
        "After the fast tier narrows to a few candidates with promising "
        "correlation, run the robust tier for confident identification. "
        "Use 'hanawalt' by default; switch to 'mip' for multi-phase or "
        "constrained matching, or when zero-shift and lattice scale need "
        "to be reported as fitted parameters."
    ),
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_xrd_match_robust(
    sim_two_theta: Sequence[float],
    sim_intensity: Sequence[float],
    *,
    exp_two_theta: Optional[Sequence[float]] = None,
    exp_intensity: Optional[Sequence[float]] = None,
    exp_peaks: Optional[dict] = None,
    algorithm: str = "hanawalt",
    tol_deg: float = 0.3,
    max_exp_peaks: int = 20,
    max_sim_peaks: int = 30,
    scale_search: tuple = (0.99, 1.01, 0.0025),
    shift_search: tuple = (-0.4, 0.4),
) -> dict[str, Any]:
    """Robust peak-list scoring. See ``TOOL_SPEC`` for full contract."""
    if algorithm not in {"hanawalt", "mip"}:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. Must be 'hanawalt' or 'mip'."
        )

    exp_pl = _resolve_exp_peaks(exp_peaks, exp_two_theta, exp_intensity, max_exp_peaks)
    sim_pl = _build_sim_peak_list(sim_two_theta, sim_intensity, max_sim_peaks)

    if not exp_pl.positions:
        return _empty_result(algorithm, exp_pl, sim_pl, "no experimental peaks")
    if not sim_pl.positions:
        return _empty_result(algorithm, exp_pl, sim_pl, "no simulated peaks")

    if algorithm == "hanawalt":
        return _score_hanawalt(exp_pl, sim_pl, tol_deg=tol_deg)
    return _score_mip(
        exp_pl, sim_pl,
        tol_deg=tol_deg,
        scale_search=scale_search,
        shift_search=shift_search,
    )


# ---------------------------------------------------------------------------
# Internal peak-list representation
# ---------------------------------------------------------------------------

@dataclass
class _PeakList:
    positions: list[float]
    intensities: list[float]
    intensities_norm: list[float]  # max-normalized to 1.0

    @property
    def n(self) -> int:
        return len(self.positions)


def _resolve_exp_peaks(
    exp_peaks: Optional[dict],
    exp_two_theta: Optional[Sequence[float]],
    exp_intensity: Optional[Sequence[float]],
    max_exp_peaks: int,
) -> _PeakList:
    if exp_peaks is not None:
        positions = list(exp_peaks.get("positions", []))
        intensities = list(exp_peaks.get("intensities", []))
    elif exp_two_theta is not None and exp_intensity is not None:
        extracted = extract_peaks(
            exp_two_theta, exp_intensity, max_peaks=max_exp_peaks,
        )
        positions = extracted["positions"]
        intensities = extracted["intensities"]
    else:
        raise ValueError(
            "Provide either exp_peaks or both exp_two_theta and exp_intensity"
        )
    return _to_peak_list(positions, intensities, max_exp_peaks)


def _build_sim_peak_list(
    sim_two_theta: Sequence[float],
    sim_intensity: Sequence[float],
    max_sim_peaks: int,
) -> _PeakList:
    positions = list(sim_two_theta)
    intensities = list(sim_intensity)
    return _to_peak_list(positions, intensities, max_sim_peaks)


def _to_peak_list(
    positions: list[float],
    intensities: list[float],
    max_keep: int,
) -> _PeakList:
    if len(positions) != len(intensities):
        raise ValueError("positions and intensities must have the same length")
    if not positions:
        return _PeakList([], [], [])
    pairs = sorted(zip(positions, intensities), key=lambda p: -p[1])[:max_keep]
    pos_sorted = [float(p) for p, _ in pairs]
    int_sorted = [float(i) for _, i in pairs]
    max_i = max(int_sorted) if int_sorted else 1.0
    if max_i <= 0:
        max_i = 1.0
    norm = [i / max_i for i in int_sorted]
    return _PeakList(pos_sorted, int_sorted, norm)


def _empty_result(algorithm: str, exp_pl: _PeakList, sim_pl: _PeakList, why: str) -> dict[str, Any]:
    score_key = "figure_of_merit" if algorithm == "hanawalt" else "cost"
    return {
        "algorithm": algorithm,
        score_key: 0.0 if algorithm == "hanawalt" else float("inf"),
        "verdict": "reject",
        "matched_peaks": [],
        "unmatched_exp": list(range(exp_pl.n)),
        "fitted_shift": 0.0,
        "fitted_scale": 1.0,
        "n_exp_peaks": exp_pl.n,
        "n_sim_peaks": sim_pl.n,
        "note": why,
    }


# ---------------------------------------------------------------------------
# Hanawalt search-match
# ---------------------------------------------------------------------------

def _score_hanawalt(
    exp_pl: _PeakList,
    sim_pl: _PeakList,
    *,
    tol_deg: float,
) -> dict[str, Any]:
    """Classical Hanawalt-style figure-of-merit with intensity weighting."""
    sim_positions = np.asarray(sim_pl.positions)
    sim_norm = np.asarray(sim_pl.intensities_norm)
    used_sim = set()
    matched_peaks = []
    unmatched_exp = []

    pos_scores = []
    int_scores = []
    weights = []

    for i, (ep, ei_norm) in enumerate(zip(exp_pl.positions, exp_pl.intensities_norm)):
        residuals = np.abs(sim_positions - ep)
        # Skip already-used sim peaks
        for j_used in used_sim:
            residuals[j_used] = np.inf
        j = int(np.argmin(residuals))
        if residuals[j] > tol_deg:
            unmatched_exp.append(i)
            continue
        used_sim.add(j)
        pos_score = 1.0 - residuals[j] / tol_deg
        int_score = max(0.0, 1.0 - abs(ei_norm - float(sim_norm[j])))
        weight = math.sqrt(ei_norm)  # weight by experimental intensity
        matched_peaks.append({
            "exp_idx": i,
            "sim_idx": j,
            "exp_pos": ep,
            "sim_pos": float(sim_positions[j]),
            "residual_deg": float(residuals[j]),
        })
        pos_scores.append(pos_score * weight)
        int_scores.append(int_score * weight)
        weights.append(weight)

    if exp_pl.n == 0:
        coverage = 0.0
    else:
        coverage = len(matched_peaks) / exp_pl.n

    if weights:
        mean_pos = sum(pos_scores) / sum(weights)
        mean_int = sum(int_scores) / sum(weights)
    else:
        mean_pos = 0.0
        mean_int = 0.0

    fom = 0.55 * coverage + 0.35 * mean_pos + 0.10 * mean_int

    if fom >= _HANAWALT_ACCEPT_MIN:
        verdict = "accept"
    elif fom >= _HANAWALT_MARGINAL_MIN:
        verdict = "marginal"
    else:
        verdict = "reject"

    return {
        "algorithm": "hanawalt",
        "figure_of_merit": float(fom),
        "coverage": float(coverage),
        "position_score": float(mean_pos),
        "intensity_score": float(mean_int),
        "verdict": verdict,
        "matched_peaks": matched_peaks,
        "unmatched_exp": unmatched_exp,
        "fitted_shift": 0.0,
        "fitted_scale": 1.0,
        "n_exp_peaks": exp_pl.n,
        "n_sim_peaks": sim_pl.n,
    }


# ---------------------------------------------------------------------------
# MIP peak-matching
# ---------------------------------------------------------------------------

def _score_mip(
    exp_pl: _PeakList,
    sim_pl: _PeakList,
    *,
    tol_deg: float,
    scale_search: tuple,
    shift_search: tuple,
) -> dict[str, Any]:
    """Joint shift + scale + assignment via MILP, gridded over scale."""
    if not PULP_AVAILABLE:
        raise RuntimeError(
            "MIP algorithm requires pulp; install via "
            "'pip install scilink[structure-matching]'"
        )

    scales = _scale_grid(scale_search)
    shift_lo, shift_hi = float(shift_search[0]), float(shift_search[1])

    best = None
    for scale in scales:
        result = _solve_mip_for_scale(
            exp_pl, sim_pl,
            scale=scale,
            tol_deg=tol_deg,
            shift_lo=shift_lo,
            shift_hi=shift_hi,
        )
        if result is None:
            continue
        if best is None or result["cost"] < best["cost"]:
            best = result
            best["fitted_scale"] = scale

    if best is None:
        return _empty_result("mip", exp_pl, sim_pl, "MILP found no feasible assignment")

    cost = best["cost"]
    if cost <= _MIP_ACCEPT_MAX:
        verdict = "accept"
    elif cost <= _MIP_MARGINAL_MAX:
        verdict = "marginal"
    else:
        verdict = "reject"

    return {
        "algorithm": "mip",
        "cost": float(cost),
        "verdict": verdict,
        "matched_peaks": best["matched_peaks"],
        "unmatched_exp": best["unmatched_exp"],
        "fitted_shift": float(best["fitted_shift"]),
        "fitted_scale": float(best["fitted_scale"]),
        "n_exp_peaks": exp_pl.n,
        "n_sim_peaks": sim_pl.n,
    }


def _scale_grid(scale_search: tuple) -> np.ndarray:
    lo, hi, step = float(scale_search[0]), float(scale_search[1]), float(scale_search[2])
    if step <= 0 or hi < lo:
        raise ValueError(f"Invalid scale_search: {scale_search!r}")
    return np.arange(lo, hi + step * 0.5, step)


def _solve_mip_for_scale(
    exp_pl: _PeakList,
    sim_pl: _PeakList,
    *,
    scale: float,
    tol_deg: float,
    shift_lo: float,
    shift_hi: float,
) -> Optional[dict[str, Any]]:
    """Solve one MILP at a fixed lattice scale; returns assignment + shift + cost.

    Two-pass approach to avoid a CBC interaction with the bilinear residual
    constraint ``r <= tol * x`` that produced bound-violating "solutions" in
    an earlier formulation.

    Pass 1 (MILP) — pure assignment with feasibility cone on shift:

        Variables:  x[i,j] in {0,1} — 1 if exp i matched to sim j
                    shift in [shift_lo, shift_hi] — zero-shift (degrees)
        Constraints:
            sum_j x[i,j] <= 1                          (each exp matched at most once)
            sum_i x[i,j] <= 1                          (each sim matched at most once)
            exp_i - scale*sim_j - shift <= tol + M(1 - x[i,j])
            -(exp_i - scale*sim_j - shift) <= tol + M(1 - x[i,j])
        Objective: maximize sum x[i,j]

    Pass 2 (post-solve scoring) — evaluate ``|exp_i - scale*sim_j - shift|``
    at the optimum for each matched pair; aggregate to a normalized cost in
    [0, 1] where every unmatched peak contributes a full tolerance unit.
    """
    nE = exp_pl.n
    nS = sim_pl.n
    if nE == 0 or nS == 0:
        return None

    # Pre-prune (i,j) pairs whose minimum possible residual exceeds tol for
    # any plausible shift — keeps the MILP small. A pair is feasible when
    # some shift in [shift_lo, shift_hi] reduces |exp_i - α·sim_j - shift|
    # below tol, equivalently raw = exp_i - α·sim_j lies in
    # [shift_lo - tol, shift_hi + tol].
    candidate_pairs: list[tuple[int, int]] = []
    raw_lookup: dict[tuple[int, int], float] = {}
    for i in range(nE):
        for j in range(nS):
            raw = exp_pl.positions[i] - scale * sim_pl.positions[j]
            if shift_lo - tol_deg <= raw <= shift_hi + tol_deg:
                candidate_pairs.append((i, j))
                raw_lookup[(i, j)] = raw
    if not candidate_pairs:
        # Nothing in this scale can match within tolerance under any shift.
        return {
            "cost": 1.0,
            "matched_peaks": [],
            "unmatched_exp": list(range(nE)),
            "fitted_shift": 0.0,
        }

    M = max(
        (max(exp_pl.positions) - min(exp_pl.positions)) if nE > 1 else 0.0,
        (max(sim_pl.positions) - min(sim_pl.positions)) if nS > 1 else 0.0,
        1.0,
    ) + max(abs(shift_lo), abs(shift_hi)) + tol_deg + 1.0

    prob = pulp.LpProblem("xrd_match", pulp.LpMaximize)
    x = {
        p: pulp.LpVariable(f"x_{p[0]}_{p[1]}", cat=pulp.LpBinary)
        for p in candidate_pairs
    }
    shift = pulp.LpVariable("shift", lowBound=shift_lo, upBound=shift_hi)

    # Objective: maximize the number of matched pairs. (Intensity-weighted
    # ties could be broken with a small bonus term; omitted for v1.)
    prob += pulp.lpSum(x.values())

    # Assignment constraints
    by_i: dict[int, list[tuple[int, int]]] = {i: [] for i in range(nE)}
    by_j: dict[int, list[tuple[int, int]]] = {j: [] for j in range(nS)}
    for p in candidate_pairs:
        by_i[p[0]].append(p)
        by_j[p[1]].append(p)
    for i, pairs in by_i.items():
        if pairs:
            prob += pulp.lpSum(x[p] for p in pairs) <= 1, f"once_per_exp_{i}"
    for j, pairs in by_j.items():
        if pairs:
            prob += pulp.lpSum(x[p] for p in pairs) <= 1, f"once_per_sim_{j}"

    # Tolerance cone: when x[i,j] = 1, |raw - shift| <= tol.
    for p in candidate_pairs:
        raw = raw_lookup[p]
        prob += raw - shift <= tol_deg + M * (1 - x[p]), f"tol_pos_{p[0]}_{p[1]}"
        prob += -(raw - shift) <= tol_deg + M * (1 - x[p]), f"tol_neg_{p[0]}_{p[1]}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        _logger.debug("MILP non-optimal at scale=%.5f: %s", scale, pulp.LpStatus[status])
        return None

    # Pass 2: with the assignment fixed, refine shift analytically. The shift
    # that minimizes Σ |raw - shift| over the matched pairs is the L1 median
    # (clipped to [shift_lo, shift_hi]). The MILP objective alone doesn't
    # discriminate among shifts inside the tolerance cone — any feasible
    # shift gives the same match count — so CBC may return a corner-of-the-
    # cone shift that inflates the cost. The median-refinement step picks
    # the best representative.
    matched_raws = [
        raw_lookup[p] for p in candidate_pairs
        if pulp.value(x[p]) is not None and pulp.value(x[p]) >= 0.5
    ]
    if matched_raws:
        median_shift = float(np.median(matched_raws))
        fitted_shift = float(np.clip(median_shift, shift_lo, shift_hi))
    else:
        fitted_shift = float(pulp.value(shift)) if pulp.value(shift) is not None else 0.0

    matched_peaks: list[dict[str, Any]] = []
    matched_exp_set: set[int] = set()
    total_residual = 0.0
    for p in candidate_pairs:
        val = pulp.value(x[p])
        if val is None or val < 0.5:
            continue
        i, j = p
        residual = abs(raw_lookup[p] - fitted_shift)
        # If median-refined shift pushed a marginal match outside the cone,
        # drop it from the matched set.
        if residual > tol_deg + 1e-9:
            continue
        matched_exp_set.add(i)
        total_residual += residual
        matched_peaks.append({
            "exp_idx": i,
            "sim_idx": j,
            "exp_pos": exp_pl.positions[i],
            "sim_pos": float(scale * sim_pl.positions[j] + fitted_shift),
            "residual_deg": float(residual),
        })
    unmatched_exp = [i for i in range(nE) if i not in matched_exp_set]
    # Normalized cost in [0, 1]: each unmatched peak costs tol, each matched
    # pair costs its residual. Divide by tol * nE so a perfect identification
    # gives ~0 and total mismatch gives 1.
    raw_cost = total_residual + tol_deg * len(unmatched_exp)
    cost = raw_cost / (tol_deg * max(nE, 1))

    return {
        "cost": float(cost),
        "matched_peaks": matched_peaks,
        "unmatched_exp": unmatched_exp,
        "fitted_shift": fitted_shift,
    }
