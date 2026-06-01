"""``fit_pattern`` tool — single global multi-peak pseudo-Voigt fit of a whole
XRD pattern.

Motivation
----------
``fit_profile`` fits one peak (or one overlapping cluster) per window and
returns a per-window R². Stitching many window fits together leaves every
*unmodelled* reflection as a large spike in the **global** residual — and the
curve-fitting agent's verifier judges the global residual, not per-window R².
On a busy pattern that mismatch drives many rejected refinement iterations.

``fit_pattern`` closes that gap: it detects *all* significant peaks in one pass,
fits them **simultaneously** as a sum of height-parameterised pseudo-Voigts on a
linear baseline (background-subtracted first), seeds each amplitude from the
measured apex so sharp peaks are never clipped, and reports the **global** R²
plus a residual-RMS-over-noise figure — the same quantities the verifier checks.
One fast call (~1 s for ~20 peaks over a few-thousand-point scan) typically lands
a clean global fit on the first attempt.

In-situ series
--------------
For a temperature/time ramp, establish the peak list once on a high-SNR frame
(``auto_detect``), then pass that fixed list back as ``peak_centers`` for every
subsequent frame. The model stays locked and consistent across the series while
each frame is still a single fast fit — warm-started, comparable frame to frame.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ..._shared._spec import ToolSpec
from .background import fit_background

_logger = logging.getLogger(__name__)

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


TOOL_SPEC = ToolSpec(
    name="fit_pattern",
    description=(
        "Global multi-peak pseudo-Voigt fit of a whole XRD pattern in one "
        "call. Auto-detects all significant peaks (or uses a supplied locked "
        "list), subtracts background, fits every peak simultaneously on a "
        "linear baseline with apex-seeded amplitudes, and returns the GLOBAL "
        "R-squared, residual-RMS-over-noise, and per-peak center/FWHM/height/"
        "area/eta. Use this for full-pattern fitting and for in-situ series; "
        "use fit_profile only to drill into a single stubborn cluster."
    ),
    import_line="from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern",
    signature=(
        "fit_pattern(exp_two_theta, exp_intensity, peak_centers=None, "
        "background='snip', snip_iterations='auto', prominence_frac=0.02, "
        "max_peaks=30, min_distance_deg=0.15, init_fwhm_deg=0.2, "
        "center_leeway_deg=0.3, max_fwhm_deg=3.0) -> dict"
    ),
    parameters={
        "exp_two_theta": {"type": "list[float]", "description": "Experimental 2-theta grid (degrees)."},
        "exp_intensity": {"type": "list[float]", "description": "Raw experimental intensity (same length). Background is handled internally."},
        "peak_centers": {
            "type": "list[float] | None",
            "description": "Fixed peak centers to fit (degrees). None => auto-detect all significant peaks. Pass a locked list to keep the model identical across an in-situ series.",
        },
        "background": {"type": "str", "description": "'snip' (default), 'polynomial', or 'none' (data already background-subtracted)."},
        "snip_iterations": {"type": "int | str", "description": "SNIP iteration count. 'auto' (default) sweeps a few counts and keeps the one with the cleanest residual at the best R² — avoids apex over-subtraction on sharp peaks without hand-tuning. Pass an int to fix it (e.g. reuse the value reported in background_method to skip the sweep on locked series frames)."},
        "prominence_frac": {"type": "float", "description": "Auto-detect: min peak prominence as a fraction of the corrected pattern range. Default 0.02 (2%). Lower (0.01) to catch weak reflections the verifier may flag as unmodelled residual; raise (0.03) if noise peaks are being fit."},
        "max_peaks": {"type": "int", "description": "Auto-detect cap. Default 30."},
        "min_distance_deg": {"type": "float", "description": "Auto-detect: minimum separation between peaks (degrees). Default 0.15."},
        "init_fwhm_deg": {"type": "float", "description": "Initial FWHM guess per peak (degrees). Default 0.2 (typical CuKa)."},
        "center_leeway_deg": {"type": "float", "description": "Each center may move +/- this much during the fit (degrees). Default 0.3."},
        "max_fwhm_deg": {"type": "float", "description": "Upper bound on fitted FWHM (degrees). Default 3.0."},
    },
    required=["exp_two_theta", "exp_intensity"],
    returns=(
        "dict with 'r_squared' (GLOBAL, over the whole corrected pattern), "
        "'residual_rms_over_noise' (global residual RMS / estimated point "
        "noise — the verifier's key statistic; < ~3 is clean), 'n_peaks', "
        "'peaks' (list of dicts: center, fwhm, amplitude (height), area, eta, "
        "each sorted by 2-theta), 'peak_centers' (the centers actually fit — "
        "feed back as the locked list for the next series frame), "
        "'intensity_corrected', 'fit_curve' (model evaluated on the full grid; "
        "use for the visualization), 'background_method'."
    ),
    when_to_use=(
        "Default tool for fitting a full XRD pattern and for every frame of an "
        "in-situ series. Detect peaks once on a strong frame, then pass "
        "peak_centers to lock the model across the series."
    ),
)


def _pseudo_voigt(x, amp, cen, fwhm, eta):
    """Height-parameterised pseudo-Voigt: amp is the peak height; eta in [0,1]
    mixes Lorentzian (eta=1) and Gaussian (eta=0)."""
    sigma = fwhm * _FWHM_TO_SIGMA
    gauss = np.exp(-((x - cen) ** 2) / (2.0 * sigma ** 2))
    gamma = fwhm / 2.0
    lorentz = gamma ** 2 / ((x - cen) ** 2 + gamma ** 2)
    return amp * (eta * lorentz + (1.0 - eta) * gauss)


def _multi(x, *p):
    n = (len(p) - 2) // 4
    out = p[-2] * x + p[-1]
    for i in range(n):
        out = out + _pseudo_voigt(x, *p[4 * i:4 * i + 4])
    return out


def _pv_area(amp, fwhm, eta):
    g = amp * fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    l = amp * fwhm * np.pi / 2.0
    return float(eta * l + (1.0 - eta) * g)


def _detect_centers(x, ycorr, step, prominence_frac, max_peaks, min_distance_deg):
    """Auto-detect significant peak centers on a background-corrected pattern."""
    noise = _estimate_noise(ycorr)
    prom = max(prominence_frac * (ycorr.max() - ycorr.min()), 3.0 * noise)
    dist = max(1, int(round(min_distance_deg / step)))
    idx, _ = find_peaks(ycorr, prominence=prom, distance=dist)
    idx = idx[np.argsort(ycorr[idx])[::-1][:max_peaks]]
    return sorted(float(x[i]) for i in idx)


def _estimate_noise(y):
    """Robust per-point noise from first differences (MAD estimator).
    diff of white noise has std sqrt(2)*sigma; MAD->std uses 1.4826."""
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0)) or 1.0


def fit_pattern(
    exp_two_theta: Sequence[float],
    exp_intensity: Sequence[float],
    peak_centers: Optional[Sequence[float]] = None,
    background: str = "snip",
    snip_iterations: Any = "auto",
    prominence_frac: float = 0.02,
    max_peaks: int = 30,
    min_distance_deg: float = 0.15,
    init_fwhm_deg: float = 0.2,
    center_leeway_deg: float = 0.3,
    max_fwhm_deg: float = 3.0,
) -> dict[str, Any]:
    x = np.asarray(exp_two_theta, dtype=float)
    y = np.asarray(exp_intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("exp_two_theta and exp_intensity must have the same length")
    if x.size < 10:
        raise ValueError("pattern too short to fit")
    step = float(np.median(np.diff(x)))

    centers_locked = (
        [float(c) for c in peak_centers]
        if peak_centers is not None and len(peak_centers) > 0 else None
    )
    fit_kw = dict(
        prominence_frac=prominence_frac, max_peaks=max_peaks,
        min_distance_deg=min_distance_deg, init_fwhm_deg=init_fwhm_deg,
        center_leeway_deg=center_leeway_deg, max_fwhm_deg=max_fwhm_deg,
    )

    # --- background + fit ---
    if background == "none":
        best = _fit_corrected(x, y.copy(), centers_locked, step, **fit_kw)
        best["background_method"] = "none"
    elif background == "polynomial":
        bg = fit_background(x.tolist(), y.tolist(), method="polynomial")
        ycorr = np.asarray(bg["intensity_corrected"], dtype=float)
        best = _fit_corrected(x, ycorr, centers_locked, step, **fit_kw)
        best["background_method"] = "polynomial"
    elif background == "snip":
        # SNIP iteration count trades off two ways: too many iterations eat into
        # the base of SHARP peaks (apex over-subtraction -> the residual spikes
        # the verifier flags), too few leave broad-background curvature (R²
        # collapses). The optimum is pattern-dependent, so sweep a few counts
        # and keep the one that minimises residual-RMS/noise while staying within
        # 0.01 R² of the best — instead of forcing the agent to hand-tune it.
        if snip_iterations == "auto":
            iters_list = [6, 10, 16, 24]
        else:
            iters_list = [int(snip_iterations)]
        # Detect the peak list ONCE on the most-aggressive background so the set
        # of peaks stays fixed while the sweep varies only the fit background.
        # (Low-iteration backgrounds leave baseline ripple that would otherwise
        # inflate the detected peak count.) A caller-supplied list overrides.
        if centers_locked is None:
            ref_bg = fit_background(
                x.tolist(), y.tolist(), method="snip", iterations=max(iters_list))
            centers_for_sweep = _detect_centers(
                x, np.asarray(ref_bg["intensity_corrected"], dtype=float), step,
                prominence_frac, max_peaks, min_distance_deg)
        else:
            centers_for_sweep = centers_locked
        trials = []
        for it in iters_list:
            bg = fit_background(x.tolist(), y.tolist(), method="snip", iterations=it)
            ycorr = np.asarray(bg["intensity_corrected"], dtype=float)
            try:
                res = _fit_corrected(x, ycorr, centers_for_sweep, step, **fit_kw)
            except (RuntimeError, ValueError):
                continue
            res["_iters"] = it
            trials.append(res)
        if not trials:
            raise RuntimeError("fit_pattern: no SNIP iteration count converged")
        # Favour R² first (attempt-1 must clear the acceptance gate, especially
        # in fast/low-iteration mode); only take a cleaner-residual background
        # when its R² is within a hair (0.002) of the best, i.e. essentially
        # free. An earlier 0.01 band over-favoured cleanliness and capped R².
        best_r2 = max(t["r_squared"] for t in trials)
        eligible = [t for t in trials if t["r_squared"] >= best_r2 - 0.002]
        best = min(eligible, key=lambda t: t["residual_rms_over_noise"])
        best["background_method"] = f"snip(iterations={best.pop('_iters')})"
    else:
        raise ValueError(f"Unknown background: {background!r}")

    return best


def _fit_corrected(
    x: np.ndarray,
    ycorr: np.ndarray,
    centers_locked: Optional[list],
    step: float,
    prominence_frac: float,
    max_peaks: int,
    min_distance_deg: float,
    init_fwhm_deg: float,
    center_leeway_deg: float,
    max_fwhm_deg: float,
) -> dict[str, Any]:
    """Global multi-peak fit of an already background-corrected pattern."""
    noise = _estimate_noise(ycorr)

    if centers_locked is not None:
        centers = list(centers_locked)
    else:
        centers = _detect_centers(
            x, ycorr, step, prominence_frac, max_peaks, min_distance_deg)
    if not centers:
        raise ValueError("no peaks detected; lower prominence_frac or pass peak_centers")

    p0, lo, hi, scale = [], [], [], []
    for c in centers:
        j = int(np.argmin(np.abs(x - c)))
        amp0 = max(ycorr[j], noise)
        p0 += [amp0, c, init_fwhm_deg, 0.5]
        lo += [0.0, c - center_leeway_deg, max(2.0 * step, 0.02), 0.0]
        hi += [5.0 * amp0 + 1.0, c + center_leeway_deg, max_fwhm_deg, 1.0]
        # Per-parameter scale: amplitudes span ~1e5 while centers ~30 and FWHM
        # ~0.3. Without x_scale the TRF optimiser thrashes (seconds -> minutes
        # on busy frames). Scale each param by its natural magnitude.
        scale += [amp0, center_leeway_deg, init_fwhm_deg, 1.0]
    p0 += [0.0, 0.0]                       # linear baseline slope, intercept
    lo += [-np.inf, -np.inf]
    hi += [np.inf, np.inf]
    scale += [max(amp0, 1.0), max(ycorr.max(), 1.0)]

    try:
        popt, _ = curve_fit(
            _multi, x, ycorr, p0=p0, bounds=(lo, hi),
            x_scale=scale, ftol=1e-4, xtol=1e-4, maxfev=20000,
        )
    except (RuntimeError, ValueError) as e:
        # Last resort: looser convergence so a single hard frame in a series
        # returns *something* fittable rather than aborting the whole run.
        try:
            popt, _ = curve_fit(
                _multi, x, ycorr, p0=p0, bounds=(lo, hi),
                x_scale=scale, ftol=1e-2, xtol=1e-2, maxfev=40000,
            )
        except (RuntimeError, ValueError):
            raise RuntimeError(
                f"fit_pattern failed to converge on {len(centers)} peaks. Try "
                "fewer peaks (raise prominence_frac) or pass explicit peak_centers."
            ) from e

    fit_curve = _multi(x, *popt)
    resid = ycorr - fit_curve
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ycorr - ycorr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rms_over_noise = float(np.sqrt(np.mean(resid ** 2)) / noise)

    peaks = []
    for i in range(len(centers)):
        amp, cen, fwhm, eta = popt[4 * i:4 * i + 4]
        peaks.append({
            "center": float(cen), "fwhm": float(fwhm), "amplitude": float(amp),
            "eta": float(eta), "area": _pv_area(amp, fwhm, eta),
        })
    peaks.sort(key=lambda d: d["center"])

    return {
        "r_squared": float(r2),
        "residual_rms_over_noise": rms_over_noise,
        "n_peaks": len(centers),
        "peaks": peaks,
        "peak_centers": [p["center"] for p in peaks],
        "intensity_corrected": [float(v) for v in ycorr],
        "fit_curve": [float(v) for v in fit_curve],
        "noise_estimate": noise,
    }
