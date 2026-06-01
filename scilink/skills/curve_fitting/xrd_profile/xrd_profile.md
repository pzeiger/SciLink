---
description: p-XRD profile fitting — per-peak pseudo-Voigt fits with FWHM, position, and intensity; Scherrer crystallite size and Williamson-Hall microstrain from the fitted line widths.
technique: [XRD, "X-ray diffraction", "powder diffraction", pXRD]
quality_gate:
  metric: r_squared
  accept_threshold: 0.95
  hard_reject_threshold: 0.85
  direction: higher_is_better
---
# p-XRD Profile Fitting Skill

## overview

Powder X-ray diffraction profile fitting for line-broadening physics.
Per-peak pseudo-Voigt fits yield calibrated peak positions, intensities,
and full widths at half maximum (FWHMs). Those widths then feed two
classical analyses:

- **Scherrer equation** for an average crystallite (coherent domain) size
  from any single peak's broadening:
  `D = K · λ / (β · cos θ)`.
- **Williamson-Hall (W-H) plot** for separating size and strain
  contributions from the 2θ dependence of broadening:
  `β · cos θ = K · λ / D + 4 · ε · sin θ`.

This skill is the *profile* half of p-XRD analysis. The *identification*
half — matching observed peaks to a candidate crystal phase — lives in
the separate `structure_matching/xrd` skill. The two are designed to be
co-activated: pass `skill=["xrd", "xrd_profile"]` and the LLM can chain
profile fitting + phase identification in one analysis script.

**Out of scope:** Rietveld refinement (atomic positions / occupancies);
quantitative phase-fraction analysis (covered in
`structure_matching/xrd`'s multi-phase MIP path); texture / preferred
orientation correction; whole-pattern Le Bail fitting.

## planning

**How many peaks to fit:**
- Scherrer alone: 3–5 strongest, well-isolated peaks is enough — report
  size per peak and the mean.
- W-H plot: at least 5 peaks spread across the 2θ range. Peaks clustered
  at similar 2θ collapse the regression's lever arm; the slope estimate
  becomes degenerate. Refuse W-H and fall back to peak-by-peak Scherrer
  when sin θ range < 0.1.

**Background first, fit second.** Subtract a background (`fit_background`
with `method='snip'` is the standard p-XRD choice) before per-peak
fitting. SNIP handles smooth amorphous backgrounds and fluorescence
floors without imposing a polynomial shape. Use `method='polynomial'`
only when a polynomial is genuinely the right model (capillary
scattering on a flat baseline).

**Line shape: pseudo-Voigt as default.** A pseudo-Voigt mixes Gaussian
and Lorentzian contributions through a single mixing parameter `eta` —
**eta = 0 is pure Gaussian, eta = 1 is pure Lorentzian** (lmfit
convention). Pseudo-Voigt captures the experimental and physical
broadenings of typical p-XRD patterns without the slower numerical
convolution of a true Voigt. Switch to pure Lorentzian (`model='lorentzian'`)
only when fitted eta consistently lands ≥ 0.9 across peaks (rare).

**Instrumental broadening subtraction.** Both Scherrer and Williamson-
Hall require the *sample* broadening, not the total. Subtract the
instrumental FWHM in quadrature:
`β_sample² = FWHM_total² − FWHM_instrumental²`. The instrumental FWHM
comes from a standard reference pattern (LaB₆, Si, Al₂O₃) measured on
the same instrument. Pass it as the `instrumental_fwhm_deg` argument to
`scherrer` and `williamson_hall`. When unknown, default to 0.0 and flag
the result as an upper-bound on broadening (lower-bound on size).

**Peak windowing.** Fit each peak over a 2θ window roughly 5–8 × the
expected FWHM, centered on the peak. Too narrow → background slope
biases the fit; too wide → neighboring peaks intrude. The `window_deg`
argument to `fit_profile` defaults to 1.0° (typical CuKa FWHM ~0.2°);
widen for nanocrystalline samples with FWHM > 0.5°.

**Overlapping peaks.** When two peaks lie within 1.5 × FWHM of each
other, fit them jointly (two pseudo-Voigt components in one
`fit_profile` window) rather than sequentially. Sequential fitting
double-counts the overlap region.

**Scherrer K constant.** Default `K = 0.9` (spherical crystallites,
FWHM-based). Use `K = 0.94` for cubic crystallites or when explicit in
the literature reference. The choice is a < 5% correction; do not
agonize over it.

**Pairing with `structure_matching/xrd`.** When both skills are active,
the recommended chaining is: `extract_peaks` (from `structure_matching/
xrd`) seeds peak centers → `fit_profile` per peak → use the refined
FWHMs as `exp_peaks={'positions': [...], 'amplitudes': [...],
'fwhms': [...]}` for `score_xrd_match_robust`. The score gets sharper
broadening per peak instead of the default uniform FWHM, which matters
for nanocrystalline patterns with peaks several times broader than the
0.15° default.

## implementation

**CRITICAL: profile fitting workflow.** The per-item analysis script
must follow this sequence:

1. Load experimental 2θ + intensity arrays.
2. Subtract background via `fit_background`.
3. Detect candidate peak centers via `extract_peaks` (reused from
   `structure_matching/xrd`) — this gives a starting list to fit.
4. For each peak, fit a pseudo-Voigt via `fit_profile` on a window
   around the center.
5. Per-peak Scherrer crystallite size via `scherrer`.
6. If ≥ 5 peaks span a useful 2θ range, run `williamson_hall` for
   size + strain.
7. Emit `FIT_RESULTS_JSON: {...}` with per-peak results + aggregated
   metrics (mean per-peak R², Scherrer mean, W-H size/strain).

**Complete profile-fitting template:**

```python
import json
import numpy as np

from scilink.skills._shared.curve_fitting_tools import load_curve_data
from scilink.skills.curve_fitting.xrd_profile.background import fit_background
from scilink.skills.curve_fitting.xrd_profile.fit_profile import fit_profile
from scilink.skills.curve_fitting.xrd_profile.scherrer import scherrer
from scilink.skills.curve_fitting.xrd_profile.williamson_hall import williamson_hall
from scilink.skills.structure_matching.xrd.extract_peaks import extract_peaks

# ---- Step 1: Load ----
data = load_curve_data(DATA_PATH)  # ndarray with X in col 0, Y in col 1
two_theta = np.asarray(data[:, 0], dtype=float)
intensity = np.asarray(data[:, 1], dtype=float)

WAVELENGTH_ANGSTROM = 1.5406  # CuKa1; replace from metadata if available
INSTRUMENTAL_FWHM_DEG = 0.05  # from LaB6/Si standard; 0.0 if unknown

# ---- Step 2: Background subtraction ----
bg = fit_background(two_theta.tolist(), intensity.tolist(), method='snip')
intensity_sub = np.asarray(bg['intensity_corrected'], dtype=float)

# ---- Step 3: Seed peak list ----
seed = extract_peaks(
    two_theta.tolist(), intensity_sub.tolist(),
    prominence_frac=0.02, max_peaks=10,
)
peak_centers = seed['positions']

# ---- Step 4: Per-peak pseudo-Voigt fit ----
fits = []
for center in peak_centers:
    result = fit_profile(
        exp_two_theta=two_theta.tolist(),
        exp_intensity=intensity_sub.tolist(),
        peak_init=center,
        model='pseudo_voigt',
        window_deg=1.0,
        background='linear',  # residual baseline inside the window
    )
    fits.append(result)

# ---- Step 5: Per-peak Scherrer size ----
sizes_nm = []
for f in fits:
    if f['r_squared'] < 0.9:
        continue  # don't trust Scherrer on a bad fit
    s = scherrer(
        fwhm_deg=f['fwhm'],
        two_theta_deg=f['center'],
        wavelength_angstrom=WAVELENGTH_ANGSTROM,
        instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
    )
    sizes_nm.append(s['size_nm'])
mean_size_nm = float(np.mean(sizes_nm)) if sizes_nm else None

# ---- Step 6: Williamson-Hall (optional) ----
wh_input = [
    {'two_theta': f['center'], 'fwhm': f['fwhm']}
    for f in fits if f['r_squared'] >= 0.9
]
wh = None
if len(wh_input) >= 5:
    wh = williamson_hall(
        peaks=wh_input,
        wavelength_angstrom=WAVELENGTH_ANGSTROM,
        instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
    )

# ---- Step 7: Emit ----
mean_r2 = float(np.mean([f['r_squared'] for f in fits])) if fits else 0.0
print("FIT_RESULTS_JSON: " + json.dumps({
    "peaks": [
        {k: f[k] for k in ('center', 'fwhm', 'amplitude', 'eta', 'r_squared')}
        for f in fits
    ],
    "scherrer_mean_size_nm": mean_size_nm,
    "scherrer_per_peak_nm": sizes_nm,
    "williamson_hall": wh,
    "fit_quality": {
        "r_squared": mean_r2,
        "verdict": "accept" if mean_r2 >= 0.95 else (
            "marginal" if mean_r2 >= 0.85 else "reject"
        ),
        "n_peaks_fitted": len(fits),
        "n_peaks_used_for_scherrer": len(sizes_nm),
    },
}))
```

**Joint fitting for overlapping peaks.** When `fit_profile` returns
`r_squared < 0.9` and the residual shows a shoulder on one side of the
peak, refit that window with two pseudo-Voigt components by passing
`peak_init=[center1, center2]` (list of two starting centers). The tool
constructs a composite model.

**NumPy compatibility.** Use `np.trapezoid` (not removed `np.trapz`).

## interpretation

**Crystallite size ranges from peak FWHM (CuKa, 2θ ≈ 30°):**

| FWHM (deg) | Size (nm) | Regime |
|------------|-----------|--------|
| < 0.10 | > 100 | Coarse / well-crystallized; instrumental-limited |
| 0.10–0.30 | 30–100 | Typical microcrystalline |
| 0.30–1.00 | 10–30 | Nanocrystalline |
| > 1.00 | < 10 | Strongly nano / poorly crystallized |

These are rough; the exact size depends on 2θ via the cos θ factor. For
peaks at higher 2θ, the same FWHM in degrees corresponds to a smaller
crystallite.

**Strain vs size from Williamson-Hall slope:**
- Slope ≈ 0 (flat W-H plot): broadening dominated by crystallite size;
  strain is negligible. Report size only.
- Positive slope: real microstrain. Slope value `m = 4ε` gives strain
  directly. Typical ε in [0.0005, 0.005] for metals and oxides;
  > 0.01 suggests defects, alloying, or measurement issues.
- Negative slope: usually unphysical — typically indicates instrumental
  miscalibration, bad background subtraction, or peak misassignment.
  Re-check fits and instrumental FWHM before reporting.

**Inconsistencies to flag:**
- Per-peak Scherrer sizes vary by > 3×: anisotropic broadening
  (anisotropic crystallite shape, hkl-dependent strain). Report the
  *range* of per-peak sizes, not just the mean, and note that an
  isotropic Scherrer mean is an oversimplification.
- W-H linearity R² < 0.7: the linear model doesn't apply. Possible
  causes: anisotropic broadening, mixed-phase sample (some peaks
  broadened by phase A, others by phase B), or fits with large
  uncertainty on FWHM.

**Quantitative confidence.** Scherrer gives an *average* coherent
domain size; for log-normal size distributions the Scherrer size is
closer to a volume-weighted mean than a number mean. Quote sizes to
2 significant figures; ±15% is the typical accuracy on a well-
calibrated instrument.

**Cross-skill follow-up.** If profile fitting succeeded but the
crystal phase has not yet been identified, recommend running
`structure_matching/xrd` next on the same pattern. The
refined FWHMs from this skill can be passed in to sharpen the scoring.

## validation

**Per-peak fit checks:**
- `r_squared` ≥ 0.95 for each fit accepts; 0.85–0.95 marginal; below
  0.85 reject the fit and either widen the window or drop the peak.
- FWHM sanity: 0.05° (typical instrumental floor for a lab CuKa
  diffractometer) ≤ FWHM ≤ 2.0° (very small crystallites; peak overlap
  dominates above this).
- `eta` (Gaussian-Lorentzian mixing) must be in [0, 1]. A fit that
  returns eta exactly at 0 or 1 with high uncertainty usually
  indicates the data don't constrain the mixing — fall back to a
  pure Gaussian or pure Lorentzian fit and compare R².
- Amplitude must be positive. Negative fitted amplitudes mean the
  initial center was wrong or background subtraction overshot —
  re-extract peaks and re-window.

**Scherrer sanity:**
- Size > 1000 nm: the peak is instrument-limited, not crystallite-
  limited. Report as "size > 100 nm (resolution-limited)" rather than a
  number.
- Size < 1 nm: physically unreasonable for a crystalline solid. Indicates
  one of: instrumental FWHM not subtracted, severe strain mistaken for
  size broadening, or peak overlap in the fit window.

**Williamson-Hall sanity:**
- W-H R² ≥ 0.9 for confident size + strain decomposition.
- W-H R² in [0.7, 0.9] reports the values with a caveat that the
  decomposition is unstable.
- W-H R² < 0.7 do not report the decomposition; fall back to per-peak
  Scherrer and report the size range only.

**Instrumental-broadening subtraction sanity:**
- If `β_sample² ≤ 0` (instrumental FWHM ≥ measured FWHM), the peak is
  resolution-limited. Report "size below sensitivity limit
  (~ Scherrer with β = instrumental)" rather than NaN or a complex
  number.
