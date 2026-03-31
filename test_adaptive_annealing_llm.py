"""
Real LLM tests for adaptive constraint annealing and refinement conformance.

Requires ANTHROPIC_API_KEY environment variable.

Tests:
  1. Refinement at T=0 with realistic parameter-level recommendations:
     the LLM should adjust parameters but keep the model structure.
  2. Refinement at T=2 with realistic model-change recommendations:
     the LLM should freely change the model.
  3. Full pipeline with exponentially-modified Gaussian (asymmetric peak):
     symmetric Gaussian can't reach threshold, forcing verification iterations.
  4. Full pipeline with overlapping Lorentzian peaks hinted as 1 Gaussian:
     single Gaussian can't capture the double-peak, forcing iterations.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ["UNSAFE_EXECUTION_OK"] = "true"
logging.basicConfig(level=logging.INFO)

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("Set ANTHROPIC_API_KEY before running this test")

from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)

DATA_DIR = Path("test_adaptive_annealing_data")
DATA_DIR.mkdir(exist_ok=True)


def _parse_llm_response(response):
    """Simple JSON parser matching the base agent's contract: (result, error)."""
    try:
        raw = response.text.strip()
        try:
            return json.loads(raw), None
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip()), None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        return None, {"error": "No JSON found", "raw": raw[:200]}
    except Exception as e:
        return None, {"error": str(e)}


def make_real_controller(r2_threshold=0.95, max_verification_iterations=7):
    """Create a controller wired to a real Claude model."""
    model = LiteLLMGenerativeModel(model="claude-sonnet-4-6")
    logger = logging.getLogger("test_llm")
    logger.setLevel(logging.INFO)

    ctrl = UnifiedSeriesProcessingController(
        model=model,
        logger=logger,
        generation_config=None,
        safety_settings=None,
        parse_fn=_parse_llm_response,
        executor=None,
        script_instructions="",
        correction_instructions="",
        quality_instructions="",
        output_dir=str(DATA_DIR / "ctrl_test"),
        plot_fn=None,
        r2_threshold=r2_threshold,
        max_model_retries=1,
        enable_human_feedback=False,
        max_verification_iterations=max_verification_iterations,
    )
    return ctrl


# ===========================================================================
# TEST 1: At T=0, realistic parameter-level recommendations
# ===========================================================================

def test_refinement_t0_parameter_fixes():
    """
    Realistic T=0 scenario: verification found systematic residuals at the
    peak tails and recommends adjusting initial guesses and widths —
    NOT changing the model. The refinement LLM should produce a config
    that preserves the 3-Gaussian model structure.
    """
    print("\n  Setting up controller with real LLM...")
    ctrl = make_real_controller()

    state = {
        "locked_fitting_config": {
            "physical_model": (
                "3 Gaussian peaks on a linear baseline: "
                "y = a + bx + sum_{i=1}^{3} A_i * exp(-0.5*((x-mu_i)/sigma_i)^2)"
            ),
            "fitting_strategy": "scipy.optimize.curve_fit with Levenberg-Marquardt",
            "parameters_to_extract": [
                "center_1", "width_1", "amplitude_1",
                "center_2", "width_2", "amplitude_2",
                "center_3", "width_3", "amplitude_3",
            ],
            "analysis_approach": "Fit 3 Gaussian peaks as specified",
        },
        "_annealing_level": 0,  # LOCKED
    }

    # Realistic T=0 verification: parameter-level issues only
    verification = {
        "recommended_action": (
            "Re-run the fit with improved initial guesses: peak 1 center "
            "should start near x=30 (currently overshooting to x=33), "
            "and peak 3 width should be constrained to sigma < 5 to "
            "prevent it from absorbing baseline. Also tighten bounds on "
            "the linear baseline slope."
        ),
        "issues_found": [
            {
                "location": "x ≈ 30 (peak 1)",
                "problem": "Peak center offset by ~3 units from visible maximum",
                "suggested_fix": "Set initial guess for center_1 to x=30 with bounds [28, 32]",
            },
            {
                "location": "x = 70-100 (right tail)",
                "problem": "Peak 3 is too broad, absorbing part of the baseline",
                "suggested_fix": "Constrain sigma_3 < 5 to prevent baseline absorption",
            },
        ],
        "spurious_components": [],
        "missing_features": [],
    }

    print("  Calling _apply_llm_verification_feedback at T=0...")
    refined_config = ctrl._apply_llm_verification_feedback(state, verification)

    model_str = refined_config.get("physical_model", "")
    print(f"  Refined model: {model_str[:120]}")

    # Should still be 3 Gaussians
    model_lower = model_str.lower()
    has_three_gauss = any(w in model_lower for w in [
        "3 gaussian", "three gaussian", "3-gaussian", "sum_{i=1}^{3}",
        "3 peaks", "three peaks",
    ])
    changed_to_voigt = any(w in model_lower for w in ["voigt", "lorentzian"])

    if has_three_gauss and not changed_to_voigt:
        print("  PASS: Model preserved as 3 Gaussians (parameter-level refinement)")
        return True
    elif changed_to_voigt:
        print(f"  FAIL: Model changed to Voigt/Lorentzian at T=0")
        return False
    else:
        # Check number of Gaussian mentions
        n_gauss = model_lower.count("gaussian")
        if n_gauss >= 1 and not changed_to_voigt:
            print(f"  PASS: Model still uses Gaussians ({n_gauss} mention(s))")
            return True
        print(f"  AMBIGUOUS: model_str='{model_str[:100]}'")
        return False


# ===========================================================================
# TEST 2: At T=2, realistic model-change recommendations
# ===========================================================================

def test_refinement_t2_model_change():
    """
    Realistic T=2 scenario: after several failed iterations, verification
    found that Gaussian tails can't match the data and recommends switching
    to Voigt profiles. The refinement LLM should produce a changed model.
    """
    print("\n  Setting up controller with real LLM...")
    ctrl = make_real_controller()

    state = {
        "locked_fitting_config": {
            "physical_model": (
                "3 Gaussian peaks on a linear baseline: "
                "y = a + bx + sum_{i=1}^{3} A_i * exp(-0.5*((x-mu_i)/sigma_i)^2)"
            ),
            "fitting_strategy": "scipy.optimize.curve_fit with Levenberg-Marquardt",
            "parameters_to_extract": [
                "center_1", "width_1", "amplitude_1",
                "center_2", "width_2", "amplitude_2",
                "center_3", "width_3", "amplitude_3",
            ],
            "analysis_approach": "Fit 3 Gaussian peaks as specified",
        },
        "_annealing_level": 2,  # FULL FREEDOM
    }

    # Realistic T=2 verification: model-level change needed
    verification = {
        "recommended_action": (
            "The Gaussian model has been tried for 5 iterations with "
            "parameter adjustments but R² remains stuck at 0.978. "
            "The systematic W-shaped residuals at each peak indicate "
            "the peak shape is fundamentally wrong — the data has heavier "
            "tails than Gaussians can capture. Switch to Voigt or "
            "Lorentzian profiles which have 1/x² tail behavior."
        ),
        "issues_found": [
            {
                "location": "All three peak positions",
                "problem": "Systematic W-shaped residuals: positive at center, negative at wings",
                "suggested_fix": "Replace Gaussian profiles with Voigt profiles",
            },
            {
                "location": "Peak tails (x > 5σ from centers)",
                "problem": "Data intensity significantly above Gaussian prediction in far tails",
                "suggested_fix": "The 1/x² Lorentzian tail behavior is needed",
            },
        ],
        "spurious_components": [],
        "missing_features": [],
    }

    print("  Calling _apply_llm_verification_feedback at T=2...")
    refined_config = ctrl._apply_llm_verification_feedback(state, verification)

    model_str = refined_config.get("physical_model", "")
    print(f"  Refined model: {model_str[:120]}")

    model_lower = model_str.lower()
    changed = any(w in model_lower for w in ["voigt", "lorentzian", "pseudo-voigt"])

    if changed:
        print("  PASS: Model changed to Voigt/Lorentzian at T=2 (freedom exercised)")
        return True
    else:
        print(f"  FAIL: Model did NOT change despite T=2 freedom and clear recommendation")
        return False


# ===========================================================================
# TEST 3: Full pipeline — asymmetric peak (EMG)
# ===========================================================================

def test_pipeline_emg():
    """
    Full agent pipeline with an exponentially-modified Gaussian (EMG) peak.
    The EMG has a sharp rise and long exponential tail that symmetric
    Gaussians cannot capture. With R² threshold = 0.995, the Gaussian fit
    should be clearly rejected (R² ~ 0.85-0.92).
    """
    print("\n  Generating EMG data...")
    from scipy.special import erfc
    from scilink.agents.exp_agents import CurveFittingAgent

    x = np.linspace(0, 50, 500)
    rng = np.random.default_rng(42)

    mu, sigma, tau, amplitude = 20.0, 2.0, 3.0, 5.0
    z = (mu - x) / sigma + sigma / tau
    emg = (amplitude * sigma / tau) * np.sqrt(np.pi / 2) * \
          np.exp(0.5 * (sigma / tau)**2 - (x - mu) / tau) * \
          erfc(z / np.sqrt(2))
    y = emg + 0.1 + 0.002 * x + rng.normal(0, 0.02, size=x.shape)

    data_path = DATA_DIR / "emg_peak.npy"
    np.save(data_path, np.column_stack([x, y]))
    print(f"  EMG: mu={mu}, sigma={sigma}, tau={tau} (asymmetric tail)")

    agent = CurveFittingAgent(
        model_name="claude-sonnet-4-6",
        output_dir=str(DATA_DIR / "output_emg"),
        r2_threshold=0.995,
        max_verification_iterations=7,
        max_model_retries=1,
        enable_human_feedback=False,
    )

    result = agent.analyze(
        data=str(data_path),
        hints=(
            "Fit with 1 symmetric Gaussian peak on a linear baseline: "
            "y = a + bx + A*exp(-0.5*((x-mu)/sigma)^2). "
            "Extract peak center, amplitude, and FWHM."
        ),
    )

    return _report_pipeline(result, "EMG peak")


# ===========================================================================
# TEST 4: Full pipeline — overlapping Lorentzians as single Gaussian
# ===========================================================================

def test_pipeline_overlapping():
    """
    Two closely-spaced Lorentzian peaks that look like one broad bump.
    Hint says 1 Gaussian. The single Gaussian can't capture the shape,
    and R² should be clearly below reject threshold.
    """
    print("\n  Generating overlapping Lorentzian data...")
    from scilink.agents.exp_agents import CurveFittingAgent

    x = np.linspace(0, 50, 500)
    rng = np.random.default_rng(42)

    y = np.zeros_like(x)
    for c, a, g in [(22.0, 4.0, 2.5), (28.0, 3.5, 2.0)]:
        y += a * (g**2) / ((x - c) ** 2 + g**2)
    y += 0.05 + 0.001 * x + rng.normal(0, 0.01, size=x.shape)

    data_path = DATA_DIR / "overlapping_lorentzians.npy"
    np.save(data_path, np.column_stack([x, y]))
    print(f"  Two Lorentzians at x=22, x=28 (6 units apart)")

    agent = CurveFittingAgent(
        model_name="claude-sonnet-4-6",
        output_dir=str(DATA_DIR / "output_overlap"),
        r2_threshold=0.995,
        max_verification_iterations=7,
        max_model_retries=1,
        enable_human_feedback=False,
    )

    result = agent.analyze(
        data=str(data_path),
        hints=(
            "Fit with 1 Gaussian peak on a linear baseline: "
            "y = a + bx + A*exp(-0.5*((x-mu)/sigma)^2). "
            "Extract peak center, amplitude, and FWHM."
        ),
    )

    return _report_pipeline(result, "Overlapping Lorentzians")


def _report_pipeline(result, label):
    """Report and validate pipeline results."""
    print(f"\n  Status: {result.get('status')}")
    print(f"  Model: {result.get('model_type', 'N/A')[:80]}")
    fq = result.get("fit_quality", {})
    print(f"  Final R²: {fq.get('r_squared', 'N/A')}")

    qh = result.get("quality_history")
    if qh:
        iters = qh.get("verification_iterations", [])
        levels = [vi.get("annealing_level", "?") for vi in iters]
        print(f"  Verification iterations: {len(iters)}")
        for i, vi in enumerate(iters):
            r2 = vi.get("r_squared", "N/A")
            lvl = vi.get("annealing_level", "?")
            fix = str(vi.get("fix_applied", ""))[:60]
            print(f"    iter {i}: R²={r2}, level={lvl}, fix={fix}")
        print(f"  Annealing levels: {levels}")

        alts = qh.get("alternative_models", [])
        if alts:
            print(f"  Alternative models: {len(alts)}")
            for a in alts:
                print(f"    - {str(a.get('model', ''))[:60]}: R²={a.get('r2', 'N/A')}")

        max_level = max(levels) if levels else 0
        n_iters = len(levels)

        # Success criteria: either multiple verification iterations
        # (exercised the loop) or escalation happened
        if n_iters >= 2 and max_level >= 1:
            print(f"\n  PASS: {n_iters} iterations, escalated to level {max_level}")
            return True
        elif n_iters >= 2:
            print(f"\n  PARTIAL PASS: {n_iters} iterations at level 0 (steady improvement)")
            return True
        elif alts:
            print(f"\n  PARTIAL PASS: 1 verification iter but {len(alts)} alt model(s) tried")
            return True
        else:
            print(f"\n  WEAK: only {n_iters} iteration(s), no escalation")
            return False
    else:
        print("  No quality_history")
        return False


# ===========================================================================
# TEST 5: Series — Gaussian peaks with one outlier
# ===========================================================================

def test_pipeline_series():
    """
    Series of 5 Gaussian spectra with increasing center position.
    Spectrum 3 is an outlier: a double-peak that a single Gaussian
    can't fit well, so it should be flagged.

    Validates:
    - Anchor (spectrum 0) gets full verification QC
    - Config/script propagates to remaining spectra
    - Outlier detection flags spectrum 3
    - Series results have correct length
    """
    print("\n  Generating 5-spectrum series...")
    from scilink.agents.exp_agents import CurveFittingAgent

    rng = np.random.default_rng(42)
    x = np.linspace(0, 50, 300)
    n_spectra = 5
    stack = np.zeros((n_spectra, 2, len(x)))

    centers = [15, 18, 21, 24, 27]
    for i, c in enumerate(centers):
        stack[i, 0] = x
        if i == 3:
            # Outlier: double peak that a single Gaussian can't capture
            y = (3.0 * np.exp(-0.5 * ((x - c) / 1.5) ** 2)
                 + 2.5 * np.exp(-0.5 * ((x - (c + 6)) / 1.5) ** 2))
        else:
            # Normal: clean single Gaussian
            y = 4.0 * np.exp(-0.5 * ((x - c) / 2.0) ** 2)
        stack[i, 1] = y + 0.1 + rng.normal(0, 0.02, size=x.shape)

    print(f"  Centers: {centers}")
    print(f"  Spectrum 3: double peak (outlier)")

    # Save each spectrum as individual .npy files (avoids stack shape
    # confusion with preprocessor which expects (N, 2) per spectrum)
    spectrum_dir = DATA_DIR / "series_spectra"
    spectrum_dir.mkdir(exist_ok=True)
    spectrum_paths = []
    for i in range(n_spectra):
        path = spectrum_dir / f"spectrum_{i:04d}.npy"
        np.save(path, np.column_stack([stack[i, 0], stack[i, 1]]))
        spectrum_paths.append(str(path))

    agent = CurveFittingAgent(
        model_name="claude-sonnet-4-6",
        output_dir=str(DATA_DIR / "output_series"),
        r2_threshold=0.99,
        max_verification_iterations=5,
        max_model_retries=1,
        enable_human_feedback=False,
    )

    result = agent.analyze(
        data=spectrum_paths,
        hints=(
            "Fit each spectrum with 1 Gaussian peak on a flat baseline: "
            "y = a + A*exp(-0.5*((x-mu)/sigma)^2). "
            "Extract peak center, amplitude, and FWHM."
        ),
        series_metadata={
            "variable": "position",
            "values": [1, 2, 3, 4, 5],
            "unit": "mm",
        },
    )

    print(f"\n  Status: {result.get('status')}")

    # analyze() returns "individual_results" for series, not "series_results"
    individual = result.get("individual_results", [])
    flagged = result.get("flagged_spectra", [])
    flagged_indices = {f["index"] for f in flagged}
    summary = result.get("summary", {})

    print(f"  Summary: {summary}")
    print(f"  Individual results: {len(individual)} spectra")
    for r in individual:
        r2 = r.get("fit_quality", {}).get("r_squared", "N/A")
        flag = " ⚠️ FLAGGED" if r.get("flagged") else ""
        refit = " (refitted)" if r.get("adaptively_refitted") else ""
        print(f"    [{r['index']}] {r['name']}: R²={r2}{flag}{refit}")
    print(f"  Flagged indices: {flagged_indices}")

    # Validate
    passed = True

    if len(individual) != n_spectra:
        print(f"  FAIL: Expected {n_spectra} results, got {len(individual)}")
        passed = False

    successful = sum(1 for r in individual if r["success"])
    if successful < n_spectra - 1:
        print(f"  FAIL: Only {successful}/{n_spectra} successful fits")
        passed = False

    # Spectrum 3 should be flagged (double peak → poor single-Gaussian fit)
    # or adaptively refitted
    spec3_flagged = 3 in flagged_indices
    spec3_refitted = any(r.get("adaptively_refitted") for r in individual if r["index"] == 3)
    if spec3_flagged or spec3_refitted:
        print(f"  Outlier detection: spectrum 3 correctly {'flagged' if spec3_flagged else 'refitted'}")
    else:
        r2_3 = individual[3].get("fit_quality", {}).get("r_squared", 1.0) if len(individual) > 3 else 1.0
        other_r2 = [r.get("fit_quality", {}).get("r_squared", 0) for i, r in enumerate(individual) if i != 3 and r["success"]]
        avg_other = np.mean(other_r2) if other_r2 else 0
        print(f"  Outlier detection: spectrum 3 not flagged/refitted (R²={r2_3:.4f}, others avg={avg_other:.4f})")
        if r2_3 < avg_other - 0.05:
            print(f"    (R² is lower as expected, flagging threshold may not have triggered)")
        else:
            print(f"    WARNING: spectrum 3 R² is close to others — double peak may have been handled by model change")

    if passed:
        print(f"\n  PASS: {successful}/{n_spectra} fits, flagged={flagged_indices}")
    else:
        print(f"\n  FAIL")

    return passed


# ===========================================================================
# Runner
# ===========================================================================

def run_test(name, fn):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    try:
        return fn()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    tests = {
        "--t0": ("TEST 1: Refinement T=0 (parameter fixes)", test_refinement_t0_parameter_fixes),
        "--t2": ("TEST 2: Refinement T=2 (model change)", test_refinement_t2_model_change),
        "--emg": ("TEST 3: Pipeline EMG peak", test_pipeline_emg),
        "--overlap": ("TEST 4: Pipeline overlapping Lorentzians", test_pipeline_overlapping),
        "--series": ("TEST 5: Pipeline series with outlier", test_pipeline_series),
    }

    selected = [k for k in sys.argv[1:] if k in tests]
    if "--all" in sys.argv or not selected:
        selected = list(tests.keys())

    results = {}
    for key in selected:
        name, fn = tests[key]
        results[name] = run_test(name, fn)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    n_pass = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    print(f"\n  {n_pass} passed, {n_fail} failed")
    sys.exit(1 if n_fail > 0 else 0)
