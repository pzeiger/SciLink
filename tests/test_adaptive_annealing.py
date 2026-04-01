"""
Tests for adaptive constraint annealing and refinement conformance.

Two targeted test groups:
  1. Adaptive escalation logic — verifies rate-based temperature escalation
     using mocked verification/refit calls (no real LLM needed).
  2. Refinement conformance — verifies the constraint annealing text is
     injected into the refinement prompt at each temperature level.
"""

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

os.environ["UNSAFE_EXECUTION_OK"] = "true"
logging.basicConfig(level=logging.INFO)

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_controller(r2_threshold=0.95, max_verification_iterations=7, max_model_retries=1):
    """Create a minimal UnifiedSeriesProcessingController with mocked deps."""
    mock_model = MagicMock()
    logger = logging.getLogger("test_adaptive")
    logger.setLevel(logging.INFO)

    ctrl = UnifiedSeriesProcessingController(
        model=mock_model,
        logger=logger,
        generation_config=None,
        safety_settings=None,
        parse_fn=lambda resp: (json.loads(resp.text), None),
        executor=MagicMock(),
        script_instructions="",
        correction_instructions="",
        quality_instructions="",
        output_dir="/tmp/test_adaptive_annealing",
        plot_fn=MagicMock(),
        r2_threshold=r2_threshold,
        max_model_retries=max_model_retries,
        enable_human_feedback=False,
        max_verification_iterations=max_verification_iterations,
        conformance_instructions="test conformance",
    )
    return ctrl


def simulate_adaptive_annealing(
    r2_initial: float,
    refit_r2_sequence: list[float],
    threshold: float = 0.95,
    max_iter: int = 7,
    n_levels: int = 3,
) -> list[int]:
    """
    Reproduce the exact adaptive escalation logic from the verification loop.

    Args:
        r2_initial: R² of the initial fit (before any verification)
        refit_r2_sequence: R² values returned by successive refits
        threshold: R² threshold
        max_iter: max_verification_iterations
        n_levels: number of annealing levels (3)

    Returns:
        List of annealing levels after each refit decision.
    """
    _annealing_level = 0
    best_r2 = r2_initial
    _prev_best_r2 = best_r2
    levels = []

    for verification_iter, verified_r2 in enumerate(refit_r2_sequence):
        # Only promote if improved (mirrors the fix)
        if verified_r2 > best_r2:
            best_r2 = verified_r2

        improvement = verified_r2 - _prev_best_r2
        remaining = max(max_iter - verification_iter - 1, 1)
        required_rate = max(threshold - best_r2, 0.0) / remaining

        if improvement < required_rate:
            _annealing_level = min(_annealing_level + 1, n_levels - 1)

        _prev_best_r2 = best_r2
        levels.append(_annealing_level)

    return levels


# ===========================================================================
# TEST GROUP 1: Adaptive escalation logic (pure, no LLM)
# ===========================================================================

def test_steady_improvement_stays_frozen():
    """When R² improves on pace to reach threshold, level stays at 0."""
    levels = simulate_adaptive_annealing(
        r2_initial=0.80,
        refit_r2_sequence=[0.85, 0.89, 0.92, 0.94, 0.96],
        threshold=0.95,
        max_iter=7,
    )
    print(f"  Steady improvement: levels={levels}")
    assert all(l == 0 for l in levels), f"Expected all level 0, got {levels}"
    print("  PASS")


def test_stalled_escalates():
    """When R² barely moves, should escalate quickly to max."""
    levels = simulate_adaptive_annealing(
        r2_initial=0.80,
        refit_r2_sequence=[0.81, 0.815, 0.818, 0.820, 0.821],
        threshold=0.95,
        max_iter=7,
    )
    print(f"  Stalled: levels={levels}")
    assert levels[0] >= 1, f"Expected escalation on first stalled refit, got level {levels[0]}"
    assert levels[-1] == 2, f"Expected max level (2) by end, got {levels[-1]}"
    print("  PASS")


def test_good_start_then_stall():
    """Improvement starts strong then stalls — escalation should happen mid-run."""
    levels = simulate_adaptive_annealing(
        r2_initial=0.80,
        refit_r2_sequence=[0.88, 0.91, 0.915, 0.917],
        threshold=0.95,
        max_iter=7,
    )
    print(f"  Good start then stall: levels={levels}")
    # First two refits are on pace → level 0
    assert levels[0] == 0, f"Expected level 0 after large improvement, got {levels[0]}"
    assert levels[1] == 0, f"Expected level 0 after second improvement, got {levels[1]}"
    # Later refits stall → should escalate
    assert levels[-1] >= 1, f"Expected escalation after stalling, got {levels[-1]}"
    print("  PASS")


def test_regression_escalates():
    """When a refit makes R² worse, should escalate."""
    levels = simulate_adaptive_annealing(
        r2_initial=0.85,
        refit_r2_sequence=[0.80, 0.82],  # worse then partial recovery
        threshold=0.95,
        max_iter=7,
    )
    print(f"  Regression: levels={levels}")
    assert levels[0] >= 1, f"Expected escalation after regression, got {levels[0]}"
    print("  PASS")


def test_above_threshold_stays():
    """When already above threshold, any non-negative improvement keeps level."""
    levels = simulate_adaptive_annealing(
        r2_initial=0.96,
        refit_r2_sequence=[0.965, 0.97],
        threshold=0.95,
        max_iter=7,
    )
    print(f"  Above threshold: levels={levels}")
    # required_rate = max(0.95 - best_r2, 0) / remaining = 0 → any improvement >= 0 stays
    assert all(l == 0 for l in levels), f"Expected all level 0 above threshold, got {levels}"
    print("  PASS")


def test_level_caps_at_max():
    """Level should never exceed n_levels - 1 (=2)."""
    levels = simulate_adaptive_annealing(
        r2_initial=0.50,
        refit_r2_sequence=[0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        threshold=0.95,
        max_iter=7,
    )
    print(f"  Cap at max: levels={levels}")
    assert max(levels) == 2, f"Expected max level 2, got {max(levels)}"
    # Should reach level 2 quickly and stay there
    assert levels[-1] == 2
    print("  PASS")


def test_rate_depends_on_threshold():
    """Higher threshold should require faster progress to avoid escalation."""
    # Same R² sequence, different thresholds
    levels_low = simulate_adaptive_annealing(
        r2_initial=0.80,
        refit_r2_sequence=[0.84, 0.87, 0.89],
        threshold=0.90,
        max_iter=7,
    )
    levels_high = simulate_adaptive_annealing(
        r2_initial=0.80,
        refit_r2_sequence=[0.84, 0.87, 0.89],
        threshold=0.99,
        max_iter=7,
    )
    print(f"  Rate vs threshold: low_thresh={levels_low}, high_thresh={levels_high}")
    # High threshold should escalate sooner (larger gap to close)
    assert max(levels_high) >= max(levels_low), \
        f"Higher threshold should escalate more: {levels_high} vs {levels_low}"
    print("  PASS")


# ===========================================================================
# TEST GROUP 2: Refinement prompt includes constraint annealing text
# ===========================================================================

def test_refinement_prompt_has_t0_constraint():
    """At annealing level 0, refinement prompt should include 'LOCKED' directive."""
    ctrl = make_controller()

    # Mock LLM to capture the prompt
    captured_prompts = []

    def capture_prompt(contents, **kwargs):
        captured_prompts.append(contents[0] if contents else "")
        resp = MagicMock()
        resp.text = json.dumps({
            "physical_model": "2 Gaussians",  # no change
            "fitting_strategy": "same",
        })
        return resp

    ctrl.model.generate_content = capture_prompt

    state = {
        "locked_fitting_config": {
            "physical_model": "2 Gaussian peaks on linear baseline",
            "fitting_strategy": "lmfit least_squares",
        },
        "_annealing_level": 0,
    }

    verification = {
        "recommended_action": "Reposition peak centers",
        "issues_found": [{"location": "peak_1", "problem": "shifted", "suggested_fix": "move center"}],
    }

    ctrl._apply_llm_verification_feedback(state, verification)

    assert len(captured_prompts) == 1, "Expected exactly one LLM call"
    prompt = captured_prompts[0]

    # T=0 text should contain "LOCKED"
    assert "LOCKED" in prompt, (
        f"T=0 refinement prompt missing 'LOCKED' constraint.\nPrompt excerpt:\n{prompt[:500]}"
    )
    print("  T=0 constraint in refinement: PASS")
    return prompt


def test_refinement_prompt_has_t2_freedom():
    """At annealing level 2, refinement prompt should include 'full freedom' directive."""
    ctrl = make_controller()

    captured_prompts = []

    def capture_prompt(contents, **kwargs):
        captured_prompts.append(contents[0] if contents else "")
        resp = MagicMock()
        resp.text = json.dumps({
            "physical_model": "3 Voigt peaks on linear baseline",
            "fitting_strategy": "lmfit",
        })
        return resp

    ctrl.model.generate_content = capture_prompt

    state = {
        "locked_fitting_config": {
            "physical_model": "3 Gaussian peaks on linear baseline",
            "fitting_strategy": "lmfit least_squares",
        },
        "_annealing_level": 2,
    }

    verification = {
        "recommended_action": "Switch to Voigt profiles",
        "issues_found": [{"location": "all_peaks", "problem": "wrong tails", "suggested_fix": "use Voigt"}],
    }

    ctrl._apply_llm_verification_feedback(state, verification)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    assert "full freedom" in prompt, (
        f"T=2 refinement prompt missing 'full freedom' text.\nPrompt excerpt:\n{prompt[:500]}"
    )
    # Should NOT contain LOCKED
    assert "LOCKED" not in prompt, "T=2 prompt should not contain LOCKED"
    print("  T=2 freedom in refinement: PASS")
    return prompt


def test_refinement_t0_blocks_model_change():
    """At T=0, if the LLM tries to change physical_model, the locked config
    should still reflect the constraint (the LLM should respect it).

    This test verifies the prompt is correct. Whether the LLM actually
    obeys is tested via the end-to-end tests.
    """
    ctrl = make_controller()

    captured_prompts = []

    def capture_prompt(contents, **kwargs):
        captured_prompts.append(contents[0] if contents else "")
        resp = MagicMock()
        # LLM respects constraint and doesn't change model
        resp.text = json.dumps({
            "physical_model": "2 Gaussian peaks on linear baseline",
            "fitting_strategy": "adjusted initial guesses",
            "parameters_to_extract": ["center_1", "center_2"],
        })
        return resp

    ctrl.model.generate_content = capture_prompt

    state = {
        "locked_fitting_config": {
            "physical_model": "2 Gaussian peaks on linear baseline",
            "fitting_strategy": "lmfit least_squares",
        },
        "_annealing_level": 0,
    }

    verification = {
        "recommended_action": "Add 2 more Gaussian peaks",
        "issues_found": [
            {"location": "residuals", "problem": "4 peaks visible but only 2 modeled",
             "suggested_fix": "add 2 more Gaussians"}
        ],
    }

    result_config = ctrl._apply_llm_verification_feedback(state, verification)

    prompt = captured_prompts[0]
    # Verify the T=0 constraint AND the original model are both in the prompt
    assert "LOCKED" in prompt
    assert "2 Gaussian peaks" in prompt

    # The mock LLM respected the constraint — model unchanged
    assert "2 Gaussian" in result_config.get("physical_model", ""), \
        f"Model should remain 2 Gaussians at T=0, got: {result_config.get('physical_model')}"
    print("  T=0 blocks model change: PASS")


# ===========================================================================
# TEST GROUP 3: Verification loop integration (mocked LLM + refit)
# ===========================================================================

def test_verification_loop_escalation():
    """
    Integration test: mock _verify_fit_with_llm and _fit_single_spectrum
    to control the R² trajectory. Verify the loop escalates at the right time.

    Scenario: R² = 0.80 -> 0.82 -> 0.83 -> 0.835 (stalling)
    Expected: starts at level 0, escalates after first stalled refit
    """
    ctrl = make_controller(r2_threshold=0.95, max_verification_iterations=5)

    refit_r2_values = iter([0.82, 0.83, 0.835])
    observed_levels = []

    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        observed_levels.append(annealing_level)
        return {
            "fit_acceptable": False,
            "issues_found": [{"location": "residuals", "problem": "systematic", "suggested_fix": "adjust"}],
            "recommended_action": "Adjust peak widths",
            "overall_assessment": "Needs improvement",
        }

    def mock_apply_feedback(state, verification):
        # Return a slightly different config each time to avoid the
        # "no config changes" branch
        config = state.get("locked_fitting_config", {}).copy()
        config["_tweak"] = config.get("_tweak", 0) + 1
        return config

    def mock_fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script):
        try:
            r2 = next(refit_r2_values)
        except StopIteration:
            return {"success": False}
        return {
            "success": True,
            "fit_quality": {"r_squared": r2},
            "visualization_path": None,
        }

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._apply_llm_verification_feedback = mock_apply_feedback
    ctrl._fit_single_spectrum = mock_fit
    ctrl._log_verification_issues = MagicMock()

    # Simulate the verification loop manually (extracted from run())
    best_r2 = 0.80
    best_result = {"success": True, "fit_quality": {"r_squared": 0.80}}
    state = {
        "locked_fitting_config": {"physical_model": "3 Gaussians", "_tweak": 0},
        "_annealing_level": 0,
    }
    verification_history = []

    _annealing_level = 0
    _prev_best_r2 = best_r2
    _n_anneal_levels = len(ctrl._CONSTRAINT_ANNEALING_SCHEDULE)

    for verification_iter in range(ctrl.max_verification_iterations):
        verification = ctrl._verify_fit_with_llm(
            state, best_result,
            history=verification_history,
            verification_iter=verification_iter,
            annealing_level=_annealing_level,
        )

        if verification is None:
            break

        _cur_level = _annealing_level

        verification_history.append({
            "r_squared": best_r2,
            "annealing_level": _cur_level,
            "issues_found": verification.get("issues_found", []),
            "overall_assessment": verification.get("overall_assessment", ""),
            "recommended_action": verification.get("recommended_action", ""),
        })

        if verification.get("fit_acceptable", True):
            break

        refined_config = ctrl._apply_llm_verification_feedback(state, verification)

        if refined_config == state.get("locked_fitting_config", {}):
            _annealing_level = min(_annealing_level + 1, _n_anneal_levels - 1)
            if _annealing_level == _cur_level:
                break
            continue

        state["locked_fitting_config"] = refined_config
        state["_annealing_level"] = _annealing_level

        verified_result = ctrl._fit_single_spectrum(
            state=state, curve_data=None, data_path="",
            spectrum_name="test", spectrum_idx=0, base_script=None,
        )

        if verified_result["success"]:
            verified_r2 = verified_result["fit_quality"]["r_squared"]

            if verified_r2 > best_r2:
                best_r2 = verified_r2
                best_result = verified_result

            improvement = verified_r2 - _prev_best_r2
            remaining = max(ctrl.max_verification_iterations - verification_iter - 1, 1)
            required_rate = max(ctrl.r2_threshold - best_r2, 0.0) / remaining
            if improvement < required_rate:
                _annealing_level = min(_annealing_level + 1, _n_anneal_levels - 1)
            _prev_best_r2 = best_r2
        else:
            break

    print(f"  Loop integration: observed_levels={observed_levels}")
    print(f"  Final best_r2={best_r2:.4f}")

    # Should start at 0 and escalate
    assert observed_levels[0] == 0, f"Should start at level 0, got {observed_levels[0]}"
    assert max(observed_levels) >= 1, f"Should have escalated, max level was {max(observed_levels)}"
    # Level should be monotonically non-decreasing
    for i in range(1, len(observed_levels)):
        assert observed_levels[i] >= observed_levels[i - 1], \
            f"Levels should be non-decreasing: {observed_levels}"
    print("  PASS")


# ===========================================================================
# Runner
# ===========================================================================

def run_group(name, tests):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    passed = 0
    failed = 0
    for test_fn in tests:
        print(f"\n  [{test_fn.__name__}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    return passed, failed


if __name__ == "__main__":
    total_pass, total_fail = 0, 0

    p, f = run_group("GROUP 1: Adaptive escalation logic (pure)", [
        test_steady_improvement_stays_frozen,
        test_stalled_escalates,
        test_good_start_then_stall,
        test_regression_escalates,
        test_above_threshold_stays,
        test_level_caps_at_max,
        test_rate_depends_on_threshold,
    ])
    total_pass += p
    total_fail += f

    p, f = run_group("GROUP 2: Refinement conformance", [
        test_refinement_prompt_has_t0_constraint,
        test_refinement_prompt_has_t2_freedom,
        test_refinement_t0_blocks_model_change,
    ])
    total_pass += p
    total_fail += f

    p, f = run_group("GROUP 3: Verification loop integration", [
        test_verification_loop_escalation,
    ])
    total_pass += p
    total_fail += f

    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed")
    print(f"{'=' * 70}")

    sys.exit(1 if total_fail > 0 else 0)
