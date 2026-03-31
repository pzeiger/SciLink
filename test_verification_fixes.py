"""
Tests for verification loop fixes:
  1. Config sync — best_config stays in sync with best_result
  2. Always-promote — latest refit replaces best_result regardless of R²
  3. fit_was_approved bypass — verifier approval skips R² threshold check
  4. Judge indexing — 1-indexed prompt, 0-indexed code, bounds check
  5. Parameterized accept_threshold in verifier prompt
"""

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["UNSAFE_EXECUTION_OK"] = "true"
logging.basicConfig(level=logging.INFO)

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_controller(r2_threshold=0.95, max_verification_iterations=5, max_model_retries=0):
    """Create a minimal controller with mocked deps."""
    mock_model = MagicMock()
    logger = logging.getLogger("test_verification_fixes")
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
        output_dir="/tmp/test_verification_fixes",
        plot_fn=MagicMock(),
        r2_threshold=r2_threshold,
        max_model_retries=max_model_retries,
        enable_human_feedback=False,
        max_verification_iterations=max_verification_iterations,
        conformance_instructions="",
    )
    return ctrl


def make_fit_result(r2, config_model="Model A", success=True):
    """Create a minimal fit result dict."""
    return {
        "success": success,
        "fit_quality": {"r_squared": r2},
        "model_type": config_model,
        "parameters": {},
        "visualization_path": None,
        "visualization_bytes": b"fake_png",
        "script": "# fake",
        "script_errors": [],
        "index": 0,
        "name": "test_spectrum",
    }


# ===========================================================================
# TEST GROUP 1: Config sync (issue #1)
# ===========================================================================

def test_config_syncs_after_nonim_proving_refit():
    """After a refit that doesn't improve R², state config should be
    restored to the config that produced best_result."""
    ctrl = make_controller(r2_threshold=0.95, max_verification_iterations=3)

    # Track configs seen by verify calls
    configs_at_verify = []
    call_count = [0]

    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        configs_at_verify.append(state.get("locked_fitting_config", {}).copy())
        return {
            "fit_acceptable": False,
            "issues_found": [{"location": "peak", "problem": "shifted", "suggested_fix": "move"}],
            "recommended_action": "Adjust parameters",
            "overall_assessment": "Needs work",
        }

    def mock_apply_feedback(state, verification):
        config = state.get("locked_fitting_config", {}).copy()
        call_count[0] += 1
        config["physical_model"] = f"Refined-{call_count[0]}"
        return config

    # R² sequence: 0.90 (initial) -> 0.92 (improved) -> 0.88 (worse)
    refit_iter = iter([0.92, 0.88])

    def mock_fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None):
        try:
            r2 = next(refit_iter)
        except StopIteration:
            return {"success": False}
        return make_fit_result(r2, config_model=state.get("locked_fitting_config", {}).get("physical_model", "?"))

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._apply_llm_verification_feedback = mock_apply_feedback
    ctrl._fit_single_spectrum = mock_fit
    ctrl._suggest_alternative_model = MagicMock(return_value=None)
    ctrl._log_verification_issues = MagicMock()

    import numpy as np
    state = {
        "locked_fitting_config": {"physical_model": "Original"},
        "_annealing_level": 0,
    }

    result = ctrl._fit_with_quality_control(
        state=state, curve_data=np.zeros(10), data_path="test.npy",
        spectrum_name="test", spectrum_idx=0,
    )

    # After iter 1: refit R²=0.92 > 0.90 → promoted, config="Refined-1"
    # After iter 2: refit R²=0.88 < 0.92 → NOT promoted, but config should
    #   be restored to "Refined-1" (the config that produced best)
    # Iter 3: verify should see config="Refined-1", not "Refined-2"

    if len(configs_at_verify) >= 3:
        third_verify_config = configs_at_verify[2]
        assert third_verify_config.get("physical_model") == "Refined-1", (
            f"Expected config 'Refined-1' at 3rd verify (synced to best), "
            f"got '{third_verify_config.get('physical_model')}'"
        )
    print(f"  Configs at verify: {[c.get('physical_model') for c in configs_at_verify]}")
    print("  PASS")


# ===========================================================================
# TEST GROUP 2: Always-promote (issue #2)
# ===========================================================================

def test_always_promote_lower_r2_refit():
    """A refit with lower R² should still become best_result so the
    verifier examines it next iteration."""
    ctrl = make_controller(r2_threshold=0.99, max_verification_iterations=3)

    verified_results = []

    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        verified_results.append(fit_result.get("fit_quality", {}).get("r_squared"))
        if verification_iter == 2:
            return {"fit_acceptable": True}  # approve on 3rd check
        return {
            "fit_acceptable": False,
            "issues_found": [{"location": "all", "problem": "spurious", "suggested_fix": "remove"}],
            "recommended_action": "Remove spurious component",
            "overall_assessment": "Overfitting",
        }

    def mock_apply_feedback(state, verification):
        config = state.get("locked_fitting_config", {}).copy()
        config["_iter"] = config.get("_iter", 0) + 1
        return config

    # R² goes DOWN: 0.99 (initial, spurious) -> 0.93 (refit, cleaner)
    fit_r2_sequence = iter([0.99, 0.93])

    def mock_fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None):
        try:
            r2 = next(fit_r2_sequence)
        except StopIteration:
            return {"success": False}
        return make_fit_result(r2)

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._apply_llm_verification_feedback = mock_apply_feedback
    ctrl._fit_single_spectrum = mock_fit
    ctrl._suggest_alternative_model = MagicMock(return_value=None)
    ctrl._log_verification_issues = MagicMock()

    import numpy as np
    state = {
        "locked_fitting_config": {"physical_model": "Overfitted model"},
        "_annealing_level": 0,
    }

    result = ctrl._fit_with_quality_control(
        state=state, curve_data=np.zeros(10), data_path="test.npy",
        spectrum_name="test", spectrum_idx=0,
    )

    # Iter 1: verify initial (R²=0.99) → rejected
    # Refit → R²=0.93 → always-promote makes this best_result
    # Iter 2: verify R²=0.93 (the new best, not 0.99 again)
    print(f"  R² values verified: {verified_results}")
    assert verified_results[0] == 0.99, f"First verify should see initial R²=0.99"
    assert len(verified_results) >= 2 and verified_results[1] == 0.93, (
        f"Second verify should see promoted R²=0.93, got {verified_results[1] if len(verified_results) >= 2 else 'N/A'}"
    )
    print("  PASS")


def test_fit_was_approved_bypasses_threshold():
    """When verifier approves a fit, result should be returned even if
    R² is below the threshold."""
    ctrl = make_controller(r2_threshold=0.95, max_verification_iterations=2)

    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        # Approve immediately
        return {"fit_acceptable": True}

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._fit_single_spectrum = lambda *a, **kw: make_fit_result(0.85)  # below threshold
    ctrl._suggest_alternative_model = MagicMock(return_value=None)

    import numpy as np
    state = {
        "locked_fitting_config": {"physical_model": "Simple model"},
        "_annealing_level": 0,
    }

    result = ctrl._fit_with_quality_control(
        state=state, curve_data=np.zeros(10), data_path="test.npy",
        spectrum_name="test", spectrum_idx=0,
    )

    # Should return the approved fit despite R²=0.85 < 0.95
    assert result["success"], "Should return successful result"
    assert result["fit_quality"]["r_squared"] == 0.85
    # Should NOT have gone through alternative models
    ctrl._suggest_alternative_model.assert_not_called()
    print("  Approved R²=0.85 returned despite threshold=0.95: PASS")


def test_broken_high_r2_fit_displaced():
    """A broken fit with R²=1.0 should be displaced by a real fit with
    lower R² so the verifier can examine the real fit."""
    ctrl = make_controller(r2_threshold=0.95, max_verification_iterations=3)

    verified_r2s = []

    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        r2 = fit_result.get("fit_quality", {}).get("r_squared", 0)
        verified_r2s.append(r2)
        if r2 < 1.0 and r2 > 0.9:
            return {"fit_acceptable": True}  # approve real fit
        return {
            "fit_acceptable": False,
            "issues_found": [{"location": "entire", "problem": "trivial line fit", "suggested_fix": "real model"}],
            "recommended_action": "Use proper peak model",
            "overall_assessment": "Broken: fitting a line to peaked data",
        }

    def mock_apply_feedback(state, verification):
        config = state.get("locked_fitting_config", {}).copy()
        config["physical_model"] = "Real peak model"
        return config

    refit_iter = iter([0.94])  # real fit

    def mock_fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None):
        if state.get("locked_fitting_config", {}).get("physical_model") == "Real peak model":
            try:
                r2 = next(refit_iter)
            except StopIteration:
                return {"success": False}
            return make_fit_result(r2, "Real peak model")
        return make_fit_result(1.0, "Broken line")  # initial broken fit

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._apply_llm_verification_feedback = mock_apply_feedback
    ctrl._fit_single_spectrum = mock_fit
    ctrl._suggest_alternative_model = MagicMock(return_value=None)
    ctrl._log_verification_issues = MagicMock()

    import numpy as np
    state = {
        "locked_fitting_config": {"physical_model": "Linear"},
        "_annealing_level": 0,
    }

    result = ctrl._fit_with_quality_control(
        state=state, curve_data=np.zeros(10), data_path="test.npy",
        spectrum_name="test", spectrum_idx=0,
    )

    print(f"  R² values verified: {verified_r2s}")
    # First verify: R²=1.0 (broken) → rejected
    # Refit: R²=0.94 (real) → promoted (always-promote)
    # Second verify: R²=0.94 → approved
    assert verified_r2s[0] == 1.0, "Should first verify broken fit"
    assert any(r2 == 0.94 for r2 in verified_r2s), "Should verify real fit after promotion"
    assert result["fit_quality"]["r_squared"] == 0.94, (
        f"Should return real fit R²=0.94, got {result['fit_quality']['r_squared']}"
    )
    print("  PASS")


# ===========================================================================
# TEST GROUP 3: Judge indexing (off-by-one fix + bounds check)
# ===========================================================================

def test_judge_1indexed_converted_to_0indexed():
    """Judge returns 1-indexed selected_index (matching prompt labels),
    code should convert to 0-indexed."""
    ctrl = make_controller()

    # Mock LLM to return selected_index=2 (meaning "Attempt 2")
    resp = MagicMock()
    resp.text = json.dumps({
        "selected_index": 2,
        "acceptable": True,
        "reasoning": "Attempt 2 has better residuals",
        "issues_with_selected": None,
    })
    ctrl.model.generate_content = MagicMock(return_value=resp)

    attempts = [
        {"model": "Model A", "r2": 0.90, "result": make_fit_result(0.90)},
        {"model": "Model B", "r2": 0.93, "result": make_fit_result(0.93)},
        {"model": "Model C", "r2": 0.88, "result": make_fit_result(0.88)},
    ]

    result = ctrl._judge_select_best_fit(attempts)

    # selected_index=2 (1-indexed) → 1 (0-indexed) → "Model B"
    assert result["selected_index"] == 1, (
        f"Expected 0-indexed 1 (Attempt 2), got {result['selected_index']}"
    )
    print(f"  Judge returned 2, converted to index 1: PASS")


def test_judge_out_of_range_returns_none():
    """Judge returns an out-of-range index → should be set to None."""
    ctrl = make_controller()

    resp = MagicMock()
    resp.text = json.dumps({
        "selected_index": 10,  # way out of range
        "acceptable": True,
        "reasoning": "Selected attempt 10",
        "issues_with_selected": None,
    })
    ctrl.model.generate_content = MagicMock(return_value=resp)

    attempts = [
        {"model": "Model A", "r2": 0.90, "result": make_fit_result(0.90)},
        {"model": "Model B", "r2": 0.93, "result": make_fit_result(0.93)},
    ]

    result = ctrl._judge_select_best_fit(attempts)

    assert result["selected_index"] is None, (
        f"Expected None for out-of-range index, got {result['selected_index']}"
    )
    print("  Out-of-range index → None: PASS")


def test_judge_zero_index_returns_none():
    """Judge returns 0 (confused about indexing) → after -1 = -1, out of range."""
    ctrl = make_controller()

    resp = MagicMock()
    resp.text = json.dumps({
        "selected_index": 0,
        "acceptable": True,
        "reasoning": "First attempt",
        "issues_with_selected": None,
    })
    ctrl.model.generate_content = MagicMock(return_value=resp)

    attempts = [
        {"model": "Model A", "r2": 0.90, "result": make_fit_result(0.90)},
    ]

    result = ctrl._judge_select_best_fit(attempts)

    assert result["selected_index"] is None, (
        f"Expected None for index 0 (converts to -1), got {result['selected_index']}"
    )
    print("  Index 0 → None (out of range after conversion): PASS")


def test_judge_null_index_passthrough():
    """Judge returns null → should stay None."""
    ctrl = make_controller()

    resp = MagicMock()
    resp.text = json.dumps({
        "selected_index": None,
        "acceptable": False,
        "reasoning": "All unacceptable",
        "issues_with_selected": None,
    })
    ctrl.model.generate_content = MagicMock(return_value=resp)

    attempts = [
        {"model": "Model A", "r2": 0.90, "result": make_fit_result(0.90)},
    ]

    result = ctrl._judge_select_best_fit(attempts)

    assert result["selected_index"] is None, (
        f"Expected None passthrough, got {result['selected_index']}"
    )
    print("  Null index passthrough: PASS")


# ===========================================================================
# TEST GROUP 4: Parameterized accept_threshold in prompt (issue #3 / GH #118)
# ===========================================================================

def test_verification_prompt_uses_parameterized_threshold():
    """The FIT_VERIFICATION_PROMPT reminder line should use the
    parameterized threshold, not hardcoded 0.98."""
    ctrl = make_controller(r2_threshold=0.995)

    # Format the prompt as the code does
    prompt = ctrl.FIT_VERIFICATION_PROMPT.format(
        r_squared=0.99,
        model_type="Test",
        n_components=1,
        parameters="{}",
        accept_threshold=ctrl.r2_threshold,
        reject_threshold=ctrl.r2_threshold - 0.05,
    )

    assert "0.98" not in prompt, (
        "Prompt still contains hardcoded 0.98 — should use parameterized threshold"
    )
    # The reminder line should have the user's threshold
    assert "R² > 1.00" in prompt or "R² > 0.99" in prompt or f"R² > {ctrl.r2_threshold:.2f}" in prompt, (
        f"Prompt should contain 'R² > {ctrl.r2_threshold:.2f}' but doesn't.\n"
        f"Last 200 chars: ...{prompt[-200:]}"
    )
    print(f"  Prompt uses threshold={ctrl.r2_threshold}: PASS")


# ===========================================================================
# TEST GROUP 5: Spectrum series
# ===========================================================================

def test_series_anchor_config_propagates_to_followers():
    """In a 3-spectrum series, after the anchor's verification loop updates
    the config, non-anchor spectra should use the anchor's base_script
    (which embeds the updated model)."""
    import numpy as np

    ctrl = make_controller(r2_threshold=0.95, max_verification_iterations=2, max_model_retries=0)

    # Track which scripts non-anchor spectra receive
    scripts_used = []

    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        if verification_iter == 0:
            return {
                "fit_acceptable": False,
                "issues_found": [{"location": "peak", "problem": "wrong", "suggested_fix": "fix"}],
                "recommended_action": "Change model",
                "overall_assessment": "Needs fixing",
            }
        return {"fit_acceptable": True}

    def mock_apply_feedback(state, verification):
        config = state.get("locked_fitting_config", {}).copy()
        config["physical_model"] = "Improved model"
        return config

    fit_call_count = [0]

    def mock_fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None):
        fit_call_count[0] += 1
        if base_script is not None:
            scripts_used.append({"idx": spectrum_idx, "base_script": base_script})
        model = state.get("locked_fitting_config", {}).get("physical_model", "Unknown")
        return {
            "success": True,
            "fit_quality": {"r_squared": 0.96},
            "model_type": model,
            "parameters": {},
            "visualization_path": None,
            "visualization_bytes": b"fake",
            "script": f"# script for {model}",
            "script_errors": [],
            "index": spectrum_idx,
            "name": spectrum_name,
        }

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._apply_llm_verification_feedback = mock_apply_feedback
    ctrl._fit_single_spectrum = mock_fit
    ctrl._log_verification_issues = MagicMock()

    os.makedirs("/tmp/test_verification_fixes", exist_ok=True)

    state = {
        "locked_fitting_config": {"physical_model": "Original model"},
        "_annealing_level": 0,
        "num_spectra": 3,
        "is_single_spectrum": False,
        "spectrum_stack": np.random.rand(3, 2, 50),
        "analysis_images": [],
        "original_plot_bytes": b"fake_plot",
        "data_statistics": {"n_points": 50, "x_range": [0, 1], "y_range": [0, 1]},
    }

    result_state = ctrl.execute(state)

    series_results = result_state.get("series_results", [])
    assert len(series_results) == 3, f"Expected 3 results, got {len(series_results)}"
    assert all(r["success"] for r in series_results), "All spectra should succeed"

    # Non-anchor spectra (idx 1, 2) should have received a base_script
    # that contains the improved model (from anchor's QC)
    assert len(scripts_used) >= 1, "Non-anchor spectra should use base_script"
    for entry in scripts_used:
        assert "Improved model" in entry["base_script"], (
            f"Spectrum {entry['idx']} got base_script without improved model: {entry['base_script'][:50]}"
        )
    print(f"  Series: {len(series_results)} results, {len(scripts_used)} used base_script")
    print("  PASS")


def test_series_approved_anchor_no_alternatives():
    """When the anchor fit is verifier-approved, the series should proceed
    without trying alternative models — even if R² is below threshold."""
    import numpy as np

    ctrl = make_controller(r2_threshold=0.98, max_verification_iterations=2, max_model_retries=1)

    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        return {"fit_acceptable": True}  # approve immediately

    def mock_fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None):
        return {
            "success": True,
            "fit_quality": {"r_squared": 0.93},  # below threshold
            "model_type": "Simple",
            "parameters": {},
            "visualization_path": None,
            "visualization_bytes": b"fake",
            "script": "# simple fit",
            "script_errors": [],
            "index": spectrum_idx,
            "name": spectrum_name,
        }

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._fit_single_spectrum = mock_fit
    ctrl._suggest_alternative_model = MagicMock(return_value=None)
    ctrl._log_verification_issues = MagicMock()

    os.makedirs("/tmp/test_verification_fixes", exist_ok=True)

    state = {
        "locked_fitting_config": {"physical_model": "Simple"},
        "_annealing_level": 0,
        "num_spectra": 3,
        "is_single_spectrum": False,
        "spectrum_stack": np.random.rand(3, 2, 50),
        "analysis_images": [],
        "original_plot_bytes": b"fake_plot",
        "data_statistics": {"n_points": 50, "x_range": [0, 1], "y_range": [0, 1]},
    }

    result_state = ctrl.execute(state)

    series_results = result_state.get("series_results", [])
    assert len(series_results) == 3, f"Expected 3 results, got {len(series_results)}"

    # Alternative models should NOT have been tried (verifier approved anchor)
    ctrl._suggest_alternative_model.assert_not_called()

    # Anchor result should have quality_history with approved=True
    anchor = series_results[0]
    qh = anchor.get("quality_history", {})
    assert qh.get("approved") is True, (
        f"Anchor quality_history.approved should be True, got {qh.get('approved')}"
    )
    assert qh.get("approved_by") == "verifier", (
        f"Anchor should be approved_by='verifier', got {qh.get('approved_by')}"
    )
    print(f"  Series: anchor approved by verifier, no alternatives tried")
    print("  PASS")


def test_series_outlier_detection_with_always_promote():
    """Outlier detection should still work correctly after always-promote
    changes — flagging spectra with low R² relative to the series."""
    import numpy as np

    ctrl = make_controller(r2_threshold=0.95, max_verification_iterations=1, max_model_retries=0)

    # No verification for simplicity — approve immediately
    def mock_verify(state, fit_result, history=None, verification_iter=0, annealing_level=None):
        return {"fit_acceptable": True}

    # Spectrum 0: R²=0.96, Spectrum 1: R²=0.95, Spectrum 2: R²=0.70 (outlier)
    r2_by_idx = {0: 0.96, 1: 0.95, 2: 0.70}

    def mock_fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None):
        r2 = r2_by_idx.get(spectrum_idx, 0.95)
        return {
            "success": True,
            "fit_quality": {"r_squared": r2},
            "model_type": "Gaussian",
            "parameters": {},
            "visualization_path": None,
            "visualization_bytes": b"fake",
            "script": "# fit script",
            "script_errors": [],
            "index": spectrum_idx,
            "name": f"spectrum_{spectrum_idx:04d}",
        }

    ctrl._verify_fit_with_llm = mock_verify
    ctrl._fit_single_spectrum = mock_fit
    ctrl._log_verification_issues = MagicMock()

    os.makedirs("/tmp/test_verification_fixes", exist_ok=True)

    state = {
        "locked_fitting_config": {"physical_model": "Gaussian"},
        "_annealing_level": 0,
        "num_spectra": 3,
        "is_single_spectrum": False,
        "spectrum_stack": np.random.rand(3, 2, 50),
        "analysis_images": [],
        "original_plot_bytes": b"fake_plot",
        "data_statistics": {"n_points": 50, "x_range": [0, 1], "y_range": [0, 1]},
    }

    result_state = ctrl.execute(state)

    flagged = result_state.get("flagged_spectra", [])
    flagged_indices = {f["index"] for f in flagged}

    # Spectrum 2 (R²=0.70) should be flagged
    assert 2 in flagged_indices, (
        f"Spectrum 2 (R²=0.70) should be flagged, flagged indices: {flagged_indices}"
    )
    # Spectra 0 and 1 should NOT be flagged
    assert 0 not in flagged_indices, "Spectrum 0 (R²=0.96) should not be flagged"
    assert 1 not in flagged_indices, "Spectrum 1 (R²=0.95) should not be flagged"

    print(f"  Series: flagged={flagged_indices}, correct outlier detection")
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
            import traceback
            traceback.print_exc()
            failed += 1
    return passed, failed


if __name__ == "__main__":
    total_pass, total_fail = 0, 0

    p, f = run_group("GROUP 1: Config sync", [
        test_config_syncs_after_nonim_proving_refit,
    ])
    total_pass += p
    total_fail += f

    p, f = run_group("GROUP 2: Always-promote + fit_was_approved", [
        test_always_promote_lower_r2_refit,
        test_fit_was_approved_bypasses_threshold,
        test_broken_high_r2_fit_displaced,
    ])
    total_pass += p
    total_fail += f

    p, f = run_group("GROUP 3: Judge indexing", [
        test_judge_1indexed_converted_to_0indexed,
        test_judge_out_of_range_returns_none,
        test_judge_zero_index_returns_none,
        test_judge_null_index_passthrough,
    ])
    total_pass += p
    total_fail += f

    p, f = run_group("GROUP 4: Parameterized threshold in prompt (GH #118)", [
        test_verification_prompt_uses_parameterized_threshold,
    ])
    total_pass += p
    total_fail += f

    p, f = run_group("GROUP 5: Spectrum series", [
        test_series_anchor_config_propagates_to_followers,
        test_series_approved_anchor_no_alternatives,
        test_series_outlier_detection_with_always_promote,
    ])
    total_pass += p
    total_fail += f

    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed")
    print(f"{'=' * 70}")

    sys.exit(1 if total_fail > 0 else 0)
