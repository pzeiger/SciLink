"""
Regression tests for the non-anchor parallel fan-out in
UnifiedSeriesProcessingController.

These tests exercise the regime-aware series loop with mocked
``_fit_with_quality_control`` and ``_fit_single_spectrum`` so they run
fast and deterministically. They verify:

  * Serial mode (parallel_workers=1) is byte-equivalent to the
    pre-feature behavior: anchors run in order, each non-anchor gets its
    regime's base_script inline, results are tagged and ordered correctly.
  * Parallel mode (parallel_workers>1) routes non-anchors through a
    thread pool but produces the same regime tagging, the same
    per-spectrum base_script binding, the same result count, and the
    same in-order series_results list.
  * Multi-regime case: each regime anchor produces its own base_script,
    and non-anchors of regime R receive base_scripts[R], not the
    base_script of a different regime.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
logging.basicConfig(level=logging.INFO)

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)


def _make_controller(parallel_workers: int | None = None, output_dir: str | Path = "/tmp/_pfanout"):
    return UnifiedSeriesProcessingController(
        model=MagicMock(),
        logger=logging.getLogger("test_parallel_fanout"),
        generation_config=None,
        safety_settings=None,
        parse_fn=lambda resp: (json.loads(resp.text), None),
        executor=MagicMock(),
        script_instructions="",
        correction_instructions="",
        quality_instructions="",
        output_dir=str(output_dir),
        plot_fn=MagicMock(return_value=b""),
        r2_threshold=0.9,
        enable_human_feedback=False,
        max_verification_iterations=1,
        parallel_workers=parallel_workers,
        conformance_instructions="",
    )


def _build_state(num_spectra: int, regimes: list[list[int]] | None = None,
                 regime_configs_per_regime: list[dict] | None = None) -> dict:
    """Build a synthetic state with optional regime structure."""
    x = np.linspace(280, 290, 51)
    stack = np.zeros((num_spectra, 2, len(x)))
    for i in range(num_spectra):
        stack[i, 0] = x
        stack[i, 1] = np.exp(-((x - 285 - 0.1 * i) ** 2) / 0.5) + 0.05

    state = {
        "num_spectra": num_spectra,
        "is_single_spectrum": (num_spectra == 1),
        "spectrum_stack": stack,
        "spectrum_paths": [],
        "system_info": {"technique": "synthetic"},
        "data_statistics": {},
        "original_plot_bytes": b"",
        "locked_fitting_config": {
            "analysis_approach": "two_gaussian",
            "physical_model": "two_gaussian_default",
            "parameters_to_extract": ["amp", "mu", "sigma"],
            "fitting_strategy": "default",
        },
        "skill_sections": {},
        "skill_name": "test",
    }
    if regimes is not None:
        plan = {"regimes": []}
        cfgs: dict[int, dict] = {}
        for i, indices in enumerate(regimes):
            name = f"R{i}"
            plan["regimes"].append({"name": name, "spectrum_indices": indices})
            cfg = (regime_configs_per_regime or [None] * len(regimes))[i] or {
                "analysis_approach": f"approach_{name}",
                "physical_model": f"model_{name}",
                "parameters_to_extract": [],
                "fitting_strategy": f"strategy_{name}",
            }
            for idx in indices:
                cfgs[idx] = cfg
        state["series_analysis_plan"] = plan
        state["regime_configs"] = cfgs
    return state


def _install_mock_fits(ctrl):
    """Replace fit methods with deterministic stubs that record what they saw.

    Anchor stub builds a script string that identifies the regime, so
    non-anchor stubs can assert they received the right base_script.
    """
    calls_lock = threading.Lock()
    qc_calls: list[dict] = []
    single_calls: list[dict] = []

    def mock_qc(state, curve_data, data_path, spectrum_name, spectrum_idx, is_regime_anchor=False):
        regime_name = "default"
        plan = state.get("series_analysis_plan") or {}
        for r in plan.get("regimes", []):
            if spectrum_idx in r.get("spectrum_indices", []):
                regime_name = r.get("name", "default")
                break
        script = f"# anchor-script for regime {regime_name}, idx={spectrum_idx}"
        with calls_lock:
            qc_calls.append({"idx": spectrum_idx, "regime": regime_name, "thread": threading.get_ident()})
        return {
            "index": spectrum_idx, "name": spectrum_name, "data_path": data_path,
            "success": True, "error": None, "model_type": "two_gaussian",
            "parameters": {}, "fit_quality": {"r_squared": 0.98},
            "deviation_note": None, "visualization_path": None,
            "visualization_bytes": None, "statistics": {},
            "script": script, "script_errors": [],
        }

    def mock_single(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None, **_):
        with calls_lock:
            single_calls.append({
                "idx": spectrum_idx,
                "base_script": base_script,
                "locked_cfg": state.get("locked_fitting_config", {}).get("physical_model"),
                "thread": threading.get_ident(),
            })
        # Slight sleep to make parallel speedup observable
        time.sleep(0.2)
        return {
            "index": spectrum_idx, "name": spectrum_name, "data_path": data_path,
            "success": True, "error": None, "model_type": "two_gaussian",
            "parameters": {}, "fit_quality": {"r_squared": 0.97},
            "deviation_note": None, "visualization_path": None,
            "visualization_bytes": None, "statistics": {},
            "script": base_script, "script_errors": [],
        }

    ctrl._fit_with_quality_control = mock_qc
    ctrl._fit_single_spectrum = mock_single
    return qc_calls, single_calls


# ---------------------------------------------------------------------------
# Single-regime cases
# ---------------------------------------------------------------------------


def test_serial_single_regime_preserves_order_and_count(tmp_path):
    ctrl = _make_controller(parallel_workers=1, output_dir=tmp_path)
    qc_calls, single_calls = _install_mock_fits(ctrl)
    state = _build_state(num_spectra=4)

    out = ctrl.execute(state)
    series = out["series_results"]

    assert [r["index"] for r in series] == [0, 1, 2, 3]
    assert all(r["success"] for r in series)
    assert len(qc_calls) == 1 and qc_calls[0]["idx"] == 0
    assert [c["idx"] for c in single_calls] == [1, 2, 3]
    # All non-anchors should have received the anchor's base_script
    assert all(c["base_script"] == "# anchor-script for regime default, idx=0" for c in single_calls)


def test_parallel_single_regime_preserves_order_and_uses_pool(tmp_path):
    ctrl = _make_controller(parallel_workers=3, output_dir=tmp_path)
    qc_calls, single_calls = _install_mock_fits(ctrl)
    state = _build_state(num_spectra=4)

    t0 = time.perf_counter()
    out = ctrl.execute(state)
    elapsed = time.perf_counter() - t0

    series = out["series_results"]
    assert [r["index"] for r in series] == [0, 1, 2, 3]
    assert all(r["success"] for r in series)
    assert len(qc_calls) == 1
    assert sorted(c["idx"] for c in single_calls) == [1, 2, 3]
    # Non-anchors should have run on at least 2 distinct threads when workers=3
    thread_ids = {c["thread"] for c in single_calls}
    assert len(thread_ids) >= 2, f"Expected >=2 worker threads, saw {thread_ids}"
    # Parallel run should be noticeably faster than 3 * 0.2s sleep
    assert elapsed < 0.55, f"Expected <0.55s with 3 parallel sleeps of 0.2s, got {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Multi-regime cases
# ---------------------------------------------------------------------------


def test_serial_multi_regime_each_anchor_gets_full_qc(tmp_path):
    ctrl = _make_controller(parallel_workers=1, output_dir=tmp_path)
    qc_calls, single_calls = _install_mock_fits(ctrl)
    state = _build_state(
        num_spectra=6,
        regimes=[[0, 1, 2], [3, 4, 5]],
        regime_configs_per_regime=[
            {"analysis_approach": "a", "physical_model": "mA", "parameters_to_extract": [], "fitting_strategy": "sA"},
            {"analysis_approach": "b", "physical_model": "mB", "parameters_to_extract": [], "fitting_strategy": "sB"},
        ],
    )

    out = ctrl.execute(state)
    series = out["series_results"]

    # 6 results in index order
    assert [r["index"] for r in series] == list(range(6))
    # 2 anchors (regime starts), 4 non-anchors
    assert sorted(c["idx"] for c in qc_calls) == [0, 3]
    assert sorted(c["idx"] for c in single_calls) == [1, 2, 4, 5]
    # Regime tagging on every result
    assert [r["regime"] for r in series] == ["R0", "R0", "R0", "R1", "R1", "R1"]
    # Non-anchors of R0 get R0's anchor script; same for R1
    by_idx = {c["idx"]: c for c in single_calls}
    assert by_idx[1]["base_script"] == "# anchor-script for regime R0, idx=0"
    assert by_idx[2]["base_script"] == "# anchor-script for regime R0, idx=0"
    assert by_idx[4]["base_script"] == "# anchor-script for regime R1, idx=3"
    assert by_idx[5]["base_script"] == "# anchor-script for regime R1, idx=3"


def test_parallel_multi_regime_routes_each_to_own_script(tmp_path):
    ctrl = _make_controller(parallel_workers=4, output_dir=tmp_path)
    qc_calls, single_calls = _install_mock_fits(ctrl)
    state = _build_state(
        num_spectra=6,
        regimes=[[0, 1, 2], [3, 4, 5]],
        regime_configs_per_regime=[
            {"analysis_approach": "a", "physical_model": "mA", "parameters_to_extract": [], "fitting_strategy": "sA"},
            {"analysis_approach": "b", "physical_model": "mB", "parameters_to_extract": [], "fitting_strategy": "sB"},
        ],
    )

    out = ctrl.execute(state)
    series = out["series_results"]

    assert [r["index"] for r in series] == list(range(6))
    assert sorted(c["idx"] for c in qc_calls) == [0, 3]
    assert sorted(c["idx"] for c in single_calls) == [1, 2, 4, 5]
    # Regime tagging
    assert [r["regime"] for r in series] == ["R0", "R0", "R0", "R1", "R1", "R1"]
    by_idx = {c["idx"]: c for c in single_calls}
    assert by_idx[1]["base_script"] == "# anchor-script for regime R0, idx=0"
    assert by_idx[2]["base_script"] == "# anchor-script for regime R0, idx=0"
    assert by_idx[4]["base_script"] == "# anchor-script for regime R1, idx=3"
    assert by_idx[5]["base_script"] == "# anchor-script for regime R1, idx=3"
    # Each non-anchor worker saw its regime's locked_fitting_config snapshot
    assert by_idx[1]["locked_cfg"] == "mA"
    assert by_idx[2]["locked_cfg"] == "mA"
    assert by_idx[4]["locked_cfg"] == "mB"
    assert by_idx[5]["locked_cfg"] == "mB"
    # Pool used
    thread_ids = {c["thread"] for c in single_calls}
    assert len(thread_ids) >= 2, f"Expected >=2 worker threads, saw {thread_ids}"


def test_parallel_single_spectrum_short_circuits_to_serial(tmp_path):
    """num_spectra=1 must always be serial regardless of parallel_workers."""
    ctrl = _make_controller(parallel_workers=8, output_dir=tmp_path)
    qc_calls, single_calls = _install_mock_fits(ctrl)
    state = _build_state(num_spectra=1)

    out = ctrl.execute(state)
    assert len(out["series_results"]) == 1
    assert qc_calls and qc_calls[0]["idx"] == 0
    assert not single_calls  # no non-anchors, no pool needed


def test_parallel_failed_non_anchor_does_not_corrupt_series(tmp_path):
    ctrl = _make_controller(parallel_workers=3, output_dir=tmp_path)

    qc_calls: list[dict] = []
    lock = threading.Lock()

    def mock_qc(state, curve_data, data_path, spectrum_name, spectrum_idx, is_regime_anchor=False):
        with lock:
            qc_calls.append(spectrum_idx)
        return {
            "index": spectrum_idx, "name": spectrum_name, "data_path": data_path,
            "success": True, "error": None, "model_type": "m",
            "parameters": {}, "fit_quality": {"r_squared": 0.98},
            "deviation_note": None, "visualization_path": None,
            "visualization_bytes": None, "statistics": {},
            "script": "anchor_script", "script_errors": [],
        }

    def flaky_single(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None, **_):
        if spectrum_idx == 2:
            raise RuntimeError("boom from worker")
        return {
            "index": spectrum_idx, "name": spectrum_name, "data_path": data_path,
            "success": True, "error": None, "model_type": "m",
            "parameters": {}, "fit_quality": {"r_squared": 0.94},
            "deviation_note": None, "visualization_path": None,
            "visualization_bytes": None, "statistics": {},
            "script": base_script, "script_errors": [],
        }

    ctrl._fit_with_quality_control = mock_qc
    ctrl._fit_single_spectrum = flaky_single
    state = _build_state(num_spectra=4)

    out = ctrl.execute(state)
    series = out["series_results"]
    assert [r["index"] for r in series] == [0, 1, 2, 3]
    assert series[2]["success"] is False
    assert "boom" in (series[2]["error"] or "")
    # Other spectra still succeeded
    assert all(series[i]["success"] for i in (0, 1, 3))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
