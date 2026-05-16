"""Flatten a completed analysis run into a tabular feature file.

One row per analyzed unit (a spectrum, an image, …); columns = the unit's
experimental conditions (from its per-unit sidecar JSON) + extracted scalar
features. The CSV feeds the planning Scalarizer / Bayesian optimization via
the meta-agent — see ``analysis_bo_feature_table_plan.md``.

The numeric path is LLM-free: this reads the structured result files the
analysis pipeline already persisted and writes a deterministic flatten.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _flatten_scalars(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively collect scalar leaves of a nested dict as flat columns.

    Lists / arrays / maps are skipped — only scalars belong in a feature row.
    """
    flat: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return flat
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_scalars(value, name + "_"))
        elif isinstance(value, (int, float, str)):  # bool is an int subclass
            flat[name] = value
    return flat


def _sidecar_conditions(data_path: Optional[str]) -> Dict[str, Any]:
    """Per-unit experimental conditions from the data file's sidecar JSON
    (``spec_5K.csv`` -> ``spec_5K.json``). Empty dict if absent / unreadable."""
    if not data_path:
        return {}
    sidecar = Path(data_path).with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        cond = json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001 - a bad sidecar must not break the run
        return {}
    if not isinstance(cond, dict):
        return {}
    return {k: v for k, v in cond.items() if isinstance(v, (int, float, str))}


def _curve_fit_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """One row per spectrum from a curve-fitting run's series_fit_results.json."""
    sfr = output_dir / "series_fit_results.json"
    if not sfr.is_file():
        return []
    try:
        data = json.loads(sfr.read_text())
    except Exception:  # noqa: BLE001
        return []
    rows: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        if not isinstance(r, dict) or not r.get("success"):
            continue
        row: Dict[str, Any] = {"unit": r.get("name") or f"index_{r.get('index')}"}
        row.update(_sidecar_conditions(r.get("data_path")))
        row.update(_flatten_scalars(r.get("parameters")))
        row.update(_flatten_scalars(r.get("fit_quality"), "fit_"))
        rows.append(row)
    return rows


def _extracted_feature_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """One row from an agent that records an ``extracted_features`` dict in
    analysis_results.json (e.g. image analysis)."""
    ar = output_dir / "analysis_results.json"
    if not ar.is_file():
        return []
    try:
        data = json.loads(ar.read_text())
    except Exception:  # noqa: BLE001
        return []
    feats = data.get("extracted_features")
    if not isinstance(feats, dict) or not feats:
        return []
    row: Dict[str, Any] = {"unit": output_dir.name}
    row.update(_flatten_scalars(feats))
    return [row]


def write_feature_table(output_dir) -> Optional[str]:
    """Write ``<output_dir>/features.csv`` — a flat per-unit feature table
    derived from the run's structured result files.

    Returns the absolute path, or ``None`` if no adapter applies or the run
    produced no scalar features. Never raises — a failure here must not break
    the analysis.
    """
    try:
        output_dir = Path(output_dir)
        # Curve-fitting series first (richest, explicitly per-unit); fall back
        # to the generic ``extracted_features`` dict (image analysis, etc.).
        rows = _curve_fit_rows(output_dir) or _extracted_feature_rows(output_dir)
        if not rows:
            return None
        columns: List[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        dest = output_dir / "features.csv"
        with open(dest, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(dest.resolve())
    except Exception as e:  # noqa: BLE001 - never break the analysis on this
        logger.warning(f"feature table emit failed: {e}")
        return None
