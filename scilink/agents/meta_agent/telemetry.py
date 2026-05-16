"""Read-only telemetry snapshot of a meta (Explore) session.

Aggregates state the agents already persist — the meta's delegation ledger,
the specialist orchestrators' counters, and each worker agent's
``action_history`` (the uniform ``_log_action`` audit trail) — into one plain
dict for the Telemetry UI tab.

The reader is LLM-free, stdlib-only, and never raises: a malformed state file
degrades that one agent's row to nothing rather than breaking the tab.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RATIONALE_MAX = 200

# Friendly names for the known worker state files; anything else falls back to
# a title-cased form of the file stem (so a new agent still shows up sanely).
_AGENT_LABELS = {
    "curve_fitting": "Curve Fitting",
    "bo": "Bayesian Optimization",
    "scalarizer": "Scalarizer",
}


def _agent_label(agent_type: Optional[str], stem: str) -> str:
    """Human-readable agent name from ``agent_type`` or the state-file stem."""
    if isinstance(agent_type, str) and agent_type.strip():
        key = agent_type.strip().lower()
        return _AGENT_LABELS.get(key, agent_type.strip())
    base = stem[:-6] if stem.endswith("_state") else stem  # "bo_state" -> "bo"
    return _AGENT_LABELS.get(base, base.replace("_", " ").title())


def _action_status(result: Any) -> str:
    """Best-effort outcome label for one action_history entry's ``result``.

    BO / Scalarizer results carry an explicit ``status``; curve-fitting results
    do not — for those a completed action is reported as ``ok``.
    """
    if isinstance(result, dict):
        status = result.get("status")
        if isinstance(status, str) and status:
            return status
        if result.get("error"):
            return "error"
        return "ok"
    return "ok" if result else "-"


def _specialist_of(path: Path) -> str:
    """Which specialist subtree (``analysis`` / ``planning``) a file sits in."""
    parts = set(path.parts)
    if "analysis" in parts:
        return "analysis"
    if "planning" in parts:
        return "planning"
    return "other"


def _worker_telemetry(state_path: Path) -> Optional[Dict[str, Any]]:
    """One agent row from a ``*_state.json`` file, or None if it has no
    ``action_history`` (so non-agent ``*_state.json`` files are skipped)."""
    try:
        data = json.loads(state_path.read_text())
    except Exception:  # noqa: BLE001 - a bad state file must not break the tab
        return None
    if not isinstance(data, dict):
        return None
    history = data.get("action_history")
    if not isinstance(history, list) or not history:
        return None

    actions: List[Dict[str, Any]] = []
    by_type: Dict[str, int] = {}
    outcomes = {"success": 0, "error": 0, "other": 0}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action") or "?")
        status = _action_status(entry.get("result"))
        by_type[action] = by_type.get(action, 0) + 1
        if status in ("success", "ok"):
            outcomes["success"] += 1
        elif status == "error":
            outcomes["error"] += 1
        else:
            outcomes["other"] += 1
        rationale = entry.get("rationale")
        if isinstance(rationale, str) and len(rationale) > _RATIONALE_MAX:
            rationale = rationale[:_RATIONALE_MAX - 1] + "…"
        actions.append({
            "timestamp": entry.get("timestamp"),
            "action": action,
            "status": status,
            "rationale": rationale,
        })

    stamps = [a["timestamp"] for a in actions if a["timestamp"]]
    return {
        "specialist": _specialist_of(state_path),
        "name": _agent_label(data.get("agent_type"), state_path.stem),
        "status": data.get("status"),
        "action_count": len(actions),
        "actions_by_type": by_type,
        "outcomes": outcomes,
        "first_timestamp": min(stamps) if stamps else None,
        "last_timestamp": max(stamps) if stamps else None,
        "actions": actions,
        "source_file": str(state_path),
    }


def _csv_row_count(path: Any) -> int:
    """Row count of a CSV, or 0 if absent / unreadable."""
    try:
        import pandas as pd
        if path and Path(path).exists():
            return len(pd.read_csv(path))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _specialists(meta_agent: Any) -> Dict[str, Any]:
    """Per-specialist counters — the same fields the meta's
    ``_session_state_summary`` reads, defensively via ``getattr``."""
    children = getattr(meta_agent, "_children", {}) or {}
    out: Dict[str, Any] = {}
    for name in ("analysis", "planning"):
        child = children.get(name)
        if child is None:
            out[name] = {"instantiated": False}
            continue
        info: Dict[str, Any] = {
            "instantiated": True,
            "message_count": getattr(child, "message_count", 0),
        }
        if name == "analysis":
            info["analyses_run"] = len(
                getattr(child, "analysis_results", []) or [])
        else:
            info["optimization_targets"] = list(
                getattr(child, "expected_target_columns", []) or [])
            info["bo_data_points"] = _csv_row_count(
                getattr(child, "bo_data_path", None))
        out[name] = info
    return out


def collect_session_telemetry(meta_agent: Any) -> Dict[str, Any]:
    """Read-only telemetry snapshot of a meta session. Never raises.

    Works for a live session (ledger in memory) and a resumed one (ledger
    restored from the checkpoint; worker state files re-read from disk).
    """
    try:
        meta_mode = getattr(meta_agent, "meta_mode", None)
        meta_mode_str = getattr(meta_mode, "value", None) or str(meta_mode)
        # No configured base_dir -> scan nothing (never fall back to cwd, which
        # would rglob the whole tree).
        base_dir_raw = getattr(meta_agent, "base_dir", None)
        ledger = list(getattr(meta_agent, "_delegation_ledger", []) or [])

        delegations = [{
            "index": e.get("index"),
            "mode": e.get("mode"),
            "label": (e.get("label") or "").strip()
            or " ".join(str(e.get("task") or "").split())[:60],
            "status": e.get("status"),
            "context_from": e.get("context_from") or [],
            "timestamp": e.get("timestamp"),
            "completed_at": e.get("completed_at"),
            "files": len(e.get("files_produced") or []),
            "feature_tables": len(e.get("feature_tables") or []),
            "warnings": len(e.get("warnings") or []),
        } for e in ledger]

        agents: List[Dict[str, Any]] = []
        if base_dir_raw and Path(base_dir_raw).is_dir():
            for sp in sorted(Path(base_dir_raw).rglob("*_state.json")):
                row = _worker_telemetry(sp)
                if row:
                    agents.append(row)

        return {
            "meta": {
                "meta_mode": meta_mode_str,
                "session_dir": str(base_dir_raw or ""),
                "delegations_total": len(ledger),
                "delegations": delegations,
            },
            "specialists": _specialists(meta_agent),
            "agents": agents,
        }
    except Exception as e:  # noqa: BLE001 - telemetry must never break the UI
        logger.warning(f"telemetry collection failed: {e}")
        return {"meta": {}, "specialists": {}, "agents": [], "error": str(e)}
