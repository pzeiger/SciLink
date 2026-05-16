"""Telemetry tab — live snapshot of meta + specialist + worker-agent state.

Explore (meta) mode only. Reads ``collect_session_telemetry`` and renders it
with native Streamlit widgets, refreshing on the same cadence as the sidebar
delegation tree (2s while a chat task runs).
"""

import streamlit as st

from scilink.agents.meta_agent.telemetry import collect_session_telemetry

_SPECIALIST_TITLES = {"analysis": "🧪 Analysis", "planning": "📋 Planning"}


def render_telemetry_tab() -> None:
    """Render the Explore-mode Telemetry tab."""
    task = st.session_state.get("chat_task")
    _interval = "2s" if (task is not None
                         and getattr(task, "is_running", False)) else None

    @st.fragment(run_every=_interval)
    def _panel() -> None:
        agent = st.session_state.get("agent")
        if agent is None or not hasattr(agent, "_delegation_ledger"):
            st.info("Telemetry is available once an Explore session is running.")
            return

        tel = collect_session_telemetry(agent)
        meta = tel.get("meta", {})

        # ── Meta header ──────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", str(meta.get("meta_mode", "—")).title())
        c2.metric("Delegations", meta.get("delegations_total", 0))
        c3.metric("Worker agents", len(tel.get("agents", [])))
        st.caption(f"Session: {meta.get('session_dir', '—')}")

        if not meta.get("delegations"):
            st.info("No delegations yet — describe a goal and the meta "
                    "routes it to a specialist.")
            return

        # ── Specialist cards ─────────────────────────────────────────
        st.subheader("Specialists")
        specs = tel.get("specialists", {})
        for col, name in zip(st.columns(2), ("analysis", "planning")):
            with col:
                _specialist_card(name, specs.get(name, {}))

        # ── Worker action histories ──────────────────────────────────
        st.subheader("Worker activity")
        agents = tel.get("agents", [])
        if not agents:
            st.caption("No worker actions recorded yet.")
        for ag in agents:
            _worker_expander(ag)

        # ── Delegation ledger ────────────────────────────────────────
        st.subheader("Delegation ledger")
        _ledger_table(meta.get("delegations", []))

    _panel()


def _specialist_card(name: str, info: dict) -> None:
    """One bordered card summarizing a specialist orchestrator."""
    title = _SPECIALIST_TITLES.get(name, name.title())
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if not info.get("instantiated"):
            st.caption("Not yet engaged.")
            return
        st.caption(f"{info.get('message_count', 0)} messages")
        if name == "analysis":
            st.metric("Analyses run", info.get("analyses_run", 0))
        else:
            st.metric("BO data points", info.get("bo_data_points", 0))
            targets = info.get("optimization_targets", [])
            if targets:
                st.caption("Targets: " + ", ".join(str(t) for t in targets))


def _worker_expander(ag: dict) -> None:
    """One expander with a worker agent's action_history table."""
    import pandas as pd

    oc = ag.get("outcomes", {})
    title = (f"{ag.get('name', '?')} — {ag.get('action_count', 0)} actions  "
             f"({oc.get('success', 0)}✓ / {oc.get('error', 0)}✗)")
    with st.expander(title):
        by_type = ag.get("actions_by_type", {})
        st.caption(
            f"Specialist: {ag.get('specialist', '—')}"
            + ("  ·  " + ", ".join(f"{k}×{v}" for k, v in by_type.items())
               if by_type else "")
        )
        rows = ag.get("actions", [])
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "action",
                                             "status", "rationale"])
            st.dataframe(df, use_container_width=True, hide_index=True)


def _ledger_table(delegations: list) -> None:
    """The delegation ledger as a flat table (richer than the sidebar tree)."""
    import pandas as pd

    rows = []
    for d in delegations:
        cf = d.get("context_from") or []
        rows.append({
            "#": d.get("index"),
            "specialist": d.get("mode"),
            "task": d.get("label"),
            "status": d.get("status"),
            "context from": ", ".join(f"#{n}" for n in cf) if cf else "",
            "files": d.get("files", 0),
            "feature tables": d.get("feature_tables", 0),
            "warnings": d.get("warnings", 0),
            "started": d.get("timestamp"),
            "completed": d.get("completed_at"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
