"""Skills tab — upload custom skills and view available built-in skills."""

from pathlib import Path

import streamlit as st

from scilink.skills.loader import list_all_skills


def render_skills_tab() -> None:
    """Render the Skills tab content."""
    agent = st.session_state.get("agent")

    if agent is None:
        st.info("Start a session to view skills.")
        return

    left_col, _, right_col = st.columns([10, 1, 10])

    with left_col:
        _render_upload_section(agent)

    with right_col:
        _render_available_skills(agent)

    st.divider()
    _render_memory_section()


def _render_upload_section(agent) -> None:
    """Upload custom skill files."""
    st.subheader("Upload Skills")
    st.caption("Available for this session only — not saved to persistent memory.")
    uploaded = st.file_uploader(
        "Upload a custom skill file (.md)",
        type=["md"],
        key="skill_file_uploader",
        accept_multiple_files=True,
        help=(
            "Markdown file with structured sections (## Overview, ## Planning, "
            "## Analysis, ## Interpretation, ## Validation) providing "
            "domain-specific guidance for analysis agents."
        ),
    )

    if uploaded:
        for f in uploaded:
            upload_key = ("custom_skill", f.name)
            if upload_key in st.session_state._processed_uploads:
                continue
            _load_skill_file(agent, f)
            st.session_state._processed_uploads.add(upload_key)


def _load_skill_file(agent, uploaded_file) -> None:
    """Save an uploaded skill .md file and register it with the agent."""
    session_dir = st.session_state.get("session_dir")
    if session_dir is None:
        st.error("No active session.")
        return

    skills_dir = Path(session_dir) / "custom_skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    dest = skills_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())

    try:
        name = agent.register_skill(str(dest))
        st.success(f"Registered skill '{name}' from {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to register {uploaded_file.name}: {e}")


def _render_available_skills(agent) -> None:
    """Show built-in and custom skills."""
    st.subheader("Available Skills")
    st.caption("Active in this session: shipped built-in skills plus any you uploaded.")

    # Built-in subsection
    st.markdown("**Built-in**")
    builtin = list_all_skills()
    if builtin:
        for domain, names in builtin.items():
            label = domain.replace("_", " ").title()
            with st.expander(f"{label} ({len(names)})", expanded=False):
                for name in names:
                    st.markdown(f"- `{name}`")
    else:
        st.caption("No built-in skills found.")

    # Custom subsection
    st.markdown("**Custom**")
    custom = getattr(agent, "_custom_skills", {})
    if custom:
        for name in sorted(custom.keys()):
            st.markdown(f"- `{name}`")
    else:
        st.caption("No custom skills registered yet.")


def _render_memory_section() -> None:
    """Persistent memory — graduated and auto-distilled skills under ~/.scilink.

    Provisional skills (auto-distilled from hard fits the agent solved only after
    escalating — T=2 auto-distillation, today wired in the curve-fitting agent;
    the provisional mechanism itself is domain-agnostic) are shown with a badge
    and can be promoted (made auto-routable) or pruned. Promoted skills survive
    sessions and pip upgrades and are auto-discovered by the loader.
    """
    from scilink.skills._shared import _memory

    st.subheader("Persistent Memory")
    st.caption(
        "Graduated and auto-distilled skills stored under `~/.scilink` — they "
        "survive sessions and upgrades. Provisional skills (auto-distilled from "
        "hard fits the agent had to solve from scratch — currently curve fitting) "
        "are held out of auto-routing until you promote them."
    )

    try:
        rows = _memory.list_memory()
    except Exception as e:
        st.error(f"Could not read persistent memory: {e}")
        return

    provisional = [r for r in rows if r["provisional"]]
    promoted = [r for r in rows if not r["provisional"]]

    if provisional:
        st.markdown(f"**Provisional — awaiting review ({len(provisional)})**")
        for r in provisional:
            _render_memory_row(_memory, r, provisional=True)
    if promoted:
        st.markdown(f"**Promoted ({len(promoted)})**")
        for r in promoted:
            _render_memory_row(_memory, r, provisional=False)
    if not rows:
        st.caption("No persisted skills yet.")

    _render_staged_section()


def _render_staged_section() -> None:
    """Staged raw T=2 solutions — distill into skills (upgrade an existing skill,
    or consolidate N of a technique into a new one)."""
    from scilink.skills._shared import _staging, _memory

    st.markdown("---")
    st.markdown("**Staged T=2 solutions**")
    st.caption(
        "Hard problems the agent solved from scratch (T=2). Upgrade an existing "
        "skill from one, or consolidate several of the same technique into a new "
        "skill. Both use the active session's model."
    )

    groups = {}
    for rec in _staging.list_staged():
        groups.setdefault((rec["domain"], rec.get("technique") or "unlabeled"), []).append(rec)
    if not groups:
        st.caption("No staged solutions.")
        return

    agent = st.session_state.get("agent")
    model = getattr(agent, "model", None)

    def _llm_call(prompt: str) -> str:
        r = model.generate_content(contents=[prompt])
        return r.text if hasattr(r, "text") else str(r)

    for (domain, technique), recs in sorted(groups.items()):
        with st.expander(f"`{domain}/{technique}` — {len(recs)} staged", expanded=False):
            for r in recs:
                metric = r.get("r_squared") or r.get("quality_score")
                meta_col, view_col = st.columns([3, 1])
                meta_col.caption(
                    f"id={r['id']} · session={r.get('session','?')}"
                    + (f" · metric={metric}" if metric is not None else "")
                )
                with view_col.popover("View", use_container_width=True):
                    _render_staged_record(r)
            if model is None:
                st.info("Start a session to enable upgrade/consolidate (needs a model).")
                continue
            # candidate existing skills to upgrade into (same domain).
            # Keep the action buttons in narrow columns + a trailing spacer so
            # they land at a modest size (like the sidebar's Reset/Quit) rather
            # than stretching across the full-width memory panel.
            targets = [f"{s['domain']}/{s['name']}" for s in _memory.list_memory(domain=domain)]
            c1, c2, _ = st.columns([2, 2, 4])
            with c1:
                if targets:
                    tgt = st.selectbox("Upgrade into", targets, key=f"tgt::{domain}/{technique}")
                    sid = recs[0]["id"]
                    if st.button("Upgrade (use newest)", key=f"up::{domain}/{technique}"):
                        from scilink.agents.exp_agents.instruct import (
                            KNOWLEDGE_TO_SKILL_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS)
                        td, tn = tgt.split("/", 1)
                        res = _staging.upgrade_skill_from_staged(
                            domain, [sid], target_domain=td, target_name=tn,
                            llm_call=_llm_call,
                            fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                            update_template=SKILL_UPDATE_INSTRUCTIONS)
                        st.success(f"Upgraded {tgt} ({res.get('method')}).")
                        st.rerun()
                else:
                    st.caption("No existing skills in this domain to upgrade into.")
            with c2:
                # New-skill consolidation accumulates first: only suggest it once
                # enough examples of this technique are staged. Below the threshold
                # the agent is still gathering evidence (one fit is too idiosyncratic
                # to generalize into a standalone skill). Upgrading an existing skill
                # is exempt — that's the upgrade@1 path on the left.
                need = _staging.consolidate_min_n()
                ready = len(recs) >= need
                if st.button("Consolidate → new skill", key=f"con::{domain}/{technique}",
                             disabled=not ready,
                             help=(None if ready else
                                   f"Accumulating {len(recs)}/{need} — consolidation into a "
                                   f"new skill unlocks once {need} solutions of this technique "
                                   f"are staged (set SCILINK_CONSOLIDATE_N to change; "
                                   f"`scilink memory consolidate` can force it).")):
                    from scilink.agents.exp_agents.instruct import (
                        T2_CONSOLIDATION_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS)
                    res = _staging.consolidate_technique(
                        domain, technique, llm_call=_llm_call,
                        consolidation_template=T2_CONSOLIDATION_INSTRUCTIONS,
                        update_template=SKILL_UPDATE_INSTRUCTIONS)
                    st.success(f"Consolidated {res.get('n_examples')} → auto_{technique} (provisional).")
                    st.rerun()
                if not ready:
                    st.caption(f"Accumulating {len(recs)}/{need} examples before a new skill.")


# Bookkeeping keys not worth showing in the per-record viewer.
_STAGED_HIDDEN_KEYS = {"id", "domain", "technique", "session", "working_script", "script"}


def _render_staged_record(r: dict) -> None:
    """Show one staged T=2 solution's actual content (planned vs final model,
    deviation, metric, and the working script) so it can be inspected before
    upgrading/consolidating."""
    for k, v in r.items():
        if k in _STAGED_HIDDEN_KEYS or v in (None, "", [], {}):
            continue
        st.markdown(f"**{k.replace('_', ' ')}:** {v}")
    script = (r.get("working_script") or r.get("script") or "").strip()
    if script:
        st.markdown("**working script:**")
        st.code(script, language="python")


def _render_memory_row(_memory, r, *, provisional: bool) -> None:
    ref = f"{r['domain']}/{r['name']}"
    badge = "🟡 provisional" if provisional else "✅ promoted"
    r2 = f" · R²={r['r_squared']}" if r.get("r_squared") is not None else ""
    with st.expander(f"{badge} · `{ref}`{r2}", expanded=False):
        if r.get("description"):
            st.markdown(f"_{r['description']}_")
        if r.get("provenance"):
            st.caption(f"provenance: {r['provenance']}"
                       + (f" · session: {r['session']}" if r.get("session") else ""))
        try:
            st.markdown(_memory.show_memory(r["domain"], r["name"]))
        except Exception as e:
            st.warning(f"Could not render skill: {e}")

        c1, c2, _ = st.columns([2, 2, 4])
        if provisional:
            if c1.button("Promote", key=f"promote::{ref}", type="primary"):
                _memory.promote_memory(r["domain"], r["name"])
                st.success(f"Promoted {ref} — now auto-routable.")
                st.rerun()
        if c2.button("Prune", key=f"prune::{ref}"):
            _memory.prune_memory(r["domain"], r["name"])
            st.warning(f"Pruned {ref}.")
            st.rerun()
