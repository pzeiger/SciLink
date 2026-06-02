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

    Provisional skills (auto-distilled from successful T=2 curve fits) are shown
    with a badge and can be promoted (made auto-routable) or pruned. Promoted
    skills survive sessions and pip upgrades and are auto-discovered by the loader.
    """
    from scilink.skills._shared import _memory

    st.subheader("🧠 Persistent Memory")
    st.caption(
        "Graduated and auto-distilled skills stored under `~/.scilink` — they "
        "survive sessions and upgrades. Provisional skills (from hard T=2 fits) "
        "are held out of auto-routing until you promote them."
    )

    try:
        rows = _memory.list_memory()
    except Exception as e:
        st.error(f"Could not read persistent memory: {e}")
        return

    if not rows:
        st.caption("No persisted skills yet.")
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

        c1, c2 = st.columns(2)
        if provisional:
            if c1.button("Promote", key=f"promote::{ref}", type="primary"):
                _memory.promote_memory(r["domain"], r["name"])
                st.success(f"Promoted {ref} — now auto-routable.")
                st.rerun()
        if c2.button("Prune", key=f"prune::{ref}"):
            _memory.prune_memory(r["domain"], r["name"])
            st.warning(f"Pruned {ref}.")
            st.rerun()
