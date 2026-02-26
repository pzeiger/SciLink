"""Sidebar component: configuration, file upload, agent status."""

import base64
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from ..config import (
    EMBEDDING_MODEL_OPTIONS,
    MODEL_OPTIONS,
    SESSION_DIR_PREFIXES,
    SUPPORTED_CODE_EXTENSIONS,
    SUPPORTED_DATA_EXTENSIONS,
    SUPPORTED_KNOWLEDGE_EXTENSIONS,
    SUPPORTED_METADATA_EXTENSIONS,
    SUPPORTED_PLANNING_DATA_EXTENSIONS,
)


_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "scilink_logo_v3_dark.svg"


def render_sidebar() -> None:
    with st.sidebar:
        if st.session_state.agent_initialized and _LOGO_PATH.exists():
            _b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
            st.markdown(
                '<style>'
                '@keyframes logo-spin{to{transform:rotate(360deg)}}'
                '.logo-glow-sm{position:relative;padding:2px;border-radius:10px;'
                'overflow:hidden;width:140px;margin:0 auto}'
                '.logo-glow-sm::before{content:"";position:absolute;'
                'top:-40%;left:-40%;width:180%;height:180%;'
                'background:conic-gradient('
                'transparent 0deg,transparent 270deg,#3A4556 300deg,'
                '#82B1FF 330deg,#FFF 345deg,#82B1FF 355deg,transparent 360deg);'
                'animation:logo-spin 4s linear infinite;z-index:0}'
                '.logo-glow-sm>img{position:relative;z-index:1;border-radius:8px;'
                'display:block;width:100%}'
                '</style>'
                f'<div class="logo-glow-sm">'
                f'<img src="data:image/svg+xml;base64,{_b64}"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.title("SciLink")
        _locked = st.session_state.agent_initialized
        preset = st.selectbox(
            "Model",
            MODEL_OPTIONS + ["Custom"],
            key="cfg_model_preset",
            disabled=_locked,
        )
        if preset == "Custom":
            model = st.text_input("Model name", key="cfg_model_custom", disabled=_locked)
        else:
            model = preset
        api_key = st.text_input("API key", type="password", key="cfg_api_key", disabled=_locked)
        base_url = st.text_input("Base URL (optional)", key="cfg_base_url", disabled=_locked)
        fh_api_key = st.text_input("FutureHouse API key (optional)", type="password", key="cfg_fh_api_key", disabled=_locked)

        # Planning mode: embedding model
        if st.session_state.app_mode == "plan":
            st.selectbox(
                "Embedding model",
                EMBEDDING_MODEL_OPTIONS + ["Custom"],
                key="cfg_embedding_preset",
                disabled=_locked,
            )
            if st.session_state.get("cfg_embedding_preset") == "Custom":
                st.text_input("Embedding model name", key="cfg_embedding_custom", disabled=_locked)

        mode = st.selectbox(
            "Autonomy mode",
            ["supervised", "co-pilot", "autonomous"],
            key="cfg_mode",
            disabled=_locked,
        )
        consent = st.checkbox(
            "I understand that the agent will execute generated Python code on my machine",
            key="cfg_consent",
            disabled=_locked,
        )
        if not consent:
            with st.expander("What does the agent execute?"):
                from scilink.executors import LLM_EXECUTION_DESCRIPTION
                st.text(LLM_EXECUTION_DESCRIPTION)
                st.warning(
                    "**Recommendation:** Run this app inside a Docker container "
                    "or a VM to sandbox generated code execution. "
                    "Example: `docker run -p 8501:8501 scilink-ui`"
                )
        col1, col2 = st.columns(2)

        with col1:
            start_disabled = st.session_state.agent_initialized or not consent
            if st.button("Start Session", disabled=start_disabled, width="stretch"):
                # Stash config and let the main app handle the heavy init
                # outside the sidebar context so the spinner is visible.
                _embed_preset = st.session_state.get("cfg_embedding_preset", "")
                _embed_model = (
                    st.session_state.get("cfg_embedding_custom", "")
                    if _embed_preset == "Custom" else _embed_preset
                ) or None
                st.session_state._pending_init = {
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "mode": mode,
                    "fh_api_key": fh_api_key,
                    "embedding_model": _embed_model,
                }

        with col2:
            if st.button("Reset Session", disabled=not st.session_state.agent_initialized,
                         width="stretch"):
                _reset_session()

        # ── File upload (mode-specific) ──────────────────────────
        if st.session_state.agent_initialized:
            st.divider()
            app_mode = st.session_state.app_mode

            if app_mode == "analyze":
                _render_analyze_sidebar_uploads()
            elif app_mode == "plan":
                _render_planning_sidebar_uploads()

            # ── Agent status (mode-specific) ─────────────────────
            st.divider()
            st.subheader("Agent Status")
            if app_mode == "analyze":
                _render_analyze_status()
            elif app_mode == "plan":
                _render_planning_status()

        # ── Vibes sliders ───────────────────────────────────
        st.divider()
        st.slider(
            "\U0001f49c Hearts",
            min_value=0, max_value=50, value=7,
            key="vibe_hearts",
        )
        st.slider(
            "\u2795 Pluses",
            min_value=0, max_value=50, value=7,
            key="vibe_pluses",
        )

        # ── Quit button (always visible at bottom) ────────────
        st.divider()
        if st.button("Quit App", use_container_width=True):
            # Stop any running agent thread
            task = st.session_state.get("chat_task")
            if task and task.is_running:
                task.stopped = True
                if task.live_capture:
                    task.live_capture.request_stop()
                if task.feedback_request is not None:
                    task.feedback_request.response = ""
                    task.feedback_request.event.set()
            # Inject JS to replace the page with a goodbye message,
            # then kill the server after a short delay.
            import streamlit.components.v1 as components
            components.html(
                '<script>'
                'window.parent.document.body.innerHTML = '
                '\'<div style="display:flex;align-items:center;justify-content:center;'
                'height:100vh;font-family:sans-serif;color:#888;background:#0e1117;">'
                '<h2>Server stopped. You can close this window.</h2></div>\';'
                '</script>',
                height=0,
            )
            import time, signal
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)


def _render_analyze_sidebar_uploads() -> None:
    """Sidebar upload widgets for analyze mode."""
    st.subheader("Upload Files")

    data_files = st.file_uploader(
        "Data file(s)",
        type=[e.lstrip(".") for e in SUPPORTED_DATA_EXTENSIONS],
        key="uploader_data",
        accept_multiple_files=True,
    )
    if data_files:
        if len(data_files) == 1:
            save_upload(data_files[0], "data")
        else:
            save_upload_batch(data_files, "data")

    meta_file = st.file_uploader(
        "Metadata file",
        type=[e.lstrip(".") for e in SUPPORTED_METADATA_EXTENSIONS],
        key="uploader_meta",
    )
    if meta_file is not None:
        save_upload(meta_file, "metadata")


def _render_planning_sidebar_uploads() -> None:
    """Compact sidebar upload widgets for plan mode."""
    st.subheader("Upload Files")

    knowledge_files = st.file_uploader(
        "Knowledge",
        type=[e.lstrip(".") for e in SUPPORTED_KNOWLEDGE_EXTENSIONS],
        key="sidebar_uploader_knowledge",
        accept_multiple_files=True,
    )
    if knowledge_files:
        from .chat_uploads import save_planning_uploads
        save_planning_uploads(knowledge_files, "knowledge")

    code_files = st.file_uploader(
        "Code",
        type=[e.lstrip(".") for e in SUPPORTED_CODE_EXTENSIONS],
        key="sidebar_uploader_code",
        accept_multiple_files=True,
    )
    if code_files:
        from .chat_uploads import save_planning_uploads
        save_planning_uploads(code_files, "code")

    data_files = st.file_uploader(
        "Data",
        type=[e.lstrip(".") for e in SUPPORTED_PLANNING_DATA_EXTENSIONS],
        key="sidebar_uploader_planning_data",
        accept_multiple_files=True,
    )
    if data_files:
        from .chat_uploads import save_planning_uploads
        save_planning_uploads(data_files, "data")


def _render_analyze_status() -> None:
    """Show analysis-specific agent status metrics."""
    agent = st.session_state.agent
    if agent.selected_agent_id is not None:
        entry = agent._agent_registry.get(agent.selected_agent_id, {})
        agent_label = entry.get("name", f"Agent {agent.selected_agent_id}")
    else:
        agent_label = "None"

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.metric("Selected Agent", agent_label)
    with row1_c2:
        st.metric("Analyses", len(agent.analysis_results))

    data_path = agent.current_data_path
    if data_path:
        st.metric("Data File", Path(data_path).name)
    else:
        st.metric("Data File", "None")

    if agent.current_data_type:
        st.metric("Data Type", agent.current_data_type)


def _render_planning_status() -> None:
    """Show planning-specific agent status metrics."""
    objective = st.session_state.get("planning_objective", "")
    n_messages = len(st.session_state.chat_messages)
    mode = st.session_state.agent_config.get("mode", "supervised")

    if objective:
        st.metric("Objective", objective[:40] + ("..." if len(objective) > 40 else ""))
    st.metric("Messages", n_messages)
    st.metric("Autonomy", mode.replace("-", " ").title())


# ── helpers ──────────────────────────────────────────────────────

def start_session(model: str, api_key: str, base_url: str, mode: str, fh_api_key: str = "", embedding_model: str = None) -> None:
    """Initialize the agent and session directory.

    Dispatches to the appropriate agent based on ``st.session_state.app_mode``.
    Designed to be called from the **main** content area (not inside
    ``with st.sidebar``) so that ``st.spinner`` is visible to the user.
    """
    import scilink.executors as executors

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key and not base_url:
        st.sidebar.error("Provide an API key or set an environment variable (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).")
        return

    app_mode = st.session_state.app_mode or "analyze"
    prefix = SESSION_DIR_PREFIXES.get(app_mode, "session")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"{prefix}_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Auto-approve sandbox for the Streamlit session (user checked consent box)
    executors._GLOBAL_SANDBOX_APPROVED = True

    try:
        if app_mode == "plan":
            agent = _init_planning_agent(
                session_dir, resolved_key, model, base_url, mode, fh_api_key,
                embedding_model=embedding_model,
            )
        else:
            agent = _init_analysis_agent(
                session_dir, resolved_key, model, base_url, mode, fh_api_key,
            )
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return

    st.session_state.agent = agent
    st.session_state.agent_initialized = True
    st.session_state.session_dir = str(session_dir)
    st.session_state.agent_config = {
        "model": model,
        "mode": mode,
    }
    st.session_state.chat_messages = []
    st.session_state.known_images = set()
    st.rerun()


def _init_analysis_agent(session_dir, api_key, model, base_url, mode, fh_api_key):
    """Create an AnalysisOrchestratorAgent."""
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )

    mode_map = {
        "co-pilot": AnalysisMode.CO_PILOT,
        "supervised": AnalysisMode.SUPERVISED,
        "autonomous": AnalysisMode.AUTONOMOUS,
    }
    return AnalysisOrchestratorAgent(
        base_dir=str(session_dir),
        api_key=api_key,
        model_name=model,
        base_url=base_url or None,
        analysis_mode=mode_map[mode],
        futurehouse_api_key=fh_api_key or None,
    )


def _init_planning_agent(session_dir, api_key, model, base_url, mode, fh_api_key,
                         embedding_model=None):
    """Create a PlanningOrchestratorAgent."""
    from scilink.agents.planning_agents.planning_orchestrator import (
        AutonomyLevel,
        PlanningOrchestratorAgent,
    )

    mode_map = {
        "co-pilot": AutonomyLevel.CO_PILOT,
        "supervised": AutonomyLevel.SUPERVISED,
        "autonomous": AutonomyLevel.AUTONOMOUS,
    }
    objective = st.session_state.get("planning_objective", "").strip() or "Undefined Research Goal"

    # Create subdirectories for planning uploads
    knowledge_dir = session_dir / "knowledge"
    code_dir = session_dir / "code"
    data_dir = session_dir / "data"
    knowledge_dir.mkdir(exist_ok=True)
    code_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    kwargs = {}
    if embedding_model:
        kwargs["embedding_model"] = embedding_model

    return PlanningOrchestratorAgent(
        objective=objective,
        base_dir=str(session_dir),
        api_key=api_key,
        model_name=model,
        base_url=base_url or None,
        autonomy_level=mode_map[mode],
        futurehouse_api_key=fh_api_key or None,
        knowledge_dir=str(knowledge_dir),
        code_dir=str(code_dir),
        data_dir=str(data_dir),
        **kwargs,
    )


def _reset_session() -> None:
    # Stop the agent thread if it's still running
    task = st.session_state.get("chat_task")
    if task and task.is_running and task.feedback_request is not None:
        task.feedback_request.response = ""
        task.feedback_request.event.set()

    # Clear all state keys — both app state and widget keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Use query params to force a clean page load, which fully resets
    # widget keys that would otherwise re-initialize with stale values
    st.query_params["reset"] = "1"
    st.rerun()


def save_upload(uploaded_file, kind: str, auto_dispatch: bool = True) -> None:
    """Save an uploaded file and optionally queue an auto-examine/load."""
    session_dir = Path(st.session_state.session_dir)
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)

    dest = upload_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())
    dest_str = str(dest)

    # Track the path so the sidebar shows the latest upload
    if kind == "data":
        st.session_state.uploaded_data_path = dest_str
    else:
        st.session_state.uploaded_metadata_path = dest_str

    # Only queue an auto-message the first time we see this file
    upload_key = (kind, dest_str)
    if upload_key not in st.session_state._processed_uploads:
        st.session_state._processed_uploads.add(upload_key)
        if auto_dispatch:
            if kind == "data":
                st.session_state.pending_auto_examine = dest_str
            else:
                st.session_state.pending_auto_load_metadata = dest_str
        st.sidebar.success(f"Uploaded {kind} file: {uploaded_file.name}")


def save_upload_batch(uploaded_files: list, kind: str, auto_dispatch: bool = True) -> None:
    """Save multiple uploaded files into a subdirectory for series/batch analysis."""
    session_dir = Path(st.session_state.session_dir)
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)

    # Create a subdirectory for the series
    series_dir = upload_dir / "series"
    series_dir.mkdir(exist_ok=True)

    for f in uploaded_files:
        dest = series_dir / f.name
        dest.write_bytes(f.getvalue())

    series_dir_str = str(series_dir)
    st.session_state.uploaded_data_path = series_dir_str

    upload_key = (kind, series_dir_str)
    if upload_key not in st.session_state._processed_uploads:
        st.session_state._processed_uploads.add(upload_key)
        if auto_dispatch:
            st.session_state.pending_auto_examine = series_dir_str
        st.sidebar.success(f"Uploaded {len(uploaded_files)} files to series/")
