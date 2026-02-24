"""Sidebar component: configuration, file upload, agent status."""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from ..config import MODEL_OPTIONS, SUPPORTED_DATA_EXTENSIONS, SUPPORTED_METADATA_EXTENSIONS


_LOGO_PATH = Path(__file__).resolve().parent.parent.parent.parent / "misc" / "scilink_logo_v3_dark.svg"


def render_sidebar() -> None:
    with st.sidebar:
        if st.session_state.agent_initialized and _LOGO_PATH.exists():
            col_l, col_c, col_r = st.columns([1, 3, 1])
            with col_c:
                st.image(str(_LOGO_PATH), width=140)
        else:
            st.title("SciLink")
            st.markdown("")
        preset = st.selectbox(
            "Model",
            MODEL_OPTIONS + ["Custom"],
            key="cfg_model_preset",
        )
        if preset == "Custom":
            model = st.text_input("Model name", key="cfg_model_custom")
        else:
            model = preset
        api_key = st.text_input("API key", type="password", key="cfg_api_key")
        base_url = st.text_input("Base URL (optional)", key="cfg_base_url")
        fh_api_key = st.text_input("FutureHouse API key (optional)", type="password", key="cfg_fh_api_key")
        mode = st.selectbox(
            "Analysis mode",
            ["co-pilot", "supervised", "autonomous"],
            key="cfg_mode",
        )
        consent = st.checkbox(
            "I understand that the agent will execute generated Python code on my machine",
            key="cfg_consent",
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
                _start_session(model, api_key, base_url, mode, fh_api_key)

        with col2:
            if st.button("Reset Session", disabled=not st.session_state.agent_initialized,
                         width="stretch"):
                _reset_session()

        # ── File upload ──────────────────────────────────────────
        if st.session_state.agent_initialized:
            st.divider()
            st.subheader("Upload Files")

            data_file = st.file_uploader(
                "Data file",
                type=[e.lstrip(".") for e in SUPPORTED_DATA_EXTENSIONS],
                key="uploader_data",
            )
            if data_file is not None:
                save_upload(data_file, "data")

            meta_file = st.file_uploader(
                "Metadata file",
                type=[e.lstrip(".") for e in SUPPORTED_METADATA_EXTENSIONS],
                key="uploader_meta",
            )
            if meta_file is not None:
                save_upload(meta_file, "metadata")

            # ── Agent status ─────────────────────────────────────
            st.divider()
            st.subheader("Agent Status")
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


# ── helpers ──────────────────────────────────────────────────────

def _start_session(model: str, api_key: str, base_url: str, mode: str, fh_api_key: str = "") -> None:
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )
    import scilink.executors as executors

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key and not base_url:
        st.sidebar.error("Provide an API key or set an environment variable (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).")
        return

    mode_map = {
        "co-pilot": AnalysisMode.CO_PILOT,
        "supervised": AnalysisMode.SUPERVISED,
        "autonomous": AnalysisMode.AUTONOMOUS,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"analysis_session_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Auto-approve sandbox for the Streamlit session (user checked consent box)
    executors._GLOBAL_SANDBOX_APPROVED = True

    with st.sidebar:
        with st.spinner("Initializing agent..."):
            try:
                agent = AnalysisOrchestratorAgent(
                    base_dir=str(session_dir),
                    api_key=resolved_key,
                    model_name=model,
                    base_url=base_url or None,
                    analysis_mode=mode_map[mode],
                    futurehouse_api_key=fh_api_key or None,
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
