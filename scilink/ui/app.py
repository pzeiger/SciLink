"""SciLink Streamlit UI — single-page app."""

import builtins
import threading
from pathlib import Path

import streamlit as st

from scilink.ui.state import init_session_state, ChatTask, FeedbackRequest
from scilink.ui.components.sidebar import render_sidebar, save_upload
from scilink.ui.components.file_viewer import render_file_preview
from scilink.ui.output_capture import OutputCapture
from scilink.ui.theme import inject_theme
from scilink.ui.config import AVATAR_USER, AVATAR_AGENT, SUPPORTED_DATA_EXTENSIONS, SUPPORTED_METADATA_EXTENSIONS

_LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "misc" / "scilink_logo_v3_dark.svg"

st.set_page_config(page_title="SciLink", layout="wide")

# Handle reset: clear the query param so the next rerun is clean
if st.query_params.get("reset"):
    st.query_params.clear()

inject_theme()
init_session_state()
render_sidebar()

# ── helpers ──────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _find_new_images() -> list[str]:
    """Return image paths in the session dir not yet shown in chat.

    Skips temporary review files (e.g. first_spectrum_fit_review.png)
    to avoid showing the same fit plot twice during the feedback step.
    """
    session_dir = st.session_state.session_dir
    if session_dir is None:
        return []
    new = []
    for ext in IMAGE_EXTENSIONS:
        for p in Path(session_dir).rglob(f"*{ext}"):
            if "review" in p.stem:
                continue
            s = str(p)
            if s not in st.session_state.known_images:
                st.session_state.known_images.add(s)
                new.append(s)
    return new


def _find_new_html_reports() -> list[str]:
    """Return HTML report paths in the session dir not yet shown."""
    session_dir = st.session_state.session_dir
    if session_dir is None:
        return []
    new = []
    for p in Path(session_dir).rglob("*.html"):
        s = str(p)
        if s not in st.session_state.known_images:  # reuse the same set
            st.session_state.known_images.add(s)
            new.append(s)
    return new


def _run_agent_chat(task: ChatTask, agent, user_input: str) -> None:
    """Target for the background thread."""
    import logging

    original_input = builtins.input

    def _streamlit_input(prompt: str = "") -> str:
        context = cap.getvalue()
        req = FeedbackRequest(prompt=prompt, context=context)
        task.feedback_request = req
        req.event.wait()
        task.feedback_request = None
        return req.response or ""

    cap = OutputCapture()
    task.live_capture = cap

    # Route logging output through the capture buffer so that
    # logging.info() messages (used by the verification-correction
    # loop) appear in the live verbose panel alongside print() output.
    log_handler = logging.StreamHandler(cap._buffer)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        builtins.input = _streamlit_input
        with cap:
            result = agent.chat(user_input)
        task.result = result
        task.verbose_log = cap.getvalue()
    except Exception as exc:
        task.error = str(exc)
        task.verbose_log = cap.getvalue()
    finally:
        builtins.input = original_input
        root_logger.removeHandler(log_handler)
        task.live_capture = None
        task.is_running = False


def _start_task(prompt: str) -> None:
    """Create a ChatTask, launch the agent thread, and rerun."""
    new_task = ChatTask(is_running=True)
    st.session_state.chat_task = new_task
    t = threading.Thread(
        target=_run_agent_chat,
        args=(new_task, st.session_state.agent, prompt),
        daemon=True,
    )
    new_task.thread = t
    t.start()
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# Welcome screen (before session is started)
# ══════════════════════════════════════════════════════════════════
if not st.session_state.agent_initialized:
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width="stretch")
        else:
            st.title("SciLink")
        st.markdown(
            '<p style="text-align:center;color:#9E9E9E;font-size:1.1em;'
            'margin-top:-8px;margin-bottom:24px">'
            "LLM-powered analysis for experimental science</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="text-align:center;color:#B0B0B0;font-size:0.95em;'
            'letter-spacing:0.3px">'
            '<span style="color:#E0E0E0">Configure model & API key</span>'
            '&ensp;<span style="color:#82B1FF">&rarr;</span>&ensp;'
            '<span style="color:#E0E0E0">Upload data</span>'
            '&ensp;<span style="color:#82B1FF">&rarr;</span>&ensp;'
            '<span style="color:#E0E0E0">Chat with the agent</span>'
            "</p>",
            unsafe_allow_html=True,
        )
    st.stop()


# ══════════════════════════════════════════════════════════════════
# Active session — Chat + File Explorer tabs
# ══════════════════════════════════════════════════════════════════
chat_tab, files_tab = st.tabs(["Chat", "File Explorer"])

# ── Chat tab ─────────────────────────────────────────────────────
with chat_tab:
    # Show a prominent upload zone until the chat conversation starts
    if not st.session_state.chat_messages:
        st.markdown(
            '<div style="border:2px dashed #4A5568;border-radius:10px;'
            "padding:32px 16px;text-align:center;margin-bottom:16px;"
            'background:#1E2530">'
            '<p style="color:#82B1FF;font-size:1.1em;margin:0 0 4px 0">'
            "Upload your data to get started</p>"
            '<p style="color:#6B7A8C;font-size:0.85em;margin:0">'
            "Images, CSV, NumPy arrays, and more</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        up_data, up_meta = st.columns(2)
        with up_data:
            main_data = st.file_uploader(
                "Data file",
                type=[e.lstrip(".") for e in SUPPORTED_DATA_EXTENSIONS],
                key="main_uploader_data",
            )
            if main_data is not None:
                save_upload(main_data, "data", auto_dispatch=False)
        with up_meta:
            main_meta = st.file_uploader(
                "Metadata (optional)",
                type=[e.lstrip(".") for e in SUPPORTED_METADATA_EXTENSIONS],
                key="main_uploader_meta",
            )
            if main_meta is not None:
                save_upload(main_meta, "metadata", auto_dispatch=False)

        # Show "Analyze" button once data is uploaded
        if st.session_state.uploaded_data_path:
            if st.button("Analyze", type="primary", width="stretch"):
                data_path = st.session_state.uploaded_data_path
                meta_path = st.session_state.uploaded_metadata_path
                if data_path and meta_path:
                    prompt = (
                        f"I uploaded a data file at `{data_path}` "
                        f"and a metadata file at `{meta_path}`. "
                        f"Please examine the data and load the metadata."
                    )
                else:
                    prompt = (
                        f"I uploaded a data file at `{data_path}`. "
                        f"Please examine it."
                    )
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                _start_task(prompt)

    _avatars = {"user": AVATAR_USER, "assistant": AVATAR_AGENT}
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar=_avatars.get(msg["role"])):
            st.markdown(msg["content"])
            for img_path in msg.get("images", []):
                try:
                    st.image(img_path)
                except Exception:
                    st.caption(f"(image not found: {img_path})")
            for html_path in msg.get("html_reports", []):
                p = Path(html_path)
                if p.exists():
                    with st.expander(f"Report: {p.name}"):
                        st.components.v1.html(
                            p.read_text(encoding="utf-8"),
                            height=600,
                            scrolling=True,
                        )
                    st.download_button(
                        f"Download {p.name}",
                        data=p.read_bytes(),
                        file_name=p.name,
                        mime="text/html",
                        key=f"dl_html_{html_path}",
                    )
            if msg.get("verbose"):
                with st.expander("Verbose output"):
                    st.code(msg["verbose"], language="text")

    # ── Agent monitoring fragment ─────────────────────────────────
    # Uses @st.fragment to rerun only this section (not the full page)
    # when the agent is working.  run_every="1s" polls the background
    # thread; scope="app" escalates to a full rerun when needed.
    task: ChatTask = st.session_state.chat_task
    _needs_polling = task.is_running and task.feedback_request is None
    _monitor_interval = "1s" if _needs_polling else None

    @st.fragment(run_every=_monitor_interval)
    def _agent_monitor():
        task: ChatTask = st.session_state.chat_task

        # ── 1. Completion — append result, full rerun to render it ──
        if not task.is_running and (task.result is not None or task.error is not None):
            content = task.result if task.result is not None else f"Error: {task.error}"
            new_images = _find_new_images()
            new_reports = _find_new_html_reports()
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": content,
                "images": new_images,
                "html_reports": new_reports,
                "verbose": task.verbose_log or "",
            })
            st.session_state.chat_task = ChatTask()
            st.rerun(scope="app")
            return

        # ── 2. Feedback — render review UI, wait for user action ────
        if task.is_running and task.feedback_request is not None:
            req: FeedbackRequest = task.feedback_request

            # Cache preview images so fragment reruns don't lose them
            # (_find_new_images marks images as known, returning [] on
            # subsequent calls — but the fragment clears its output each
            # rerun, so we need to keep the paths around).
            if "_feedback_preview_images" not in st.session_state:
                st.session_state._feedback_preview_images = _find_new_images()
            for img_path in st.session_state._feedback_preview_images:
                st.image(img_path)

            if req.context:
                import html as _html
                import re

                display_ctx = req.context
                lines = display_ctx.split("\n")
                # Find the last separator block (===…) followed by a
                # non-empty content line — this is the start of the
                # review section (e.g. fit result or plan summary).
                start = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("=" * 20) and i + 1 < len(lines) and lines[i + 1].strip():
                        start = i
                if start:
                    display_ctx = "\n".join(lines[start:])
                # Strip === separator lines
                display_ctx = re.sub(r"^[=]{10,}\s*$", "", display_ctx, flags=re.MULTILINE)
                # Strip whitespace-only lines (so they count as truly blank)
                display_ctx = re.sub(r"^[ \t]+$", "", display_ctx, flags=re.MULTILINE)
                # Collapse all runs of blank lines into a single newline
                display_ctx = re.sub(r"\n{2,}", "\n", display_ctx).strip()
                # Re-insert one blank line before each section header
                # (lines starting with an emoji) for visual breathing room,
                # but not between header and its body.
                display_ctx = re.sub(
                    r"\n(?=[\U0001f300-\U0001fAFF\u2600-\u27BF\u2700-\u27BF])",
                    "\n\n",
                    display_ctx,
                )
                # Add an extra blank line after the top-level title
                # (e.g. "📋 PROPOSED FITTING PLAN" or "📊 FIRST SPECTRUM FIT RESULT")
                display_ctx = re.sub(
                    r"^(.+(?:PLAN|RESULT|REVIEW).*)$",
                    r"\1\n",
                    display_ctx,
                    count=1,
                    flags=re.MULTILINE,
                )
                escaped_ctx = _html.escape(display_ctx)
                # Estimate height: ~20px per line, clamped to 150-400px
                n_lines = escaped_ctx.count("\n") + 1
                box_h = max(150, min(400, n_lines * 20 + 32))
                st.components.v1.html(
                    f'<pre style="background:#1e1e1e;margin:0;'
                    f"border:1px solid #333;border-radius:6px;"
                    f"padding:12px 16px;font-family:monospace;"
                    f"font-size:13px;color:#e0e0e0;"
                    f"white-space:pre-wrap;word-wrap:break-word;"
                    f'overflow-y:auto;line-height:1.5">'
                    f"{escaped_ctx}</pre>",
                    height=box_h,
                    scrolling=True,
                )

            feedback = st.text_area(
                "Your feedback (optional):",
                key="feedback_input",
            )
            col_submit, col_accept = st.columns(2)
            with col_submit:
                if st.button("Submit feedback", disabled=not feedback.strip(),
                             width="stretch"):
                    req.response = feedback.strip()
                    req.event.set()
                    st.session_state.pop("_feedback_preview_images", None)
                    st.rerun(scope="app")
            with col_accept:
                if st.button("Accept as-is", width="stretch"):
                    req.response = ""
                    req.event.set()
                    st.session_state.pop("_feedback_preview_images", None)
                    st.rerun(scope="app")
            return

        # ── 3. Live monitoring — fragment auto-reruns, no blocking ──
        if task.is_running:
            st.markdown(
                '<div class="agent-spinner-container">'
                '  <div class="agent-spinner-dot"></div>'
                '  <div class="agent-spinner-dot"></div>'
                '  <div class="agent-spinner-dot"></div>'
                '  <span class="agent-spinner-label">Agent is working...</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            live = ""
            if task.live_capture is not None:
                try:
                    live = task.live_capture.getvalue()
                except Exception:
                    pass
            if live:
                show = st.toggle("Show verbose output", key="live_verbose_toggle")
                if show:
                    import html as _html

                    tail = "\n".join(live.split("\n")[-200:])
                    escaped = _html.escape(tail)
                    st.components.v1.html(
                        f'<pre style="height:280px;overflow-y:auto;margin:0;'
                        f"background:#1e1e1e;padding:8px;border-radius:4px;"
                        f"border:1px solid #333;font-family:monospace;"
                        f"font-size:13px;white-space:pre-wrap;"
                        f'color:#e0e0e0" id="log">{escaped}</pre>'
                        f"<script>var e=document.getElementById('log');"
                        f"e.scrollTop=e.scrollHeight;</script>",
                        height=300,
                        scrolling=False,
                    )

    _agent_monitor()

    # ── Auto-dispatch uploads ────────────────────────────────────
    auto_prompt = None
    if not task.is_running:
        data_path = st.session_state.pop("pending_auto_examine", None)
        meta_path = st.session_state.pop("pending_auto_load_metadata", None)
        if data_path and meta_path:
            auto_prompt = (
                f"I uploaded a data file at `{data_path}` "
                f"and a metadata file at `{meta_path}`. "
                f"Please examine the data and load the metadata."
            )
        elif data_path:
            auto_prompt = (
                f"I uploaded a data file at `{data_path}`. "
                f"Please examine it."
            )
        elif meta_path:
            auto_prompt = (
                f"I uploaded a metadata file at `{meta_path}`. "
                f"Please load it."
            )

    if auto_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": auto_prompt})
        _start_task(auto_prompt)

    if prompt := st.chat_input("Message the analysis agent...", disabled=task.is_running):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        _start_task(prompt)

# ── File Explorer tab ────────────────────────────────────────────

_FILE_TYPE_ICONS = {
    ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".tif": "🖼", ".tiff": "🖼",
    ".csv": "📊", ".tsv": "📊", ".xlsx": "📊",
    ".json": "📋", ".npy": "🔢",
    ".html": "📄", ".txt": "📝", ".md": "📝", ".log": "📝",
    ".py": "🐍",
}


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


with files_tab:
    if st.session_state.session_dir is None:
        st.info("Start a session first to browse output files.")
    else:
        session_path = Path(st.session_state.session_dir)
        tree_col, preview_col = st.columns([1, 2])

        with tree_col:
            st.subheader("Session Files")

            files = sorted(
                [p for p in session_path.rglob("*") if p.is_file()],
                key=lambda p: str(p),
            )

            if not files:
                st.caption("No files found yet.")
                selected_file = None
            else:
                # Group files by parent directory and build labels
                from collections import OrderedDict
                groups: OrderedDict[str, list[Path]] = OrderedDict()
                for f in files:
                    rel = f.relative_to(session_path)
                    group = str(rel.parent) if str(rel.parent) != "." else "root"
                    groups.setdefault(group, []).append(f)

                # Build a single flat list with group-prefixed labels
                all_labels: list[str] = []
                label_to_path: dict[str, Path] = {}
                group_boundaries: dict[int, str] = {}  # index → group name

                for group_name, group_files in groups.items():
                    group_boundaries[len(all_labels)] = group_name
                    for f in group_files:
                        icon = _FILE_TYPE_ICONS.get(f.suffix.lower(), "📁")
                        size = _format_size(f.stat().st_size)
                        label = f"{icon}  {f.name}  ({size})"
                        all_labels.append(label)
                        label_to_path[label] = f

                # Render group headers above the radio
                for idx, gname in group_boundaries.items():
                    display_name = gname.replace("/", " / ")
                    count = len(groups[gname])
                    st.markdown(
                        f'<div class="file-group-header">'
                        f'{display_name} ({count} file{"s" if count != 1 else ""})'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                selected_label = st.radio(
                    "Files",
                    all_labels,
                    key="file_explorer_selection",
                    label_visibility="collapsed",
                )
                selected_file = label_to_path.get(selected_label)

        with preview_col:
            if selected_file and selected_file.exists():
                render_file_preview(selected_file)
            else:
                st.caption("Select a file to preview.")
