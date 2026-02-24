"""SciLink Streamlit UI — single-page app."""

import base64
import builtins
import threading
from pathlib import Path

import streamlit as st

from scilink.ui.state import init_session_state, ChatTask, FeedbackRequest
from scilink.ui.components.sidebar import render_sidebar, save_upload, save_upload_batch, start_session
from scilink.ui.components.file_viewer import render_file_preview
from scilink.ui.output_capture import OutputCapture
from scilink.ui.theme import inject_theme
from scilink.ui.config import AVATAR_USER, AVATAR_AGENT, SUPPORTED_DATA_EXTENSIONS, SUPPORTED_METADATA_EXTENSIONS

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "scilink_logo_v3_dark.svg"

st.set_page_config(page_title="SciLink", layout="wide")

# Handle reset: clear the query param so the next rerun is clean
if st.query_params.get("reset"):
    st.query_params.clear()

inject_theme()
init_session_state()
render_sidebar()

# ── helpers ──────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _find_new_images(summary_only: bool = False) -> list[str]:
    """Return image paths in the session dir not yet shown in chat.

    Skips temporary review files (e.g. first_spectrum_fit_review.png)
    to avoid showing the same fit plot twice during the feedback step.

    If *summary_only* is True, only returns summary grid images
    (files containing ``Summary_Grid`` in their name). This is used
    during the human-feedback step so the user sees only the final
    NMF/PCA grid rather than every per-component plot.
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
    if summary_only:
        new = [p for p in new if "Summary_Grid" in Path(p).stem]
    return new


def _find_feedback_preview_images() -> list[str]:
    """Return preview images specifically meant for the feedback step.

    Finds curve-fitting review images (``*review*``) and hyperspectral
    Summary_Grid images (``*Summary_Grid*``).  The search is scoped to
    the most recently modified ``analysis_*`` directory so that images
    from earlier analyses in the same session are not shown.
    """
    session_dir = st.session_state.session_dir
    if session_dir is None:
        return []
    # Scope to the most recently modified analysis directory so we
    # don't pick up stale Summary_Grid / review images from previous runs.
    search_root = Path(session_dir)
    results_dir = search_root / "results"
    if results_dir.exists():
        analysis_dirs = sorted(
            [d for d in results_dir.iterdir()
             if d.is_dir() and d.name.startswith("analysis_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if analysis_dirs:
            search_root = analysis_dirs[0]
    previews = []
    for ext in IMAGE_EXTENSIONS:
        for p in search_root.rglob(f"*{ext}"):
            if "review" in p.stem or "Summary_Grid" in p.stem:
                previews.append(str(p))
    return previews


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
    # A thread filter ensures each session only captures its own agent's
    # logs, preventing cross-talk when multiple analyses run concurrently.
    log_handler = logging.StreamHandler(cap._buffer)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    _this_thread = threading.get_ident()
    log_handler.addFilter(lambda record: record.thread == _this_thread)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        builtins.input = _streamlit_input
        with cap:
            result = agent.chat(user_input)
        if not task.stopped:
            task.result = result
            task.verbose_log = cap.getvalue()
    except Exception as exc:
        if not task.stopped:
            task.error = str(exc)
            task.verbose_log = cap.getvalue()
    finally:
        builtins.input = original_input
        root_logger.removeHandler(log_handler)
        task.live_capture = None
        if not task.stopped:
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
    # Check if session init was requested from the sidebar
    _pending = st.session_state.pop("_pending_init", None)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if _LOGO_PATH.exists():
            _b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
            st.markdown(
                '<style>'
                '@keyframes logo-spin{to{transform:rotate(360deg)}}'
                '.logo-glow{position:relative;padding:2px;border-radius:14px;overflow:hidden}'
                '.logo-glow::before{content:"";position:absolute;'
                'top:-40%;left:-40%;width:180%;height:180%;'
                'background:conic-gradient('
                'transparent 0deg,transparent 270deg,#3A4556 300deg,'
                '#82B1FF 330deg,#FFF 345deg,#82B1FF 355deg,transparent 360deg);'
                'animation:logo-spin 4s linear infinite;z-index:0}'
                '.logo-glow>img{position:relative;z-index:1;border-radius:12px;'
                'display:block;width:100%}'
                '</style>'
                f'<div class="logo-glow">'
                f'<img src="data:image/svg+xml;base64,{_b64}"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.title("SciLink")
        st.markdown(
            '<p style="text-align:center;color:#9E9E9E;font-size:1.1em;'
            'margin-top:12px;margin-bottom:24px">'
            "LLM-powered agents for scientific research automation</p>",
            unsafe_allow_html=True,
        )

        if _pending is not None:
            st.markdown(
                '<p style="text-align:center;color:#82B1FF;font-size:1em">'
                '⏳ Initializing agents...</p>',
                unsafe_allow_html=True,
            )
            start_session(**_pending)
            # start_session calls st.rerun() on success, so we only
            # reach here if initialization failed.
            st.stop()

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
                "Data file(s)",
                type=[e.lstrip(".") for e in SUPPORTED_DATA_EXTENSIONS],
                key="main_uploader_data",
                accept_multiple_files=True,
            )
            if main_data:
                if len(main_data) == 1:
                    save_upload(main_data[0], "data", auto_dispatch=False)
                else:
                    save_upload_batch(main_data, "data", auto_dispatch=False)
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
            # When an HTML report is present it already embeds the
            # relevant figures — skip showing raw images separately
            # to avoid duplicate clutter (matches curve fitting UX).
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": content,
                "images": [] if new_reports else new_images,
                "html_reports": new_reports,
                "verbose": task.verbose_log or "",
            })
            st.session_state.chat_task = ChatTask()
            st.rerun(scope="app")
            return

        # ── 2. Feedback — render review UI, wait for user action ────
        if task.is_running and task.feedback_request is not None:
            req: FeedbackRequest = task.feedback_request

            # Cache preview images so fragment reruns don't lose them.
            # First call _find_new_images() to mark all current images
            # as known (side-effect) so they don't re-appear in the
            # completion message.  Then use _find_feedback_preview_images
            # to locate the actual preview images the user should see:
            #   • curve fitting  → *review*.png  (fit preview)
            #   • hyperspectral  → *Summary_Grid*.jpeg  (NMF/PCA grid)
            if "_feedback_preview_images" not in st.session_state:
                _find_new_images()  # mark intermediate images as known
                st.session_state._feedback_preview_images = _find_feedback_preview_images()
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
            _spin_col, _stop_col = st.columns([4, 1])
            with _spin_col:
                st.markdown(
                    '<div class="agent-spinner-container">'
                    '  <div class="agent-spinner-dot"></div>'
                    '  <div class="agent-spinner-dot"></div>'
                    '  <div class="agent-spinner-dot"></div>'
                    '  <span class="agent-spinner-label">Agent is working...</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with _stop_col:
                if st.button("Stop", type="secondary", key="stop_agent_btn",
                             use_container_width=True):
                    task.stopped = True
                    task.is_running = False
                    task.verbose_log = (
                        task.live_capture.getvalue() if task.live_capture else ""
                    )
                    task.live_capture = None
                    # Unblock any pending feedback wait
                    if task.feedback_request is not None:
                        task.feedback_request.response = ""
                        task.feedback_request.event.set()
                        task.feedback_request = None
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "Analysis stopped by user.",
                        "verbose": task.verbose_log,
                    })
                    st.session_state.chat_task = ChatTask()
                    st.rerun(scope="app")
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


def _discover_sessions(current_session_dir: str | None) -> list[tuple[str, Path]]:
    """Return (display_label, path) for every analysis_session_* directory."""
    if current_session_dir is None:
        return []
    parent = Path(current_session_dir).parent
    sessions = sorted(
        parent.glob("analysis_session_*"),
        key=lambda p: p.name,
        reverse=True,
    )
    current = Path(current_session_dir)
    result: list[tuple[str, Path]] = []
    for s in sessions:
        # Parse timestamp from directory name like analysis_session_20260223_201729
        parts = s.name.removeprefix("analysis_session_")
        try:
            label = f"{parts[:4]}-{parts[4:6]}-{parts[6:8]} {parts[9:11]}:{parts[11:13]}:{parts[13:15]}"
        except (IndexError, ValueError):
            label = s.name
        if s.resolve() == current.resolve():
            label += " (current)"
        result.append((label, s))
    return result


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


with files_tab:
    sessions = _discover_sessions(st.session_state.session_dir)
    if not sessions:
        st.info("Start a session first to browse output files.")
    else:
        labels = [s[0] for s in sessions]
        paths = [s[1] for s in sessions]
        selected_idx = st.selectbox(
            "Session",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key="session_selector",
        )
        session_path = paths[selected_idx]
        tree_col, preview_col = st.columns([1, 2])

        # Clear selected file when switching sessions
        sel = st.session_state.get("selected_preview_file")
        if sel and not str(sel).startswith(str(session_path)):
            st.session_state.selected_preview_file = None

        with tree_col:
            st.subheader("Session Files")

            all_files = list(session_path.rglob("*"))
            has_files = any(f.is_file() for f in all_files)

            if not has_files:
                st.caption("No files found yet.")
            else:
                def _render_dir(dir_path: Path, depth: int = 0) -> None:
                    """Render directory tree using expanders."""
                    items = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
                    dirs = [i for i in items if i.is_dir()]
                    files = [i for i in items if i.is_file()]

                    for d in dirs:
                        child_count = sum(1 for _ in d.rglob("*") if _.is_file())
                        if child_count == 0:
                            continue
                        with st.expander(f"📁 {d.name} ({child_count})", expanded=(depth == 0)):
                            _render_dir(d, depth + 1)

                    def _render_files(file_list: list[Path]) -> None:
                        for f in file_list:
                            icon = _FILE_TYPE_ICONS.get(f.suffix.lower(), "📄")
                            size = _format_size(f.stat().st_size)
                            is_sel = st.session_state.get("selected_preview_file") == str(f)
                            btn_type = "primary" if is_sel else "secondary"
                            if st.button(
                                f"{icon}  {f.name}  ({size})",
                                key=f"fbtn_{f.relative_to(session_path)}",
                                use_container_width=True,
                                type=btn_type,
                            ):
                                st.session_state.selected_preview_file = str(f)
                                st.rerun()

                    if files and depth == 0:
                        with st.expander(f"📄 Session ({len(files)})", expanded=True):
                            _render_files(files)
                    else:
                        _render_files(files)

                _render_dir(session_path)

        selected_file = None
        sel_path = st.session_state.get("selected_preview_file")
        if sel_path:
            selected_file = Path(sel_path)

        with preview_col:
            if selected_file and selected_file.exists():
                render_file_preview(selected_file)
            else:
                st.caption("Select a file to preview.")
