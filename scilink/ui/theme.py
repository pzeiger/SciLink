"""Material Design Dark theme — custom CSS that complements .streamlit/config.toml."""

import streamlit as st

_MATERIAL_CSS = """
<style>
/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #1E2530;
    border-right: 1px solid #3A4556;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child > div:first-child {
    padding-top: 0 !important;
}
/* Tighten sidebar vertical spacing */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    gap: 0.4rem;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.4em !important;
    margin: 0 0 0.25rem 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.95em !important;
    margin: 0.25rem 0 0.15rem 0 !important;
    padding: 0 !important;
}
/* Pull the logo up toward the top of the sidebar */
section[data-testid="stSidebar"] [data-testid="stMarkdown"]:has(.logo-glow-sm) {
    margin-top: -2rem !important;
    padding-top: 0 !important;
}

/* ── Sidebar section dividers ───────────────────────── */
section[data-testid="stSidebar"] hr {
    border-color: #3A4556;
    margin: 0.4rem 0;
}

/* ── Sidebar metric styling ─────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background-color: #2A3340;
    border: 1px solid #3A4556;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #9E9E9E;
    font-size: 0.75em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #E0E0E0;
    font-size: 0.9em;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
    border: none;
    border-radius: 4px;
    font-weight: 500;
    letter-spacing: 0.3px;
    transition: background-color 0.2s, box-shadow 0.2s;
    white-space: nowrap;
}
/* Primary buttons — purple (default for all) */
button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #6200EE !important;
    color: #FFFFFF !important;
    font-weight: 600;
}
/* Uppercase only for action buttons inside the chat tabs */
.stTabs .stButton > button[kind="primary"] {
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background-color: #7C4DFF !important;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.35);
}
/* Sidebar buttons — original purple */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #6200EE;
    color: #FFFFFF;
    font-size: 0.85em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 0.4rem 0.6rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #7C4DFF;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.35);
}
/* File explorer tree buttons — soft gray, blue on selection */
[data-testid="stExpander"] .stButton > button {
    background-color: #2A3340;
    color: #B0BEC5;
    border: 1px solid #3A4556;
    text-transform: none;
    font-weight: 400;
    letter-spacing: 0;
    font-size: 0.85em;
    padding: 0.25rem 0.5rem;
}
[data-testid="stExpander"] .stButton > button:hover {
    background-color: #344155;
    border-color: #5B8DEF;
    color: #E0E0E0;
    box-shadow: none;
}
[data-testid="stExpander"] .stButton > button[kind="primary"] {
    background-color: #1A3A5C;
    border-color: #5B8DEF;
    color: #82B1FF;
}

/* ── Success-style button (used via st.markdown class) ─ */
.success-btn > button {
    background-color: #03DAC6 !important;
    color: #121212 !important;
    font-weight: 600;
}
.success-btn > button:hover {
    background-color: #04F1DB !important;
    box-shadow: 0 2px 8px rgba(3, 218, 198, 0.35) !important;
}

/* ── Chat messages ──────────────────────────────────── */
.stChatMessage {
    border-radius: 8px;
    border: 1px solid #3A4556;
}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    color: #B0B0B0;
}
.stTabs [aria-selected="true"] {
    color: #82B1FF;
    border-bottom-color: #82B1FF;
}

/* ── Expanders ──────────────────────────────────────── */
.streamlit-expanderHeader {
    color: #B0B0B0;
    font-size: 0.85em;
}

/* ── Text inputs & text areas ───────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #1E2530;
    color: #E0E0E0;
    border: 1px solid #4A5568;
    border-radius: 4px;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #BB86FC;
    box-shadow: 0 0 4px rgba(187, 134, 252, 0.3);
}

/* ── Select boxes ───────────────────────────────────── */
.stSelectbox > div > div {
    background-color: #1E2530;
    border: 1px solid #4A5568;
    border-radius: 4px;
}

/* ── File uploader ──────────────────────────────────── */
.stFileUploader > div {
    border: 1px dashed #4A5568;
    border-radius: 4px;
}

/* ── Code blocks ────────────────────────────────────── */
.stCodeBlock {
    border: 1px solid #3A4556;
    border-radius: 4px;
}

/* ── Scrollbar (webkit) ─────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #252D38;
}
::-webkit-scrollbar-thumb {
    background: #4A5568;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #82B1FF;
}

/* ── Headings ───────────────────────────────────────── */
h1 {
    color: #82B1FF !important;
}
h2, h3 {
    color: #E0E0E0 !important;
}

/* ── Agent working spinner ──────────────────────────── */
@keyframes scilink-pulse {
    0%, 100% { opacity: 0.4; transform: scale(1); }
    50%      { opacity: 1;   transform: scale(1.05); }
}
.agent-spinner-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: linear-gradient(135deg, #1E2530 0%, #252D38 100%);
    border: 1px solid #3A4556;
    border-left: 3px solid #4FC3F7;
    border-radius: 6px;
}
.agent-spinner-heart {
    font-size: 1.1em;
    animation: scilink-pulse 1.4s ease-in-out infinite;
}
.agent-spinner-heart:nth-child(2) { animation-delay: 0.2s; }
.agent-spinner-heart:nth-child(3) { animation-delay: 0.4s; }
.agent-spinner-label {
    color: #E0E0E0;
    font-size: 0.9em;
    font-weight: 500;
}

/* ── Stop button (square icon beside spinner) ──────── */
/* Push the button wrapper down to align with the spinner bar */
[data-testid="stHorizontalBlock"]:has(.agent-spinner-container)
    > [data-testid="stColumn"]:last-child .stButton {
    padding-top: 10px;
}
.stTabs .stButton > button[kind="secondary"] {
    width: 100% !important;
    height: 58px !important;
    min-height: 58px !important;
    padding: 0 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1em;
    border-radius: 6px;
    background-color: #3A4556 !important;
    color: #E0E0E0 !important;
    border: 1px solid #4A5568 !important;
    line-height: 1;
    text-transform: none;
}
.stTabs .stButton > button[kind="secondary"]:hover {
    background-color: #D32F2F !important;
    border-color: #D32F2F !important;
    color: #FFFFFF !important;
}

/* ── Live log viewer ────────────────────────────────── */
.live-log-viewer {
    height: 280px;
    overflow-y: auto;
    margin: 0;
    background: #1E2530;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #3A4556;
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
    color: #e0e0e0;
}

/* ── Hide Streamlit chrome (deploy, menu, stop) ────── */
.stDeployButton,
[data-testid="stAppDeployButton"],
#MainMenu,
[data-testid="stMainMenu"],
header [data-testid="stStatusWidget"] {
    display: none !important;
}

/* ── Floating background emojis ───────────────────── */
.floating-emojis {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    animation: emojis-fade-in 2s ease-out forwards;
}
@keyframes emojis-fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.floating-emojis span {
    position: absolute;
    display: block;
    font-size: var(--emoji-size, 28px);
    opacity: 0;
    animation: emoji-float var(--duration, 18s) var(--delay, 0s) ease-in-out infinite;
}
@keyframes emoji-float {
    0% {
        opacity: 0;
        transform: translateY(100vh) rotate(0deg);
    }
    3% {
        opacity: var(--peak-opacity, 0.12);
    }
    93% {
        opacity: var(--peak-opacity, 0.12);
    }
    100% {
        opacity: 0;
        transform: translateY(-10vh) rotate(var(--rotation, 360deg));
    }
}
</style>
"""

_FLOATING_HTML = """
<div class="floating-emojis" aria-hidden="true">
{spans}
</div>
"""


def _build_emoji_spans(n_hearts: int = 7, n_pluses: int = 7) -> str:
    """Generate absolutely-positioned emoji spans with randomised CSS vars."""
    import random

    emojis: list[str] = (
        ["\U0001f49c"] * n_hearts  # purple hearts
        + ["\u2795"] * n_pluses     # plus signs
    )
    random.shuffle(emojis)

    spans: list[str] = []
    for emoji in emojis:
        left = random.randint(2, 96)
        size = random.randint(20, 40)
        duration = round(random.uniform(16, 30), 1)
        delay = round(random.uniform(0, 3), 1)
        rotation = random.choice([-360, -180, 180, 360])
        is_plus = emoji == "\u2795"
        opacity = round(random.uniform(0.12, 0.22) if is_plus else random.uniform(0.10, 0.20), 2)
        spans.append(
            f'<span style="left:{left}%;'
            f"--emoji-size:{size}px;"
            f"--duration:{duration}s;"
            f"--delay:{delay}s;"
            f"--rotation:{rotation}deg;"
            f'--peak-opacity:{opacity}">{emoji}</span>'
        )
    return "\n".join(spans)


def inject_theme() -> None:
    """Inject the Material Design CSS into the current page."""
    st.markdown(_MATERIAL_CSS, unsafe_allow_html=True)

    n_hearts = st.session_state.get("vibe_hearts", 7)
    n_pluses = st.session_state.get("vibe_pluses", 7)
    if n_hearts or n_pluses:
        st.markdown(
            _FLOATING_HTML.format(spans=_build_emoji_spans(n_hearts, n_pluses)),
            unsafe_allow_html=True,
        )
