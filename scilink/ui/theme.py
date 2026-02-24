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
/* Push the first element (logo/title) flush to top */
section[data-testid="stSidebar"] [data-testid="stImage"] {
    margin-top: 0 !important;
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
button[kind="primary"],
.stButton > button {
    background-color: #6200EE;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    text-transform: uppercase;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: background-color 0.2s, box-shadow 0.2s;
}
.stButton > button:hover {
    background-color: #7C4DFF;
    box-shadow: 0 2px 8px rgba(98, 0, 238, 0.35);
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
    margin-bottom: 8px;
}
.agent-spinner-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #4FC3F7;
    animation: scilink-pulse 1.4s ease-in-out infinite;
}
.agent-spinner-dot:nth-child(2) { animation-delay: 0.2s; }
.agent-spinner-dot:nth-child(3) { animation-delay: 0.4s; }
.agent-spinner-label {
    color: #E0E0E0;
    font-size: 0.9em;
    font-weight: 500;
}

/* ── File explorer ──────────────────────────────────── */
.file-group-header {
    color: #82B1FF;
    font-size: 0.8em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 0 4px 0;
    border-bottom: 1px solid #3A4556;
    margin-bottom: 4px;
}
.file-entry {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 2px 0;
    font-size: 0.85em;
}
.file-icon {
    font-size: 0.9em;
    width: 20px;
    text-align: center;
    flex-shrink: 0;
}
.file-meta {
    color: #6B7A8C;
    font-size: 0.8em;
    margin-left: auto;
    flex-shrink: 0;
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

/* ── Hide the Streamlit "Stop" button ───────────────── */
button[data-testid="baseButton-header"],
.stDeployButton,
header [data-testid="stStatusWidget"] {
    display: none !important;
}
</style>
"""


def inject_theme() -> None:
    """Inject the Material Design CSS into the current page."""
    st.markdown(_MATERIAL_CSS, unsafe_allow_html=True)
