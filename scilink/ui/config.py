"""Shared constants and defaults for the SciLink Streamlit UI."""

from pathlib import Path
from typing import Dict, Optional, Tuple

from .. import auth

MODEL_OPTIONS = [
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
    # Amazon Bedrock (Claude Opus 4.8) via the US geo cross-region inference
    # profile (exact ID from the AWS model card; Opus 4.8 has no date stamp and
    # no version suffix). Invoke-able only through an inference profile, hence
    # the ``us.`` prefix. For EU/JP/AU keys swap the prefix:
    # ``eu.``/``jp.``/``au.``. (Base model ID: anthropic.claude-opus-4-8.)
    "bedrock/us.anthropic.claude-opus-4-8",
]

EMBEDDING_MODEL_OPTIONS = [
    "gemini-embedding-001",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

# ── Mode registry ────────────────────────────────────────────────
APP_MODES = [
    {"key": "meta",    "label": "🧪 📋 ⚛️",  "beta": True, "description": "Routes your research goal to the Analyze & Plan specialists"},
    {"key": "analyze", "label": "Analyze", "description": "Multi-modal data analysis"},
    {"key": "plan",    "label": "Plan",    "description": "Experimental design & optimization"},
    {"key": "simulate", "label": "Simulate", "description": "Submit and monitor DFT/MD simulations"},
]

SESSION_DIR_PREFIXES = {
    "meta": "meta_session",
    "analyze": "analysis_session",
    "plan": "planning_session",
    "simulate": "simulation_session",
}

# ── File extensions ──────────────────────────────────────────────
SUPPORTED_DATA_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".npy", ".csv", ".txt", ".tsv", ".xlsx",
    ".h5", ".hdf5", ".nxs",
)

# Vendor formats that SciLink itself cannot read; surfaced in the upload
# whitelist only when an MCP server exposing a compatible reader is
# connected (see ``extra_data_extensions_for``).
VENDOR_DATA_EXTENSIONS = (
    ".dm3", ".dm4", ".emd", ".ndata", ".ndata1", ".ndata2",
    ".ibw", ".ardf",
    ".gwy", ".gsf",
    ".spe", ".spc", ".spx",
    ".asc", ".dat",
)


def extra_data_extensions_for(agent) -> tuple:
    """Return upload extensions enabled by external readers on *agent*.

    Recognises the SciFiReaders MCP server via its ``read_scifireaders_file``
    tool. Returns an empty tuple if no compatible reader server is
    connected, so the uploader does not advertise formats SciLink cannot
    actually handle.
    """
    if agent is None:
        return ()
    conns = getattr(agent, "_mcp_connections", None) or {}
    for conn in conns.values():
        for schema in getattr(conn, "tool_schemas", []) or []:
            if schema.get("function", {}).get("name") == "read_scifireaders_file":
                return VENDOR_DATA_EXTENSIONS
    return ()


SUPPORTED_METADATA_EXTENSIONS = (".json", ".txt")

SUPPORTED_KNOWLEDGE_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".csv", ".xlsx", ".tsv", ".json")
SUPPORTED_CODE_EXTENSIONS = (".py", ".txt", ".md", ".json", ".yaml", ".yml")
SUPPORTED_PLANNING_DATA_EXTENSIONS = (".csv", ".xlsx", ".tsv", ".txt", ".npy", ".json")

# Explore (meta) mode accepts everything the specialist modes accept — one
# combined uploader; the meta-agent routes each file to the right child.
SUPPORTED_META_EXTENSIONS = tuple(sorted(set(
    SUPPORTED_DATA_EXTENSIONS
    + SUPPORTED_METADATA_EXTENSIONS
    + SUPPORTED_KNOWLEDGE_EXTENSIONS
    + SUPPORTED_CODE_EXTENSIONS
)))

AVATAR_USER = str(Path(__file__).resolve().parent / "assets" / "avatar_user.svg")
AVATAR_AGENT = str(Path(__file__).resolve().parent / "assets" / "avatar_agent.svg")


# ── Credential prefill resolution ────────────────────────────────

def resolve_prefill(
    model: str, existing_base_url: str = ""
) -> Dict[str, Tuple[str, Optional[str]]]:
    """Resolve which environment variables prefill the credential form fields.

    Pure function — no Streamlit / session_state — so the resolution rules
    (especially the proxy-vs-vendor safety guard) are unit-testable in
    isolation. The sidebar wraps this to seed widget state and render captions.

    Returns ``{field: (value, env_var_name_or_None)}`` for fields
    ``api_key``, ``base_url``, ``fh``, ``mp``. ``env_var_name`` is the variable
    a value came from (for a "✓ from X" caption), or ``None`` when nothing was
    detected for that field.

    Resolution mirrors the backend's API-key handling:
      - Proxy path: ``SCILINK_API_KEY`` fills the main key ONLY when a base URL
        is available (``SCILINK_BASE_URL`` env, or one the user already entered),
        because the proxy key is rejected by vendor endpoints without a base
        URL. It is never routed to a vendor on its own.
      - Otherwise the main key is the vendor key matching the model's provider
        (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` …).
      - ``base_url`` prefills from ``SCILINK_BASE_URL`` when set.
      - FutureHouse / Materials Project keys come from their own env vars,
        independent of the model.
    """
    proxy_key = auth.get_internal_proxy_key()
    proxy_url = auth.get_internal_proxy_base_url()

    if proxy_key and (proxy_url or existing_base_url):
        api: Tuple[str, Optional[str]] = (proxy_key, auth.INTERNAL_PROXY_KEY)
    else:
        found = auth.find_env_var_for_model(model)
        api = (found[1], found[0]) if found else ("", None)

    base: Tuple[str, Optional[str]] = (
        (proxy_url, auth.INTERNAL_PROXY_BASE_URL) if proxy_url else ("", None)
    )

    fh = auth.find_env_var("futurehouse")
    mp = auth.find_env_var("materials_project")

    return {
        "api_key": api,
        "base_url": base,
        "fh": (fh[1], fh[0]) if fh else ("", None),
        "mp": (mp[1], mp[0]) if mp else ("", None),
    }
