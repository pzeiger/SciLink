"""MCP server that exposes SciLink's analysis and planning tools.

Allows any MCP client (Claude Desktop, Cursor, etc.) to use SciLink as a
tool provider.  Start with::

    scilink serve --model gemini-3.1-pro-preview

Supports three autonomy modes:
- **autonomous** (default) — tools execute without human approval.
- **supervised** — tools run but return key decisions for review.
- **co-pilot** — tools that need approval return a ``needs_input``
  response; the MCP client must call ``scilink_respond`` to continue.

Requires the ``mcp`` optional dependency::

    pip install scilink[mcp]
"""

import asyncio
import contextlib
import io
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _require_mcp():
    if not HAS_MCP:
        raise ImportError(
            "MCP server support requires the 'mcp' package. "
            "Install with: pip install scilink[mcp]"
        )


# ── Schema conversion ───────────────────────────────────────────────────

def _openai_to_mcp_tool(schema: dict, prefix: str = "scilink") -> types.Tool:
    """Convert an OpenAI function-calling schema to an MCP Tool object."""
    fn = schema.get("function", schema)
    name = fn.get("name", "")
    return types.Tool(
        name=f"{prefix}_{name}" if prefix else name,
        description=fn.get("description", ""),
        inputSchema=fn.get("parameters", {"type": "object", "properties": {}}),
    )


# ── Stdout capture ──────────────────────────────────────────────────────

def _execute_tool_captured(tools, tool_name: str, kwargs: dict) -> str:
    """Execute a tool while capturing stdout so it doesn't corrupt stdio MCP transport."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = tools.execute_tool(tool_name, **kwargs)

    log_output = captured.getvalue().strip()
    if log_output:
        logging.info(f"[tool:{tool_name}] {log_output}")

    return result


# ── Pending action support (co-pilot / supervised) ──────────────────────

class PendingAction:
    """Holds a pending tool call that needs user approval before execution."""

    def __init__(self, tool_name: str, kwargs: dict, prompt: str, context: dict = None):
        self.tool_name = tool_name
        self.kwargs = kwargs
        self.prompt = prompt
        self.context = context or {}


# ── Server factory ──────────────────────────────────────────────────────

def create_server(
    *,
    api_key: str = None,
    model_name: str = "gemini-3.1-pro-preview",
    base_url: str = None,
    mode: str = "both",
    session_dir: str = None,
    analysis_mode: str = "autonomous",
    futurehouse_api_key: str = None,
) -> "Server":
    """Create and return a configured MCP Server instance.

    Args:
        api_key: LLM API key.
        model_name: Model identifier.
        base_url: Optional OpenAI-compatible endpoint.
        mode: ``"analyze"``, ``"plan"``, or ``"both"``.
        session_dir: Directory for session outputs.
        analysis_mode: ``"autonomous"``, ``"supervised"``, or ``"co-pilot"``.
        futurehouse_api_key: Optional FutureHouse/Edison API key.

    Returns:
        A configured ``mcp.server.lowlevel.Server``.
    """
    _require_mcp()

    server = Server("scilink")

    # ── State (initialized lazily on first tool list) ────────────────
    state: Dict[str, Any] = {
        "analysis_orch": None,
        "planning_orch": None,
        "pending_action": None,
        "initialized": False,
        "config": {
            "api_key": api_key,
            "model_name": model_name,
            "base_url": base_url,
            "mode": mode,
            "session_dir": session_dir,
            "analysis_mode": analysis_mode,
            "futurehouse_api_key": futurehouse_api_key,
        },
    }

    # Tool name → (orchestrator_key, original_name) mapping
    tool_map: Dict[str, tuple] = {}

    def _ensure_initialized():
        if state["initialized"]:
            return

        config = state["config"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            _init_orchestrators(state, config)

        log = captured.getvalue().strip()
        if log:
            logging.info(f"[init] {log}")

        # Build tool map
        if state["analysis_orch"]:
            for schema in state["analysis_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                if fn_name:
                    mcp_name = f"scilink_{fn_name}"
                    tool_map[mcp_name] = ("analysis_orch", fn_name)

        if state["planning_orch"]:
            for schema in state["planning_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                if fn_name:
                    # Avoid collisions with analysis tools
                    mcp_name = f"scilink_{fn_name}"
                    if mcp_name in tool_map:
                        mcp_name = f"scilink_plan_{fn_name}"
                    tool_map[mcp_name] = ("planning_orch", fn_name)

        state["initialized"] = True

    # ── Eager init (call before starting transport) ────────────────

    def _eager_init():
        _ensure_initialized()

    server.eager_init = _eager_init

    # ── tools/list ───────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        _ensure_initialized()
        tools = []

        if state["analysis_orch"]:
            for schema in state["analysis_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                mcp_name = f"scilink_{fn_name}"
                tools.append(_openai_to_mcp_tool(schema, prefix="scilink"))

        if state["planning_orch"]:
            for schema in state["planning_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                mcp_name = f"scilink_{fn_name}"
                # Use the same collision-avoidance as tool_map
                if mcp_name in tool_map and tool_map[mcp_name][0] != "planning_orch":
                    prefix = "scilink_plan"
                else:
                    prefix = "scilink"
                tools.append(_openai_to_mcp_tool(schema, prefix=prefix))

        # Add the respond tool for co-pilot/supervised modes
        if state["config"]["analysis_mode"] != "autonomous":
            tools.append(types.Tool(
                name="scilink_respond",
                description=(
                    "Send a response to a pending SciLink action that requires "
                    "human approval. Call this after receiving a 'needs_input' "
                    "status from another tool."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": (
                                "User's response: 'yes'/'approve' to proceed, "
                                "'no'/'reject' to cancel, or free-text feedback."
                            ),
                        },
                    },
                    "required": ["response"],
                },
            ))

        return tools

    # ── tools/call ───────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> List[types.TextContent]:
        _ensure_initialized()

        # Handle the respond tool
        if name == "scilink_respond":
            return await _handle_respond(state, arguments)

        # Look up which orchestrator owns this tool
        if name not in tool_map:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Unknown tool: {name}",
                }),
            )]

        orch_key, original_name = tool_map[name]
        orch = state[orch_key]

        result = await asyncio.to_thread(
            _execute_tool_captured, orch.tools, original_name, arguments
        )

        return [types.TextContent(type="text", text=result)]

    # ── resources/list ───────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> List[types.Resource]:
        _ensure_initialized()
        resources = []

        if state["analysis_orch"]:
            resources.append(types.Resource(
                uri="scilink://session/status",
                name="Session Status",
                description="Current analysis session state",
                mimeType="application/json",
            ))
            resources.append(types.Resource(
                uri="scilink://session/metadata",
                name="Current Metadata",
                description="Loaded sample/experiment metadata",
                mimeType="application/json",
            ))
            resources.append(types.Resource(
                uri="scilink://session/agents",
                name="Available Agents",
                description="Registered analysis agents",
                mimeType="application/json",
            ))

        return resources

    # ── resources/read ───────────────────────────────────────────────

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        _ensure_initialized()
        orch = state.get("analysis_orch")

        if uri == "scilink://session/status":
            data = {
                "current_data_path": getattr(orch, "current_data_path", None),
                "current_data_type": getattr(orch, "current_data_type", None),
                "selected_agent_id": getattr(orch, "selected_agent_id", None),
                "analysis_count": len(getattr(orch, "analysis_results", [])),
                "message_count": getattr(orch, "message_count", 0),
            }
            return json.dumps(data, indent=2)

        elif uri == "scilink://session/metadata":
            meta = getattr(orch, "current_metadata", None)
            return json.dumps(meta or {}, indent=2)

        elif uri == "scilink://session/agents":
            registry = getattr(orch, "_agent_registry", {})
            agents = {}
            for aid, entry in registry.items():
                agents[str(aid)] = {
                    "name": entry.get("name", ""),
                    "description": entry.get("description", ""),
                    "short_name": entry.get("short_name", ""),
                }
            return json.dumps(agents, indent=2)

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


# ── Orchestrator initialization ──────────────────────────────────────────

def _init_orchestrators(state: dict, config: dict) -> None:
    """Initialize orchestrator(s) based on config."""
    import os
    from datetime import datetime
    from pathlib import Path

    session_dir = config["session_dir"]
    if not session_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path.home() / "scilink_mcp_sessions" / f"session_{ts}")
    Path(session_dir).mkdir(parents=True, exist_ok=True)

    # Resolve analysis mode enum
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
    mode_map = {
        "co-pilot": AnalysisMode.CO_PILOT,
        "co_pilot": AnalysisMode.CO_PILOT,
        "copilot": AnalysisMode.CO_PILOT,
        "supervised": AnalysisMode.SUPERVISED,
        "autonomous": AnalysisMode.AUTONOMOUS,
    }
    analysis_mode = mode_map.get(
        config["analysis_mode"].lower(), AnalysisMode.AUTONOMOUS
    )

    api_key = config["api_key"]
    model_name = config["model_name"]
    base_url = config["base_url"]
    fh_key = config["futurehouse_api_key"] or os.environ.get("FUTUREHOUSE_API_KEY")

    if config["mode"] in ("analyze", "both"):
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent,
        )
        state["analysis_orch"] = AnalysisOrchestratorAgent(
            base_dir=session_dir,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            analysis_mode=analysis_mode,
            futurehouse_api_key=fh_key,
        )

    if config["mode"] in ("plan", "both"):
        try:
            from scilink.agents.planning_agents.planning_orchestrator import (
                PlanningOrchestratorAgent,
            )
            state["planning_orch"] = PlanningOrchestratorAgent(
                base_dir=session_dir,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
            )
        except Exception as exc:
            logging.warning(f"Planning orchestrator not available: {exc}")


# ── Respond handler (co-pilot / supervised) ──────────────────────────────

async def _handle_respond(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Handle user response to a pending action."""
    pending = state.get("pending_action")
    if pending is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": "No pending action to respond to.",
            }),
        )]

    response = arguments.get("response", "").strip().lower()
    state["pending_action"] = None

    if response in ("yes", "y", "approve", "ok", "proceed"):
        # Execute the pending tool
        orch_key = pending.context.get("orch_key", "analysis_orch")
        orch = state.get(orch_key)
        if orch is None:
            return [types.TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": "Orchestrator not found."}),
            )]

        result = await asyncio.to_thread(
            _execute_tool_captured, orch.tools, pending.tool_name, pending.kwargs
        )
        return [types.TextContent(type="text", text=result)]

    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "cancelled",
                "message": f"Action cancelled by user: {response}",
                "tool": pending.tool_name,
            }),
        )]


# ── Server runner ────────────────────────────────────────────────────────

async def run_stdio(server: Server, real_stdout=None) -> None:
    """Run the MCP server over stdio transport.

    Args:
        server: The configured MCP Server.
        real_stdout: The original ``sys.stdout`` before redirection.
            Needed because the CLI redirects ``sys.stdout`` to stderr
            to protect the JSON-RPC stream from stray ``print()`` calls.
    """
    _require_mcp()
    import anyio
    from io import TextIOWrapper

    stdout_arg = None
    if real_stdout is not None:
        stdout_arg = anyio.wrap_file(
            TextIOWrapper(real_stdout.buffer, encoding="utf-8")
        )

    async with stdio_server(stdout=stdout_arg) as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )
