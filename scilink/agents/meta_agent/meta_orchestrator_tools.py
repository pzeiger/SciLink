"""Tool registry for MetaOrchestratorAgent.

Mirrors the AnalysisOrchestratorTools shape — a ``ToolsClass(orchestrator)``
that builds ``functions_map`` + ``openai_schemas`` and exposes
``execute_tool``. The meta-agent's tools delegate to child orchestrators via
their ``run_task`` contract and introspect the delegation ledger. See
CLAUDE.md "The meta agent".

The duplication with AnalysisOrchestratorTools is intentional and acceptable
at this development stage — see CLAUDE.md "Why no BaseChatOrchestrator
refactor".
"""

import json
import logging
from typing import Any, Callable, Dict


class MetaOrchestratorTools:
    """Tool definitions, schemas, and execution for MetaOrchestratorAgent."""

    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: the parent MetaOrchestratorAgent.
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)

        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []

        self._register_all_tools()

    def _register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: list = None,
    ):
        """Register a tool in OpenAI function-calling format."""
        self.functions_map[name] = func
        self.openai_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        })

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name; always returns a JSON string."""
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found",
            })
        try:
            return self.functions_map[tool_name](**kwargs)
        except Exception as e:
            logging.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name,
            })

    def _register_all_tools(self):
        """Register the meta-agent's delegation and introspection tools."""

        # -- delegate_to_analysis -------------------------------------------
        def delegate_to_analysis(task: str, context: dict = None) -> str:
            print(f"  🧪 Delegating to analysis specialist: {task[:80]}...")
            return self.orch._delegate("analysis", task, context)

        self._register_tool(
            func=delegate_to_analysis,
            name="delegate_to_analysis",
            description=(
                "Delegate an experimental-data-analysis task to the analysis "
                "specialist (microscopy, spectroscopy, curve fitting, "
                "hyperspectral datacubes, quality assessment, feature "
                "extraction, novelty checks). The specialist runs autonomously "
                "with no interactive user and returns a structured JSON result "
                "(status, summary, key_findings, files_produced, "
                "suggested_followups, warnings, delegation_index). `task` must "
                "be a complete, self-contained instruction including absolute "
                "paths to any data files."
            ),
            parameters={
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained analysis instruction.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Optional upstream findings / file paths (e.g. from an "
                        "earlier delegation) to inform the task."
                    ),
                },
            },
            required=["task"],
        )

        # -- delegate_to_planning -------------------------------------------
        def delegate_to_planning(task: str, context: dict = None) -> str:
            print(f"  📋 Delegating to planning specialist: {task[:80]}...")
            return self.orch._delegate("planning", task, context)

        self._register_tool(
            func=delegate_to_planning,
            name="delegate_to_planning",
            description=(
                "Delegate an experimental-campaign-planning task to the "
                "planning specialist (experiment design, multi-objective "
                "Bayesian optimization, hypothesis generation, deciding what "
                "to measure or run next). The specialist runs autonomously "
                "with no interactive user and returns a structured JSON result "
                "(status, summary, key_findings, files_produced, "
                "suggested_followups, warnings, delegation_index). `task` must "
                "be a complete, self-contained instruction."
            ),
            parameters={
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained planning instruction.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Optional upstream findings / file paths (e.g. analysis "
                        "key_findings) to inform the task."
                    ),
                },
            },
            required=["task"],
        )

        # -- delegate_to_simulation: DEFERRED lazy seam, intentionally NOT built
        #
        # v1 covers analysis + planning only. When simulation delegation is
        # added, register a tool whose body does a GUARDED import INSIDE the
        # function — never at module scope, because scilink.agents.sim_agents
        # hard-imports `ase`, an optional dependency, and the meta-agent module
        # must stay importable without ASE:
        #
        #   def delegate_to_simulation(task, context=None):
        #       try:
        #           from ..sim_agents.simulation_orchestrator import (
        #               SimulationOrchestratorAgent, SimulationMode)
        #       except ImportError as e:
        #           return json.dumps({"status": "error",
        #               "message": "Simulation support requires the optional "
        #               "[sim] extra (pip install scilink[sim]).",
        #               "detail": str(e)})
        #       return self.orch._delegate("simulation", task, context)
        #
        # MetaOrchestratorAgent._delegate / _get_*_child would gain a
        # "simulation" branch using a self.orch.simulation_dir sub-directory.

        # -- summarize_session_state ----------------------------------------
        def summarize_session_state() -> str:
            return self.orch._session_state_summary()

        self._register_tool(
            func=summarize_session_state,
            name="summarize_session_state",
            description=(
                "Report the cross-specialist session state: which specialists "
                "have been instantiated, how many delegations have run, and "
                "per-specialist counters (analyses run, optimization targets, "
                "collected data points). Read-only."
            ),
            parameters={},
            required=[],
        )

        # -- get_delegation_history -----------------------------------------
        def get_delegation_history(limit: int = None) -> str:
            return self.orch._delegation_history(limit)

        self._register_tool(
            func=get_delegation_history,
            name="get_delegation_history",
            description=(
                "Retrieve the delegation ledger — the results of prior "
                "delegations (status, summary, key_findings, files_produced, "
                "suggested_followups). Use it to pull an earlier specialist's "
                "result and thread the relevant pieces as the `context` "
                "argument of the next delegate_to_* call. Optional `limit` "
                "returns only the most recent N entries."
            ),
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Return only the most recent N delegations.",
                },
            },
            required=[],
        )
