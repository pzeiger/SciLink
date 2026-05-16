"""Meta-Agent Orchestrator — the orchestrator-of-orchestrators.

Presents a single conversational surface and routes each piece of work to the
right specialist mode orchestrator (analysis or planning), consuming them
through their non-interactive ``run_task`` contract. See CLAUDE.md
"The meta agent".

It is **not a fourth mode** — it sits on top of the three modes with a
different role (router + context bridge). v1 covers analysis + planning;
simulation delegation is a deferred lazy seam (see meta_orchestrator_tools.py).

The chat-loop / checkpoint / history shape is copied from
AnalysisOrchestratorAgent. The duplication is intentional and acceptable at
this development stage — see CLAUDE.md "Why no BaseChatOrchestrator refactor".
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .meta_orchestrator_tools import MetaOrchestratorTools


class MetaMode(Enum):
    """Autonomy level for the meta orchestrator — **two levels, not three**.

    The individual modes use a three-level paradigm (co-pilot / autopilot /
    autonomous). The meta agent cannot: a delegation runs the child through
    its one-shot ``run_task``, which gives the child a single turn. Co-pilot's
    model — pause after *every* step and wait for the user's next message —
    needs many turns, so it cannot complete a delegated task. AUTOPILOT and
    AUTONOMOUS each finish a task within one turn.

    Each delegated child runs under this same mode (mapped by enum name):
    AUTOPILOT keeps the child's human-feedback prompts — it pauses at
    decision points for the user to approve / edit plans and outputs;
    AUTONOMOUS runs end to end without pausing.
    """
    AUTOPILOT = "autopilot"      # Children pause for feedback at decision points
    AUTONOMOUS = "autonomous"    # Children run end to end without pausing

    @classmethod
    def _missing_(cls, value):
        # Back-compat: the AUTOPILOT level was named "supervised" before.
        if isinstance(value, str) and value.strip().lower() == "supervised":
            return cls.AUTOPILOT
        return None


_AUTOPILOT_DIRECTIVE = """**OPERATING MODE: AUTOPILOT (default)**
- Delegate on your own judgement; briefly announce each delegation before
  making it, and ask a clarifying question when the goal or the right
  specialist is genuinely ambiguous.
- Each delegated specialist runs in autopilot mode — it pauses at its own
  decision points for the user to approve or edit plans and outputs. Let
  that happen.
- Chain delegations when the path is clear; pause to report if a child
  returns an error or an ambiguous result.

"""

_AUTONOMOUS_DIRECTIVE = """**OPERATING MODE: AUTONOMOUS**
- Fulfil the user's goal end to end. Chain `delegate_to_*` calls as needed.
- Thread results forward: pass a child's relevant `key_findings` /
  `files_produced` into the next delegation's `context`.
- Only stop for an unrecoverable error. Summarize the whole effort at the end.

"""

_SYSTEM_PROMPT_BODY = """You are the **Meta-Agent** — a research orchestrator that coordinates
SciLink's specialist mode orchestrators. You do not analyze data or design
experiments yourself; you route each piece of work to the right specialist
and weave their results into one coherent response for the user.

**WHEN TO USE EACH SPECIALIST:**
- `delegate_to_analysis` — experimental data already exists and needs to be
  interpreted: microscopy / spectroscopy images, 1D curves, hyperspectral
  datacubes, "analyze this file", data-quality assessment, feature
  extraction, novelty checks against the literature. It can also read a few
  user-provided papers or reports directly as reference context for the
  analysis.
- `delegate_to_planning` — deciding what to do next: experimental campaign
  design, multi-objective Bayesian optimization, "what should I measure or
  run next", hypothesis generation, trade-off analysis.

Computational simulation (DFT / MD) is NOT available in this build — do not
attempt to delegate simulation work.

**INSPECTING UPLOADED FILES:**
- When the user refers to uploaded files — or points you at a folder — call
  `inspect_uploads` FIRST, before delegating. It returns a content probe of
  each file (array shape/dtype, table columns, document text, JSON keys) so
  you route from evidence rather than guessing from filenames.
- Route data from the probe, not the name: images and measurement arrays,
  and data tables with experimental columns → `delegate_to_analysis`; code
  → `delegate_to_planning`.
- Papers / reports / notes route by INTENT, not file type. A few documents
  that are reference context for interpreting the data — a methods paper, a
  prior analysis report — go WITH the data to `delegate_to_analysis`, which
  can read documents directly. Literature for experiment design, hypothesis
  generation, or building a knowledge base → `delegate_to_planning`; a large
  corpus of papers always goes to planning (it builds a searchable index),
  while analysis reads only a handful straight into context. Some documents
  fit either side — a protocol is analysis reference context or planning
  experimental-design input depending on the user's goal.
- Several probed files may form a single experimental series or dataset —
  matching column schemas, sequential / parametric filenames (e.g.
  `spec_5K`, `spec_10K`, ...), or a shared sidecar-JSON pattern. Recognize
  this from the probes and the user's goal, and delegate the whole set as
  ONE batched task (pass the file list or their shared directory) so the
  specialist's batch tools engage — never one delegation per file.
- `inspect_uploads` is for routing only — do not use its output to interpret
  or analyze the data yourself; hand that to the specialist.

**EXPERIMENTAL METADATA:**
- Experimental data needs metadata — measurement technique, instrument,
  sample, and conditions — for a meaningful analysis, and planning data
  needs the experimental conditions behind each measurement.
- Before delegating an analysis or planning task, check whether the user
  supplied it: in their message, an uploaded metadata JSON, or per-file
  sidecar files. `inspect_uploads` shows each JSON's keys and filename — a
  JSON whose name matches a data file's stem (`spec_5K.json` ↔
  `spec_5K.csv`) is that file's sidecar metadata.
- If it is missing, ask the user for it conversationally FIRST, then put it
  into the delegation's `task` / `context`. Do not delegate a data task with
  no metadata and let the specialist stop midway to ask for it.

**THE DELEGATION CONTRACT:**
- `delegate_to_analysis(task, context)` and `delegate_to_planning(task,
  context)` run the specialist and return a structured JSON result: status,
  summary, key_findings, files_produced, suggested_followups, warnings,
  delegation_index.
- The specialist runs in the SAME autonomy mode as you. In autopilot mode it
  pauses at its decision points for the user to approve or edit plans and
  outputs (via its own human-feedback prompts) — that is expected and good;
  let it happen. In autonomous mode it runs the task end to end.
- `task` must still be a complete, self-contained instruction (including
  absolute paths to any data files) — the specialist cannot see this
  conversation.
- Pass upstream findings via the `context` dict, not by re-typing them into
  `task`.
- Give each call a short `label` — the data type for an analysis (e.g.
  "1-D Raman spectra", "STEM image") or the focus for a planning task (e.g.
  "follow-up BO campaign"). It is how the delegation is shown in the UI.

**BRIDGING CONTEXT BETWEEN SPECIALISTS:**
- Every delegation's result is kept in a delegation ledger. Call
  `get_delegation_history` to retrieve an earlier result, then pass the
  relevant `key_findings` / `files_produced` as the `context` argument of the
  next `delegate_to_*` call. That is how analysis findings inform planning.
- When the `context` you pass draws on earlier delegations, also list those
  delegations' `delegation_index` numbers in the `context_from` argument —
  this records the provenance of the threaded findings.
- `summarize_session_state` reports what each specialist has done so far.

**RESPONSE STYLE:**
- Do not dump raw tool JSON back to the user — synthesize it into plain
  language.
- Make clear which specialist produced which result.
- Surface the specialists' `suggested_followups` when proposing next steps.
"""


def get_system_prompt(meta_mode: MetaMode) -> str:
    """Return the system prompt for the given meta autonomy mode."""
    directives = {
        MetaMode.AUTOPILOT: _AUTOPILOT_DIRECTIVE,
        MetaMode.AUTONOMOUS: _AUTONOMOUS_DIRECTIVE,
    }
    return directives[meta_mode] + _SYSTEM_PROMPT_BODY


class MetaOrchestratorAgent:
    """Orchestrator-of-orchestrators.

    Presents one chat surface; delegates analysis and planning sub-tasks to
    persistent child orchestrators (one per mode, reused across delegations)
    nested under the meta-session directory. Children are consumed only
    through their public ``run_task`` contract — duck-typed, no base class.

    Args:
        base_dir: Base directory for the meta session.
        api_key: API key for the LLM provider (forwarded to children).
        model_name: Model name (forwarded to children).
        base_url: Base URL for an internal proxy endpoint (forwarded).
        embedding_model: Embedding model name (forwarded).
        embedding_api_key: API key for the embedding provider (forwarded).
        futurehouse_api_key: Optional FutureHouse API key (forwarded).
        restore_checkpoint: Whether to restore from a previous checkpoint.
        meta_mode: Autonomy level (AUTOPILOT or AUTONOMOUS). The meta has
            only these two — see MetaMode.
    """

    # Configuration constants (match the child orchestrators).
    MAX_TOOL_ITERATIONS = 20
    MAX_HISTORY_MESSAGES = 100
    CHECKPOINT_INTERVAL = 10

    def __init__(
        self,
        base_dir: str = "./meta_session",
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        embedding_api_key: Optional[str] = None,
        futurehouse_api_key: Optional[str] = None,
        restore_checkpoint: bool = False,
        meta_mode: MetaMode = MetaMode.AUTOPILOT,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
            if embedding_api_key is not None:
                logging.warning(
                    "⚠️ embedding_api_key is ignored for internal proxy. "
                    "Using api_key for all requests."
                )
            embedding_api_key = api_key
        else:
            if embedding_api_key is None:
                embedding_api_key = api_key

        # Store configuration — these are forwarded verbatim to child
        # orchestrators when they are lazily created.
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.futurehouse_api_key = (
            futurehouse_api_key or os.environ.get("FUTUREHOUSE_API_KEY")
        )

        self.meta_mode = meta_mode
        self._enable_human_feedback = self._should_enable_human_feedback()
        logging.info(f"🎛️  Meta Mode: {meta_mode.value.upper()}")

        # Setup directories. Children live in fixed sub-directories so a
        # persistent child can be re-created (with restore_checkpoint=True)
        # after a meta restore simply by probing for its checkpoint.json.
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.base_dir / "chat_history.json"
        self.checkpoint_path = self.base_dir / "checkpoint.json"
        self.analysis_dir = self.base_dir / "analysis"
        self.planning_dir = self.base_dir / "planning"

        # Session state — kept shallow; children own their deep state.
        self._children: Dict[str, Any] = {}          # "analysis"/"planning" -> agent
        self._delegation_ledger: List[Dict[str, Any]] = []

        self.message_count = 0
        self.last_checkpoint_message_count = 0

        if restore_checkpoint and self.checkpoint_path.exists():
            self._restore_checkpoint()

        # Tools registry.
        self.tools = MetaOrchestratorTools(self)

        system_prompt = get_system_prompt(self.meta_mode)

        # Initialize LLM (dual path, copied from AnalysisOrchestratorAgent).
        if base_url:
            logging.info(f"🏛️ Meta orchestrator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name, api_key=api_key, base_url=base_url
            )
            self.use_openai = True
            self.tools_for_model = self.tools.openai_schemas
        else:
            logging.info(f"🌐 Meta orchestrator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key,
                system_instruction=system_prompt,
                tools=self._convert_tools_to_litellm_format(),
            )
            self.use_openai = False
            self.tools_for_model = self._convert_tools_to_litellm_format()

        self._system_prompt = system_prompt

        # Initialize message history.
        history = self._load_history()
        self.messages = [{"role": "system", "content": system_prompt}]
        if history:
            self.messages.extend(
                self._trim_history(history, max_messages=self.MAX_HISTORY_MESSAGES)
            )

        logging.info(f"✅ MetaOrchestratorAgent initialized. Session: {self.base_dir}")

    # =========================================================================
    # Mode / configuration
    # =========================================================================

    def _convert_tools_to_litellm_format(self) -> List[Dict]:
        """Convert OpenAI tool schemas to LiteLLM format (passthrough)."""
        return self.tools.openai_schemas

    def _should_enable_human_feedback(self) -> bool:
        """Human feedback is on unless the meta is fully autonomous."""
        return self.meta_mode != MetaMode.AUTONOMOUS

    def set_meta_mode(self, mode: MetaMode) -> None:
        """Change the meta autonomy mode at runtime."""
        old_mode = self.meta_mode
        self.meta_mode = mode
        self._enable_human_feedback = self._should_enable_human_feedback()

        new_system_prompt = get_system_prompt(mode)
        self._system_prompt = new_system_prompt
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = new_system_prompt

        logging.info(f"🔄 Meta mode changed: {old_mode.value} → {mode.value}")

    def get_human_feedback_setting(self) -> bool:
        """Returns the current human feedback setting."""
        return self._enable_human_feedback

    # =========================================================================
    # Child orchestrators (lazy, persistent, one per mode)
    # =========================================================================

    def _get_analysis_child(self):
        """Lazily create (or restore) the persistent analysis child.

        Imported lazily inside the method so importing the meta-agent module
        does not pull the analysis stack until a delegation actually happens.
        """
        if "analysis" not in self._children:
            from ..exp_agents.analysis_orchestrator import (
                AnalysisOrchestratorAgent, AnalysisMode,
            )
            restore = (self.analysis_dir / "checkpoint.json").exists()
            self.logger.info(
                f"🧩 {'Restoring' if restore else 'Creating'} analysis child "
                f"at {self.analysis_dir}"
            )
            # Resting mode CO_PILOT; each delegation's run_task sets the
            # autonomy mode for that call to match the meta's.
            self._children["analysis"] = AnalysisOrchestratorAgent(
                base_dir=str(self.analysis_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                embedding_model=self.embedding_model,
                embedding_api_key=self.embedding_api_key,
                futurehouse_api_key=self.futurehouse_api_key,
                restore_checkpoint=restore,
                analysis_mode=AnalysisMode.CO_PILOT,
            )
        return self._children["analysis"]

    def _get_planning_child(self):
        """Lazily create (or restore) the persistent planning child.

        Built in CO_PILOT with data_dir=None: PlanningOrchestratorAgent only
        requires data_dir under AUTOPILOT/AUTONOMOUS at *construction* time.
        Each delegation's run_task sets the autonomy level via
        set_autonomy_level, which does not re-validate data_dir — so a
        autopilot/autonomous delegation is still safe. Per-task data files
        arrive as absolute paths in the delegation `task`.
        """
        if "planning" not in self._children:
            from ..planning_agents.planning_orchestrator import (
                PlanningOrchestratorAgent, AutonomyLevel,
            )
            restore = (self.planning_dir / "checkpoint.json").exists()
            self.logger.info(
                f"🧩 {'Restoring' if restore else 'Creating'} planning child "
                f"at {self.planning_dir}"
            )
            self._children["planning"] = PlanningOrchestratorAgent(
                objective="Delegated by meta-agent",
                base_dir=str(self.planning_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                embedding_model=self.embedding_model,
                embedding_api_key=self.embedding_api_key,
                futurehouse_api_key=self.futurehouse_api_key,
                restore_checkpoint=restore,
                autonomy_level=AutonomyLevel.CO_PILOT,
                data_dir=None,
            )
        return self._children["planning"]

    # =========================================================================
    # Delegation + ledger (used by MetaOrchestratorTools)
    # =========================================================================

    def _delegate(self, mode: str, task: str, context: Optional[dict] = None,
                  context_from: Optional[list] = None,
                  label: Optional[str] = None) -> str:
        """Run a task on a child orchestrator, record it, return a JSON summary.

        The child runs under the meta's own autonomy mode (mapped by enum
        name), so an autopilot delegation keeps the specialist's
        human-feedback prompts — they reach the user driving the meta exactly
        as in a direct single-mode session. Called by the delegate_to_* tools.
        Never raises — child/setup failures are captured into an error result.

        A provisional 'running' ledger entry is opened before the child runs,
        so the UI delegation tree shows the delegation live; it is finalized
        with the result on completion.
        """
        if mode == "analysis":
            from ..exp_agents.analysis_orchestrator import AnalysisMode
            get_child, autonomy_enum = self._get_analysis_child, AnalysisMode
        elif mode == "planning":
            from ..planning_agents.planning_orchestrator import AutonomyLevel
            get_child, autonomy_enum = self._get_planning_child, AutonomyLevel
        else:
            return json.dumps({
                "status": "error",
                "message": f"Unknown delegation target: {mode}",
            })

        entry = self._open_delegation(mode, task, context, context_from, label)
        try:
            child = get_child()
            result = child.run_task(
                task, context=context,
                autonomy=autonomy_enum[self.meta_mode.name],
            )
        except Exception as e:
            self.logger.exception(f"Delegation to {mode} failed: {e}")
            result = {
                "status": "error", "error": str(e), "summary": "",
                "key_findings": [], "files_produced": [],
                "suggested_followups": [], "warnings": [],
            }
        self._close_delegation(entry, result)
        return self._summarize_delegation_result(mode, result, entry["index"])

    def _open_delegation(self, mode, task, context, context_from,
                         label=None) -> Dict[str, Any]:
        """Append a provisional 'running' ledger entry before the child runs.

        Finalized later by ``_close_delegation``. Having the entry present up
        front lets the UI delegation tree show the delegation while it runs.
        """
        index = len(self._delegation_ledger) + 1
        # Normalize context_from to valid prior delegation indices (the LLM may
        # pass ints, strings, or "#n" forms; drop anything out of range).
        raw = context_from
        if isinstance(raw, (int, str)):
            raw = [raw]
        elif not isinstance(raw, (list, tuple)):
            raw = []
        sources = sorted({
            int(s) for s in (str(v).strip().lstrip("#") for v in raw)
            if s.isdigit() and 0 < int(s) < index
        })
        entry = {
            "index": index,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "task": task,
            "label": (label or "").strip(),
            "context_keys": sorted(context.keys()) if isinstance(context, dict) else [],
            "context_from": sources,
            "status": "running",
            "summary": "",
            "key_findings": [],
            "files_produced": [],
            "suggested_followups": [],
            "warnings": [],
            "error": None,
        }
        self._delegation_ledger.append(entry)
        return entry

    def _close_delegation(self, entry: Dict[str, Any], result: dict) -> None:
        """Finalize a provisional ledger entry with the child's result."""
        entry.update({
            "status": result.get("status"),
            "summary": result.get("summary", ""),
            "key_findings": result.get("key_findings", []),
            "files_produced": result.get("files_produced", []),
            "suggested_followups": result.get("suggested_followups", []),
            "warnings": result.get("warnings", []),
            "error": result.get("error"),
            "completed_at": datetime.now().isoformat(),
        })

    def _summarize_delegation_result(self, mode, result, index) -> str:
        """Build the JSON string a delegate_to_* tool returns to the meta LLM."""
        summary = {
            "delegation_index": index,
            "mode": mode,
            "status": result.get("status"),
            "summary": result.get("summary", ""),
            "key_findings": result.get("key_findings", []),
            "files_produced": result.get("files_produced", []),
            "suggested_followups": result.get("suggested_followups", []),
            "warnings": result.get("warnings", []),
        }
        if result.get("error"):
            summary["error"] = result["error"]
        # Domain-specific field, passed through lightly.
        if "analyses" in result:
            summary["analyses"] = result["analyses"]
        if "campaign_state" in result:
            summary["campaign_state"] = result["campaign_state"]
        return json.dumps(summary, indent=2, default=str)

    def _session_state_summary(self) -> str:
        """JSON snapshot of cross-specialist session state."""
        children = {}
        for name in ("analysis", "planning"):
            child = self._children.get(name)
            if child is None:
                children[name] = {"instantiated": False}
                continue
            info = {
                "instantiated": True,
                "base_dir": str(getattr(child, "base_dir", "")),
                "message_count": getattr(child, "message_count", 0),
            }
            if name == "analysis":
                info["analyses_run"] = len(getattr(child, "analysis_results", []) or [])
            else:
                info["optimization_targets"] = (
                    getattr(child, "expected_target_columns", []) or []
                )
                bo_path = getattr(child, "bo_data_path", None)
                try:
                    import pandas as pd
                    info["bo_data_points"] = (
                        len(pd.read_csv(bo_path))
                        if bo_path and Path(bo_path).exists() else 0
                    )
                except Exception:
                    info["bo_data_points"] = 0
            children[name] = info

        last = self._delegation_ledger[-1] if self._delegation_ledger else None
        return json.dumps({
            "meta_mode": self.meta_mode.value,
            "session_dir": str(self.base_dir),
            "delegations_total": len(self._delegation_ledger),
            "last_delegation": (
                f"{last['mode']} / {last['status']}" if last else None
            ),
            "children": children,
        }, indent=2, default=str)

    def _delegation_history(self, limit: Optional[int] = None) -> str:
        """JSON dump of the delegation ledger (optionally the most recent N)."""
        ledger = self._delegation_ledger
        if limit is not None and limit > 0:
            ledger = ledger[-limit:]
        return json.dumps(ledger, indent=2, default=str)

    # =========================================================================
    # Checkpoint / history
    # =========================================================================

    def _restore_checkpoint(self):
        """Restore shallow meta state from checkpoint. Children are re-created
        lazily on the next delegation (their own checkpoint.json survives in
        the fixed sub-directory)."""
        print(f"  📂 Restoring checkpoint from: {self.checkpoint_path}")
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)

            if "meta_mode" in state:
                try:
                    self.meta_mode = MetaMode(state["meta_mode"])
                    self._enable_human_feedback = self._should_enable_human_feedback()
                except ValueError:
                    pass

            self.message_count = state.get("message_count", 0)
            self._delegation_ledger = state.get("delegation_ledger", [])

            print(f"    ✅ Restored state:")
            print(f"       - Meta mode: {self.meta_mode.value}")
            print(f"       - Delegations: {len(self._delegation_ledger)}")

        except Exception as e:
            logging.warning(f"Failed to restore checkpoint: {e}")

    def _auto_checkpoint(self):
        """Internal auto-checkpoint without LLM interaction."""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "meta_mode": self.meta_mode.value,
                "message_count": self.message_count,
                "children_instantiated": sorted(self._children.keys()),
                "delegation_ledger": self._delegation_ledger,
            }
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            print(f"    ✅ Auto-checkpoint saved")
        except Exception as e:
            logging.warning(f"Auto-checkpoint failed: {e}")

    def _trim_history(self, history: List[Dict], max_messages: int = None) -> List[Dict]:
        """Keep only recent messages to avoid context window overflow."""
        if max_messages is None:
            max_messages = self.MAX_HISTORY_MESSAGES

        if len(history) <= max_messages:
            return history

        print(f"  ⚠️  Trimming history: {len(history)} → {max_messages} messages")

        context_window = 10
        recent_window = max_messages - context_window

        trimmed = history[:context_window] + history[-recent_window:]

        summary_marker = {
            "role": "system",
            "content": f"[{len(history) - max_messages} messages omitted for context management]",
        }
        trimmed.insert(context_window, summary_marker)

        return trimmed

    def _load_history(self) -> List[Dict]:
        """Load conversation history from disk."""
        if not self.history_path.exists():
            return []
        print("  🧠 Memory: Loading previous conversation...")
        try:
            with open(self.history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")
            return []

    def _save_history(self):
        """Save conversation history to disk."""
        try:
            history_data = [m for m in self.messages if m["role"] != "system"]
            with open(self.history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save history: {e}")

    @classmethod
    def restore_from_checkpoint(cls, base_dir: str, **kwargs):
        """Factory method to create a MetaOrchestratorAgent from a checkpoint."""
        return cls(base_dir=base_dir, restore_checkpoint=True, **kwargs)

    # =========================================================================
    # Chat
    # =========================================================================

    def chat(self, user_input: str) -> str:
        """Main chat interface with robust function calling support."""
        self.message_count += 1

        # Auto-checkpoint every N messages.
        if self.message_count - self.last_checkpoint_message_count >= self.CHECKPOINT_INTERVAL:
            print(f"  💾 Auto-checkpoint triggered (every {self.CHECKPOINT_INTERVAL} messages)...")
            self._auto_checkpoint()
            self.last_checkpoint_message_count = self.message_count

        try:
            if self.use_openai:
                response_text = self._handle_openai_chat(user_input)
            else:
                response_text = self._handle_litellm_chat(user_input)

            # The response is returned to the caller (CLI / UI), which is
            # responsible for displaying it — chat() does not print it.
            self._save_history()

            if self.message_count > 80:
                response_text += (
                    "\n\n⚠️ Note: Conversation is getting long. "
                    "Consider starting a fresh meta session."
                )

            return response_text

        except Exception as e:
            logging.error(f"Chat Error: {e}", exc_info=True)
            print("  💾 Error detected - saving emergency checkpoint...")
            self._auto_checkpoint()
            return f"❌ Error: {e}\n\n(Emergency checkpoint saved to {self.checkpoint_path})"

    def _handle_openai_chat(self, user_input: str) -> str:
        """Handle chat with OpenAI-compatible models — manual tool-calling loop."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url,
            timeout=120.0,
        )

        self.messages.append({"role": "user", "content": user_input})

        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=self.MAX_HISTORY_MESSAGES)
            self.messages = [system_msg] + recent_msgs

        iteration = 0
        while iteration < self.MAX_TOOL_ITERATIONS:
            iteration += 1
            print(f"  ⏳ Waiting for meta-orchestrator response ...")

            try:
                response = client.chat.completions.create(
                    model=self.model.model,
                    messages=self.messages,
                    tools=self.tools_for_model,
                    tool_choice="auto",
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"  ⚠️ API timeout on iteration {iteration}. Retrying...")
                    if iteration < 3:
                        continue
                raise

            message = response.choices[0].message

            if not message.tool_calls:
                text = message.content
                if not text and iteration > 0:
                    self.messages.append({
                        "role": "user",
                        "content": "Please briefly summarize what you just did and suggest next steps.",
                    })
                    followup = client.chat.completions.create(
                        model=self.model.model,
                        messages=self.messages,
                        tools=self.tools_for_model,
                        tool_choice="none",
                    )
                    text = followup.choices[0].message.content or ""
                self.messages.append({"role": "assistant", "content": text})
                return text

            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  🔧 Calling tool: {func_name}")
                result = self.tools.execute_tool(func_name, **args)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "⚠️ Maximum tool iterations reached. Please simplify your request."

    def _handle_litellm_chat(self, user_input: str) -> str:
        """Handle chat with LiteLLM models — manual tool-calling loop."""
        import litellm

        self.messages.append({"role": "user", "content": user_input})

        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=self.MAX_HISTORY_MESSAGES)
            self.messages = [system_msg] + recent_msgs

        iteration = 0
        while iteration < self.MAX_TOOL_ITERATIONS:
            iteration += 1
            print(f"  ⏳ Waiting for meta-orchestrator response ...")

            try:
                response = litellm.completion(
                    model=self.model.model,
                    messages=self.messages,
                    tools=self.tools_for_model,
                    tool_choice="auto",
                    api_key=self.model.api_key,
                    api_base=self.model.base_url,
                    timeout=120,
                    request_timeout=120,
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"  ⚠️ API timeout on iteration {iteration}. Retrying...")
                    if iteration < 3:
                        continue
                raise

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            content = getattr(message, "content", None)

            if not tool_calls:
                if not content and iteration > 0:
                    self.messages.append({
                        "role": "user",
                        "content": "Please briefly summarize what you just did and suggest next steps.",
                    })
                    followup = litellm.completion(
                        model=self.model.model,
                        messages=self.messages,
                        tools=self.tools_for_model,
                        tool_choice="none",
                    )
                    content = getattr(followup.choices[0].message, "content", None) or ""
                self.messages.append({"role": "assistant", "content": content or ""})
                return content or ""

            self.messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in tool_calls
                ],
            })

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  🔧 Calling tool: {func_name}")
                result = self.tools.execute_tool(func_name, **args)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "⚠️ Maximum tool iterations reached. Please simplify your request."
