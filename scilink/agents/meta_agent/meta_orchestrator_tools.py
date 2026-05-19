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
from pathlib import Path
from typing import Any, Callable, Dict


# ── Upload inspection ────────────────────────────────────────────────
# Lightweight, content-based file probes so the meta-agent routes each
# uploaded file from evidence (array shape, table columns, document text)
# rather than guessing from the filename. Every heavy / optional import is
# done lazily inside the relevant branch, so this module stays importable
# without those packages and a missing reader degrades one file's probe
# instead of breaking the whole tool.

_PROBE_MAX_FILES = 60
_PROBE_TEXT_HEAD = 400


def _probe_file(path: Path) -> Dict[str, Any]:
    """Content-probe a single file for routing. Never raises — any failure
    is reported in the returned dict's ``note`` field."""
    ext = path.suffix.lower()
    info: Dict[str, Any] = {"file": str(path), "ext": ext}
    try:
        info["size_kb"] = round(path.stat().st_size / 1024, 1)
    except OSError:
        pass
    try:
        if ext == ".npy":
            import numpy as np
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            info.update(kind="array", shape=list(arr.shape), dtype=str(arr.dtype))
        elif ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            from PIL import Image
            with Image.open(path) as im:
                info.update(kind="image", height=im.height, width=im.width,
                            mode=im.mode, n_frames=getattr(im, "n_frames", 1))
        elif ext in (".csv", ".tsv"):
            import pandas as pd
            df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", nrows=200)
            info.update(kind="table", n_columns=int(df.shape[1]),
                        sampled_rows=int(df.shape[0]),
                        columns=[str(c) for c in df.columns[:40]],
                        dtypes={str(c): str(t)
                                for c, t in list(df.dtypes.items())[:40]})
        elif ext == ".xlsx":
            import pandas as pd
            df = pd.read_excel(path, nrows=200)
            info.update(kind="table", n_columns=int(df.shape[1]),
                        sampled_rows=int(df.shape[0]),
                        columns=[str(c) for c in df.columns[:40]])
        elif ext == ".json":
            with open(path, "r", errors="replace") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                info.update(kind="json", json_type="object",
                            top_level_keys=[str(k) for k in list(obj)[:40]])
            elif isinstance(obj, list):
                info.update(kind="json", json_type="array", length=len(obj))
            else:
                info.update(kind="json", json_type=type(obj).__name__)
        elif ext == ".pdf":
            info.update(kind="document", doc_type="pdf")
            try:
                from scilink.parsers import extract_text
                # max_pages=1 keeps the probe cheap — only a text head is needed.
                doc_info = extract_text(path, max_pages=1)
                info["n_pages"] = doc_info.get("n_pages")
                info["text_head"] = doc_info["text"][:_PROBE_TEXT_HEAD].strip()
            except Exception as e:  # noqa: BLE001 - optional reader / bad PDF
                info["note"] = f"page/text probe unavailable: {e}"
        elif ext == ".docx":
            info.update(kind="document", doc_type="docx")
            try:
                from scilink.parsers import extract_text
                doc_info = extract_text(path)
                info["n_paragraphs"] = doc_info.get("n_paragraphs")
                info["text_head"] = doc_info["text"][:_PROBE_TEXT_HEAD].strip()
            except Exception as e:  # noqa: BLE001 - optional reader / bad docx
                info["note"] = f"text probe unavailable: {e}"
        elif ext in (".md", ".txt"):
            text = path.read_text(errors="replace")
            info.update(kind="text", n_chars=len(text),
                        text_head=text[:_PROBE_TEXT_HEAD].strip())
        elif ext == ".py":
            text = path.read_text(errors="replace")
            info.update(kind="code", n_lines=text.count("\n") + 1,
                        text_head=text[:_PROBE_TEXT_HEAD].strip())
        elif ext in (".yaml", ".yml"):
            text = path.read_text(errors="replace")
            info.update(kind="config", text_head=text[:_PROBE_TEXT_HEAD].strip())
        else:
            info["kind"] = "unknown"
    except Exception as e:  # noqa: BLE001 - probe must never break the tool
        info.setdefault("kind", "unreadable")
        info["note"] = f"probe failed: {e}"
    return info


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
        def delegate_to_analysis(task: str, context: dict = None,
                                 context_from: list = None,
                                 label: str = None) -> str:
            print(f"  🧪 Delegating to analysis specialist: {task[:80]}...")
            return self.orch._delegate("analysis", task, context, context_from, label)

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
                "context_from": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "delegation_index numbers of earlier delegations whose "
                        "findings you threaded into `context` — records provenance."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "REQUIRED short label for the UI delegation tree — a "
                        "2-5 word noun phrase naming the data type being "
                        "analyzed (e.g. '1-D Raman spectra', 'STEM image', "
                        "'hyperspectral datacube'). NOT a sentence or a "
                        "restatement of the task."
                    ),
                },
            },
            required=["task", "label"],
        )

        # -- delegate_to_planning -------------------------------------------
        def delegate_to_planning(task: str, context: dict = None,
                                 context_from: list = None,
                                 label: str = None) -> str:
            print(f"  📋 Delegating to planning specialist: {task[:80]}...")
            return self.orch._delegate("planning", task, context, context_from, label)

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
                "context_from": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "delegation_index numbers of earlier delegations whose "
                        "findings you threaded into `context` — records provenance."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "REQUIRED short label for the UI delegation tree — a "
                        "2-5 word noun phrase naming the focus of the planning "
                        "task (e.g. 'follow-up BO campaign', 'experiment "
                        "design'). NOT a sentence."
                    ),
                },
            },
            required=["task", "label"],
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

        # -- inspect_uploads ------------------------------------------------
        def inspect_uploads(path: str = None) -> str:
            base = Path(path) if path else (self.orch.base_dir / "uploads")
            print(f"  🔍 Inspecting uploads at {base} ...")
            if not base.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Path not found: {base}",
                })
            if base.is_file():
                files = [base]
                directory = str(base.parent)
            else:
                files = sorted(
                    f for f in base.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                )
                directory = str(base)
            probes = [_probe_file(f) for f in files[:_PROBE_MAX_FILES]]
            return json.dumps({
                "status": "success",
                "directory": directory,
                "n_files": len(files),
                "truncated": len(files) > _PROBE_MAX_FILES,
                "files": probes,
            }, default=str)

        self._register_tool(
            func=inspect_uploads,
            name="inspect_uploads",
            description=(
                "Inspect uploaded files to decide how to route them. Returns a "
                "lightweight CONTENT probe of each file — array shape/dtype, "
                "table column names, document text snippets, JSON keys — so you "
                "classify from evidence, not from filenames. Call this FIRST "
                "whenever the user refers to uploaded files or points you at a "
                "folder. With no argument it inspects the meta session's "
                "uploads/ directory; pass `path` for a specific file or folder. "
                "Read-only — use the result only to choose a specialist, never "
                "to interpret the data yourself."
            ),
            parameters={
                "path": {
                    "type": "string",
                    "description": (
                        "Optional file or directory to inspect. Defaults to "
                        "the meta session's uploads/ directory."
                    ),
                },
            },
            required=[],
        )

        # ----- view_image -----------------------------------------------------
        # Generic "view & describe an arbitrary image" — the meta itself has no
        # multimodal input path, so this is how a notebook photo / diagram /
        # screenshot / figure becomes content the meta can reason about. NOT
        # for scientific images (route those to analysis for quantification).

        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff",
                       ".bmp", ".gif", ".webp"}
        _VIEW_IMAGE_DEFAULT_PROMPT = (
            "Describe the contents of this image in detail. If it contains "
            "any readable text — printed or handwritten — transcribe it "
            "faithfully, rendering tables as GitHub-flavored Markdown. "
            "Return your description and any transcription as plain text, "
            "with no extra commentary."
        )

        def view_image(paths, question: str = None) -> str:
            """Open one or more images and have the vision model describe
            (and transcribe text/tables in) them."""
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                return json.dumps({"status": "error",
                                   "message": "No image path provided."})
            print(f"  🖼️  Tool: Viewing {len(paths)} image(s)...")

            import io
            from PIL import Image as _PILImage
            from scilink.parsers.ocr import describe_image

            prompt = question.strip() if question else _VIEW_IMAGE_DEFAULT_PROMPT
            results, errors = [], []
            for p in paths:
                pp = Path(p)
                if not pp.is_file():
                    errors.append(f"Not a file: {p}")
                    continue
                if pp.suffix.lower() not in _IMAGE_EXTS:
                    errors.append(f"Not an image file: {p}")
                    continue
                try:
                    img = _PILImage.open(pp)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    # Cap the longest side at 2048 px — keeps fine print legible
                    # while keeping payload size reasonable.
                    img.thumbnail((2048, 2048))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    description = describe_image(
                        buf.getvalue(), self.orch.model, prompt
                    )
                    results.append({"name": pp.name,
                                    "description": description})
                except Exception as e:  # noqa: BLE001 - one bad image must not break the tool
                    logging.error(f"view_image failed for {p}: {e}")
                    errors.append(f"Could not view {pp.name}: {e}")
            if not results:
                return json.dumps({
                    "status": "error",
                    "message": "No images could be viewed.",
                    "errors": errors,
                })
            return json.dumps({
                "status": "success",
                "n_images": len(results),
                "images": results,
                "errors": errors or None,
            })

        self._register_tool(
            func=view_image,
            name="view_image",
            description=(
                "Open one or more image files and have the vision model "
                "describe them — including faithfully transcribing any "
                "readable text or tables (printed or handwritten). Use this "
                "for a photo of a notebook page, a diagram, a screenshot, a "
                "figure, or any image that needs to be interpreted as "
                "content. NOT for scientific images that need feature "
                "extraction or quantification — route those to analysis. "
                "Accepts .png, .jpg/.jpeg, .tif/.tiff, .bmp, .gif, .webp."
            ),
            parameters={
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Absolute path(s) to the image file(s) to view."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "Optional question or instruction (e.g. "
                        "'transcribe the table' or 'what does the diagram "
                        "show?'). Default is to describe + transcribe."
                    ),
                },
            },
            required=["paths"],
        )
