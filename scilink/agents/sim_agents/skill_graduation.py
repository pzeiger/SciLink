"""Shared infrastructure for on-the-fly skill graduation in sim_agents.

Mirrors the planning orchestrator's `graduate_to_skill` pattern
(`scilink/agents/planning_agents/orchestrator_tools.py:4257`), but
persistent across sessions and agent-agnostic so `VaspUpdater` and
`VaspQualityAgent` (and any future sim agent) can share it.

Two-tier memory:
  - active_knowledge (in-memory, session-scoped): observations the
    agent records during a run. Cheap to add, easy to discard.
  - graduated skills (.md files on disk): durable rules the agent
    has crystallized. Loaded back into LLM context next session.

Graduation uses an LLM to integrate a knowledge entry into either a
fresh skill or an existing one (auto-detected based on whether a file
with the target name already exists). This is the same merge-not-
append pattern the planning side uses.

Default storage: ~/.scilink/graduated_skills/<domain>/<name>/<name>.md
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


GRADUATED_SKILLS_DIR = Path.home() / ".scilink" / "graduated_skills"
WORD_COUNT_WARN_THRESHOLD = 8000


# ──────────────────────────────────────────────────────────────
# In-session knowledge store
# ──────────────────────────────────────────────────────────────

class KnowledgeStore:
    """Session-scoped store of observations awaiting graduation.

    Attached to an agent instance; lives only as long as the agent.
    Intentionally simple — list of dicts with auto-assigned ids."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def record(self, observation: Dict[str, Any]) -> str:
        """Append an observation; return its assigned id."""
        entry_id = uuid.uuid4().hex[:8]
        entry = {"id": entry_id, **observation}
        self._entries.append(entry)
        self.logger.info(
            f"Recorded knowledge entry {entry_id}: "
            f"{str(observation.get('summary', ''))[:80]}"
        )
        return entry_id

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        for e in self._entries:
            if e.get("id") == entry_id:
                return e
        return None

    def list(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def remove(self, entry_id: Optional[str] = None) -> int:
        """Remove a specific entry, or all when entry_id is None.
        Returns count removed."""
        if entry_id is None:
            n = len(self._entries)
            self._entries.clear()
            return n
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.get("id") != entry_id]
        return before - len(self._entries)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _format_knowledge(entry: Dict[str, Any]) -> str:
    """Format a knowledge entry as readable text for the LLM prompt.

    Skips the auto-assigned id; renders each remaining key as a
    `**Key:** value` line."""
    lines = []
    for key, value in entry.items():
        if key == "id":
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"**{pretty_key}:** {value}")
    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    """Strip ```markdown ... ``` wrapping if the LLM fenced its output."""
    stripped = text.strip()
    fenced = re.match(
        r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$",
        stripped,
        re.DOTALL,
    )
    if fenced:
        return fenced.group(1).strip() + "\n"
    return stripped + "\n"


# Regex for a well-formed YAML frontmatter block at the top of a file.
# Mirrors the pattern in scilink/skills/loader.py so we can detect when
# the LLM forgot to wrap the description.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _yaml_safe_single_line(value: str) -> str:
    """Quote a single-line scalar so YAML parses it as a literal string,
    even when it begins with characters YAML treats specially ({ [ * & ! |
    > ' " % @ #).

    Uses single-quoted style: any embedded single quotes are doubled. Newlines
    are collapsed (single-quoted style cannot encode them; descriptions are
    declared single-line in the prompt anyway)."""
    collapsed = " ".join(value.split())
    escaped = collapsed.replace("'", "''")
    return f"'{escaped}'"


def _ensure_frontmatter(text: str) -> str:
    """Repair a graduated-skill body whose LLM forgot the YAML frontmatter.

    The fresh-graduation prompt asks the LLM to emit::

        ---
        description: <one-line>
        ---

        ## overview
        ...

    In practice the LLM occasionally drops the opening ``---`` and emits
    the description as a bare top line. The skill loader then can't
    parse the frontmatter and the rule never reaches downstream prompts.
    Auto-repair: if the text doesn't start with a frontmatter block,
    promote whatever's before the first ``## <section>`` heading into a
    ``description:`` field and wrap it.
    """
    if _FRONTMATTER_RE.match(text):
        return text
    # Find the first markdown section heading; everything before it is
    # candidate description text.
    match = re.search(r"^##\s", text, re.MULTILINE)
    if match is None:
        # No body to anchor on -- give up, return as-is.
        return text
    head = text[: match.start()].strip()
    body = text[match.start():]
    # Promote any leading "description:" prefix the LLM might have used,
    # otherwise take the head verbatim. Strip a trailing standalone "---"
    # if the LLM emitted one (the closing delimiter without an opener).
    head = re.sub(r"^description:\s*", "", head, flags=re.IGNORECASE)
    head = re.sub(r"\n?---\s*$", "", head).strip()
    if not head:
        head = "auto-generated description (LLM omitted frontmatter)"
    # YAML-quote the value so descriptions starting with `{`, `"`, etc.
    # don't trip the loader's YAML parser.
    return f"---\ndescription: {_yaml_safe_single_line(head)}\n---\n\n{body}"


# ──────────────────────────────────────────────────────────────
# Graduation
# ──────────────────────────────────────────────────────────────

def graduate_to_skill_file(
    *,
    knowledge_entry: Dict[str, Any],
    skill_name: str,
    domain: str,
    llm_call: Callable[[str], str],
    fresh_template: str,
    update_template: str,
    skills_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Graduate a knowledge entry into a skill .md file.

    If a skill with this name exists at
    ``<skills_root>/<domain>/<skill_name>/<skill_name>.md``, the LLM is
    asked to merge the new knowledge into it via `update_template`.
    Otherwise a fresh skill is created via `fresh_template`. Either way
    the result lands at that same path so subsequent sessions auto-load
    it via `load_graduated_skills`.

    Args:
        knowledge_entry: dict from `KnowledgeStore.get` or hand-built.
        skill_name: target skill name (used as both directory and .md
            filename).
        domain: skill domain (e.g. "vasp").
        llm_call: callable taking a prompt and returning the LLM's
            response text.
        fresh_template: prompt template for first-time graduation. Must
            accept {skill_name}, {domain}, {knowledge_text} keys.
        update_template: prompt template for merging into an existing
            skill. Must accept {skill_name}, {existing_skill},
            {new_knowledge} keys.
        skills_root: where graduated skills live. Defaults to
            GRADUATED_SKILLS_DIR (~/.scilink/graduated_skills).

    Returns:
        dict with status, method ("created"/"updated"), skill_path,
        word_count, and a soft warning if the file is getting long.
    """
    skills_root = skills_root or GRADUATED_SKILLS_DIR
    domain_dir = skills_root / domain
    skill_dir = domain_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / f"{skill_name}.md"

    knowledge_text = _format_knowledge(knowledge_entry)

    is_update = skill_path.exists()
    if is_update:
        existing = skill_path.read_text()
        prompt = update_template.format(
            skill_name=skill_name,
            existing_skill=existing,
            new_knowledge=knowledge_text,
        )
    else:
        prompt = fresh_template.format(
            skill_name=skill_name,
            domain=domain,
            knowledge_text=knowledge_text,
        )

    raw = llm_call(prompt)
    skill_content = _strip_code_fences(raw)
    # Auto-repair if the LLM forgot the frontmatter -- otherwise the
    # rule never makes it into downstream prompts via the loader.
    skill_content = _ensure_frontmatter(skill_content)

    # Loader-style bundles include __init__.py; harmless for path-loaded
    # skills, but keeps the layout consistent with built-ins.
    (skill_dir / "__init__.py").touch()
    skill_path.write_text(skill_content)

    word_count = len(skill_content.split())
    warning = None
    if word_count > WORD_COUNT_WARN_THRESHOLD:
        warning = (
            f"Graduated skill '{skill_name}' is now {word_count} words long; "
            f"consider running a manual consolidation pass."
        )

    return {
        "status": "success",
        "method": "updated" if is_update else "created",
        "skill_name": skill_name,
        "domain": domain,
        "skill_path": str(skill_path),
        "word_count": word_count,
        "warning": warning,
    }


# ──────────────────────────────────────────────────────────────
# Loading + prompt injection
# ──────────────────────────────────────────────────────────────

def load_graduated_skills(
    domain: str,
    skills_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return parsed skill dicts for every graduated skill in this domain.

    Each dict has the same shape as `scilink.skills.loader.load_skill`'s
    return value. Returns an empty list if no graduated skills exist.

    Loading uses the standard skill loader by passing the .md path
    directly (the loader supports path-as-skill — bypasses bundle
    lookup)."""
    skills_root = skills_root or GRADUATED_SKILLS_DIR
    domain_dir = skills_root / domain
    if not domain_dir.is_dir():
        return []

    from ...skills.loader import load_skill

    skills: List[Dict[str, Any]] = []
    for entry in sorted(domain_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        md_path = entry / f"{entry.name}.md"
        if not md_path.exists():
            continue
        try:
            parsed = load_skill(str(md_path), domain=domain)
            skills.append(parsed)
        except Exception as exc:
            logging.warning(
                f"Could not load graduated skill at {md_path}: {exc}"
            )
    return skills


def format_graduated_skills_block(
    skills: List[Dict[str, Any]],
    *,
    sections: Optional[List[str]] = None,
) -> str:
    """Format graduated skills as a single markdown block for prompt injection.

    Each skill's description + selected sections rendered as nested
    markdown. Returns an empty string when `skills` is empty (so the
    caller can unconditionally concatenate this into a prompt).

    Args:
        skills: list returned by `load_graduated_skills`.
        sections: which canonical sections to include. Defaults to
            ["planning", "implementation", "validation"] -- the
            sections most likely to carry actionable rules.
    """
    if not skills:
        return ""
    sections = sections or ["planning", "implementation", "validation"]
    lines = ["", "## LEARNED RULES (graduated skills)", ""]
    for sk in skills:
        name = sk.get("name", "graduated")
        desc = (sk.get("meta") or {}).get("description", "")
        lines.append(f"### {name}")
        if desc:
            lines.append(f"_{desc}_")
        lines.append("")
        for section in sections:
            content = (sk.get(section) or "").strip()
            if content:
                lines.append(f"#### {section}")
                lines.append(content)
                lines.append("")
    return "\n".join(lines)
