"""
Skill loader for domain-specific knowledge files.

Skills are markdown files with structured sections (## headings) that provide
domain-specific guidance for LLM-driven analysis pipelines. They can be
built-in (shipped with the package) or user-provided (custom .md files).

A skill file may optionally begin with a YAML frontmatter block delimited by
``---`` lines, e.g.::

    ---
    description: One-line LLM-facing blurb
    domain: force_field
    applies_to: [amber, gaff2, ff14sb]
    ---
    ## overview
    ...

Frontmatter, when present, is exposed under the ``meta`` key of the parsed
skill. Sections whose heading isn't in the canonical vocabulary are preserved
under the ``extras`` key (lowercased heading → body) and a warning is logged.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import yaml

_SKILLS_DIR = Path(__file__).parent

_KNOWN_SECTIONS = {"overview", "planning", "analysis", "interpretation", "validation", "implementation"}

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_logger = logging.getLogger(__name__)


# ─── User-provided skill roots ─────────────────────────────────────
# Users can add their own skill bundles without modifying SciLink source
# by pointing the ``SCILINK_SKILLS_PATH`` env var at one (or several,
# os.pathsep-separated) extra skill-root directories. Each user root
# follows the same layout as the built-in tree:
#     <root>/<domain>/<name>/<name>.md
# User roots are searched BEFORE the built-in tree, so a user-provided
# ``<root>/periodic_dft/vasp/vasp.md`` overrides the bundled one of the
# same name. Each agent's capability-discovery (``supported_software``)
# automatically picks up new bundles dropped into a user root with no
# code changes required.

def _skill_roots() -> List[Path]:
    """Return ordered list of directories to search for skill bundles.

    User-provided roots from ``$SCILINK_SKILLS_PATH`` come first
    (highest precedence), then the built-in ``scilink/skills/`` tree.
    Re-evaluated on each call so adding to the env var mid-process is
    visible without a reload.
    """
    roots: List[Path] = []
    extra = os.environ.get("SCILINK_SKILLS_PATH", "").strip()
    if extra:
        for raw in extra.split(os.pathsep):
            if not raw.strip():
                continue
            p = Path(raw).expanduser().resolve()
            if p.is_dir():
                roots.append(p)
            else:
                _logger.warning(
                    "SCILINK_SKILLS_PATH entry not found or not a directory: %s", p
                )
    roots.append(_SKILLS_DIR)
    return roots


def list_skills(domain: str = "curve_fitting") -> list:
    """Return all available skill names for a domain, across all roots.

    A skill is a folder ``<root>/<domain>/<name>/`` containing a
    matching ``<name>.md`` (Anthropic-Skills-style bundle). Folders
    starting with ``_`` or ``.`` (e.g. ``_shared/``) are skipped.
    User-provided skill roots (via ``$SCILINK_SKILLS_PATH``) and the
    built-in tree are unioned; duplicate names across roots collapse
    to one entry (user root wins at load time).

    Args:
        domain: Skill domain subdirectory (default: "curve_fitting").

    Returns:
        Sorted list of skill name strings.
    """
    names: set = set()
    for root in _skill_roots():
        domain_dir = root / domain
        if domain_dir.is_dir():
            names.update(_iter_skill_names(domain_dir))
    return sorted(names)


def list_all_skills() -> dict:
    """Auto-discover all skill domains across user + built-in roots.

    Scans subdirectories of every skill root (user-provided via
    ``$SCILINK_SKILLS_PATH`` and the built-in ``scilink/skills/``) for
    skill bundles. Domains from different roots are merged.

    Returns:
        Dict mapping domain names to sorted lists of skill names,
        e.g. ``{"periodic_dft": ["vasp", "cp2k"], ...}``.
        Empty domains are omitted.
    """
    result: dict = {}
    for root in _skill_roots():
        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or sub.name.startswith(("_", ".")):
                continue
            names = set(_iter_skill_names(sub))
            if names:
                result.setdefault(sub.name, set()).update(names)
    return {k: sorted(v) for k, v in result.items()}


def _iter_skill_names(domain_dir: Path):
    """Yield skill names found as ``<domain_dir>/<name>/<name>.md`` bundles."""
    for child in domain_dir.iterdir():
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        if (child / f"{child.name}.md").exists():
            yield child.name


def load_skill(skill: str, domain: str = "curve_fitting") -> Dict:
    """Load and parse a skill markdown file.

    Args:
        skill: Either a built-in skill name (e.g. "xps") which resolves to
            ``scilink/skills/{domain}/{name}.md``, or a path to a custom
            ``.md`` file.
        domain: Skill domain subdirectory (default: "curve_fitting").

    Returns:
        Dict with keys: ``name`` (file stem), ``meta`` (frontmatter dict,
        empty if no frontmatter), one entry per canonical section in
        :data:`_KNOWN_SECTIONS` (missing sections default to empty string),
        and ``extras`` (dict of non-canonical-heading → body).

    Raises:
        FileNotFoundError: If the skill file cannot be found.
        ValueError: If the file is empty or cannot be parsed.
    """
    path = _resolve_skill_path(skill, domain)
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"Skill file is empty: {path}")

    meta, body = _split_frontmatter(text, source=str(path))
    sections, extras = _parse_sections(body, source=str(path))

    sections["name"] = path.stem
    sections["meta"] = meta
    sections["extras"] = extras
    return sections


def _resolve_skill_path(skill: str, domain: str) -> Path:
    """Resolve a skill name or path to an actual file path.

    Skills live at ``<root>/<domain>/<name>/<name>.md`` (Tier C bundle
    layout). Searches user-provided roots from ``$SCILINK_SKILLS_PATH``
    first (so user bundles can override built-in ones of the same
    name), then the built-in ``scilink/skills/`` tree.

    Three accepted forms for ``skill``:

    1. ``"<name>"`` — bare skill name. First searched under the
       requested ``domain``; if not found there, falls back to a
       cross-domain search of every visible skill root. Cross-domain
       lookup is what makes cross-cutting skills (e.g. ``"xrd"`` under
       ``structure_matching/``) usable from agents whose default
       domain is something else (e.g. ``CurveFittingAgent`` →
       ``curve_fitting``).
    2. ``"<domain>/<name>"`` — explicit domain-qualified name. Bypasses
       the fallback search and goes straight to the named domain.
    3. ``"/path/to/<name>.md"`` — direct path to a markdown file.

    A bare-name lookup that matches the same ``<name>`` under multiple
    domains is ambiguous and raises with the list of candidates.
    """
    candidate = Path(skill)
    if candidate.suffix.lower() == ".md":
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Skill file not found: {candidate}")

    # Domain-qualified form: "<domain>/<name>"
    if "/" in skill and not skill.startswith("/"):
        explicit_domain, explicit_name = skill.split("/", 1)
        if "/" in explicit_name:
            raise ValueError(
                f"Skill name '{skill}' is malformed; expected "
                "'<domain>/<name>' or a bare '<name>'."
            )
        for root in _skill_roots():
            bundle = root / explicit_domain / explicit_name / f"{explicit_name}.md"
            if bundle.exists():
                return bundle
        raise FileNotFoundError(
            f"Skill '{explicit_name}' not found in domain '{explicit_domain}'."
        )

    # Bare-name lookup: requested domain first, then fall back to all domains.
    for root in _skill_roots():
        bundle = root / domain / skill / f"{skill}.md"
        if bundle.exists():
            return bundle

    matches = _find_skill_across_domains(skill)
    if len(matches) == 1:
        _logger.debug(
            "Skill '%s' not found in domain '%s' — resolved cross-domain to %s.",
            skill, domain, matches[0],
        )
        return matches[0]
    if len(matches) > 1:
        rendered = ", ".join(f"{p.parent.parent.name}/{skill}" for p in matches)
        raise FileNotFoundError(
            f"Skill '{skill}' is ambiguous across domains ({rendered}). "
            f"Qualify it as '<domain>/{skill}'."
        )

    available_here = list_skills(domain=domain)
    all_domains = list_all_skills()
    raise FileNotFoundError(
        f"Skill '{skill}' not found for domain '{domain}'. "
        f"Available in '{domain}': {available_here}. "
        f"All known skills: {all_domains}."
    )


def _find_skill_across_domains(name: str) -> list[Path]:
    """Return every ``<root>/<domain>/<name>/<name>.md`` across all roots."""
    matches: list[Path] = []
    seen: set[str] = set()
    for root in _skill_roots():
        if not root.is_dir():
            continue
        for domain_dir in sorted(root.iterdir()):
            if not domain_dir.is_dir() or domain_dir.name.startswith(("_", ".")):
                continue
            bundle = domain_dir / name / f"{name}.md"
            if bundle.exists() and str(bundle) not in seen:
                seen.add(str(bundle))
                matches.append(bundle)
    return matches


def _split_frontmatter(text: str, source: str = "<skill>") -> tuple[dict, str]:
    """Strip an optional leading ``---``-fenced YAML frontmatter block.

    Returns ``(meta_dict, remaining_text)``. If no frontmatter is present,
    returns ``({}, text)``. Malformed YAML logs a warning and yields an
    empty meta dict but does not raise — the body is returned with the
    frontmatter stripped so section parsing can proceed.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    raw = m.group(1)
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        _logger.warning("Malformed frontmatter in %s: %s", source, e)
        return {}, text[m.end():]

    if not isinstance(parsed, dict):
        _logger.warning(
            "Frontmatter in %s did not parse to a mapping (got %s); ignoring.",
            source, type(parsed).__name__,
        )
        return {}, text[m.end():]

    return parsed, text[m.end():]


def _parse_sections(text: str, source: str = "<skill>") -> tuple[Dict[str, str], Dict[str, str]]:
    """Parse markdown into sections keyed by ``## heading`` name.

    Returns ``(known_sections, extras)`` where ``known_sections`` covers
    the canonical vocabulary (missing entries default to empty string) and
    ``extras`` captures any other ``## ...`` sections (lowercased heading
    → body). Unknown headings emit a warning so authors get feedback when
    content would otherwise be silently dropped.
    """
    sections = {k: "" for k in _KNOWN_SECTIONS}
    extras: Dict[str, str] = {}

    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)

    for part in parts[1:]:
        lines = part.split("\n", 1)
        heading = lines[0].strip().lower()
        body = lines[1].strip() if len(lines) > 1 else ""
        if heading in _KNOWN_SECTIONS:
            sections[heading] = body
        else:
            extras[heading] = body
            _logger.warning(
                "Skill %s: section '## %s' is not in the canonical vocabulary "
                "(%s); preserved under 'extras'.",
                source, heading, sorted(_KNOWN_SECTIONS),
            )

    return sections, extras
