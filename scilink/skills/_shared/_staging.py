"""Staging buffer for raw T=2 solutions, and the two distillation consumers.

The T=2 auto-distill hooks no longer write a skill per win. Instead they
**stage** the raw solution (the agent's knowledge entry + the verbatim working
script + an LLM-assigned technique label) here, and skills are produced later,
review-gated, two ways:

  upgrade@1  — a single staged solution enriches an EXISTING skill
               (``upgrade_skill_from_staged`` → ``graduate_to_skill_file`` update branch).
  new-skill@N — N staged solutions for one technique are consolidated into a
               NEW skill (``consolidate_technique``).

Staging lives at ``scilink_home()/distill_staging/<domain>/<id>.json`` — a sibling
of ``graduated_skills/`` so the skill loader never mistakes a staged record for a
skill. The module is package-neutral (``ase``-free): stdlib + the ``_graduation``
helper + the loader, same as ``_memory``.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..loader import scilink_home, load_skill, list_skills
from ._graduation import graduate_to_skill_file


def staging_dir() -> Path:
    """Root of the raw-solution staging buffer (honors ``$SCILINK_HOME``)."""
    return scilink_home() / "distill_staging"


def _domain_dir(domain: str, *, root: Optional[Path] = None) -> Path:
    return (root or staging_dir()) / domain


# ──────────────────────────────────────────────────────────────
# Buffer CRUD
# ──────────────────────────────────────────────────────────────

def stage_solution(
    domain: str,
    technique: str,
    record: Dict[str, Any],
    *,
    root: Optional[Path] = None,
) -> str:
    """Persist one raw T=2 solution; return its assigned id.

    ``record`` is the agent's ``knowledge_entry`` (model/deviation/metric/script,
    etc.); ``technique`` is the normalized grouping label. The stored JSON adds
    ``id``, ``domain``, ``technique``.
    """
    d = _domain_dir(domain, root=root)
    d.mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex[:8]
    payload = {"id": sid, "domain": domain, "technique": technique, **record}
    (d / f"{sid}.json").write_text(json.dumps(payload, indent=2, default=str))
    return sid


def list_staged(
    domain: Optional[str] = None,
    technique: Optional[str] = None,
    *,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return staged solution records, optionally filtered by domain/technique."""
    base = root or staging_dir()
    out: List[Dict[str, Any]] = []
    if not base.is_dir():
        return out
    domains = [base / domain] if domain else [
        p for p in sorted(base.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_"))
    ]
    for dd in domains:
        if not dd.is_dir():
            continue
        for f in sorted(dd.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            if technique is not None and rec.get("technique") != technique:
                continue
            rec.setdefault("domain", dd.name)
            out.append(rec)
    return out


def get_staged(domain: str, sid: str, *, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    f = _domain_dir(domain, root=root) / f"{sid}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def remove_staged(domain: str, ids: List[str], *, root: Optional[Path] = None) -> int:
    """Delete staged records by id; return count removed."""
    d = _domain_dir(domain, root=root)
    n = 0
    for sid in ids:
        f = d / f"{sid}.json"
        if f.exists():
            f.unlink()
            n += 1
    return n


def group_by_technique(domain: str, *, root: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Group a domain's staged records by their technique label."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in list_staged(domain, root=root):
        groups.setdefault(rec.get("technique") or "unlabeled", []).append(rec)
    return groups


def assign_technique_label(
    domain: str,
    model: str,
    deviation: str,
    llm_call: Callable[[str], str],
    label_template: str,
    *,
    root: Optional[Path] = None,
) -> str:
    """LLM-assign a normalized snake_case technique label (reuse-or-create).

    Uses ``existing_techniques(domain)`` as the reuse vocabulary. Returns a
    sanitized label; falls back to ``"uncategorized"`` if the model yields
    nothing usable.
    """
    import re
    existing = existing_techniques(domain, root=root)
    prompt = label_template.format(
        model=model or "(unknown model)",
        deviation=deviation or "(no explicit note)",
        existing="\n".join(f"- {t}" for t in existing) or "(none)",
    )
    raw = (llm_call(prompt) or "").strip()
    first = raw.splitlines()[0] if raw else ""
    label = re.sub(r"[^a-z0-9]+", "_", first.lower()).strip("_")[:48]
    return label or "uncategorized"


def existing_techniques(domain: str, *, root: Optional[Path] = None) -> List[str]:
    """Vocabulary for the labeling prompt + target hints.

    Union of (a) technique labels already on staged records and (b) existing
    skill names in this domain (persistent + built-in, via the loader)."""
    labels = set(group_by_technique(domain, root=root).keys())
    try:
        labels.update(list_skills(domain))
    except Exception:
        pass
    return sorted(t for t in labels if t and t != "unlabeled")


# ──────────────────────────────────────────────────────────────
# Shared: build a reference block + knowledge text from a record
# ──────────────────────────────────────────────────────────────

def _script_of(rec: Dict[str, Any]) -> str:
    return (rec.get("working_script") or rec.get("script") or "").strip()


def _ref_block(rec: Dict[str, Any], *, label: str = "fit") -> str:
    script = _script_of(rec)
    if not script:
        return ""
    return (
        f"### Reference implementation (verbatim {label} that produced this skill)\n\n"
        "```python\n" + script + "\n```"
    )


def _record_for_prompt(rec: Dict[str, Any]) -> Dict[str, Any]:
    """A record trimmed for the LLM prompt — drop bookkeeping + the (large)
    script, which is appended verbatim separately."""
    drop = {"id", "domain", "technique", "working_script", "script", "created_at"}
    return {k: v for k, v in rec.items() if k not in drop}


# ──────────────────────────────────────────────────────────────
# upgrade@1 — merge staged solution(s) into an EXISTING skill
# ──────────────────────────────────────────────────────────────

def upgrade_skill_from_staged(
    domain: str,
    staged_ids: List[str],
    *,
    target_domain: str,
    target_name: str,
    llm_call: Callable[[str], str],
    fresh_template: str,
    update_template: str,
    root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Enrich an existing skill from one (or a few) staged solution(s).

    The target skill must already exist, so ``graduate_to_skill_file`` takes its
    update/merge branch — the same path the demo exercises (EELS v1→v2). The
    verbatim script of the first staged record is appended as a reference. On
    success the consumed staged records are removed.
    """
    recs = [r for r in (get_staged(domain, sid, root=root) for sid in staged_ids) if r]
    if not recs:
        return {"status": "error", "message": "No staged records found for the given ids."}

    primary = recs[0]
    knowledge_entry = {
        "new_t2_solutions": [_record_for_prompt(r) for r in recs],
    }
    result = graduate_to_skill_file(
        knowledge_entry=knowledge_entry,
        skill_name=target_name,
        domain=target_domain,
        llm_call=llm_call,
        fresh_template=fresh_template,
        update_template=update_template,
        skills_root=skills_root,
        append_sections={"implementation": _ref_block(primary)},
    )
    if result.get("status") == "success":
        remove_staged(domain, [r["id"] for r in recs if r.get("id")], root=root)
        result["consumed_staged"] = [r.get("id") for r in recs]
    return result


# ──────────────────────────────────────────────────────────────
# new-skill@N — consolidate N staged solutions into a NEW skill
# ──────────────────────────────────────────────────────────────

def _best_record(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the highest-quality staged record (r_squared or quality_score)."""
    def _metric(r: Dict[str, Any]) -> float:
        for k in ("r_squared", "quality_score", "metric"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    return max(recs, key=_metric)


def consolidate_technique(
    domain: str,
    technique: str,
    *,
    llm_call: Callable[[str], str],
    consolidation_template: str,
    update_template: str,
    root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Distill all staged solutions for one technique into a single new skill.

    Builds a multi-example knowledge entry, calls ``graduate_to_skill_file`` with
    the consolidation template (skill name ``auto_<technique>``), appends the best
    example's script verbatim, tags ``provenance=t2_consolidated`` + ``n_examples``,
    and removes the consumed staged records.
    """
    recs = list_staged(domain, technique, root=root)
    if not recs:
        return {"status": "error", "message": f"No staged solutions for {domain}/{technique}."}

    best = _best_record(recs)
    knowledge_entry = {
        "technique": technique,
        "n_examples": len(recs),
        "examples": [_record_for_prompt(r) for r in recs],
    }
    skill_name = f"auto_{technique}"
    result = graduate_to_skill_file(
        knowledge_entry=knowledge_entry,
        skill_name=skill_name,
        domain=domain,
        llm_call=llm_call,
        fresh_template=consolidation_template,
        update_template=update_template,
        skills_root=skills_root,
        extra_meta={
            "provisional": True,
            "provenance": "t2_consolidated",
            "n_examples": len(recs),
        },
        append_sections={"implementation": _ref_block(best)},
    )
    if result.get("status") == "success":
        remove_staged(domain, [r["id"] for r in recs if r.get("id")], root=root)
        result["consumed_staged"] = [r.get("id") for r in recs]
        result["n_examples"] = len(recs)
    return result
