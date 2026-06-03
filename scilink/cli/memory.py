#!/usr/bin/env python3
"""
scilink memory - Manage SciLink's persistent memory

The persistent store (``~/.scilink/graduated_skills``, relocatable via
``SCILINK_HOME``) holds graduated and auto-distilled skills. It survives a
``pip`` upgrade and is auto-discovered by the skill loader.

Auto-distilled skills (from successful T=2 "hot annealing" curve fits) are
written **provisional**: discoverable and explicitly usable, but kept out of
the auto-routing menu until you review and promote them.

Skills (graduated_skills) subcommands:
  list      List persisted skills (use --provisional-only to triage)
  show      Print a skill's markdown
  promote   Clear a skill's provisional flag so it routes normally
  prune     Delete a skill bundle

Staged T=2 solutions (distill_staging) subcommands:
  staged       List staged raw T=2 solutions, grouped by technique
  upgrade      Merge a staged solution INTO an existing skill (--into <domain>/<name>)
  consolidate  Distill all staged solutions of a technique into a NEW skill
  prune-staged Delete staged solution(s)

`upgrade`/`consolidate` call an LLM; configure with --model / --base-url / --api-key.
"""

import argparse
import sys

from scilink.skills._shared._memory import (
    list_memory,
    show_memory,
    promote_memory,
    prune_memory,
)
from scilink.skills._shared import _staging


def _split_ref(ref: str):
    """Parse a 'domain/name' reference. Returns (domain, name)."""
    if "/" not in ref:
        raise SystemExit(
            f"❌ Expected '<domain>/<name>' (e.g. 'curve_fitting/auto_voigt_ab12cd34'), got: {ref}"
        )
    domain, name = ref.split("/", 1)
    return domain, name


def _make_llm_call(args):
    """Build an llm_call callable from the standard --model/--base-url/--api-key.

    Mirrors how the other CLI commands construct a model (LiteLLMGenerativeModel).
    """
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
    model = LiteLLMGenerativeModel(
        model=getattr(args, "model", None) or "claude-opus-4-6",
        api_key=getattr(args, "api_key", None),
        base_url=getattr(args, "base_url", None),
    )

    def _call(prompt: str) -> str:
        r = model.generate_content(contents=[prompt])
        return r.text if hasattr(r, "text") else str(r)

    return _call


def _add_model_args(p):
    p.add_argument("--model", default="claude-opus-4-6",
                   help="Model for the distillation LLM (default: claude-opus-4-6)")
    p.add_argument("--base-url", dest="base_url", default=None,
                   help="Custom API base URL (e.g. internal proxy)")
    p.add_argument("--api-key", dest="api_key", default=None,
                   help="API key (else taken from the conventional vendor env var)")


def _cmd_list(args) -> int:
    provisional = None
    if args.provisional_only:
        provisional = True
    elif args.promoted_only:
        provisional = False
    rows = list_memory(domain=args.domain, provisional=provisional)
    if not rows:
        print("No skills in persistent memory.")
        return 0
    for r in rows:
        tag = "  [provisional]" if r["provisional"] else ""
        r2 = f"  R²={r['r_squared']}" if r.get("r_squared") is not None else ""
        prov = f"  ({r['provenance']})" if r.get("provenance") else ""
        print(f"{r['domain']}/{r['name']}{tag}{r2}{prov}")
        if r.get("description"):
            print(f"    {r['description']}")
    print(f"\n{len(rows)} skill(s).")
    return 0


def _cmd_show(args) -> int:
    domain, name = _split_ref(args.ref)
    try:
        print(show_memory(domain, name))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    return 0


def _cmd_promote(args) -> int:
    domain, name = _split_ref(args.ref)
    try:
        res = promote_memory(domain, name, to_domain=args.to_domain)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    print(f"✅ Promoted {domain}/{name} → {res['domain']}/{name} (now auto-routable).")
    print(f"   {res['path']}")
    return 0


def _cmd_prune(args) -> int:
    domain, name = _split_ref(args.ref)
    if not args.yes:
        resp = input(f"Delete {domain}/{name} from persistent memory? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1
    try:
        prune_memory(domain, name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    print(f"🗑️  Pruned {domain}/{name}.")
    return 0


def _consolidate_n() -> int:
    import os
    try:
        return int(os.environ.get("SCILINK_CONSOLIDATE_N", "3"))
    except ValueError:
        return 3


def _cmd_staged(args) -> int:
    groups = _staging.group_by_technique(args.domain) if args.domain else {}
    if args.domain:
        domains = {args.domain: groups}
    else:
        # all domains
        domains = {}
        for rec in _staging.list_staged():
            domains.setdefault(rec["domain"], {}).setdefault(
                rec.get("technique") or "unlabeled", []).append(rec)
    total = 0
    threshold = _consolidate_n()
    for dom, by_tech in sorted(domains.items()):
        for tech, recs in sorted(by_tech.items()):
            total += len(recs)
            ready = " — ready to consolidate" if len(recs) >= threshold else ""
            print(f"{dom}/{tech}: {len(recs)} staged{ready}")
            for r in recs:
                metric = r.get("r_squared") or r.get("quality_score")
                mtxt = f"  metric={metric}" if metric is not None else ""
                print(f"    · id={r['id']}  session={r.get('session','?')}{mtxt}")
    if not total:
        print("No staged T=2 solutions.")
    else:
        print(f"\n{total} staged solution(s). "
              f"Upgrade an existing skill (`upgrade <domain>/<id> --into ...`) "
              f"or consolidate (`consolidate <domain>/<technique>`).")
    return 0


def _cmd_upgrade(args) -> int:
    from scilink.agents.exp_agents.instruct import (
        KNOWLEDGE_TO_SKILL_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS,
    )
    domain, sid = _split_ref(args.ref)  # staged ref = <domain>/<id>
    tdomain, tname = _split_ref(args.into)
    res = _staging.upgrade_skill_from_staged(
        domain, [sid], target_domain=tdomain, target_name=tname,
        llm_call=_make_llm_call(args),
        fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
        update_template=SKILL_UPDATE_INSTRUCTIONS,
    )
    if res.get("status") != "success":
        print(f"❌ {res.get('message', res)}")
        return 1
    print(f"✅ Upgraded {tdomain}/{tname} ({res.get('method')}) from staged {sid}.")
    print(f"   {res.get('skill_path')}")
    return 0


def _cmd_consolidate(args) -> int:
    from scilink.agents.exp_agents.instruct import (
        T2_CONSOLIDATION_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS,
    )
    domain, technique = _split_ref(args.ref)  # <domain>/<technique>
    res = _staging.consolidate_technique(
        domain, technique,
        llm_call=_make_llm_call(args),
        consolidation_template=T2_CONSOLIDATION_INSTRUCTIONS,
        update_template=SKILL_UPDATE_INSTRUCTIONS,
    )
    if res.get("status") != "success":
        print(f"❌ {res.get('message', res)}")
        return 1
    print(f"✅ Consolidated {res.get('n_examples')} staged solution(s) → "
          f"{domain}/auto_{technique} (provisional).")
    print(f"   {res.get('skill_path')}")
    return 0


def _cmd_prune_staged(args) -> int:
    domain, sid = _split_ref(args.ref)
    if not args.yes:
        resp = input(f"Delete staged solution {domain}/{sid}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1
    n = _staging.remove_staged(domain, [sid])
    print(f"🗑️  Removed {n} staged solution(s).")
    return 0 if n else 1


def main():
    """Entry point for 'scilink memory'."""
    parser = argparse.ArgumentParser(
        prog="scilink memory",
        description="Manage SciLink's persistent memory (graduated + auto-distilled skills).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action")

    p_list = sub.add_parser("list", help="List persisted skills")
    p_list.add_argument("--domain", help="Restrict to one domain (e.g. curve_fitting)")
    p_list.add_argument("--provisional-only", action="store_true",
                        help="Show only provisional (auto-distilled, unreviewed) skills")
    p_list.add_argument("--promoted-only", action="store_true",
                        help="Show only promoted (non-provisional) skills")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Print a skill's markdown")
    p_show.add_argument("ref", help="Skill reference '<domain>/<name>'")
    p_show.set_defaults(func=_cmd_show)

    p_promote = sub.add_parser("promote", help="Clear a skill's provisional flag")
    p_promote.add_argument("ref", help="Skill reference '<domain>/<name>'")
    p_promote.add_argument("--to-domain", help="Optionally move the bundle to a curated domain")
    p_promote.set_defaults(func=_cmd_promote)

    p_prune = sub.add_parser("prune", help="Delete a skill bundle")
    p_prune.add_argument("ref", help="Skill reference '<domain>/<name>'")
    p_prune.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_prune.set_defaults(func=_cmd_prune)

    # --- staged T=2 solutions ---
    p_staged = sub.add_parser("staged", help="List staged raw T=2 solutions by technique")
    p_staged.add_argument("--domain", help="Restrict to one domain")
    p_staged.set_defaults(func=_cmd_staged)

    p_up = sub.add_parser("upgrade", help="Merge a staged solution INTO an existing skill")
    p_up.add_argument("ref", help="Staged solution ref '<domain>/<id>'")
    p_up.add_argument("--into", required=True, help="Target skill '<domain>/<name>'")
    _add_model_args(p_up)
    p_up.set_defaults(func=_cmd_upgrade)

    p_con = sub.add_parser("consolidate", help="Distill all staged solutions of a technique into a NEW skill")
    p_con.add_argument("ref", help="Technique ref '<domain>/<technique>'")
    _add_model_args(p_con)
    p_con.set_defaults(func=_cmd_consolidate)

    p_ps = sub.add_parser("prune-staged", help="Delete a staged solution")
    p_ps.add_argument("ref", help="Staged solution ref '<domain>/<id>'")
    p_ps.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_ps.set_defaults(func=_cmd_prune_staged)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
