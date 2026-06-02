#!/usr/bin/env python3
"""
scilink memory - Manage SciLink's persistent memory

The persistent store (``~/.scilink/graduated_skills``, relocatable via
``SCILINK_HOME``) holds graduated and auto-distilled skills. It survives a
``pip`` upgrade and is auto-discovered by the skill loader.

Auto-distilled skills (from successful T=2 "hot annealing" curve fits) are
written **provisional**: discoverable and explicitly usable, but kept out of
the auto-routing menu until you review and promote them.

Subcommands:
  list      List persisted skills (use --provisional-only to triage)
  show      Print a skill's markdown
  promote   Clear a skill's provisional flag so it routes normally
  prune     Delete a skill bundle
"""

import argparse
import sys

from scilink.skills._shared._memory import (
    list_memory,
    show_memory,
    promote_memory,
    prune_memory,
)


def _split_ref(ref: str):
    """Parse a 'domain/name' reference. Returns (domain, name)."""
    if "/" not in ref:
        raise SystemExit(
            f"❌ Expected '<domain>/<name>' (e.g. 'curve_fitting/auto_voigt_ab12cd34'), got: {ref}"
        )
    domain, name = ref.split("/", 1)
    return domain, name


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

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
