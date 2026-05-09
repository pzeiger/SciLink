"""
Tests for the skill loader's frontmatter parsing and extras capture.

These cover the loader-level behaviors only — agent-side consumption is
exercised in domain-specific test modules (e.g. test_amber_skill).
"""

import logging
import textwrap

import pytest

from scilink.skills.loader import load_skill, _split_frontmatter, _parse_sections


def _write_skill(tmp_path, body: str) -> str:
    path = tmp_path / "my_skill.md"
    path.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(path)


class TestFrontmatter:
    def test_no_frontmatter_yields_empty_meta(self, tmp_path):
        path = _write_skill(tmp_path, """
            ## overview
            Body text.
        """)
        skill = load_skill(path)
        assert skill["meta"] == {}
        assert skill["overview"].startswith("Body text")

    def test_frontmatter_is_parsed(self, tmp_path):
        path = _write_skill(tmp_path, """
            ---
            description: One-line blurb
            domain: force_field
            applies_to: [amber, gaff2]
            ---
            ## overview
            Body text.
        """)
        skill = load_skill(path)
        assert skill["meta"]["description"] == "One-line blurb"
        assert skill["meta"]["domain"] == "force_field"
        assert skill["meta"]["applies_to"] == ["amber", "gaff2"]
        assert skill["overview"].startswith("Body text")

    def test_malformed_frontmatter_logs_warning_but_does_not_raise(self, tmp_path, caplog):
        path = _write_skill(tmp_path, """
            ---
            description: : :: not yaml
              indent_broken
             - bad
            ---
            ## overview
            Body text.
        """)
        with caplog.at_level(logging.WARNING, logger="scilink.skills.loader"):
            skill = load_skill(path)
        assert skill["meta"] == {}
        assert skill["overview"].startswith("Body text")
        assert any("Malformed frontmatter" in r.message for r in caplog.records)

    def test_non_mapping_frontmatter_is_ignored(self, tmp_path, caplog):
        # Frontmatter that parses to a list, not a dict.
        path = _write_skill(tmp_path, """
            ---
            - just
            - a
            - list
            ---
            ## overview
            Body text.
        """)
        with caplog.at_level(logging.WARNING, logger="scilink.skills.loader"):
            skill = load_skill(path)
        assert skill["meta"] == {}
        assert any("did not parse to a mapping" in r.message for r in caplog.records)


class TestExtras:
    def test_unknown_section_is_captured_under_extras(self, tmp_path, caplog):
        path = _write_skill(tmp_path, """
            ## overview
            Overview body.

            ## common pitfalls
            Watch out for X.
        """)
        with caplog.at_level(logging.WARNING, logger="scilink.skills.loader"):
            skill = load_skill(path)
        assert skill["overview"].startswith("Overview body")
        assert "common pitfalls" in skill["extras"]
        assert skill["extras"]["common pitfalls"].startswith("Watch out for X")
        assert any("not in the canonical vocabulary" in r.message for r in caplog.records)

    def test_known_sections_are_not_in_extras(self, tmp_path):
        path = _write_skill(tmp_path, """
            ## overview
            X.
            ## planning
            Y.
        """)
        skill = load_skill(path)
        assert skill["extras"] == {}

    def test_aimsgb_common_pitfalls_is_now_captured(self):
        """The shipped aimsgb skill has a 'Common pitfalls' section that
        used to be silently dropped. Verify it now flows through extras."""
        skill = load_skill("aimsgb", domain="structure_generation")
        assert "common pitfalls" in skill["extras"]
        assert len(skill["extras"]["common pitfalls"]) > 0


class TestBackwardsCompatibility:
    def test_canonical_section_keys_still_present(self, tmp_path):
        """All known section keys must always exist in the result, even
        when missing from the file — existing consumers rely on this."""
        path = _write_skill(tmp_path, """
            ## overview
            Just overview.
        """)
        skill = load_skill(path)
        for key in ["overview", "planning", "analysis", "interpretation",
                    "validation", "implementation"]:
            assert key in skill, f"Missing key: {key}"
        assert skill["overview"].startswith("Just overview")
        assert skill["planning"] == ""

    def test_existing_amber_skill_still_loads_cleanly(self):
        skill = load_skill("amber", domain="force_field")
        assert skill["name"] == "amber"
        assert "meta" in skill
        assert "extras" in skill
        # AMBER ships with a description in frontmatter; no off-vocabulary
        # headings.
        assert isinstance(skill["meta"].get("description"), str)
        assert skill["extras"] == {}
        assert len(skill["overview"]) > 0
