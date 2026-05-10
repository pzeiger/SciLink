"""Unit tests for sim_agents skill graduation.

No real LLM calls. The graduate_to_skill_file helper takes the LLM as
a callable, so we inject a fake one whose return value we control.
"""

import json
from pathlib import Path

import pytest

from scilink.agents.sim_agents.skill_graduation import (
    GRADUATED_SKILLS_DIR,
    KnowledgeStore,
    WORD_COUNT_WARN_THRESHOLD,
    _ensure_frontmatter,
    _format_knowledge,
    _strip_code_fences,
    format_graduated_skills_block,
    graduate_to_skill_file,
    load_graduated_skills,
)


# ── Minimal templates that just echo their inputs back. The real
# prompts are in scilink.agents.sim_agents.instruct; we don't test
# their text here — we test the plumbing.

FAKE_FRESH = "FRESH skill={skill_name} domain={domain}\n{knowledge_text}"
FAKE_UPDATE = "UPDATE skill={skill_name}\nEXISTING:\n{existing_skill}\nNEW:\n{new_knowledge}"

# ── A canonical valid skill .md the loader can actually parse, used
# both as a fake-LLM "response" for fresh graduation and as a fixture
# for update tests.

VALID_SKILL_MD = """\
---
description: test rule for graduation
---
## overview
Covers a single VASP error class.

## planning
Apply this rule when X.

## implementation
Set INCAR key Y to value Z.

## interpretation
Log line "X" indicates the rule applies.

## validation
After applying, verify W.
"""


# ──────────────────────────────────────────────────────────────
# KnowledgeStore
# ──────────────────────────────────────────────────────────────

class TestKnowledgeStore:
    def test_record_and_get(self):
        ks = KnowledgeStore()
        i1 = ks.record({"summary": "first"})
        i2 = ks.record({"summary": "second"})
        assert i1 != i2
        assert ks.get(i1)["summary"] == "first"
        assert ks.get(i2)["summary"] == "second"

    def test_get_unknown_id_returns_none(self):
        ks = KnowledgeStore()
        ks.record({"summary": "x"})
        assert ks.get("does-not-exist") is None

    def test_list_returns_copy(self):
        ks = KnowledgeStore()
        ks.record({"summary": "x"})
        first = ks.list()
        first.append({"id": "fake", "summary": "should not stick"})
        # Mutating the returned list shouldn't affect the store.
        assert len(ks.list()) == 1

    def test_remove_specific(self):
        ks = KnowledgeStore()
        i1 = ks.record({"summary": "one"})
        i2 = ks.record({"summary": "two"})
        n = ks.remove(i1)
        assert n == 1
        assert ks.get(i1) is None
        assert ks.get(i2) is not None

    def test_remove_all(self):
        ks = KnowledgeStore()
        ks.record({"summary": "a"})
        ks.record({"summary": "b"})
        ks.record({"summary": "c"})
        n = ks.remove()
        assert n == 3
        assert ks.list() == []


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class TestFormatKnowledge:
    def test_skips_id_field(self):
        text = _format_knowledge({"id": "abc", "summary": "thing"})
        assert "abc" not in text
        assert "Summary" in text and "thing" in text

    def test_pretty_prints_keys(self):
        text = _format_knowledge({"error_pattern": "ZBRENT"})
        assert "Error Pattern" in text


class TestEnsureFrontmatter:
    """The LLM occasionally forgets the opening `---`; auto-repair so the
    skill loader can still parse the resulting file."""

    def test_passthrough_when_frontmatter_present(self):
        good = "---\ndescription: x\n---\n\n## overview\nbody\n"
        assert _ensure_frontmatter(good) == good

    def test_repairs_missing_opening_delimiter(self):
        broken = (
            "VASP rule about ALGO=All + ISMEAR=-5 incompatibility\n"
            "---\n"
            "\n"
            "## overview\n"
            "body content\n"
        )
        repaired = _ensure_frontmatter(broken)
        assert repaired.startswith("---\n")
        assert "description: VASP rule about ALGO=All + ISMEAR=-5" in repaired
        # Body is preserved.
        assert "## overview" in repaired
        assert "body content" in repaired

    def test_repairs_completely_missing_frontmatter(self):
        broken = "Some description text\n\n## overview\nbody\n"
        repaired = _ensure_frontmatter(broken)
        assert repaired.startswith("---\n")
        assert "description: Some description text" in repaired

    def test_strips_description_prefix_if_llm_included_it(self):
        broken = (
            "description: foo bar baz\n"
            "---\n"
            "\n"
            "## overview\nbody\n"
        )
        repaired = _ensure_frontmatter(broken)
        assert "description: foo bar baz" in repaired
        # Should not have nested "description: description:".
        assert "description: description:" not in repaired

    def test_collapses_multiline_description_to_single_line(self):
        broken = (
            "Multiline description\n"
            "with a wrapped second line\n"
            "---\n"
            "\n"
            "## overview\nbody\n"
        )
        repaired = _ensure_frontmatter(broken)
        # Description survives as one line in the frontmatter.
        first_lines = repaired.splitlines()[:3]
        assert first_lines[0] == "---"
        assert first_lines[1].startswith("description:")
        assert "Multiline description with a wrapped second line" in first_lines[1]


class TestStripCodeFences:
    def test_passthrough_when_no_fence(self):
        out = _strip_code_fences("---\nfoo: bar\n---\n## overview\nhi\n")
        assert out.startswith("---")
        assert out.endswith("\n")

    def test_strips_markdown_fence(self):
        wrapped = "```markdown\n---\ndescription: x\n---\n## overview\nhi\n```"
        assert _strip_code_fences(wrapped).startswith("---")

    def test_strips_bare_fence(self):
        wrapped = "```\n---\ndescription: x\n---\n## overview\nhi\n```"
        assert _strip_code_fences(wrapped).startswith("---")


# ──────────────────────────────────────────────────────────────
# Graduation: fresh
# ──────────────────────────────────────────────────────────────

def _make_fake_llm(response):
    """Returns a callable matching the llm_call signature."""
    def _fn(prompt: str) -> str:
        # Return whatever was configured; tests can use this to fake
        # the LLM's structured-skill response.
        return response
    return _fn


class TestGraduateFresh:
    def test_creates_file_at_expected_path(self, tmp_path):
        result = graduate_to_skill_file(
            knowledge_entry={"id": "abc", "summary": "the thing"},
            skill_name="my_rule",
            domain="vasp",
            llm_call=_make_fake_llm(VALID_SKILL_MD),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["status"] == "success"
        assert result["method"] == "created"
        skill_path = Path(result["skill_path"])
        assert skill_path == tmp_path / "vasp" / "my_rule" / "my_rule.md"
        assert skill_path.exists()
        # __init__.py also created in the bundle dir for layout parity.
        assert (skill_path.parent / "__init__.py").exists()

    def test_file_content_matches_llm_output(self, tmp_path):
        graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="r",
            domain="vasp",
            llm_call=_make_fake_llm(VALID_SKILL_MD),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        content = (tmp_path / "vasp" / "r" / "r.md").read_text()
        assert "## overview" in content
        assert "## planning" in content

    def test_no_warning_below_word_threshold(self, tmp_path):
        result = graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="r",
            domain="vasp",
            llm_call=_make_fake_llm(VALID_SKILL_MD),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["warning"] is None
        assert 0 < result["word_count"] < WORD_COUNT_WARN_THRESHOLD


class TestGraduateLargeFile:
    def test_warning_fires_above_threshold(self, tmp_path):
        # Synthetic huge skill — way over 8000 words.
        big = VALID_SKILL_MD + "\n\n" + ("word " * (WORD_COUNT_WARN_THRESHOLD + 100))
        result = graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="big_rule",
            domain="vasp",
            llm_call=_make_fake_llm(big),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["warning"] is not None
        assert "consolidation" in result["warning"]
        assert result["word_count"] > WORD_COUNT_WARN_THRESHOLD


# ──────────────────────────────────────────────────────────────
# Graduation: update existing
# ──────────────────────────────────────────────────────────────

class TestGraduateUpdate:
    def test_uses_update_template_when_skill_exists(self, tmp_path):
        # First call: creates the skill.
        graduate_to_skill_file(
            knowledge_entry={"summary": "first"},
            skill_name="r",
            domain="vasp",
            llm_call=_make_fake_llm(VALID_SKILL_MD),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )

        # Second call: should detect the existing file and use update_template.
        # Capture the prompt the fake LLM saw to verify which template was used.
        seen = {"prompt": None}

        def capture_llm(prompt):
            seen["prompt"] = prompt
            return VALID_SKILL_MD  # fake an updated skill

        result = graduate_to_skill_file(
            knowledge_entry={"summary": "second"},
            skill_name="r",
            domain="vasp",
            llm_call=capture_llm,
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["method"] == "updated"
        # The prompt must have been built from update_template.
        assert seen["prompt"].startswith("UPDATE skill=r")
        assert "EXISTING:" in seen["prompt"]
        assert "NEW:" in seen["prompt"]


# ──────────────────────────────────────────────────────────────
# Loading + prompt-block formatting
# ──────────────────────────────────────────────────────────────

class TestLoadGraduatedSkills:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert load_graduated_skills("vasp", skills_root=tmp_path) == []

    def test_loads_a_graduated_skill_after_creation(self, tmp_path):
        graduate_to_skill_file(
            knowledge_entry={"summary": "first"},
            skill_name="my_rule",
            domain="vasp",
            llm_call=_make_fake_llm(VALID_SKILL_MD),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        skills = load_graduated_skills("vasp", skills_root=tmp_path)
        assert len(skills) == 1
        sk = skills[0]
        assert sk["name"] == "my_rule"
        # Description came from frontmatter.
        assert sk["meta"]["description"] == "test rule for graduation"

    def test_skips_dotted_and_underscored_dirs(self, tmp_path):
        # Make a real graduated skill alongside two ignored dirs.
        graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="real",
            domain="vasp",
            llm_call=_make_fake_llm(VALID_SKILL_MD),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        # These shouldn't get loaded.
        (tmp_path / "vasp" / "_internal").mkdir()
        (tmp_path / "vasp" / ".cache").mkdir()
        skills = load_graduated_skills("vasp", skills_root=tmp_path)
        assert len(skills) == 1
        assert skills[0]["name"] == "real"


class TestFormatGraduatedSkillsBlock:
    def test_empty_list_returns_empty_string(self):
        # Caller should be able to unconditionally concatenate the result.
        assert format_graduated_skills_block([]) == ""

    def test_renders_skill_with_description(self):
        skill = {
            "name": "demo",
            "meta": {"description": "demo rule for testing"},
            "planning": "Do X.",
            "implementation": "Set Y = Z.",
            "validation": "Verify W.",
        }
        block = format_graduated_skills_block([skill])
        assert "## LEARNED RULES" in block
        assert "### demo" in block
        assert "demo rule for testing" in block
        assert "#### planning" in block
        assert "#### implementation" in block
        assert "#### validation" in block

    def test_section_filter(self):
        skill = {
            "name": "demo",
            "meta": {"description": "x"},
            "planning": "P",
            "implementation": "I",
            "validation": "V",
        }
        block = format_graduated_skills_block(
            [skill], sections=["planning"],
        )
        assert "#### planning" in block
        # Only requested sections rendered.
        assert "#### implementation" not in block
        assert "#### validation" not in block
