"""Tests for the persistent-memory store (PR1).

Covers:
  - the loader's discovery of the persistent graduated-skills store
    (precedence ordering, de-dup, SCILINK_HOME relocation);
  - the graduation helper's extra-meta (provisional) passthrough and
    its byte-identical legacy output for description-only callers;
  - the ase-free import guarantee for the relocated helper;
  - the sim-side back-compat shim.

No real LLM calls — the graduation helper takes the LLM as a callable.
"""

import json
import sys

from scilink.skills import loader
from scilink.skills._shared._graduation import (
    format_skill_as_markdown,
    graduate_to_skill_file,
)


def _fake_llm(response: str):
    def _fn(prompt: str) -> str:
        return response
    return _fn


VALID_JSON = json.dumps({
    "description": "distilled curve-fit recipe",
    "overview": "ov",
    "implementation": "recipe\n\n```python\nprint(1)\n```",
})


# ──────────────────────────────────────────────────────────────
# Loader: persistent store discovery
# ──────────────────────────────────────────────────────────────

class TestSkillRootsDiscovery:
    def test_home_redirect_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        assert loader.graduated_skills_dir() == tmp_path / "graduated_skills"

    def test_graduated_dir_appears_in_roots_after_creation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.delenv("SCILINK_SKILLS_PATH", raising=False)
        gd = loader.graduated_skills_dir()
        # Absent until it exists on disk.
        assert gd.resolve() not in [r.resolve() for r in loader._skill_roots()]
        (gd / "curve_fitting" / "foo").mkdir(parents=True)
        (gd / "curve_fitting" / "foo" / "foo.md").write_text(
            "---\ndescription: t\n---\n## overview\nhi\n"
        )
        roots = [r.resolve() for r in loader._skill_roots()]
        assert gd.resolve() in roots
        assert "foo" in loader.list_all_skills().get("curve_fitting", [])

    def test_precedence_order_env_then_graduated_then_builtin(self, tmp_path, monkeypatch):
        env_root = tmp_path / "env_skills"
        (env_root / "curve_fitting" / "x").mkdir(parents=True)
        (env_root / "curve_fitting" / "x" / "x.md").write_text("---\ndescription: e\n---\n## overview\ne\n")
        home = tmp_path / "home"
        gd = home / "graduated_skills"
        (gd / "curve_fitting" / "y").mkdir(parents=True)
        (gd / "curve_fitting" / "y" / "y.md").write_text("---\ndescription: g\n---\n## overview\ng\n")

        monkeypatch.setenv("SCILINK_HOME", str(home))
        monkeypatch.setenv("SCILINK_SKILLS_PATH", str(env_root))

        roots = [r.resolve() for r in loader._skill_roots()]
        idx_env = roots.index(env_root.resolve())
        idx_grad = roots.index(gd.resolve())
        idx_builtin = roots.index(loader._SKILLS_DIR.resolve())
        assert idx_env < idx_grad < idx_builtin

    def test_dedupes_when_home_also_in_skills_path(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        gd = home / "graduated_skills"
        (gd / "curve_fitting" / "y").mkdir(parents=True)
        (gd / "curve_fitting" / "y" / "y.md").write_text("---\ndescription: g\n---\n## overview\ng\n")
        monkeypatch.setenv("SCILINK_HOME", str(home))
        # Point SCILINK_SKILLS_PATH at the same graduated dir.
        monkeypatch.setenv("SCILINK_SKILLS_PATH", str(gd))
        roots = [r.resolve() for r in loader._skill_roots()]
        assert roots.count(gd.resolve()) == 1


# ──────────────────────────────────────────────────────────────
# Graduation: provisional / extra-meta passthrough
# ──────────────────────────────────────────────────────────────

class TestExtraMeta:
    def test_format_passthrough_allowlist(self):
        out = format_skill_as_markdown({
            "description": "d",
            "provisional": True,
            "provenance": "t2_autodistill",
            "r_squared": 0.97,
            "not_allowed": "should be dropped",
            "overview": "o",
        })
        assert "provisional: true" in out
        assert "provenance: t2_autodistill" in out
        assert "r_squared: 0.97" in out
        assert "not_allowed" not in out

    def test_description_only_is_byte_identical_to_legacy(self):
        # The legacy formatter emitted exactly this for a description-only
        # input; the allowlist must not perturb it.
        out = format_skill_as_markdown({"description": "d", "overview": "o"})
        assert out == "---\ndescription: d\n---\n\n## overview\n\no\n"

    def test_graduate_writes_provisional_frontmatter(self, tmp_path):
        result = graduate_to_skill_file(
            knowledge_entry={"summary": "x", "script": "print(1)"},
            skill_name="auto_test",
            domain="curve_fitting",
            llm_call=_fake_llm(VALID_JSON),
            fresh_template="{skill_name} {domain} {knowledge_text}",
            update_template="{skill_name} {existing_skill} {new_knowledge}",
            skills_root=tmp_path,
            extra_meta={"provisional": True, "provenance": "t2_autodistill", "r_squared": 0.97},
        )
        parsed = loader.load_skill(result["skill_path"], domain="curve_fitting")
        assert parsed["meta"].get("provisional") is True
        assert parsed["meta"].get("provenance") == "t2_autodistill"
        assert "print(1)" in parsed["implementation"]


# ──────────────────────────────────────────────────────────────
# ase-free import + shim back-compat
# ──────────────────────────────────────────────────────────────

class TestImportSafety:
    def test_graduation_helper_is_ase_free(self):
        """The relocated helper must import without ase, which the sim
        package hard-requires. Run in a subprocess that blocks ase, so we
        don't pollute this interpreter's module cache."""
        import subprocess
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            class Blocker:
                def find_spec(self, name, path, target=None):
                    if name == 'ase' or name.startswith('ase.'):
                        raise ImportError('ase blocked')
                    return None
            sys.meta_path.insert(0, Blocker())
            from scilink.skills._shared._graduation import graduate_to_skill_file
            from scilink.skills import loader
            assert 'ase' not in sys.modules, 'ase was imported'
            print('OK')
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_sim_shim_reexports_helper(self):
        from scilink.agents.sim_agents.skill_graduation import (
            graduate_to_skill_file as shim_fn,
        )
        from scilink.skills._shared._graduation import graduate_to_skill_file as real_fn
        assert shim_fn is real_fn


# ──────────────────────────────────────────────────────────────
# Routing: provisional skills excluded from the run_analysis menu
# ──────────────────────────────────────────────────────────────

class TestRoutingFilter:
    def _make_skill(self, gd, name, *, provisional):
        d = gd / "curve_fitting" / name
        d.mkdir(parents=True)
        fm = "description: a test skill\n"
        if provisional:
            fm += "provisional: true\n"
        (d / f"{name}.md").write_text(f"---\n{fm}---\n\n## overview\nbody\n")

    def test_provisional_excluded_but_loadable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.delenv("SCILINK_SKILLS_PATH", raising=False)
        gd = loader.graduated_skills_dir()
        self._make_skill(gd, "prov_skill", provisional=True)
        self._make_skill(gd, "normal_skill", provisional=False)

        from scilink.agents.exp_agents.analysis_orchestrator_tools import (
            _build_skill_description,
        )
        blurb = _build_skill_description()
        assert "normal_skill" in blurb
        assert "prov_skill" not in blurb

        # Still explicitly loadable.
        parsed = loader.load_skill("prov_skill", domain="curve_fitting")
        assert parsed["meta"].get("provisional") is True


# ──────────────────────────────────────────────────────────────
# T=2 auto-distill hook
# ──────────────────────────────────────────────────────────────

class TestT2DistillHook:
    def _fake_agent(self, home, model):
        import logging
        import types
        from pathlib import Path

        a = types.SimpleNamespace()
        a.model = model
        a.logger = logging.getLogger("t2test")
        a.output_dir = Path(home) / "sess"
        a.output_dir.mkdir(parents=True, exist_ok=True)
        a.generation_config = None
        a.safety_settings = None
        return a

    def _hot_state(self):
        return {
            "locked_fitting_config": {"physical_model": "3 Gaussians", "fitting_strategy": "seq"},
            "skills_loaded": [],
            "series_results": [{
                "index": 0, "name": "s0", "success": True,
                "model_type": "2 Voigt + exp tail",
                "fit_quality": {"r_squared": 0.991},
                "deviation_note": "switched model",
                "script": "import numpy as np\n# VERBATIM_MARKER\nprint('fit')\n",
                "quality_history": {"verification_iterations": [
                    {"annealing_level": 0}, {"annealing_level": 2}]},
            }],
        }

    class _Model:
        def generate_content(self, contents=None, generation_config=None, safety_settings=None):
            import types
            r = types.SimpleNamespace()
            r.text = json.dumps({
                "description": "generalized recipe",
                "overview": "ov", "planning": "pl",
                "analysis": "GENERALIZED_BODY",
                "interpretation": "in", "validation": "va",
            })
            return r

    def test_distills_provisional_with_generalized_and_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        from scilink.skills._shared import _memory

        agent = self._fake_agent(str(tmp_path), self._Model())
        out = CurveFittingAgent._maybe_distill_t2_skills(agent, self._hot_state())
        assert out, "expected a distilled skill"

        rows = _memory.list_memory(provisional=True)
        assert len(rows) == 1
        parsed = loader.load_skill(rows[0]["path"], domain="curve_fitting")
        impl = parsed["implementation"]
        assert "GENERALIZED_BODY" in parsed["analysis"]      # generalized recipe
        assert "VERBATIM_MARKER" in impl                      # verbatim appendix
        assert "Reference implementation" in impl
        assert parsed["meta"].get("provisional") is True
        assert parsed["meta"].get("provenance") == "t2_autodistill"

    def test_no_distill_when_not_hot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        state = self._hot_state()
        state["series_results"][0]["quality_history"]["verification_iterations"] = [
            {"annealing_level": 0}, {"annealing_level": 1}]
        assert CurveFittingAgent._maybe_distill_t2_skills(agent, state) == []

    def test_distill_failure_is_isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

        class Boom:
            def generate_content(self, *a, **k):
                raise RuntimeError("boom")

        agent = self._fake_agent(str(tmp_path), Boom())
        # Must not raise; returns no skills.
        assert CurveFittingAgent._maybe_distill_t2_skills(agent, self._hot_state()) == []

    def test_opt_out_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.setenv("SCILINK_T2_AUTODISTILL", "0")
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        assert CurveFittingAgent._maybe_distill_t2_skills(agent, self._hot_state()) == []
