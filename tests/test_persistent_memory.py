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


# ──────────────────────────────────────────────────────────────
# Meta-agent review tool (PR3b)
# ──────────────────────────────────────────────────────────────

class TestMetaReviewTool:
    def _seed_provisional(self):
        from scilink.skills._shared._graduation import graduate_to_skill_file
        graduate_to_skill_file(
            knowledge_entry={"summary": "s"},
            skill_name="auto_demo_x", domain="curve_fitting",
            llm_call=lambda p: json.dumps({"description": "d", "overview": "o", "analysis": "recipe"}),
            fresh_template="{skill_name}{domain}{knowledge_text}",
            update_template="{skill_name}{existing_skill}{new_knowledge}",
            extra_meta={"provisional": True, "provenance": "t2_autodistill", "r_squared": 0.99},
        )

    def _tools(self):
        import types
        from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
        # Closures only touch the orchestrator at call time, not registration.
        return MetaOrchestratorTools(types.SimpleNamespace())

    def test_review_tool_registered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        assert "review_distilled_skills" in self._tools().functions_map

    def test_list_show_promote_discard(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        self._seed_provisional()
        fn = self._tools().functions_map["review_distilled_skills"]

        listed = json.loads(fn(action="list"))
        assert len(listed["provisional_skills"]) == 1

        shown = json.loads(fn(action="show", skill="curve_fitting/auto_demo_x"))
        assert "recipe" in shown["markdown"]

        promoted = json.loads(fn(action="promote", skill="curve_fitting/auto_demo_x"))
        assert promoted["status"] == "success"
        assert _memory.list_memory(provisional=True) == []
        assert len(_memory.list_memory(provisional=False)) == 1

        discarded = json.loads(fn(action="discard", skill="curve_fitting/auto_demo_x"))
        assert discarded["status"] == "success"
        assert _memory.list_memory() == []

    def test_meta_tools_import_is_ase_free(self):
        import subprocess, sys, textwrap
        script = textwrap.dedent("""
            import sys
            class B:
                def find_spec(self, n, p, t=None):
                    if n == 'ase' or n.startswith('ase.'): raise ImportError('blocked')
                    return None
            sys.meta_path.insert(0, B())
            from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
            assert 'ase' not in sys.modules
            print('OK')
        """)
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert p.returncode == 0 and "OK" in p.stdout, p.stderr


# ──────────────────────────────────────────────────────────────
# Hot-deviation stamping (controllers) — makes a hot success "novel"
# ──────────────────────────────────────────────────────────────

class TestHotDeviationStamp:
    def test_curve_fitting_stamp(self):
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
            UnifiedSeriesProcessingController as U,
        )
        c = U.__new__(U)
        hot = len(U._CONSTRAINT_ANNEALING_SCHEDULE) - 1
        # hot success, no note -> synthesized
        r = {"success": True, "model_type": "2 Voigt", "_produced_at_level": hot}
        c._stamp_hot_deviation(r)
        assert r.get("deviation_note")
        # existing note preserved
        r2 = {"success": True, "_produced_at_level": hot, "deviation_note": "LLM note"}
        c._stamp_hot_deviation(r2)
        assert r2["deviation_note"] == "LLM note"
        # T=0 best -> no false positive
        r3 = {"success": True, "_produced_at_level": 0}
        c._stamp_hot_deviation(r3)
        assert not r3.get("deviation_note")
        # no level recorded / failed -> no note
        assert not _stamped(c, {"success": True})
        assert not _stamped(c, {"success": False, "_produced_at_level": hot})

    def test_image_stamp(self):
        from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
            UnifiedImageProcessingController as U,
        )
        c = U.__new__(U)
        hot = len(U._CONSTRAINT_ANNEALING_SCHEDULE) - 1
        r = {"success": True, "analysis_type": "atom-finder", "_produced_at_level": hot}
        c._stamp_hot_deviation(r)
        assert r.get("plan_deviation_summary")
        r0 = {"success": True, "_produced_at_level": 0}
        c._stamp_hot_deviation(r0)
        assert not r0.get("plan_deviation_summary")


def _stamped(controller, result):
    controller._stamp_hot_deviation(result)
    return result.get("deviation_note")


# ──────────────────────────────────────────────────────────────
# Image-agent T=2 auto-distill (mirror of curve fitting)
# ──────────────────────────────────────────────────────────────

class TestImageT2Distill:
    class _Model:
        def generate_content(self, contents=None, generation_config=None, safety_settings=None):
            import types
            r = types.SimpleNamespace()
            r.text = json.dumps({
                "description": "generalized image recipe", "overview": "ov",
                "planning": "pl", "analysis": "GENERALIZED_IMG_BODY",
                "interpretation": "in", "validation": "va",
            })
            return r

    def _fake_agent(self, home, model):
        import logging
        import types
        from pathlib import Path
        a = types.SimpleNamespace()
        a.model = model
        a.logger = logging.getLogger("imgt2")
        a.output_dir = Path(home) / "imgsess"
        a.output_dir.mkdir(parents=True, exist_ok=True)
        a.generation_config = None
        a.safety_settings = None
        return a

    def _hot_state(self):
        return {
            "analysis_approach": "threshold + watershed",
            "skills_loaded": [],
            "series_results": [{
                "index": 0, "name": "img0", "success": True,
                "analysis_type": "atom segmentation + RDF",
                "plan_deviation_summary": "switched to watershed after threshold failed",
                "script": "import numpy as np\n# IMG_VERBATIM_MARKER\nprint('seg')\n",
                "quality_history": {"final_score": 0.93, "verification_iterations": [
                    {"annealing_level": 0}, {"annealing_level": 2}]},
            }],
        }

    def test_distills_provisional_with_generalized_and_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
        from scilink.skills._shared import _memory
        agent = self._fake_agent(str(tmp_path), self._Model())
        out = ImageAnalysisAgent._maybe_distill_t2_skills(agent, self._hot_state())
        assert out
        rows = _memory.list_memory(provisional=True)
        assert len(rows) == 1 and rows[0]["domain"] == "image_analysis"
        parsed = loader.load_skill(rows[0]["path"], domain="image_analysis")
        assert "GENERALIZED_IMG_BODY" in parsed["analysis"]
        impl = parsed["implementation"]
        assert "IMG_VERBATIM_MARKER" in impl and "Reference implementation" in impl
        assert parsed["meta"].get("provisional") is True
        assert parsed["meta"].get("quality_score") == 0.93

    def test_no_distill_when_not_hot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        state = self._hot_state()
        state["series_results"][0]["quality_history"]["verification_iterations"] = [
            {"annealing_level": 0}, {"annealing_level": 1}]
        assert ImageAnalysisAgent._maybe_distill_t2_skills(agent, state) == []

    def test_distill_failure_is_isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

        class Boom:
            def generate_content(self, *a, **k):
                raise RuntimeError("boom")

        agent = self._fake_agent(str(tmp_path), Boom())
        assert ImageAnalysisAgent._maybe_distill_t2_skills(agent, self._hot_state()) == []
