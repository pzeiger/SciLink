"""
Tests for AMBER skill loading and integration with ForceFieldAgent.
No AmberTools or LLM API key required.
"""

import pytest
import os

# ─── Skill loader tests ──────────────────────────────────────────

class TestSkillLoader:
    """Test that the skill file loads and parses correctly."""

    def test_amber_skill_is_discoverable(self):
        """The amber skill should appear in list_skills('force_field')."""
        from scilink.skills.loader import list_skills
        skills = list_skills(domain="force_field")
        assert "amber" in skills, f"Expected 'amber' in {skills}"

    def test_amber_skill_loads(self):
        """load_skill('amber', 'force_field') should return all sections."""
        from scilink.skills.loader import load_skill
        skill = load_skill("amber", domain="force_field")

        assert skill["name"] == "amber"

        expected_sections = ["overview", "planning", "analysis",
                            "interpretation", "validation", "implementation"]
        for section in expected_sections:
            assert section in skill, f"Missing section: {section}"
            assert len(skill[section]) > 0, f"Section '{section}' is empty"

    def test_amber_skill_overview_content(self):
        """Overview should mention AMBER and key tools."""
        from scilink.skills.loader import load_skill
        skill = load_skill("amber", domain="force_field")
        overview = skill["overview"].lower()

        assert "amber" in overview
        assert "ambertools" in overview or "antechamber" in overview
        assert "parmed" in overview or "parmtop" in overview
        assert "lammps" in overview

    def test_amber_skill_planning_has_tables(self):
        """Planning section should have FF selection guidance."""
        from scilink.skills.loader import load_skill
        skill = load_skill("amber", domain="force_field")
        planning = skill["planning"].lower()

        # Should mention key force fields
        assert "ff19sb" in planning
        assert "ff14sb" in planning
        assert "gaff2" in planning or "gaff" in planning

        # Should mention water models
        assert "tip3p" in planning
        assert "opc" in planning

    def test_amber_skill_validation_has_ranges(self):
        """Validation section should include parameter sanity ranges."""
        from scilink.skills.loader import load_skill
        skill = load_skill("amber", domain="force_field")
        validation = skill["validation"].lower()

        assert "epsilon" in validation or "sigma" in validation
        assert "charge" in validation
        assert "kcal/mol" in validation or "kcal" in validation

    def test_amber_skill_implementation_has_pipeline(self):
        """Implementation section should describe the full pipeline."""
        from scilink.skills.loader import load_skill
        skill = load_skill("amber", domain="force_field")
        impl = skill["implementation"].lower()

        assert "antechamber" in impl
        assert "tleap" in impl
        assert "parmed" in impl or "parmtop" in impl

    def test_list_all_skills_includes_force_field(self):
        """list_all_skills() should include the force_field domain."""
        from scilink.skills.loader import list_all_skills
        all_skills = list_all_skills()

        assert "force_field" in all_skills
        assert "amber" in all_skills["force_field"]


# ─── Agent skill integration tests (no LLM needed) ──────────────

class TestAgentSkillIntegration:
    """Test that ForceFieldAgent correctly loads and uses skills.
    These tests mock the LLM to avoid needing an API key."""

    @pytest.fixture
    def agent_with_skill(self, tmp_path):
        """Create an agent with the amber skill pre-loaded, mocking the LLM."""
        from unittest.mock import MagicMock, patch

        # Mock the LLM model so we don't need an API key
        with patch('scilink.agents.sim_agents.force_field_agent.normalize_params',
                   return_value=("fake_key", None)):
            with patch('scilink.agents.sim_agents.force_field_agent.LiteLLMGenerativeModel'):
                agent = ForceFieldAgent.__new__(ForceFieldAgent)
                agent.working_dir = str(tmp_path)
                os.makedirs(agent.working_dir, exist_ok=True)

                # Set up logging
                import logging
                agent.logger = logging.getLogger("test_agent")
                agent.logger.setLevel(logging.INFO)

                # Mock model
                agent.model = MagicMock()
                agent.generation_config = None

                # Build mass lookup
                agent._mass_to_element = agent._build_mass_lookup()

                # Initialize skill state (PR 3: self.skills is the canonical
                # list; skill_name / skill_sections are read-only properties).
                agent.skills = []
                try:
                    from scilink.skills.loader import list_skills
                    agent._available_ff_skills = list_skills(domain="force_field")
                except Exception:
                    agent._available_ff_skills = []

                # Load the skill
                agent._load_skill("amber")

                return agent

    def test_skill_loaded(self, agent_with_skill):
        """After _load_skill('amber'), skill state should be set."""
        agent = agent_with_skill
        assert agent.skill_name == "amber"
        assert agent.skill_sections is not None
        assert "overview" in agent.skill_sections

    def test_get_skill_context_overview(self, agent_with_skill):
        """_get_skill_context() should return overview by default."""
        agent = agent_with_skill
        context = agent._get_skill_context()
        assert "AMBER" in context or "amber" in context.lower()
        assert len(context) > 100

    def test_get_skill_context_section(self, agent_with_skill):
        """_get_skill_context(section='planning') should return planning."""
        agent = agent_with_skill
        context = agent._get_skill_context(section="planning")
        assert "ff19sb" in context.lower() or "ff14sb" in context.lower()

    def test_get_skill_context_all(self, agent_with_skill):
        """_get_skill_context(include_all=True) should return everything."""
        agent = agent_with_skill
        context = agent._get_skill_context(include_all=True)
        assert "PLANNING" in context or "planning" in context.lower()
        assert "VALIDATION" in context or "validation" in context.lower()
        assert "IMPLEMENTATION" in context or "implementation" in context.lower()

    def test_get_skill_context_empty_when_no_skill(self, tmp_path):
        """_get_skill_context() should return '' when no skill is loaded."""
        from unittest.mock import MagicMock
        agent = MagicMock()
        agent.skills = []

        # Call the real method
        from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent
        result = ForceFieldAgent._get_skill_context(agent)
        assert result == ""

    def test_auto_select_skill_amber(self, agent_with_skill):
        """_auto_select_skill should match AMBER FF names."""
        agent = agent_with_skill
        # Reset skill state
        agent.skills = []

        assert agent._auto_select_skill("AMBER ff19SB") is True
        assert agent.skill_name == "amber"

    def test_auto_select_skill_gaff(self, agent_with_skill):
        agent = agent_with_skill
        agent.skills = []

        assert agent._auto_select_skill("GAFF2") is True
        assert agent.skill_name == "amber"

    def test_auto_select_skill_non_amber(self, agent_with_skill):
        agent = agent_with_skill
        agent.skills = []

        # OPLS shouldn't match amber skill
        result = agent._auto_select_skill("OPLS-AA")
        # Depends on whether other skills exist
        assert agent.skill_name != "amber" or result is False

    def test_is_amber_force_field(self, agent_with_skill):
        agent = agent_with_skill
        assert agent._is_amber_force_field("AMBER ff19SB") is True
        assert agent._is_amber_force_field("GAFF2") is True
        assert agent._is_amber_force_field("ff14SB") is True
        assert agent._is_amber_force_field("OPLS-AA") is False
        assert agent._is_amber_force_field("CHARMM36m") is False
        assert agent._is_amber_force_field("") is False

    def test_resolve_protein_ff(self, agent_with_skill):
        agent = agent_with_skill
        assert agent._resolve_protein_ff("AMBER ff19SB") == "ff19SB"
        assert agent._resolve_protein_ff("ff14sb") == "ff14SB"
        assert agent._resolve_protein_ff("ff19sb") == "ff19SB"
        assert agent._resolve_protein_ff("CustomFF") == "CustomFF"


# ─── Import the agent for test fixtures ──────────────────────────

try:
    from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent
except ImportError:
    # Allow test collection even if full import fails
    ForceFieldAgent = None
    pytestmark = pytest.mark.skip("Could not import ForceFieldAgent")
