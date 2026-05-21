# PR draft — `structure-matching` branch

Copy-paste the body below when ready to open a PR against `main`. Branch is `structure-matching`; 29 commits, all pushed.

Recommended PR title:

```
structure_matching: XRD phase identification skill + configurable framework quality gate
```

---

## Summary

Adds a new skill bundle, `structure_matching/xrd`, that gives SciLink the ability to **identify a crystalline phase from an experimental XRD pattern by matching against crystal-structure databases**. The workflow is the one an XRD scientist runs by hand: query one or more databases for candidate structures, simulate each candidate's diffraction pattern, score against the experiment, rank.

The skill is **multi-source from day one** via a `StructureBackend` protocol. v1 ships two concrete backends — **Materials Project** (`mp-api`) and **local CIF directory** (a folder of `.cif` files on disk, set via `SCILINK_LOCAL_CIF_DIR`) — plus an explicit **COD** stub. Adding ICSD, OQMD, AFLOW, NOMAD, or any custom source later is a one-class change: implement `is_available()` + `query(QuerySpec) → list[StructureCandidate]` and either call `register_backend(name, factory)` or declare an entry-point in your `pyproject.toml`. SciLink discovers it on import.

Two scoring tiers ship together:

- **Fast triage** (`score_xrd_match_fast`) — cross-correlation of the broadened simulated pattern against the experiment, fitting zero-shift and lattice scale jointly. ~50 ms per candidate. Suitable for ranking N candidates while a measurement is still running.
- **Robust confirmation** (`score_xrd_match_robust`) with two algorithm choices:
  - `'hanawalt'` (default) — classical Hanawalt / Fink figure-of-merit search-match, the same shape used in commercial XRD packages (Jade, HighScore, EVA). Coverage is intensity-weighted so weak noise peaks don't drag the FOM on otherwise-clean spectra.
  - `'mip'` — mixed-integer linear programming via PuLP. Jointly optimizes peak assignment + zero-shift + lattice scale; reports the fitted shift and scale as parameters. Use when you want a provably optimal tie-breaker, suspect a multi-phase mixture, or need the shift / scale numbers for downstream Rietveld initialization.

The two tiers are designed to run together: fast for triage, robust for the actual identification call. Worked example from the skill markdown:

```python
from scilink.skills.structure_matching.xrd.search_structures import search_structures
from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
from scilink.skills.structure_matching.xrd.score_match_fast import score_xrd_match_fast
from scilink.skills.structure_matching.xrd.extract_peaks import extract_peaks
from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_robust

# 1. Query whichever databases are available (auto-detected from env)
hits = search_structures(query={"chemistry": ["Ti", "O"], "top_n": 5})

# 2. Fast tier — rank candidates
ranked = []
for cand in hits["candidates"]:
    sim = simulate_xrd_pattern(cand["structure_path"], wavelength="CuKa")
    fast = score_xrd_match_fast(exp_2theta, exp_intensity,
                                sim["two_theta"], sim["intensities"])
    ranked.append((cand, sim, fast))

# 3. Robust tier — confirm on the survivors
exp_peaks = extract_peaks(exp_2theta, exp_intensity)
for cand, sim, fast in ranked:
    if fast["verdict"] in {"accept", "marginal"}:
        robust = score_xrd_match_robust(
            sim_two_theta=sim["two_theta"], sim_intensity=sim["intensities"],
            exp_peaks=exp_peaks, algorithm="hanawalt",
        )
        print(cand["formula"], cand["space_group"], robust["verdict"], robust["figure_of_merit"])
```

For the **no-chemistry-hint** workflow, `search_structures` now accepts a list-of-lists chemistry so multiple element hypotheses can be tried in one call (instead of looping over single-hypothesis calls):

```python
search_structures(query={"chemistry": [["Si"], ["C"], ["Ge"], ["Ti", "O"]]})
```

## What's also new at the framework level

This branch introduces one piece of framework-wide infrastructure that benefits every analyze-mode skill, not just XRD:

**Configurable quality gate.** Historically the framework gated every fit on `R² ≥ 0.95` — a hardcoded curve-fitting default. For workflow-style skills like `structure_matching/xrd`, the natural metric is the figure-of-merit, not R². The new `QualityGate` dataclass lets a skill declare its own framework gate via frontmatter:

```yaml
---
description: XRD structure matching ...
quality_gate:
  metric: figure_of_merit
  accept_threshold: 0.70
  hard_reject_threshold: 0.40
  direction: higher_is_better
---
```

Resolution chain (highest priority first):
1. `analyze(quality_gate=...)` — per-call Python API override
2. `CurveFittingAgent(quality_gate=...)` — agent-level default
3. Skill frontmatter `quality_gate:` block
4. Legacy `r2_threshold=` kwarg shortcut (still works)
5. Framework default — R² ≥ 0.95

**Full backward compatibility** for every skill that doesn't declare `quality_gate` (xps, etc.) and every call that doesn't pass `quality_gate=` — verified by 116 offline tests passing unchanged and a live conventional-curve-fit regression on a 2-Gaussian Raman spectrum (status=success, R²=0.996).

## Performance — `LocalCIFBackend` parquet index

Without an index, `LocalCIFBackend` was parsing every CIF on every query (~10 ms/file via pymatgen). For a curated set of a few hundred files that's fine; for a full COD bulk mirror (~500 k CIFs) it was an 80-minute query.

The new lazy parquet index pre-extracts chemistry / symmetry / lattice metadata into `~/.scilink/cache/local_cif_index_<root>.parquet`. Subsequent queries filter the DataFrame in pandas (milliseconds for 500 k rows) and only parse the top-N survivors. Incremental rebuild reuses cached rows when CIF mtimes haven't drifted. Falls back to direct-walk-and-parse when pyarrow is unavailable.

## Install

The skill ships with its dependencies declared as an extra so existing SciLink installs are unaffected:

```
pip install -e .[structure-matching]
```

This pulls in `pymatgen` (full, for the XRD analysis module), `mp-api`, `pulp` (for the MIP solver, bundles CBC), and `pyarrow` (for the parquet index). `MP_API_KEY` must be set in the environment for Materials Project queries; the local CIF backend works without it.

## What was tested (live + offline)

**Offline:** 132 tests pass, 1 skipped. Covers QualityGate (18), CIF index (16), backend registration (12), structure backends (20), tool unit tests (26+), score_match_robust including MIP (15), extract_peaks (10), stdout parser (10), plus 5 dedicated multi-chemistry tests.

**Live (claude-opus-4-6, MP + local CIF):**
- Conventional curve-fit regression (no skill, 2-Gaussian Raman) — R²=0.996, status=success → backward compat verified.
- Edge cases (3 scenarios: clean Si with hint / no-hint discovery / anatase TiO2) — all 3 successful, all 5 tools invoked.
- Stress sweep (5 scenarios: lab combo / nanocrystalline / preferred orientation / limited range / Si+Ge overlapping phases) — 5/5 success, verdicts calibrated honestly (accept on K_limited_range FOM 0.71; marginal on H/I where compound artifacts depress the score; multi-phase L falls back to single-phase identification — a known v1 scope-out).
- Orchestrator path (`AnalysisOrchestratorAgent.run_task`) on noisy Si pattern — identified **Si Fd-3m, a = 5.4309 Å** (literature match to 4 sig figs), FOM 0.65, scientist-readable summary.

## What to test (reviewer checklist)

- [ ] **Offline tests pass after install**: `pip install -e .[structure-matching] && python -m pytest tests/ -k "structure_matching or quality_gate or cif_index or backend_registration or curve_fitting_stdout"` — 132 tests, 1 skipped.
- [ ] **Live smoke**: `UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=… MP_API_KEY=… python -m pytest tests/test_structure_matching_smoke_live.py -v -s`.
- [ ] **Try a real diffractogram**: `scilink analyze --skill xrd <pattern.csv>` with wavelength + (optional) chemistry hint in system_info; verify the agent emits a tool-driven plan, calls all five tools, and reports `figure_of_merit` + `verdict`.
- [ ] **Try with a CIF mirror**: `export SCILINK_LOCAL_CIF_DIR=~/cifs` before invocation; first query on a >200-file mirror should print a "Building CIF index" message; subsequent queries should be milliseconds.

---

## Technical details (for reviewers / framework maintainers)

Internals-flavored. Skip if you're just looking to use the feature.

**Skill bundle layout** at `scilink/skills/structure_matching/`:

- `_backends/` — private subpackage, invisible to skill discovery (leading underscore filters it out in `loader.py:117-118`). Contains `_base.py` (StructureBackend protocol + QuerySpec + StructureCandidate), `materials_project.py`, `local_cif.py`, `cod.py` (stub), `_cif_index.py` (parquet indexer), and `__init__.py` (registry: `register_backend`, `unregister_backend`, `registered_backends`, `get_backend_factory`, entry-points loader).
- `xrd/` — the v1 anchor skill. Markdown (`xrd.md`) plus five `TOOL_SPEC`-registered helpers (`search_structures.py`, `simulate_xrd.py`, `score_match_fast.py`, `extract_peaks.py`, `score_match_robust.py`). The registry walker (`_registry.py:122-148`) picks them up automatically when `'xrd'` is in `active_skills`. Cross-domain skill loading (`scilink/skills/loader.py:_resolve_skill_path`) means `CurveFittingAgent.analyze(skill='xrd')` resolves to `structure_matching/xrd/xrd.md` even though the agent defaults to the `curve_fitting` domain.

**Framework touchpoints** (additive only, no behavior changes for existing skills):

- `scilink/agents/exp_agents/quality_gate.py` — new `QualityGate` dataclass + `resolve_gate()` resolver. `R_SQUARED_DEFAULT` preserves the pre-existing R²≥0.95 / hard-reject<0.90 semantics.
- `scilink/agents/exp_agents/curve_fitting_agent.py` — adds `quality_gate=` kwarg to `__init__` and `analyze()`; resolves the effective gate from (call → agent → skill frontmatter → legacy r2_threshold → default); stashes in `state['quality_gate']` so controllers read via `_gate(state)`.
- `scilink/agents/exp_agents/controllers/curve_fitting_controllers.py` — `_verify_fit_with_llm` short-circuits to a structured accept/marginal/reject verdict when `gate.metric != 'r_squared'` (skill workflow's own scoring is the verification; R²-shaped verifier prompt would not apply). When metric IS r_squared, every existing code path runs unchanged.
- `scilink/agents/exp_agents/controllers/curve_fitting_controllers.py:_parse_script_markers` — extended to recognize `DB_MATCHES_JSON:` (emitted by `search_structures`) and `MATCH_RESULTS_JSON:` alongside the existing `FIT_RESULTS_JSON:`. Result lands at `fit_results['db_matches']` via `setdefault` (so a script that already embeds it inside FIT_RESULTS_JSON wins). Existing first-wins semantics preserved.
- `scilink/agents/exp_agents/controllers/curve_fitting_controllers.py` — both the planner (`HumanFeedbackRefinementController._plan_analysis`) and the code-gen / correction prompts (`UnifiedSeriesProcessingController._generate_script`, `_correct_script`) inject `format_tool_inventory(...)` so the LLM sees the skill's registered tools. Mirrors `image_analysis_controllers` which already did this.

**Import-path shim** (`scilink/agents/skills/__init__.py`, ~50 lines): claude-opus-4-6 routinely synthesizes `from scilink.agents.skills.<X> import ...` instead of the canonical `scilink.skills.<X>`. Rather than fight the hallucination through prompt hardening (empirically didn't stick), this shim walks `scilink.skills` via `pkgutil.walk_packages` at import time and registers every submodule under the equivalent `scilink.agents.skills.*` alias in `sys.modules`. Identity preserved (verified by `is` against canonical imports).

**MIP formulation** (`score_match_robust.py:_solve_mip_for_scale`): two-pass approach. Pass 1 is a pure assignment MILP maximizing `Σ x[i,j]` subject to unique assignment + position-tolerance cone. Pass 2 takes the L1 median of `raw` over matched pairs (clipped to `[shift_lo, shift_hi]`) as the optimal shift, then drops any matched pair whose post-median residual exceeds tol. Sidesteps a CBC presolve interaction with bilinear `r ≤ tol*x` residual constraints that produced bound-violating "optimal" solutions in an earlier formulation. Cost is computed post-solve as `(Σ |raw - shift| + tol * n_unmatched) / (tol * n_exp)`, giving a normalized score in [0, 1].

**Scale handling**: pymatgen's XRD calculator is single-crystal kinematic at scale = 1, but real-lab lattice parameters drift ±0.5% from MP's DFT-relaxed values. Both tiers scan a small lattice-scale grid (default ±2% at 0.002 step) and report the best per candidate. CBC solves each per-scale MILP in tens of ms.

**Existing utilities reused, not reimplemented**: `MaterialsProjectHelper` at `sim_agents/utils.py:116` (MPRester + per-query cache; `MaterialsProjectBackend` is a thin adapter); `APIKeyManager.get_key('materials_project')` for auth (UI sidebar input already exists at `ui/components/sidebar.py:226`); the `ToolSpec` registry; `_get_skill_context(section=...)` for prompt injection.

**Out of scope (follow-up branches)**: COD backend implementation (stub interface only in v1); OQMD / AFLOW / NOMAD backends (users wire via the registration API); EELS / HRTEM / EBSD sibling skills under the `structure_matching` domain; `~/.scilink/config.toml` persistence tier and skill-driven sidebar form; cross-mode delegation to simulate-mode for DFT-grade simulation; Rietveld refinement / quantitative phase fractions; `AdaptiveRefitController` extension for explicit candidate iteration (current per-item retry handles it via LLM re-prompting; revisit if observed traces show the LLM doesn't iterate productively).

🤖 Co-authored with Claude Code
