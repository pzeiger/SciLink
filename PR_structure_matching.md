# PR draft — `structure-matching` branch

Copy-paste the body below when ready to open a PR against `main`. Branch is `structure-matching`; commits are already pushed (see `git log main..structure-matching`).

Recommended PR title:

```
structure_matching: XRD phase identification skill (fast triage + robust confirmation)
```

---

## Summary

Adds a new skill bundle, `structure_matching/xrd`, that gives SciLink the ability to **identify a crystalline phase from an experimental XRD pattern by matching against crystal-structure databases**. The workflow is the one an XRD scientist would run by hand: query one or more databases for candidate structures, simulate each candidate's diffraction pattern, score against the experiment, rank.

The skill is **multi-source from day one** via a `StructureBackend` protocol that the dispatching tool iterates over. v1 ships two concrete backends — **Materials Project** (`mp-api`) and **local CIF directory** (a folder of `.cif` files on disk, set via `SCILINK_LOCAL_CIF_DIR`) — plus an explicit **COD** stub. Adding ICSD, OQMD, AFLOW, NOMAD, or any other source later is "write a class that implements `is_available()` + `query(QuerySpec) → list[StructureCandidate]` and register it in `_BACKEND_FACTORY`" — no changes to the skill markdown, the tool surface, the dispatch loop, or the dedup logic. The protocol-level dedup already handles cross-source duplicates and keeps the more authoritative entry.

Two scoring tiers, both shipped together:

- **Fast triage** — cross-correlation of the broadened simulated pattern against the experiment, fitting zero-shift and lattice scale jointly. About 10–50 ms per candidate. Suitable for ranking N candidates while a measurement is still running.
- **Robust confirmation** — peak-list scoring with two selectable algorithms:
  - `'hanawalt'` (default) — classical Hanawalt / Fink figure-of-merit search-match, the same shape used in commercial XRD packages (Jade, HighScore, EVA).
  - `'mip'` — mixed-integer linear programming via PuLP. Jointly optimizes peak assignment + zero-shift + lattice scale; reports the fitted shift and scale as parameters. Use when you want a provably optimal tie-breaker, suspect a multi-phase mixture, or need the shift / scale numbers for downstream Rietveld initialization.

The two tiers are designed to run together: fast for triage, robust for the actual identification call. A worked example from the skill markdown:

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

Cross-source dedup is centralized in the dispatcher (matched on reduced formula + space group; preference order is configurable). The `search_structures` tool returns a single merged, ranked candidate list regardless of how many backends contributed.

## Why two tiers instead of one

SciLink is meant for real lab data. A naive single-scorer (e.g., profile R-factor on broadened patterns) makes for clean synthetic benchmarks but folds zero-shift and lattice-parameter discrepancies into the residual — kills it on real diffractometer output where 0.1–0.2° sample-displacement zero-shifts are routine. The fast tier handles shift + scale natively via the correlation peak; the robust tier reports them as fitted parameters when MIP is used.

## Install

The skill ships with three new dependencies declared as an extra so existing SciLink installs are unaffected:

```
pip install -e .[structure-matching]
```

This pulls in `pymatgen` (full, for the XRD analysis module), `mp-api`, and `pulp` (for the MIP solver, bundles CBC).

`MP_API_KEY` must be set in the environment for Materials Project queries; the local CIF backend works without it.

## What to test (reviewer checklist)

- [ ] **Offline tests pass**: `python -m pytest tests/test_structure_backends.py tests/test_structure_matching_tools.py tests/test_extract_peaks.py tests/test_score_match_robust.py tests/test_curve_fitting_stdout_parser.py` — 81 tests, ~0 skipped after `pip install -e .[structure-matching]`
- [ ] **Live smoke test on a synthetic Si pattern**: `UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=… MP_API_KEY=… python -m pytest tests/test_structure_matching_smoke_live.py -v -s`
- [ ] **Try it on a real diffractogram** of your choosing — the skill is invoked via `scilink analyze --skill xrd <pattern.csv>` with system_info carrying the wavelength and (if known) chemistry hint.

---

## Technical details (for reviewers / framework maintainers)

This section is more internals-flavored than the rest of the PR. Skip it if you're just looking to use the feature.

**Skill bundle layout** at `scilink/skills/structure_matching/`:

- `_backends/` — private subpackage, invisible to skill discovery (leading underscore filters it out in `loader.py:117-118`). Contains `_base.py` (StructureBackend protocol + QuerySpec + StructureCandidate dataclasses), `materials_project.py`, `local_cif.py`, `cod.py` (stub).
- `xrd/` — the v1 anchor skill. Markdown (`xrd.md`) plus five `TOOL_SPEC`-registered helpers (`search_structures.py`, `simulate_xrd.py`, `score_match_fast.py`, `extract_peaks.py`, `score_match_robust.py`). The registry walker (`_registry.py:122-148`) picks them up automatically when `'xrd'` is in `active_skills`.

**One framework touchpoint**: `curve_fitting_controllers.py:_parse_script_markers` (extracted from an inline loop in `UnifiedSeriesProcessingController._fit_single_spectrum`). The existing `FIT_RESULTS_JSON:` parser was lifted into a module-level helper and extended to recognize one additional stdout marker, `DB_MATCHES_JSON:`. The marker is emitted by `search_structures` and lands at `fit_results['db_matches']` via `setdefault` (so a script that already embeds db_matches inside FIT_RESULTS_JSON wins). Preserves existing first-wins semantics exactly. 10 unit tests in `tests/test_curve_fitting_stdout_parser.py` cover both markers in either order, malformed JSON paths, and setdefault precedence.

**No other framework code changes** — no new controllers, no pipeline factory edits, no agent `__init__` changes, no planner / synthesis JSON-contract extensions. The existing `UnifiedSeriesProcessingController` per-item quality-verification loop handles "fit is poor → try a different candidate" automatically because the LLM re-plans on retry and sees the same tool registry.

**MIP formulation** (`score_match_robust.py:_solve_mip_for_scale`): a two-pass approach to sidestep a CBC presolve interaction with bilinear `r ≤ tol * x` residual constraints that produced bound-violating "optimal" solutions in an earlier formulation. Pass 1 is a pure assignment MILP maximizing `Σ x[i,j]` subject to unique assignment + position-tolerance cone (big-M-encoded `|raw - shift| ≤ tol` when `x[i,j] = 1`); Pass 2 takes the L1 median of `raw` over matched pairs (clipped to `[shift_lo, shift_hi]`) as the optimal shift, then drops any matched pair whose post-median residual exceeds tol. Cost is computed post-solve as `(Σ |raw - shift| + tol * n_unmatched) / (tol * n_exp)`, giving a normalized score in [0, 1].

**Scale handling**: pymatgen's XRD calculator is single-crystal kinematic at scale = 1, but real-lab lattice parameters drift ±0.5% from MP's DFT-relaxed values. Both tiers scan a small lattice-scale grid (default 9 scales over [0.99, 1.01]) and report the best per candidate. CBC solves each per-scale MILP in tens of ms for the test peak counts.

**Existing utilities reused, not reimplemented**: `MaterialsProjectHelper` at `sim_agents/utils.py:116` (MPRester + per-query cache; `MaterialsProjectBackend` is a thin adapter); `APIKeyManager.get_key('materials_project')` for auth (UI sidebar input already exists at `ui/components/sidebar.py:226`); the `ToolSpec` registry; `_get_skill_context(section=...)` for prompt injection.

**Test count**: 81 offline tests + 1 live smoke test. Existing `test_lit_*` and curve-fitting tests pass unchanged.

**Out of scope (follow-up branches)**: COD backend implementation; OQMD / AFLOW / NOMAD backends; EELS / HRTEM / EBSD sibling skills under the `structure_matching` domain; `~/.scilink/config.toml` persistence tier and skill-driven sidebar form; cross-mode delegation to simulate-mode for DFT-grade simulation; `AdaptiveRefitController` extension for explicit candidate iteration (current per-item retry handles it via LLM re-prompting); Rietveld refinement / quantitative phase fractions.

🤖 Co-authored with Claude Code
