---
description: 'XRD phase identification (search-match) — the default first-pass XRD analysis answering "what phase(s) is this?". Queries crystal-structure databases (COD, Materials Project, local CIF), simulates kinematic patterns, and scores by cross-correlation (fast) and Hanawalt / MIP peak-matching (robust). Use this for routine phase ID; the xrd_profile skill is the specialized follow-up for line-broadening (crystallite size / strain) once the phase is known.'
quality_gate:
  metric: figure_of_merit
  accept_threshold: 0.70
  hard_reject_threshold: 0.40
  direction: higher_is_better
---
# XRD Structure Matching Skill

## overview

Identify a crystalline phase from an experimental X-ray diffraction (XRD)
pattern by matching against database structures. **This is the default,
highest-frequency XRD question — "what phase(s) is my sample?" — and the usual
first pass for any XRD pattern.** Profile fitting (the `xrd_profile` skill:
per-peak pseudo-Voigt → Scherrer crystallite size / Williamson-Hall strain) is
the *specialized follow-up* once the phase is known, not the starting point.

The skill ships five tools the analysis script chains together:

- `search_structures` — query the **COD** (Crystallography Open Database — the
  recommended default: experimental structures, organic + inorganic, no API key),
  **Materials Project** (computed inorganic; for stability ranking / predicted
  phases), and / or a local CIF directory for candidate structures (chemistry +
  symmetry filters). COD's experimental cells avoid the DFT lattice mismatch that
  MP structures carry.
- `simulate_xrd_pattern` — kinematic XRD pattern from a CIF via pymatgen
  (CuKa default; any wavelength supported).
- `score_xrd_match_fast` — **fast tier**. Cross-correlation of the
  broadened simulated pattern against the experiment, fitting zero-shift
  and lattice scale jointly. Tens of milliseconds per candidate. Use for
  on-the-fly ranking during an experiment or scout passes over many
  candidates.
- `extract_peaks` — extract a peak list from a continuous experimental
  pattern (positions, intensities, FWHMs). Needed before the robust
  tier; can also be called standalone to inspect / report peaks.
- `score_xrd_match_robust` — **robust tier**. Peak-list-based scoring
  with two algorithms: `'hanawalt'` (default, classical figure-of-merit
  search-match) or `'mip'` (mixed-integer linear programming for joint
  shift / scale / assignment optimization). Hundreds of milliseconds
  per candidate. Use for confident identification on real-lab patterns
  after the fast tier narrows the candidate list.

Install dependencies: `pip install scilink[structure-matching]` (pymatgen
with the XRD analysis module, mp-api, pulp).

**Extending the backend list.** Materials Project, local CIF, and COD
ship in-package. ICSD, OQMD, AFLOW, NOMAD, and any custom database
plug in through a small public API. A user implements a class
satisfying the `StructureBackend` protocol
(`is_available()` + `query(spec) -> list[StructureCandidate]`) and
either calls `register_backend("icsd", ICSDBackend)` from their code,
or declares it in their package's `pyproject.toml`:

```toml
[project.entry-points."scilink.structure_backends"]
icsd = "my_package.icsd_backend:ICSDBackend"
```

SciLink discovers the entry point on import and the backend is then
addressable as `sources=["icsd"]` in `search_structures`. See
`scilink/skills/structure_matching/_backends/__init__.py` for the full
protocol and the registration helpers.

## planning

**This is pattern-MATCHING, not fitting.** The deliverable is a ranked candidate
match plus a `figure_of_merit` (the skill's quality gate) — *not* a fitted model.
The plan must not list peak-shape fit parameters (pseudo-Voigt mixing `eta`,
per-peak FWHM / amplitude as fit targets); the only quantitative outputs are the
matched phase(s), the scorer's `figure_of_merit`, and its fitted zero-shift /
lattice-scale. Every stage below chains search → simulate → score; no stage fits
the experimental pattern.

**Two-tier identification workflow** — both tiers usually run, never
either-alone:

1. **Pre-fit / triage** with the **fast tier**. Query the DB for top-N
   candidates (N between 5 and 10), simulate each one's pattern, run
   `score_xrd_match_fast` on every candidate. Rank by correlation. This
   establishes which candidates are even plausible — typically 3-4
   survive the `verdict='accept'|'marginal'` filter; the others
   (`verdict='reject'`) can be dropped.

2. **Confident identification** with the **robust tier**. Extract the
   experimental peak list once with `extract_peaks`. For each surviving
   candidate from step 1, run `score_xrd_match_robust(algorithm='hanawalt')`.
   Report the best by figure-of-merit. Switch to `algorithm='mip'` when:
   - The pattern is **suspected multi-phase** (see below — this is the
     dominant trigger).
   - You need the fitted zero-shift and lattice scale reported as
     parameters (e.g., for downstream Rietveld initialization).
   - Hanawalt's top candidates are within ~10% FOM of each other and you
     want a provably optimal tie-breaker.

**Three invocation modes for step 1:**

- **Pre-fit pattern (single-phase)** — chemistry is hypothesized as one
  compound (e.g. "this is a TiO2 sample" or `chemistry_hint=["Ti","O"]`
  for the binary). Query DB with `chemistry=[Ti, O]` (single list);
  optionally add `space_group_hints` if the user names a polymorph.

- **Multi-phase mixture** — when the metadata / notes suggest a mixture
  ("Si + Ge mixture", "suspected multi-phase", `chemistry_hint=["Si","C"]`
  for two distinct elemental phases, etc.). Use the **list-of-lists**
  chemistry form to get separate candidate lists per phase:
  `chemistry=[["Si"], ["Ge"]]` — NOT `chemistry=["Si", "Ge"]` which would
  ask the DB for Si-Ge *binary compounds* instead of Si and Ge
  separately. For step 2 use **`score_xrd_match_multiphase` from the
  start** (not the per-candidate `score_xrd_match_robust` with
  `algorithm='mip'`). The multi-phase tool accepts the full list of
  candidate phase patterns and solves one joint MILP across them, with
  per-phase activation binaries that let the solver leave a phase out
  entirely. Output includes per-phase coverage and matched-peak lists
  — strictly more informative than running per-candidate MIPs and
  comparing them, which loses the cross-phase assignment constraints.
  The per-candidate `score_xrd_match_robust(algorithm='mip')` remains
  available as a fallback when only one phase's identity matters and
  the per-phase decomposition isn't needed.

- **Post-fit pattern (no chemistry hypothesis)** — no hint at all. Run
  `extract_peaks` first to estimate the dominant peak positions, infer
  an approximate lattice from the strongest peak via Bragg's law for a
  guessed crystal system, then make a **single** `search_structures`
  call with a list-of-lists chemistry hypothesis (e.g.
  `chemistry=[["Si"], ["C"], ["Ge"], ["Ti","O"]]`). The tool dispatches
  one DB query per hypothesis, dedupes, and merges results. **Never
  loop over single-chemistry `search_structures` calls** — that path
  has historically broken on consolidation, and the tool is built to
  handle multiple hypotheses in one invocation.

**Recognizing the multi-phase trigger.** Any of these phrasings in
`system_info` / notes mean "use `score_xrd_match_multiphase`":
"mixture", "multi-phase", "two-phase", "binary mixture" (as opposed to
"binary compound"), "co-existing phases", or `chemistry_hint`
containing two elements with NO compound name (e.g. `["Si", "Ge"]`
with note "suspected mixture" — the elements are distinct phases, not
a compound). When in doubt, run `score_xrd_match_multiphase` with a
single-candidate list — it gracefully reduces to the single-phase MIP
under that input (with one phase always active) and the joint solver's
output format is the same.

**Candidate count — tight first pass, widen on failure.** The cost is the
per-candidate *simulation*, so keep the FIRST pass cheap: `search_structures(
query={"top_n": N, ...})` with N in [5, 10]. That already identifies common
single-polymorph phases.

But if that pass FAILS (best `figure_of_merit` below the accept threshold /
all candidates marginal-or-reject), the answer is often a phase that the tight
retrieval simply did not return — a *polymorph-rich* chemistry (e.g. Ti-O has
many TiO2 polymorphs plus Magnéli suboxides; the right one can be candidate #20,
not #5). On such a re-plan you SHOULD widen the retrieval — `top_n` up to ~30
(and/or relax symmetry/lattice filters, or try an alternate chemistry
hypothesis) — and re-run. This widening is *expected and allowed* on a failed
pass; do not stay capped at 10 while re-planning a search that returned no
confident match. (For an in-situ *series*, keep it tight per frame — the
establishing frame can widen once, then lock the identified phase.)

**Wavelength selection.** Default CuKa unless the experiment metadata
says otherwise. MoKa is common for high-2θ work. A wavelength mismatch
gives uniformly bad correlations / FOMs for all candidates with a
characteristic shift in peak positions — suspect that first when every
candidate scores reject.

Call `resolve_wavelength(system_info)` once near the top of the
analysis script and pass its return value to every
`simulate_xrd_pattern` call instead of hard-coding `wavelength='CuKa'`.
The resolver reads structured `experiment.wavelength` / `source` /
`x_ray_source` fields first, then falls back to a free-text scan for
canonical source names. When metadata is silent it returns the default
`'CuKa'`, so the call is safe even on patterns with no metadata at all.

**Narrowing the candidate list.** The `search_structures` `query` dict
accepts three optional filters that often pay for themselves:

- `z_range: (int, int)` — number of sites per unit cell. Useful when
  the user has a rough atom-count expectation (`(1, 8)` for simple
  binaries; `(8, 50)` for typical oxides).
- `density_range: (float, float)` — g/cm³. Narrows by physical density
  when the sample's bulk density is known from independent measurement.
- `anonymous_formula: str` — stoichiometry template, e.g. `'AB2'` for
  rutile/anatase-type, `'ABC3'` for perovskites. Pymatgen's
  `Composition.anonymized_formula` is the matching convention.

All three are optional and respected by Materials Project and the
local CIF backend; COD ignores them.

**Profile fitting is a DOWNSTREAM follow-up, not part of identification.**
Crystallite size / strain (per-peak pseudo-Voigt → Scherrer / Williamson-Hall)
is the `curve_fitting/xrd_profile` skill's job, run as a **separate step after
the phase is identified** — never an in-ID fit. Identification needs only peak
*positions and relative intensities* (the light `extract_peaks`), which the
scorers consume directly. Do **not** call `fit_profile`, fit pseudo-Voigt peaks,
or compute per-peak R² inside the identification script: it adds nothing the
match needs and pulls the run into the curve-fitting pipeline (an R²-shaped
deliverable the `figure_of_merit` gate then rejects, triggering avoidable
refinement iterations). When line broadening matters for the match on
nanocrystalline data, **widen the scorer's `fwhm` (0.3-0.5°)** rather than
fitting each peak.

## analysis

**CRITICAL: structure-matching workflow.** The per-item analysis script
must follow this exact sequence:

1. Load experimental 2-theta + intensity arrays.
2. Call `search_structures` with `top_n` 5-10 (first pass). If the run is a
   re-plan after a failed/low-FoM pass, widen `top_n` up to ~30 and/or relax
   filters (see "Candidate count — tight first pass, widen on failure").
3. For each candidate, call `simulate_xrd_pattern` and `score_xrd_match_fast`.
4. Filter to candidates with `verdict in {'accept', 'marginal'}`.
5. Call `extract_peaks` once on the experimental pattern.
6. For each surviving candidate, call `score_xrd_match_robust` (default
   `algorithm='hanawalt'`).
7. Emit a `MATCH_RESULTS_JSON: {...}` line collecting the ranked match
   list, and a `FIT_RESULTS_JSON: {...}` line with verdict + best-match
   metadata. The `search_structures` tool already prints its own
   `DB_MATCHES_JSON:` marker; the framework's stdout parser lifts it
   into `fit_results['db_matches']` automatically.

**Don't profile-fit for identification.** Two common over-builds to avoid:
- The **fast tier needs no peak extraction** — it cross-correlates the
  *continuous* (background-subtracted) pattern directly. Subtract the background,
  then correlate; do not extract or fit peaks before Step 3.
- For the **robust tier**, use the **light** `extract_peaks` (positions +
  relative intensities). Do **NOT** fit a pseudo-Voigt profile to every peak —
  that is the `xrd_profile` skill's specialized job (crystallite size / strain)
  and is unnecessary for identification, which only needs positions + relative
  intensities. FWHMs from `extract_peaks` are optional refinement, not a goal.
- **Emit `figure_of_merit`, never `r_squared`.** The ID gate scores by
  `figure_of_merit`; there is no curve fit here, so there is no R² to report.
  For the visualization, overlay the best-match **simulated** pattern (broadened
  sticks) on the experimental data — do not `curve_fit` a profile for the plot
  or compute an R² for it.

**Complete two-tier template** — adapt for the active wavelength and
chemistry hypothesis:

```python
import json
import numpy as np

from scilink.skills.structure_matching.xrd.search_structures import search_structures
from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
from scilink.skills.structure_matching.xrd.score_match_fast import score_xrd_match_fast
from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_robust
from scilink.skills.structure_matching.xrd.extract_peaks import extract_peaks

# ---- Step 1: Load experimental pattern ----
data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
exp_2theta, exp_intensity = data[:, 0], data[:, 1]

# ---- Step 2: Query DB ----
hits = search_structures(
    query={"chemistry": CHEMISTRY_HINT, "top_n": 5},
    output_dir="./candidates",
)

# ---- Step 3: Fast tier — broad ranking ----
WAVELENGTH = "CuKa"
two_theta_range = (float(exp_2theta.min()), float(exp_2theta.max()))
fast_results = []
for cand in hits["candidates"]:
    sim = simulate_xrd_pattern(
        cand["structure_path"], wavelength=WAVELENGTH,
        two_theta_range=two_theta_range,
    )
    fast = score_xrd_match_fast(
        exp_two_theta=exp_2theta.tolist(),
        exp_intensity=exp_intensity.tolist(),
        sim_two_theta=sim["two_theta"],
        sim_intensity=sim["intensities"],
    )
    fast_results.append({**cand, "fast": fast, "sim_peaks": sim})

# ---- Step 4: Filter to plausible candidates ----
plausible = [r for r in fast_results if r["fast"]["verdict"] in {"accept", "marginal"}]
if not plausible:
    plausible = sorted(fast_results, key=lambda r: -r["fast"]["correlation"])[:3]

# ---- Step 5: Extract experimental peaks once ----
exp_peaks = extract_peaks(exp_2theta.tolist(), exp_intensity.tolist())

# ---- Step 6: Robust tier — confident identification ----
final = []
for r in plausible:
    robust = score_xrd_match_robust(
        sim_two_theta=r["sim_peaks"]["two_theta"],
        sim_intensity=r["sim_peaks"]["intensities"],
        exp_peaks=exp_peaks,
        algorithm="hanawalt",  # change to 'mip' for multi-phase / fitted parameters
    )
    final.append({
        "id": r["id"], "source": r["source"], "formula": r["formula"],
        "space_group": r["space_group"],
        "correlation_fast": r["fast"]["correlation"],
        "figure_of_merit": robust["figure_of_merit"],
        "verdict": robust["verdict"],
        "fitted_shift_fast": r["fast"]["fitted_shift"],
        "fitted_scale_fast": r["fast"]["fitted_scale"],
    })

final.sort(key=lambda r: -r["figure_of_merit"])
best = final[0] if final else None

# ---- Step 7: Emit ranked match list + fit results ----
print("MATCH_RESULTS_JSON: " + json.dumps(final))
print("FIT_RESULTS_JSON: " + json.dumps({
    "best_match": best,
    "candidates_considered": len(final),
    "fit_quality": {
        "figure_of_merit": best["figure_of_merit"] if best else None,
        "verdict": best["verdict"] if best else "no_candidates",
    },
}))
```

**Multi-phase emit (REQUIRED when using `score_xrd_match_multiphase`).** The
multi-phase scorer returns ONE result with `active_phases` (not a ranked
per-candidate list), plus `figure_of_merit` (= 1 − cost) and `verdict`. The
quality gate reads `fit_quality.figure_of_merit`, so you MUST surface the
multi-phase `figure_of_merit` there or the run is hard-rejected as "metric
missing" regardless of how good the match is:

```python
mp = score_xrd_match_multiphase(exp_peaks=exp_peaks, candidates=candidates)
print("FIT_RESULTS_JSON: " + json.dumps({
    "active_phases": mp["active_phases"],          # each: id, formula, coverage,
                                                   # matched_peaks, lattice_scale
    "unmatched_exp": mp["unmatched_exp"],          # peaks no phase explains
    "fit_quality": {
        "figure_of_merit": mp["figure_of_merit"],  # gate reads THIS
        "verdict": mp["verdict"],
        "cost": mp["cost"],
    },
}))
```

**Background handling.** Both scorers default to subtracting the
experimental minimum as a flat offset. For patterns with significant
continuous background (amorphous halo, fluorescence), call
`fit_background(two_theta, intensity, method='snip')` from
`scilink.skills.curve_fitting.xrd_profile.background` to estimate the
continuous background, subtract it, and pass `background="none"` to
the scorers. SNIP handles smooth amorphous floors without imposing a
polynomial shape; use `method='polynomial'` only when a polynomial is
genuinely the right model.

**Wavelength consistency.** Pass the same `wavelength` to every
`simulate_xrd_pattern` call. Pull from experiment metadata when present;
otherwise default CuKa.

**Peak broadening.** Both scorers use a Lorentzian with FWHM=0.15° by
default. For low-resolution or strongly broadened experimental patterns
(nanocrystalline samples), bump `fwhm` to 0.3-0.5° so peaks overlap as
they do in the data.

**When the robust tier disagrees with the fast tier.** Trust the robust
tier — it factors out background, scale, and intensity-ratio effects
that the fast tier folds into the correlation. The fast tier is a
triage step; the robust tier is the identification.

## interpretation

**Verdict thresholds.**

Fast tier (correlation, in [-1, 1]):
- correlation ≥ 0.85 → "accept"
- correlation ≥ 0.60 → "marginal"
- otherwise → "reject"

Robust tier — Hanawalt (figure-of-merit, in [0, 1]):
- FOM ≥ 0.70 → "accept"
- FOM ≥ 0.40 → "marginal"
- otherwise → "reject"

Robust tier — MIP (cost, in [0, 1], lower is better):
- cost ≤ 0.25 → "accept"
- cost ≤ 0.55 → "marginal"
- otherwise → "reject"

**Reporting style for synthesis.** Phrase identification in proportion
to the robust-tier margin:

- Best FOM > 0.85 and runner-up > 1.5× lower: declare identification
  with high confidence. Name the phase, space group, and (if MIP was
  used) the fitted zero-shift and lattice scale.
- Best FOM in [0.70, 0.85]: declare identification with caveats. List
  alternative candidates within 0.1 FOM of the best.
- Best FOM in [0.40, 0.70]: do not declare a single identification.
  List the top 3 candidates as "consistent with the experimental
  pattern" and recommend higher-resolution data or a complementary
  technique (EELS, EDS).
- All FOMs below 0.40: report no confident match. Suggest broadening
  the chemistry hypothesis, checking the experimental wavelength, or
  running a peak-fit-first workflow.

**What a good score cannot tell you.** A high FOM / low MIP cost
confirms the phase is present; it does NOT confirm purity, sample
preparation correctness, or absence of minor phases. Mention this
caveat when reporting identifications.

**Multi-phase samples.** Hanawalt is single-phase by construction. For
suspected multi-phase mixtures, switch to `score_xrd_match_multiphase`
and feed it the full candidate phase list — the joint MILP allocates
peaks across phases with per-phase activation binaries and reports
per-phase coverage. Increase `max_exp_peaks` (default 30) when the
mixture has many resolved peaks per phase. The tool does NOT compute
quantitative phase fractions (peak-area weighted Rietveld refinement
is the standard for that); the `coverage` field is a peak-count
proxy, not a phase fraction. Note this caveat in the report.

## validation

**Per-candidate sanity checks.**

- The simulated pattern must cover the experimental 2-theta range. If
  `min(sim["two_theta"])` is more than 1° above `min(exp_2theta)`,
  expand the simulation range or note unobserved low-angle peaks.
- The number of simulated peaks should be reasonable for the
  candidate's symmetry (5-50 peaks typical in 10-90° 2θ; outliers
  indicate the simulation range was too narrow or too broad).
- Fast and robust tier verdicts should agree on direction. When the
  fast tier says "accept" but the robust tier says "reject", the
  correlation is being fooled by background or scale — re-run with a
  polynomial background subtraction.

**Cross-candidate sanity.** When 3+ candidates score "accept" in the
robust tier, the chemistry hypothesis is too broad — narrow it. When 0
candidates score above "reject", the chemistry hypothesis is wrong or
the wavelength is mismatched.

**Zero-shift sanity (MIP only).** A fitted zero-shift larger than 0.3°
typically indicates either a sample-displacement problem on the
instrument or a wavelength mismatch — not a true identification issue.
Flag it in the report.

**FWHM tuning.** If the best candidate scores FOM > 0.6 but visual
inspection (peak positions match) suggests it's the right phase, retry
the robust scorer with `tol_deg=0.5` (looser tolerance) or extract
peaks with a larger `min_distance_deg` to suppress noise peaks. Broad
peaks in real data require looser tolerance windows.

**Materials-Project authority.** When MP and a local CIF both return the
same (formula, space_group), `search_structures` keeps the MP entry and
drops the local one. Local CIFs without space-group metadata cannot
dedup against MP and appear as separate candidates — verify they aren't
duplicates of MP entries already in the candidate list.
