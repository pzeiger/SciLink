---
description: >
  abTEM STEM multislice / 4D-STEM image simulation — probe and potential setup
  (energy, semiangle, real-space sampling, slice thickness), detector geometry
  (annular BF/MAADF/HAADF, flexible, pixelated), frozen-phonon thermal diffuse
  scattering, and PRISM acceleration, for oriented periodic orthogonal supercells.
detect:
  binaries: []
  env_vars: []
  python_modules: [abtem, ase]
  guidance: |
    abTEM is a pure-Python / GPU library — detection is via module import, not
    a $PATH binary. GPU acceleration requires CuPy (CUDA) or a ROCm-enabled
    build and is a runtime concern, not an import-time one. Verify with:
      python -c "import abtem; print(abtem.__version__)"
    The API changed substantially at the 1.0 rewrite; always verify calls
    against the installed version before running generated scripts.
---

## Overview

abTEM (ab-initio Transmission Electron Microscopy) is a Python library for
forward simulation of STEM images, 4D-STEM diffraction patterns, and exit wave
functions from atomic structures using the multislice algorithm or the PRISM
(Projected Real-space Iterative Scattering Matrix) approximation.

### Object model

```
ase.Atoms (or FrozenPhonons wrapping it)
    └─► Potential(atoms, sampling, slice_thickness)
            ├─► Probe(energy, semiangle_cutoff)   [STEM / 4D-STEM]
            │       ├─► GridScan(start, end, sampling)
            │       ├─► AnnularDetector(inner, outer)      [ADF]
            │       ├─► FlexibleAnnularDetector(step_size) [flexible ADF]
            │       └─► PixelatedDetector()                [4D-STEM / CBED]
            └─► PlaneWave(energy)                 [TEM / exit wave]
```

Evaluation is **lazy** (backed by Dask): building objects constructs a
computation graph; `.compute()` triggers actual evaluation and returns NumPy
arrays. Large scans benefit from chunked / GPU computation.

### Units
- Energy: **eV** (e.g. 200 000 eV = 200 keV)
- Lengths / sampling: **Å** (ångström)
- Angles (semiangle, detector limits): **mrad**
- Thermal displacement parameters (sigmas): **Å RMS**

### Structure requirements
The input structure must be:
1. **Orthogonal** — all cell angles 90°. abTEM requires a rectangular supercell.
2. **Periodic in x and y** — the probe wraps at cell boundaries.
3. **Beam along +z** — orient the zone axis along the c-axis before building
   the Potential.
4. **Laterally large enough** — the supercell should contain the probe footprint
   plus its tails (typically ≥ 10 Å on each side, so tile if needed).


## Planning

### Beam energy
| Application                        | Typical energy |
|------------------------------------|---------------|
| Biological / beam-sensitive        | 60–80 keV     |
| General materials (semiconductors) | 100–120 keV   |
| Hard materials, heavy elements     | 200–300 keV   |
| Maximum resolution (aberr.-corr.)  | 300 keV       |

Higher energy → shorter wavelength → higher resolution, but also more knock-on
damage for light elements.

### Real-space sampling and antialiasing

The maximum scattering angle representable on a grid with sampling `dx` (Å/px)
is `θ_max = λ / (2·dx)` mrad, where `λ` is the relativistic electron wavelength.

**Antialiasing rule:** `θ_max ≥ 1.5 × max(θ_semiangle, θ_detector_outer)`

Practical values for 200 keV (λ ≈ 0.0251 Å):
| sampling (Å/px) | θ_max (mrad) |
|----------------|-------------|
| 0.10           | 126         |
| 0.05           | 251         |
| 0.03           | 418         |

Start with `sampling = 0.05 Å` for most HAADF work; reduce to 0.03 Å for
high-angle detectors or high-resolution phase contrast.

### Probe parameters
- `semiangle_cutoff`: convergence semi-angle of the objective aperture in mrad.
  Typical values: 10–15 mrad (uncorrected), 20–30 mrad (aberration-corrected).
- `energy`: in eV; must match beam_energy.
- Aberrations (optional): `Probe(energy=..., semiangle_cutoff=..., Cs=1e-3, defocus=0)`
  where `Cs` is in m and `defocus` in Å. Default is aberration-free.

### Scan
- `GridScan(start=[0,0], end=[1,1], fractional=True, potential=potential, sampling=...)`
  covers the whole unit cell fractionally. `sampling` should be set to
  `probe.aperture.nyquist_sampling` or finer.
- For a sub-region scan use fractional coordinates < 1.

### Detector choice
| Goal                                | Detector                                 | Notes                                      |
|-------------------------------------|------------------------------------------|--------------------------------------------|
| HAADF (high-angle ADF)              | `AnnularDetector(inner=60, outer=200)`   | Heavy-element contrast, ~Z²               |
| MAADF (medium-angle ADF)            | `AnnularDetector(inner=40, outer=100)`   | Useful for lighter elements                |
| ABF (annular bright field)          | `AnnularDetector(inner=10, outer=25)`    | Light-element sensitivity                  |
| BF (bright field)                   | `AnnularDetector(inner=0, outer=10)`     | Phase contrast                             |
| Flexible / post-hoc selection       | `FlexibleAnnularDetector(step_size=1)`   | Low memory; choose limits after simulation |
| 4D-STEM / CBED / ptychography       | `PixelatedDetector()`                    | Full diffraction pattern per probe pos.    |

Multiple detectors can be passed as a list: `detectors=[det_haadf, det_abf]`.

### Frozen phonons
Thermal diffuse scattering (TDS) dominates ADF intensities at high angles. Model
it with `FrozenPhonons`:
- `sigmas`: RMS thermal displacement in Å. Can be a single float (same for all
  elements) or a `{symbol: sigma}` dict. Typical: 0.07–0.12 Å at 300 K.
  Debye–Waller B-factors: `sigma = sqrt(B / (8π²))`.
- `num_configs`: number of phonon configurations. ≥ 8 for reasonable statistics;
  ≥ 20 for quantitative intensities. Convergence: run with 8 and 16, compare.
- `seed`: integer for reproducibility.

For qualitative imaging (phase contrast, defect visibility) frozen phonons may be
omitted; for quantitative ADF intensity comparisons they are essential.

### PRISM vs. full multislice
| Method         | When to use                                    | Tradeoff                                   |
|----------------|------------------------------------------------|--------------------------------------------|
| Full multislice| Default; all scan sizes; large convergence angles | Slowest; most accurate                  |
| PRISM (SMatrix)| Large-area scans (> 5×5 nm²); small convergence angles | Fast; interpolation error at large angles |

PRISM interpolation factor: `interpolation=4` gives 4× speedup with small angle
error; `interpolation=1` recovers full-multislice accuracy.


## Implementation

> **Verify every API call against the installed abTEM version** before running.
> The API changed substantially at the 1.0 rewrite. Check with:
> `python -c "import abtem; print(abtem.__version__)"`

### Standard HAADF GridScan (full multislice)

```python
# run_abtem.py — generated by EMSAgent
# Simulation: HAADF STEM multislice with frozen phonons
# STRUCTURE_PATH and OUTPUT_PATH are set by the agent at generation time.

import numpy as np
import ase.io
import abtem

STRUCTURE_PATH = "structure_prepped.vasp"   # prepped, orthogonal supercell
OUTPUT_PATH = "measurement.npz"

# Load structure
atoms = ase.io.read(STRUCTURE_PATH)

# Frozen phonons for thermal diffuse scattering
frozen = abtem.FrozenPhonons(atoms, num_configs=8, sigmas=0.1, seed=100)

# Build electrostatic potential
potential = abtem.Potential(frozen, sampling=0.05, slice_thickness=2.0)

# Define probe
probe = abtem.Probe(energy=200e3, semiangle_cutoff=20)
probe.grid.match(potential)

# Define scan (full unit cell, Nyquist sampling)
scan = abtem.GridScan(
    start=[0, 0],
    end=[1, 1],
    fractional=True,
    potential=potential,
    sampling=probe.aperture.nyquist_sampling,
)

# HAADF detector (50–150 mrad)
detector = abtem.AnnularDetector(inner=50, outer=150)

# Run and compute
measurement = probe.scan(potential, scan=scan, detectors=detector)
measurement.compute()

# Save output
np.savez(OUTPUT_PATH, data=measurement.array, sampling=scan.sampling)
print(f"Done. Output shape: {measurement.array.shape}")
```

### 4D-STEM / CBED variant

Replace the detector and output:

```python
detector = abtem.PixelatedDetector()
measurement = probe.scan(potential, scan=scan, detectors=detector)
measurement.compute()
measurement.to_zarr(OUTPUT_PATH.replace(".npz", ".zarr"))
```

### PRISM (SMatrix) variant for large-area scans

```python
s_matrix = abtem.SMatrix(
    potential,
    energy=200e3,
    semiangle_cutoff=20,
    interpolation=4,   # speedup factor; reduce for high-angle accuracy
)
measurement = s_matrix.scan(scan=scan, detectors=detector)
measurement.compute()
np.savez(OUTPUT_PATH, data=measurement.array, sampling=scan.sampling)
```

### Multiple detectors

```python
det_haadf = abtem.AnnularDetector(inner=60, outer=200)
det_abf   = abtem.AnnularDetector(inner=10, outer=25)
measurements = probe.scan(
    potential, scan=scan, detectors=[det_haadf, det_abf]
)
measurements.compute()
for i, det_name in enumerate(["haadf", "abf"]):
    np.savez(f"measurement_{det_name}.npz", data=measurements[i].array)
```

### GPU acceleration (CuPy required)

```python
import abtem
abtem.config.set({"device": "gpu"})   # or set ABTEM_DEVICE=gpu env var
# All subsequent abtem calls use GPU arrays via CuPy.
```

### Output formats
- **NPZ** (`np.savez`): portable, NumPy native, good for < ~1 GB.
- **Zarr** (`.to_zarr(path)`): chunked, lazy, preferred for 4D-STEM datacubes
  and multi-run ensembles. Compatible with downstream TACAW analysis.


## Validation

Deterministic checks the agent applies before and after script generation.
These are geometric/physical constraints, not LLM-based heuristics.

### Antialiasing (sampling vs. detector angle)
`θ_max = λ / (2 · dx)` where `λ` is the relativistic wavelength (Å) and
`dx` is the real-space sampling (Å/px).

**Requirement:** `θ_max ≥ 1.5 × max(semiangle_cutoff, detector_outer_mrad)`

If violated: reduce `sampling` until satisfied, or reduce detector outer angle.

### Cell orthogonality
abTEM requires a rectangular (orthogonal) periodic cell. All cell angles must
be 90° ± 0.5°. Non-orthogonal cells must be converted before building the
Potential (structure prep step).

### Thickness
- `thickness > 0` — cell must have non-zero extent along the beam (c-axis).
- `thickness < 500 Å` — warn for very thick samples (slow; channelling strong).
- `slice_thickness` between 0.5 Å and 5.0 Å; values outside this range should
  be flagged with a warning.

### Frozen phonon convergence
- `num_configs ≥ 1` (error if zero or negative).
- `num_configs ≥ 8` for ADF imaging (warn below this threshold).
- For quantitative ADF intensity comparisons, `num_configs ≥ 20` is advisable.

### GridScan coverage
- `sampling` of the scan should be ≤ `probe.aperture.nyquist_sampling`
  (`λ / (2 · semiangle_mrad / 1000)`) to satisfy the probe-sampling criterion.
- `end` coordinates must be > `start` coordinates.

### Detector angles vs. representable range
- `detector_outer_mrad < θ_max` — detector must be inside the representable
  angular range of the simulation grid.
- `detector_inner_mrad < detector_outer_mrad` — inner < outer.
