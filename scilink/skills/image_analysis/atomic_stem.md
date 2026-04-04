# STEM Atomic Resolution Image Analysis Skill

## overview

Atomic-resolution STEM image analysis (HAADF, MAADF). Individual atomic
columns are resolved as bright spots on a dark background. Applicable
to any crystalline material viewed along a zone axis. Covers column
detection, sublattice separation, lattice characterization, defect
identification, and structural variation analysis.

## planning

### foundational
Before running detection, estimate the expected number of **visible**
atomic columns from the image dimensions, pixel calibration (if
available), and known lattice parameters for the material. Count all
sublattices that are visible in HAADF — for multi-sublattice
structures, the expected total is the number of unit cells multiplied
by the number of visible column types per cell. After detection,
compare the actual count against this estimate — significant
discrepancy (outside 0.90-1.10x) suggests detection issues or a
structurally interesting region.

Choose a detection method that reliably finds atomic columns across
the full image despite intensity variations (different species,
thickness gradients). For classical detection (`detect_atoms`),
background subtraction or bandpass filtering before detection helps
with non-uniform illumination. For DCNN detection
(`detect_atoms_dcnn`), pass the raw image without preprocessing — the
model was trained to handle intensity gradients internally. Refine detected
positions with 2D Gaussian fitting for sub-pixel precision. Determine
lattice parameters from the FFT (which directly shows periodicity)
rather than solely from nearest-neighbor distances.

### advanced
**Atom finding tools:** The functions in `scilink.tools.atom_finding_tools`
handle multi-sublattice detection, refinement, and separation. Use them
instead of writing detection code from scratch:
```python
from scilink.tools.atom_finding_tools import (
    detect_atoms, detect_atoms_dcnn, refine_positions,
    find_zone_axes, find_missing_atoms, subtract_atoms
)
```
Two detection methods are available: `detect_atoms_dcnn` (AtomNet3
DCNN ensemble, requires `fov_nm` from metadata) and `detect_atoms`
(classical peak detection, works in pixel space). Both return the same
dict format so downstream tools work with either. Try `detect_atoms_dcnn`
first when spatial calibration is available. If DCNN detection fails or
produces poor results, fall back to `detect_atoms`. Use `detect_atoms`
directly when `fov_nm` cannot be determined from metadata.

Plan the pipeline around the chosen detector, then use `find_zone_axes` to identify
lattice vectors, `find_missing_atoms` to predict weaker sublattice
positions at fractional sites, and `subtract_atoms` to remove the
dominant sublattice and reveal fainter ones underneath. For materials
with multiple sublattices (perovskites, layered oxides, rock-salt),
plan for iterative detect→subtract→detect cycles.

**Sublattice separation:** Intensity-based clustering alone is not
sufficient for complex structures. For layered materials, columns of
different species form distinct rows or planes. Use both intensity AND
position within the unit cell to assign sublattices. Verify that the
assignment produces the correct stoichiometric ratios.

**Unit cell identification:** The nearest-neighbor distance is NOT the
lattice parameter for complex structures. The true unit cell repeat
may be 2x, 3x, or more of the shortest column spacing. Use the FFT
to identify the full periodicity, or count how many distinct column
rows/intensities repeat along each direction.

**Structural anomalies:** After fitting the ideal lattice, examine the
spatial distribution of displacements, intensity variations, and local
lattice parameter changes across the image. Regions where these
quantities deviate systematically from the bulk may indicate structural
features worth reporting — let the data guide the interpretation rather
than searching for specific defect types.

## analysis

### foundational
Column detection typically involves: normalization, background
subtraction or bandpass filtering, blob/peak detection, and 2D
Gaussian refinement. Measure basic statistics: column count, intensity
distribution, nearest-neighbor distances, and lattice parameters from
FFT.

### advanced

**Atom finding tools — detailed usage:**

`detect_atoms(image, separation, threshold_rel=0.02, refine=True, percent_to_nn=0.4, subtract_background=False, normalize_intensity=True)`
finds atomic column positions with optional sub-pixel Gaussian refinement.
- **separation** (int): minimum atom spacing in **pixels**. Estimate from
  known lattice parameter / pixel size, or from FFT peak position:
  `separation ≈ image_width / (2 × FFT_peak_distance_from_center)`.
  If unsure, use 70-80% of the apparent nearest-neighbor distance.
- **threshold_rel** (float): peak sensitivity. Default 0.02. Raise to
  0.05-0.1 for noisy images; lower to 0.01 for faint columns.
- **percent_to_nn** (float): Gaussian fit mask as fraction of NN distance.
  0.4 default. Increase for sparse lattices, decrease for dense.
- **subtract_background** (bool): Gaussian-blur background subtraction
  before peak finding. Default False. Enable for images with strong
  intensity gradients.
- **normalize_intensity** (bool): Normalize image to 0-1 before peak
  finding. Default True.
- Returns dict: `"positions"` (N,2 as x,y where x=col y=row),
  `"sigma_x"`, `"sigma_y"`, `"amplitude"`, `"rotation"` (all N arrays).

`detect_atoms_dcnn(image, fov_nm, model_dir=None, target_pixel_size=0.25, threshold=0.8, refine=True)`
detects atom columns using an AtomNet3 DCNN ensemble.
- **fov_nm** (float): field of view in **nanometers** (from metadata).
- **target_pixel_size** (float): target pixel size in Angstroms for
  the model. Default 0.25. May need tuning for different materials.
- **threshold** (float): detection confidence, 0-1. Default 0.8.
- Returns dict: `"positions"` (N,2 as x,y in original image pixels),
  `"heatmap"` (2D probability map, pre-threshold),
  sigma/amplitude/rotation are None. Use `refine_positions` to obtain them.

`refine_positions(image, positions, percent_to_nn=0.4)`
fits 2D Gaussians at known atom positions to get sub-pixel coordinates
and per-atom sigma, amplitude, and rotation. Use after `detect_atoms_dcnn`
or any other source that lacks Gaussian parameters (needed for
`subtract_atoms`).
- **positions** (N,2): atom positions as (x, y) from any detection method.
- Returns dict with `"positions"`, `"sigma_x"`, `"sigma_y"`,
  `"amplitude"`, `"rotation"` — same format as `detect_atoms(refine=True)`.

`find_zone_axes(positions, n_neighbors=9, distance_tolerance=None)`
detects lattice translation vectors by clustering displacement vectors.
- **n_neighbors**: 9 for simple lattices, 15-25 for complex unit cells.
- Returns list of (dx, dy) tuples, shortest first. Square lattice → 2
  vectors, hexagonal → 3. The shortest vector is the NN distance; the
  **lattice parameter** may be 2×+ for multi-sublattice structures.

`find_missing_atoms(positions, zone_vector, fraction=0.5, min_distance=3.0)`
predicts positions at fractional lattice sites along a zone vector.
- **fraction**: 0.5 = midpoint (binary compounds), 0.33/0.67 (ternary).
- **min_distance**: discard predictions within this distance of existing
  atoms (set to ~separation/3).
- Returns (M,2) predictions. Verify that the image has intensity there.

`subtract_atoms(image, positions, sigma_x, sigma_y, amplitude, rotation=None)`
removes fitted Gaussians from the image. Requires per-atom Gaussian
parameters — use `detect_atoms(refine=True)` or `refine_positions()`
to obtain them. Returns residual (clipped >= 0) where subtracted regions
drop to background. Run detection on the residual with lower threshold
to find the next sublattice.

**Multi-sublattice workflow (classical):**
```
result1 = detect_atoms(image, separation, refine=True)
zone_vecs = find_zone_axes(result1["positions"])
predicted = find_missing_atoms(result1["positions"], zone_vecs[0], fraction=0.5)
residual = subtract_atoms(image, result1["positions"],
                          result1["sigma_x"], result1["sigma_y"],
                          result1["amplitude"], result1["rotation"])
result2 = detect_atoms(residual, separation, threshold_rel=0.01, refine=True)
```

**Multi-sublattice workflow (DCNN):**
```
dcnn1 = detect_atoms_dcnn(image, fov_nm)
result1 = refine_positions(image, dcnn1["positions"])
zone_vecs = find_zone_axes(result1["positions"])
residual = subtract_atoms(image, result1["positions"],
                          result1["sigma_x"], result1["sigma_y"],
                          result1["amplitude"], result1["rotation"])
result2 = detect_atoms(residual, separation, threshold_rel=0.01, refine=True)
# Save dcnn1["heatmap"] to visualize detection confidence
```
Stop when residual has no peaks above 3× noise std. Validate: check
stoichiometric ratios, heavier atoms should have higher amplitude,
each sublattice's NN distances should be consistent.

**Sublattice separation** — three approaches, choose based on the data:
1. **Iterative detect-subtract-detect:** detect and refine the brightest
   columns, subtract them, detect the next brightest on the residual,
   repeat until no peaks remain. Separates by geometry without clustering.
2. **Local environment GMM:** crop a small square window (side length
   approximately equal to the lattice parameter) centered on each
   detected column, flatten each crop into a 1D vector, stack them into
   an (N, window*window) matrix, and cluster with GMM. Each cluster
   centroid is an average local environment image. This captures the
   full neighborhood (neighboring column arrangement, not just peak
   intensity) — useful when intensity alone is ambiguous.
3. **Intensity + positional analysis:** cluster by raw column intensity
   combined with fractional position within the unit cell.

Use raw (unnormalized) column intensities for any intensity-based
analysis — local normalization removes the Z-contrast difference
between species. Verify that each sublattice has consistent intensity
and the expected stoichiometric ratio.

**Defect identification:** Compare detected positions to ideal lattice
sites. Restrict vacancy search to the interior of the detected region
to avoid edge false positives. Verify vacancy candidates with forced
Gaussian fit at expected position.

**Strain and displacement:** Map displacements from ideal lattice
positions spatially across the image. Only report distortions exceeding
the position fit uncertainty.

## interpretation

### foundational
**HAADF intensity:** Scales as ~Z^1.6-2. Brighter columns contain
heavier atoms.

**Lattice parameters:** Compare against known bulk values. Report in
both pixels and physical units when calibration is available.

### advanced
**Sublattice assignment:** Use chemical identity from intensity and
positional analysis to interpret which sublattice corresponds to which
atomic species.

**Strain:** Distinguish fitted-lattice residuals (local disorder) from
deviations against known ideal lattice (true strain). Least-squares
fitting absorbs mean strain — use known lattice constants when
available.

**Vacancy concentration:** In pristine crystals, typically 0.01-1%.
Above 5-10% usually indicates detection or fitting error, unless the
sample was intentionally modified (irradiation, beam damage, quenching).

## validation

### foundational
**Detection completeness:** Detected vs expected column count (from
image area and unit cell) should be within 0.90-1.10.

**NN distance consistency:** CV below 15% for well-ordered crystals.

**Unit cell sanity:** Measured lattice parameters should be close to
known bulk values when spatial calibration is available.

### advanced
**Sublattice populations:** Should match expected stoichiometry for
the material. Heavy doping can shift intensities between clusters.

**Lattice fit residual:** Below 0.3x the lattice spacing.

**Displacement field:** Mean displacement from ideal lattice should be
small (<0.3x lattice spacing). Large systematic displacements indicate
fitting errors, not real strain.
