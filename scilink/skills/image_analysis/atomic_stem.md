# STEM Atomic Resolution Image Analysis Skill

## overview

Atomic-resolution STEM image analysis (HAADF, MAADF). Individual atomic
columns are resolved as bright spots on a dark background. Applicable
to any crystalline material viewed along a zone axis. Covers column
detection, sublattice separation, lattice characterization, defect
identification, and structural variation analysis.

## planning

### foundational
Choose a detection method that reliably finds atomic columns across
the full image despite intensity variations (different species,
thickness gradients). Background subtraction or bandpass filtering
before detection helps with non-uniform illumination. Refine detected
positions with 2D Gaussian fitting for sub-pixel precision. Determine
lattice parameters from the FFT (which directly shows periodicity)
rather than solely from nearest-neighbor distances.

### advanced
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
**Sublattice separation:** Combine intensity clustering with positional
analysis. After detecting all columns and fitting lattice vectors,
project each column position onto the unit cell to determine its
fractional coordinates. Columns at the same fractional position belong
to the same sublattice. Verify by checking that each sublattice has
consistent intensity (bright = heavy atoms in HAADF).

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
image area and unit cell) should be within 0.85-1.15.

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
