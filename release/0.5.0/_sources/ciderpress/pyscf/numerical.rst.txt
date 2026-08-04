PySCF Numerical Data Flow
=========================

For a molecular grid block, the custom numerical integrator obtains density,
gradient, and kinetic-energy-density ingredients from PySCF.  It evaluates
semilocal and optional nonlocal features, applies the mapped model, then
back-propagates the model derivatives to the matrix-valued XC potential.  The
restricted and unrestricted paths share feature plans but use different spin
array layouts.

NLDF evaluation
---------------

``ciderpress.pyscf.gen_cider_grid`` constructs the atom-centered grid used
by the nonlocal evaluator.  ``ciderpress.pyscf.nldf_convolutions``
expands the density-dependent kernels in atom-centered auxiliary functions
and evaluates their forward and adjoint contractions.  The plan is derived
from the model's ``NLDFSettings``; changing feature order or exponents at
inference time changes the functional.

SDMX evaluation
---------------

``ciderpress.pyscf.sdmx`` and ``ciderpress.pyscf.sdmx_slow`` compute
smoothed density-matrix quantities and their response.  The optimized and
reference implementations should agree before a new SDMX setting is exposed
to a mapped model.  SDMX gradients are not currently a supported production
property.

Derivatives
-----------

The SCF potential is the adjoint of feature generation with respect to the
density matrix.  Molecular gradients additionally require AO, orbital,
quadrature-grid, and nonlocal-feature response.  Finite-difference checks
must use the same basis, grid, feature settings, density-fitting treatment,
and electronic state as the analytical calculation.

See :doc:`../../theory/numerical_evaluation` for the backend comparison and
:doc:`../../workflows/extending` for the extension checklist.
