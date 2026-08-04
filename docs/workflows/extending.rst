Extending CiderPress
====================

New functionality should preserve the forward/derivative contract across the
model and every supported backend.  Adding a Python class that computes an
array is only one part of a usable density-functional feature.

Adding or modifying a feature
-----------------------------

1. Define the feature and its parameters in the settings layer, including its
   feature count, spin behavior, uniform-scaling behavior, and uniform-electron
   gas limit where applicable.
2. Implement a numerical plan that evaluates the raw feature and its adjoint
   derivative with respect to its electronic inputs.
3. Define normalization and bounded transforms without changing feature order
   between training and evaluation.
4. Connect the plan to each intended backend and fail explicitly in unsupported
   backends.
5. Test feature values, finite differences, spin exchange symmetry, low-density
   behavior, and serialization before fitting a model with it.

Backend data flow
-----------------

PySCF selects its numerical integrator from the feature settings stored in the
model.  A new molecular feature must work for restricted and unrestricted
density layouts and, if gradients are claimed, through the grid and orbital
response path.

GPAW separates smooth-grid evaluation from atom-centered PAW corrections.  A
new nonlocal PAW feature requires matching all-electron and pseudo forward
terms, PASDW transfer, potential back-propagation, and any claimed force or
stress derivative.  Tests should separate pseudopotential/smooth-grid behavior
from PAW behavior so a disagreement can be localized.

PAW and interpolation invariants
--------------------------------

Changes to partial-wave interpolation or full-potential/core handling can
alter energies and descriptors across many elements.  Preserve these
invariants:

* Values at radial-grid endpoints are accepted within floating-point
  tolerance, while genuine extrapolation is rejected.
* Reconstructed all-electron and pseudo quantities use the same partial-wave
  convention as their derivatives.
* Kinetic-energy-density terms remain finite at the origin and obey their
  physical core lower bound.
* Energy, potential, force, and stress paths use the same discretized feature.
* Reference values are updated only after comparison to finite differences and
  independent numerical settings.

Model and checkpoint compatibility
----------------------------------

Settings, mapped evaluators, and checkpoint dictionaries are persistent
interfaces.  When adding a field, provide a safe default for older objects and
include it in round-trip tests.  Never infer a scientifically different
functional composition from a missing field.

Before release, exercise serial and MPI layouts, restricted and unrestricted
molecular cases, PAW elements spanning light and transition-metal regimes,
and a full checkpoint restart.  See :doc:`../reference/limitations` before
advertising a new feature/property combination.
