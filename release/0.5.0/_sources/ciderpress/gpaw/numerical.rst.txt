FFT, PAW, and Radial Interpolation
==================================

Smooth-grid NLDF path
---------------------

The GPAW backend represents the density-dependent nonlocal kernel on a
geometric interpolation grid.  ``ciderpress.gpaw.nldf_interface``
coordinates the feature plan, while ``ciderpress.gpaw.cider_fft`` performs
the cell FFT convolutions and applies the adjoint feature potential.  Energy,
potential, force, and stress implementations must use the same interpolation
parameters.

PAW/PASDW correction path
-------------------------

The smooth pseudo density is incomplete inside a PAW augmentation sphere.
``ciderpress.gpaw.cider_paw``, ``ciderpress.gpaw.atom_utils``, and
``ciderpress.gpaw.atom_descriptor_utils`` construct matching all-electron
and pseudo atomic contributions, transfer their nonlocal sources through the
PASDW representation, and return the derivative contribution to GPAW.

``pasdw_ovlp_fit`` controls overlap fitting in that transfer.
``pasdw_store_funcs`` controls whether atom-centered projector functions are
cached.  The latter changes memory and repeated cost, not the definition of
the model.

Radial interpolation and core terms
-----------------------------------

``ciderpress.gpaw.interp_paw`` defines ``DiffGGA`` and ``DiffMGGA``, the
radial-grid machinery used to evaluate differentiable PAW XC corrections.
It interpolates all-electron and pseudo partial-wave density ingredients,
assembles core contributions, and back-propagates derivatives to the PAW
density matrices.  Meta-GGA handling includes kinetic-energy-density terms;
the full-core contribution is kept above its physical lower bound when a
finite radial representation would otherwise violate it.

This layer affects every PAW atom using the corresponding feature class.
Changes should therefore be checked across light atoms and transition metals,
at radial endpoints, and against finite differences for energy, force, and
stress.  It is distinct from an SCF convergence adjustment: mixer changes
cannot repair an inconsistent PAW forward/adjoint pair.

See :doc:`../../theory/numerical_evaluation` for the complete calculation
path and :doc:`../../workflows/extending` for implementation invariants.
