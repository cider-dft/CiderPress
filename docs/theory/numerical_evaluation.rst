Numerical Evaluation in PySCF and GPAW
======================================

The mapped functional is backend independent.  PySCF constructs its input
features in a molecular Gaussian basis, and GPAW constructs them on a
periodic plane-wave/PAW representation.  Both paths implement the forward and
adjoint structure described below.

Common forward and adjoint calculation
--------------------------------------

The forward calculation evaluates raw features and the mapped energy density.
The model also returns derivatives with respect to every feature.  The backend
applies the adjoint of each feature operation to obtain derivatives with
respect to the density, density gradient, kinetic-energy density, orbitals, or
density matrix.  Those quantities enter the host code's XC potential and its
analytical nuclear or cell derivatives.

Forward features, feature derivatives, density potentials, forces, and stress
must all represent the same discretized functional.

Molecular Gaussian-basis path
-----------------------------

PySCF evaluates semilocal ingredients on an atom-centered quadrature grid.
CiderPress augments that grid for NLDF features, evaluates the atom-centered
convolution and interpolation operations, and inserts the resulting energy
and potential through a custom numerical integrator.  SDMX models construct
smoothed quantities from the one-particle density matrix.

The decorated mean-field object retains PySCF controls for molecular basis
sets, grids, density fitting, occupations, and SCF convergence.  Settings
stored in the model select the CIDER grid and feature generator when the
decorated object is constructed.

Periodic plane-wave and PAW path
--------------------------------

For GPAW, version-j NLDF kernels are expanded over interpolation points and
evaluated as FFT convolutions on the uniform real-space grid.  The
``Nalpha``, ``qmax``, and ``lambd`` controls describe this interpolation:
``Nalpha`` is normally chosen automatically, ``qmax`` bounds the represented
kernel exponent, and smaller ``lambd`` gives a denser and more expensive
interpolation grid.

The smooth pseudo density excludes the core and rapidly varying all-electron
density inside augmentation spheres.  CiderPress combines the smooth-grid
calculation with PAW corrections.  PASDW transfers the nonlocal feature
sources and potentials between the uniform grid and the atom-centered
augmentation representation.  ``pasdw_ovlp_fit=True`` selects overlap fitting;
``pasdw_store_funcs=True`` trades memory for reduced repeated atomic work.

The PAW interpolation code reconstructs all-electron and pseudo partial-wave
quantities on radial grids, including kinetic-energy-density terms required by
meta-GGA models.  Core kinetic-energy contributions are constrained to their
von Weizsäcker lower bound to control finite-grid violations, and elements
above argon use the setup's denser all-electron radial grid.

Forces and stress require derivatives of both the FFT contribution and the
PAW/PASDW projection terms.  The supported meta-GGA force and stress path uses
PAW setups.

Parallel and restart state
--------------------------

FFT and augmented-grid work can be distributed with
``parallel={"augment_grids": True}``.  CiderPress and GPAW must be built with
compatible FFT, MPI, BLAS, and OpenMP runtimes.  Compatibility can be checked
with a small calculation using the launcher and rank layout planned for
larger jobs.

A full meta-GGA restart needs wavefunctions to reconstruct the kinetic-energy
density.  Write the baseline and CIDER checkpoints with ``mode="all"``.
CIDER checkpoint dictionaries also store the nonlocal interpolation grid so
that a resumed calculation evaluates the same numerical functional.

See :doc:`../usage/gpaw` for executable settings and
:doc:`../workflows/extending` for implementation guidance.
