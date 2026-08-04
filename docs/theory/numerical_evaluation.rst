Numerical Evaluation in PySCF and GPAW
======================================

The mapped functional is backend independent, but its input features are not
obtained in the same way in a Gaussian basis and on a periodic plane-wave
grid.  This page describes the common contract and the two numerical paths.

Common forward and adjoint calculation
--------------------------------------

The forward calculation evaluates raw features and the mapped energy density.
The model also returns derivatives with respect to every feature.  The backend
then applies the adjoint of each feature operation to obtain derivatives with
respect to the density, density gradient, kinetic-energy density, orbitals, or
density matrix.  Those quantities enter the host code's XC potential and its
analytical nuclear or cell derivatives.

Numerical consistency therefore requires more than matching a total energy.
Forward features, feature derivatives, density potentials, forces, and stress
must describe the same discretized functional.

Molecular Gaussian-basis path
-----------------------------

PySCF evaluates semilocal ingredients on an atom-centered quadrature grid.
CiderPress augments that grid when NLDF features are requested, evaluates the
required atom-centered convolution/interpolation operations, and inserts the
resulting energy and potential through a custom numerical integrator.  SDMX
models instead construct smoothed quantities from the one-particle density
matrix.

The decorated mean-field object retains normal PySCF controls for molecular
basis sets, grids, density fitting, occupations, and SCF convergence.  The
CIDER grid and feature generator are selected from the settings stored in the
model.  Users should not manually choose a feature implementation for a
packaged functional.

Periodic plane-wave and PAW path
--------------------------------

For GPAW, version-j NLDF kernels are expanded over interpolation points and
evaluated as FFT convolutions on the uniform real-space grid.  The
``Nalpha``, ``qmax``, and ``lambd`` controls describe this interpolation:
``Nalpha`` is normally chosen automatically, ``qmax`` bounds the represented
kernel exponent, and smaller ``lambd`` gives a denser and more expensive
interpolation grid.

Pseudo densities alone omit the core and rapidly varying all-electron density
inside augmentation spheres.  CiderPress therefore combines the smooth-grid
calculation with PAW corrections.  The PASDW machinery transfers the nonlocal
feature sources and potentials between the uniform grid and atom-centered
augmentation representation.  ``pasdw_ovlp_fit=True`` improves projection
consistency; ``pasdw_store_funcs=True`` trades memory for reduced repeated
atomic work.

The PAW interpolation code reconstructs all-electron and pseudo partial-wave
quantities on radial grids, including kinetic-energy-density terms required by
meta-GGA models.  Core kinetic-energy contributions are constrained to their
physical lower bound to avoid finite-grid violations, and heavier elements use
a denser all-electron radial treatment.  These are functional-evaluation
choices, not SCF mixer parameters, and should be kept fixed across quantities
being compared.

Forces and stress require derivatives of both the FFT contribution and the
PAW/PASDW projection terms.  Production meta-GGA forces and stress therefore
require PAW setups.  Norm-conserving pseudopotentials omit the all-electron
information needed by NLDF models and are not a supported production route.

Parallel and restart state
--------------------------

FFT and augmented-grid work can be distributed with
``parallel={"augment_grids": True}``.  CiderPress and GPAW must be built with
compatible FFT, MPI, BLAS, and OpenMP runtimes; otherwise failures can occur
before the functional is evaluated correctly.

A full meta-GGA restart needs wavefunctions to reconstruct the kinetic-energy
density.  Both the baseline and CIDER checkpoints should therefore be written
with ``mode="all"``.  CIDER checkpoint dictionaries also store the nonlocal
interpolation grid so that a resumed calculation evaluates the same numerical
functional.

See :doc:`../usage/gpaw` for executable settings and
:doc:`../workflows/extending` for the implementation contract.
