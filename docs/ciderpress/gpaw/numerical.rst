GPAW FFT, PAW, and Radial Implementation
========================================

The GPAW backend connects the mapped functional to smooth-grid FFT operations,
PAW/PASDW corrections, and their derivatives.  Calculations construct the
functional with :func:`~ciderpress.gpaw.calculator.get_cider_functional` and
use :class:`~ciderpress.gpaw.calculator.CiderGPAW`; the modules below implement
the numerical layers beneath those interfaces.

Smooth-grid forward and adjoint
-------------------------------

.. py:module:: ciderpress.gpaw.cider_kernel

:mod:`ciderpress.gpaw.cider_kernel` is the bridge between GPAW density arrays
and :class:`~ciderpress.dft.xc_evaluator.MappedXC` or
:class:`~ciderpress.dft.xc_evaluator2.MappedXC2`.  It assembles semilocal and
NLDF blocks in serialized order, applies normalization, calls the mapped
model, and distributes the returned derivatives to the density, gradient,
kinetic-energy-density, and NLDF potentials.  The GGA/MGGA kernel classes also
assemble the explicit exchange and correlation terms selected by
``get_cider_functional``.

.. py:module:: ciderpress.gpaw.cider_sl

:mod:`ciderpress.gpaw.cider_sl` connects packaged semilocal CIDER23X models to
the same GGA/MGGA and PAW lifecycle.

.. py:module:: ciderpress.gpaw.nldf_interface

:mod:`ciderpress.gpaw.nldf_interface` wraps the compiled periodic NLDF
library. It distributes the real and reciprocal arrays, evaluates the kernel
interpolation and FFT convolutions, and provides the reverse contractions
needed for the XC potential and stress.

.. py:class:: LibCiderPW

   Owns the compiled serial or MPI convolution plan and its work arrays. Its
   forward and backward operations are an adjoint pair.

.. py:class:: FFTWrapper

   Adapts GPAW's plane-wave descriptor and array distribution to the compiled
   CIDER FFT interface.

.. py:module:: ciderpress.gpaw.cider_fft

:mod:`ciderpress.gpaw.cider_fft` connects those operations to GPAW's GGA and
meta-GGA XC lifecycle. It gathers the smooth density ingredients, evaluates
the mapped energy density, adds the backward feature potential, and supplies
the smooth-grid force and stress terms.

.. py:class:: CiderGGA

   Smooth-grid GGA CIDER implementation.

.. py:class:: CiderMGGA

   Smooth-grid meta-GGA implementation, including kinetic-energy-density
   input and potential.

The interpolation controls ``qmax``, ``lambd``, and ``Nalpha`` define the
numerical representation of the density-dependent kernel. Energy, potential,
force, and stress calculations being compared must use the same controls.

Why PAW needs PASDW
-------------------

In ordinary PAW, a local or semilocal XC energy can be decomposed into a
smooth cell term and atom-centered all-electron-minus-pseudo corrections,

.. math::

   E_{\mathrm{xc}}
   = \widetilde E_{\mathrm{xc}}
   + \sum_A\left(E_{\mathrm{xc}}^A-\widetilde E_{\mathrm{xc}}^A\right).

An NLDF at a point depends on density in a finite surrounding region.  A
final on-site energy correction leaves the convolution source outside the
augmentation sphere unchanged.  CiderPress transports that all-electron
source with the PAW/PASDW construction introduced with
CIDER23X. :footcite:p:`CIDER23X`

The transfer has two coupled stages. First, localized source functions
:math:`g_i^A` augment the smooth kernel source,

.. math::

   \widetilde\theta_\alpha^{\mathrm{aug}}(\mathbf r)
   = \widetilde\theta_\alpha(\mathbf r)
   + \sum_{A i} C_{i\alpha}^A g_i^A(\mathbf r),

so that the convolved smooth feature reproduces the all-electron feature
outside each augmentation sphere. Second, localized feature partial waves
and dual projectors transfer the convolved fields back to the atomic support
grids. Their projection coefficients have the form

.. math::

   B_{j\beta}^A
   = \Delta v\sum_{g\in A}
   \widetilde F_\beta^{\mathrm{aug}}(\mathbf r_g)
   \widetilde p_j^A(\mathbf r_g).

The on-site all-electron and pseudo features use the same returned field, with
the required localized residual added on the all-electron side.  Their
difference gives the PAW correction and includes the nonlocal information
transported through the cell grid.

PAW/PASDW modules
-----------------

.. py:module:: ciderpress.gpaw.cider_paw

:mod:`ciderpress.gpaw.cider_paw` coordinates the atom-to-grid source
augmentation, grid-to-atom return projection, on-site energy/potential, and
their force and stress contributions.

.. py:class:: CiderGGAPASDW

   GGA smooth-grid functional with PASDW augmentation corrections.

.. py:class:: CiderMGGAPASDW

   Meta-GGA counterpart, including PAW kinetic-energy-density response.

.. py:module:: ciderpress.gpaw.atom_utils

:mod:`ciderpress.gpaw.atom_utils` constructs the radial/angular auxiliary
bases, localized source and projector functions, overlap fits, and the
forward/backward atomic contractions used by PASDW.

.. py:class:: FastPASDWCiderKernel

   Owns the per-setup PASDW data and applies atom-to-grid, grid-to-atom,
   energy, potential, force, and stress contractions for a calculation.

.. py:module:: ciderpress.gpaw.fit_paw_gauss_pot

:mod:`ciderpress.gpaw.fit_paw_gauss_pot` builds the localized radial source
and projector bases.

.. py:module:: ciderpress.dft.pwutil

:mod:`ciderpress.dft.pwutil` wraps the compiled spline, spherical-harmonic,
and atom/grid contraction kernels.

.. py:module:: ciderpress.gpaw.atom_descriptor_utils

:mod:`ciderpress.gpaw.atom_descriptor_utils` applies the corresponding atomic
corrections when extracting fixed-density descriptor arrays and occupation
derivatives.

``pasdw_ovlp_fit`` selects overlap fitting for the transfer; use the same
value throughout a numerical comparison.  ``pasdw_store_funcs`` controls
caching of atom-centered projector values.  Caching uses more memory to reduce
repeated cost and preserves the evaluated functional.

Radial reconstruction and on-site correction
---------------------------------------------

.. py:module:: ciderpress.gpaw.interp_paw

:mod:`ciderpress.gpaw.interp_paw` handles the radial on-site layer of the
PAW/PASDW route.  It reconstructs differentiable all-electron and pseudo
partial-wave density ingredients on an appropriate radial grid and evaluates
the on-site GGA, meta-GGA, or CIDER correction. The returned derivatives are
contracted back into GPAW's PAW density matrices.

.. py:class:: DiffPAWXCCorrection

   Container for interpolated all-electron and pseudo partial-wave densities,
   radial derivatives, kinetic-energy-density terms, and projector
   transformations for one GPAW setup.

.. py:class:: DiffGGA

   Differentiable radial GGA correction used by the CIDER PAW path.

.. py:class:: DiffMGGA

   Differentiable radial meta-GGA correction, including reconstructed
   kinetic-energy-density terms.

.. py:module:: ciderpress.gpaw.gpaw_grids

:mod:`ciderpress.gpaw.gpaw_grids` supplies the radial-grid descriptors used
for this reconstruction.

For heavier elements, the implementation retains the setup's native dense
all-electron radial form inside the required cutoff; lighter elements use the
generated CIDER radial representation.  The implementation selects this
radial treatment by element to control interpolation error.

Two safeguards preserve physical and numerical invariants:

* A mapped point that exceeds an interpolation endpoint by floating-point
  roundoff is clipped to that endpoint.  A request outside the source radial
  grid raises an error.
* Interpolated all-electron and pseudo core kinetic-energy densities are
  bounded below by the von Weizsaecker value
  :math:`\tau_{\mathrm W}=|\nabla n|^2/(8n)` (with the corresponding
  spherical-harmonic normalization used on the radial grid).  This enforces
  the Fermi-hole-curvature/iso-orbital bound in the finite radial
  representation.

Forward/derivative contract
---------------------------

The PASDW transfer is linear, so the returned potential must be the exact
adjoint of the discretized forward transfer. Forces additionally differentiate
the localized functions and projectors with respect to atomic position;
stress differentiates the reciprocal kernel and the cell-scaled transfer.

Validation covers smooth-grid and PAW feature values, potentials, forces, and
stress at the same interpolation and projection settings.  See
:doc:`../../usage/convergence` for SCF controls,
:doc:`../../theory/numerical_evaluation` for the complete backend path, and
:doc:`../../workflows/extending` for implementation invariants.

.. footbibliography::
