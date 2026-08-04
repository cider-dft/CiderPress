GPAW FFT, PAW, and Radial Implementation
========================================

This page describes the numerical implementation beneath the GPAW calculator
interface. These modules are linked so that the forward and derivative paths
can be inspected, but their classes are implementation details rather than
compatibility-stable user APIs. Production calculations should construct the
functional with :func:`~ciderpress.gpaw.calculator.get_cider_functional` and
use :class:`~ciderpress.gpaw.calculator.CiderGPAW`.

Smooth-grid forward and adjoint
-------------------------------

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

An NLDF at a point depends on density in a finite surrounding region. A
correction confined to the final on-site energy therefore cannot restore the
all-electron source seen by a convolution outside the augmentation sphere.
CiderPress uses the projector augmented-wave and smooth double-wave
(PAW/PASDW) construction introduced with CIDER23X. :footcite:p:`CIDER23X`

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

The on-site all-electron and pseudo features are then evaluated with the same
returned field, plus the localized residual required on the all-electron
side. This yields the normal PAW energy difference while retaining the
nonlocal information transported through the cell grid.

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

.. py:module:: ciderpress.gpaw.atom_descriptor_utils

:mod:`ciderpress.gpaw.atom_descriptor_utils` applies the corresponding atomic
corrections when extracting fixed-density descriptor arrays and occupation
derivatives.

``pasdw_ovlp_fit`` selects overlap fitting for the transfer and should remain
enabled for sensitive comparisons. ``pasdw_store_funcs`` controls caching of
atom-centered projector values: it changes memory use and repeated cost, not
the mathematical functional.

Radial reconstruction and on-site correction
---------------------------------------------

.. py:module:: ciderpress.gpaw.interp_paw

:mod:`ciderpress.gpaw.interp_paw` is one layer of the PAW/PASDW route, not the
whole algorithm. It reconstructs differentiable all-electron and pseudo
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

For heavier elements, the implementation retains the setup's native dense
all-electron radial form inside the required cutoff; lighter elements use the
generated CIDER radial representation. This choice limits interpolation error
without exposing an element-specific SCF setting.

Two safeguards preserve physical and numerical invariants:

* A mapped point that exceeds an interpolation endpoint only by floating-point
  roundoff is clipped to that endpoint. A genuine request outside the source
  radial grid raises an error instead of silently extrapolating partial waves.
* Interpolated all-electron and pseudo core kinetic-energy densities are
  bounded below by the von Weizsaecker value
  :math:`\tau_{\mathrm W}=|\nabla n|^2/(8n)` (with the corresponding
  spherical-harmonic normalization used on the radial grid). This removes a
  finite-representation violation of the Fermi-hole-curvature/iso-orbital
  bound; it is not a change to the model or an SCF convergence device.

Forward/derivative contract
---------------------------

The PASDW transfer is linear, so the returned potential must be the exact
adjoint of the discretized forward transfer. Forces additionally differentiate
the localized functions and projectors with respect to atomic position;
stress differentiates the reciprocal kernel and the cell-scaled transfer.
A feature is not validated by matching an energy alone: smooth-grid and PAW
feature values, potentials, forces, and stress must all be checked with the
same interpolation and projection settings.

Mixer changes can help the density reach self-consistency but cannot repair a
mismatched forward/adjoint pair. See :doc:`../../usage/convergence` for SCF
fallbacks, :doc:`../../theory/numerical_evaluation` for the complete backend
path, and :doc:`../../workflows/extending` for implementation invariants.

.. footbibliography::
