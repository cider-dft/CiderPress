Electronic Features in CiderPress
==================================

CIDER models predict a grid-resolved exchange or exchange-correlation energy
density from an electronic feature vector.  Density-based models use
functionals of :math:`n(\mathbf r)`; generalized Kohn--Sham models may also
depend on the one-particle density matrix
:math:`n_1(\mathbf r,\mathbf r')`:

.. math::

   E_{\mathrm{xc}} = \int d^3\mathbf r\,
   e_{\mathrm{xc}}\!\left(\mathbf x[n,n_1](\mathbf r)\right).

The feature representation determines what spatial information is available
to the regression, which exact scaling behavior can be imposed, which
backends can evaluate the model, and which derivatives are available.  The
serialized feature representation is part of the functional definition.

Feature families
----------------

.. list-table:: Electronic information exposed by each feature family
   :header-rows: 1
   :widths: 16 29 28 27

   * - Family
     - Inputs
     - Information represented
     - Packaged use
   * - :ref:`SL <sl_feat>`
     - Density, gradient, and optionally kinetic-energy density
     - Local GGA or meta-GGA environment
     - Present in every packaged family
   * - :ref:`NLDF <nldf_feat>`
     - Density integrated through density-dependent real-space kernels
     - Rotationally invariant finite-neighborhood density shape
     - CIDER23X and CIDER26XC
   * - :ref:`SDMX <sdmx_feat>`
     - Smoothed one-particle density matrix
     - Approximate exchange-hole and orbital information
     - CIDER24X

All models require a semilocal block, even when the learned correction is
driven mainly by nonlocal descriptors.  The density and semilocal ingredients
also define energy baselines, kernel length scales, normalization factors, and
low-density regularization.

From raw quantities to model inputs
-----------------------------------

The backend arrays pass through three layers before regression:

1. A settings object declares the raw quantities and their parameters.
2. Physical normalizers combine them into dimensionless or
   controlled-scaling descriptors.
3. Bounded transforms place the descriptors in coordinates suitable for the
   mapped Gaussian-process evaluator.

The complete ordered settings, normalizers, and transforms are serialized in
the model and are loaded as one unit.  An explicitly constructed settings
object represents the same descriptors when its feature order, spin
convention, exponents, normalization, and transforms all match.

Family lineage
--------------

CIDER23X combines semilocal ingredients with efficient version-j NLDF
descriptors.  CIDER24X uses SDMX and can train on orbital-occupation
derivatives.  CIDER26XC uses version-j NLDF with separate transformed inputs
for its learned exchange and correlation components.  The learned energy
forms and CIDER26XC feature vectors are explained in
:doc:`../theory/full_xc`; model/backend compatibility is listed in
:doc:`../usage/production_models`.

Numerical and physical constraints
----------------------------------

Scale-invariant exchange descriptors allow the exact uniform-coordinate
scaling of exchange to be built into the energy form.  Correlation has a
different scaling structure and may retain explicit density dependence.
Scalar contractions of vector or angular components provide rotational
invariance.  The model's exchange or correlation energy form determines how
spin channels are combined.

At very low density, near nuclei, and near interpolation boundaries, feature
regularization and numerical evaluation must be considered together.  A
regularization applied in the forward evaluation must have the corresponding
adjoint derivative.  See
:doc:`../theory/uniform_scaling`, :doc:`../theory/nldf_numerical`, and
:doc:`../theory/numerical_evaluation` for these constraints, and
:doc:`../workflows/extending` before changing a feature implementation.

.. toctree::
   :maxdepth: 1
   :caption: Feature definitions

   sl
   nldf
   sdmx
