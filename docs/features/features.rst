Electronic Features in CiderPress
==================================

CIDER models predict a grid-resolved exchange or exchange-correlation energy
density from an electronic feature vector.  In a pure density functional the
features are functionals of :math:`n(\mathbf r)`; generalized Kohn--Sham
models may additionally depend on the one-particle density matrix
:math:`n_1(\mathbf r,\mathbf r')`:

.. math::

   E_{\mathrm{xc}} = \int d^3\mathbf r\,
   e_{\mathrm{xc}}\!\left(\mathbf x[n,n_1](\mathbf r)\right).

The feature representation determines what spatial information is available
to the regression, which exact scaling behavior can be imposed, which
backends can evaluate the model, and which derivatives are available.  It is
therefore part of the functional definition rather than a tunable SCF option.

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
   * - :ref:`NLOF <nlof_feat>`
     - Fractional-Laplacian orbital quantities
     - Experimental orbital nonlocality
     - No packaged production model

All models require a semilocal block, even when the learned correction is
driven mainly by nonlocal descriptors.  The density and semilocal ingredients
also define energy baselines, kernel length scales, normalization factors, and
low-density regularization.

From raw quantities to model inputs
-----------------------------------

The arrays produced by a backend are not necessarily the coordinates passed
to regression.  CiderPress applies three layers:

1. A settings object declares the raw quantities and their parameters.
2. Physical normalizers combine quantities into dimensionless or
   controlled-scaling descriptors.
3. bounded transforms place the descriptors in coordinates suitable for the
   mapped Gaussian-process evaluator.

The complete ordered settings, normalizers, and transforms are serialized in
the model.  A packaged model should therefore be loaded as a unit.  Recreating
an ``NLDFSettings`` or ``SDMXSettings`` object with similar parameters is not
equivalent unless feature order, spin convention, exponents, normalization,
and transforms all match.

Family lineage
--------------

CIDER23X combines semilocal ingredients with efficient version-j NLDF
descriptors.  CIDER24X replaces the density-only nonlocal block with SDMX and
can train on orbital-occupation derivatives.  CIDER26XC uses version-j NLDF
again, but supplies separate transformed inputs to learned exchange and
correlation components.  The learned energy forms are explained in
:doc:`../theory/full_xc`; model/backend compatibility is listed in
:doc:`../usage/production_models`.

Numerical and physical constraints
----------------------------------

Scale-invariant exchange descriptors allow the exact uniform-coordinate
scaling of exchange to be built into the energy form.  Correlation has a
different scaling structure and may retain explicit density dependence.
Rotational invariance is obtained through scalar contractions of vector or
angular components, while spin channels are combined according to the model's
exchange or correlation contract.

At very low density, near nuclei, and near interpolation boundaries, feature
regularization is inseparable from numerical stability.  A change that makes
forward features finite must also preserve their adjoint derivatives.  See
:doc:`../theory/uniform_scaling`, :doc:`../theory/nldf_numerical`, and
:doc:`../theory/numerical_evaluation` for these constraints, and
:doc:`../workflows/extending` before changing a feature implementation.

.. toctree::
   :maxdepth: 1
   :caption: Feature definitions

   sl
   nldf
   sdmx
   nlof
