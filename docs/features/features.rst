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

Every packaged model begins with a semilocal block.  Its density ingredients
also enter the energy baselines, kernel length scales, normalization factors,
and low-density regularization.

Representation in the code
--------------------------

For each grid point, CiderPress evaluates raw features :math:`\mathbf X_0`,
applies the model's physical normalizers :math:`\mathcal N`, and then applies
the bounded feature maps :math:`\mathcal T` used by the regression:

.. math::

   \mathbf X_0(\mathbf r)
   \xrightarrow{\mathcal N}
   \widetilde{\mathbf X}_0(\mathbf r)
   \xrightarrow{\mathcal T}
   \mathbf X_1(\mathbf r)
   \xrightarrow{f_{\mathrm{ML}}}
   e_{\mathrm{xc}}(\mathbf r).

The corresponding objects are:

.. list-table:: Feature representation objects
   :header-rows: 1
   :widths: 24 36 40

   * - Stage
     - Class
     - Role
   * - Feature declaration
     - :class:`~ciderpress.dft.settings.FeatureSettings`
     - Stores the ordered semilocal, NLDF, or SDMX settings used by a model
   * - Physical normalization
     - :class:`~ciderpress.dft.feat_normalizer.FeatNormalizerList`
     - Converts the raw feature powers to the representation expected by the
       energy form
   * - Bounded feature map
     - :class:`~ciderpress.dft.transform_data.FeatureList`
     - Maps normalized features to the coordinates of one regression kernel
   * - Mapped evaluator
     - :class:`~ciderpress.dft.xc_evaluator.MappedXC` or
       :class:`~ciderpress.dft.xc_evaluator2.MappedXC2`
     - Evaluates the mapped exchange or full-XC model and its derivatives

:class:`~ciderpress.dft.settings.SemilocalSettings`,
:class:`~ciderpress.dft.settings.NLDFSettingsVJ`, and
:class:`~ciderpress.dft.settings.SDMXSettings` define the individual feature
blocks.  Their numerical plans are implemented by
:class:`~ciderpress.dft.plans.SemilocalPlan`,
:class:`~ciderpress.dft.plans.NLDFSplinePlan`, and
:class:`~ciderpress.dft.plans.SDMXPlan`.

The settings, normalizers, and mapped kernels are serialized together in a
CIDER model file.  The PySCF and GPAW interfaces reconstruct them when the
functional is loaded.

The equations and parameter conventions for each block are given below.
Uniform-scaling constraints are derived in :doc:`../theory/uniform_scaling`,
and the exchange and correlation energy forms used by CIDER26XC are defined
in :doc:`../theory/full_xc`.  Backend support for the packaged families is
listed in :doc:`../usage/production_models`.

.. toctree::
   :maxdepth: 1
   :caption: Feature definitions

   sl
   nldf
   sdmx
