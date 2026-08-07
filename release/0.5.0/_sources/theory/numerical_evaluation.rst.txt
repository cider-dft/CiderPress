Numerical Evaluation in PySCF and GPAW
======================================

CiderPress uses a common mapped-model layer with separate molecular and
plane-wave feature generators.  Their shared evaluation contract begins with
the feature definitions in :doc:`../features/features`; the Version-J
convolution is derived in :doc:`nldf_numerical`.

Mapped-model contract
---------------------

Let :math:`\mathbf X_0[z](\mathbf r)` denote the ordered raw features stored in
a model.  The numerical variables :math:`z` include the spin densities and
their semilocal ingredients.  For an SDMX model, :math:`z` also includes the
one-particle density matrix.  Each mapped kernel evaluates

.. math::

   \mathbf X_0
   \xrightarrow{\mathcal N}
   \overline{\mathbf X}_0
   \xrightarrow{\mathcal T_\kappa}
   \mathbf X_{1,\kappa}
   \xrightarrow{f_\kappa}
   e_\kappa,

where :math:`\mathcal N` is the model's physical normalization and
:math:`\mathcal T_\kappa` is the bounded feature map associated with kernel
:math:`\kappa`.  The total energy is

.. math::

   E_{\mathrm{xc}} = \int \mathrm d^3\mathbf r\,
   e_{\mathrm{xc}}(\mathbf r),
   \qquad
   e_{\mathrm{xc}}=\sum_\kappa e_\kappa.

:class:`~ciderpress.dft.feat_normalizer.FeatNormalizerList` applies
:math:`\mathcal N`, and each mapped kernel stores its
:class:`~ciderpress.dft.transform_data.FeatureList`.  The evaluator propagates
derivatives back through both layers to obtain

.. math::

   v_i(\mathbf r)
   = \frac{\partial e_{\mathrm{xc}}(\mathbf r)}
           {\partial X_{0,i}(\mathbf r)}.

The feature generator then applies its discrete adjoint:

.. math::

   \frac{\delta E_{\mathrm{xc}}}{\delta z}
   = \sum_i \left(D_z X_{0,i}\right)^{\!*}v_i
     + \left.\frac{\partial E_{\mathrm{xc}}}{\partial z}
       \right|_{\mathbf X_0}.

The last term is present when an energy baseline depends explicitly on the
density variables.  This notation covers local density and gradient
potentials, the generalized Kohn--Sham kinetic-energy-density term, and the
density-matrix potential produced by SDMX.

Mapped evaluators
-----------------

.. list-table:: Inference-time model objects
   :header-rows: 1
   :widths: 24 38 38

   * - Object
     - Energy form
     - Returned derivatives
   * - :class:`~ciderpress.dft.xc_evaluator.MappedXC`
     - Applies baselines expressed through the feature array.  Packaged
       CIDER23X and CIDER24X models use this representation.
     - Derivatives with respect to normalized features; the backend reverses
       normalization to obtain :math:`v_i`.
   * - :class:`~ciderpress.dft.xc_evaluator2.MappedXC2`
     - Evaluates mapped kernels whose multiplicative and additive baselines
       depend directly on density, gradient, or kinetic-energy density.
       CIDER26XC uses this representation. :footcite:p:`CIDER26XC`
     - Feature derivatives and a separate ``vrho_tuple`` containing the
       explicit derivatives of the baselines.

The kernel mode stored in the model determines its spin contraction.
``SEP`` evaluates a contribution for each spin channel; ``NPOL`` evaluates
the transformed features of their mean.  CIDER26XC uses separate-spin
exchange kernels and a spin-averaged feature vector for its correlation
kernel, consistent with the energy form in the CIDER26XC manuscript.
:footcite:p:`CIDER26XC`

PySCF molecular path
--------------------

:func:`~ciderpress.pyscf.dft.make_cider_calc` installs
:class:`~ciderpress.pyscf.numint.CiderNumInt` on a PySCF RKS or UKS object.
The numerical integrator selects feature generators from the settings loaded
with the model and assembles :math:`\mathbf X_0` in serialized order.

Semilocal and SDMX features can be evaluated on the current quadrature block.
An NLDF couples different grid points, so
:func:`~ciderpress.pyscf.numint.nr_rks` and
:func:`~ciderpress.pyscf.numint.nr_uks` use two passes:

1. Evaluate the density ingredients on the complete molecular quadrature grid
   and call ``LCAONLDFGenerator.get_features`` in
   :source:`ciderpress/dft/lcao_nldf_generator.py`.
2. Evaluate the mapped energy and feature derivatives block by block.  Store
   the weighted NLDF derivatives on the complete grid.
3. Call ``LCAONLDFGenerator.get_potential`` once, add its density, gradient,
   and kinetic-energy-density contributions, and contract the completed
   potential with the atomic orbitals.

For SDMX, :class:`~ciderpress.pyscf.sdmx.EXXSphGenerator` constructs features
from the density matrix and adds the returned adjoint directly to the AO
matrix. :footcite:p:`CIDER24X`  Restricted and unrestricted calculations use
arrays of shape ``(nspin, nfeature, ngrid)``; the mapped kernel's serialized
mode performs the spin contraction.

The mapped evaluators return an energy per unit volume.  PySCF's numerical
integration interface expects ``exc`` per particle, so ``eval_xc_cider`` in
:source:`ciderpress/pyscf/numint.py` divides the mapped result by the total
density.  PySCF then accumulates
:math:`\sum_g w_g n_g\,\mathtt{exc}_g`.

Analytical molecular gradients differentiate the atom-centered quadrature and
the NLDF auxiliary representation in
:mod:`ciderpress.pyscf.rks_grad` and :mod:`ciderpress.pyscf.uks_grad`.  The
implementation details are given in :doc:`../ciderpress/pyscf/numerical`.

GPAW plane-wave and PAW path
----------------------------

:func:`~ciderpress.gpaw.calculator.get_cider_functional` constructs a
:class:`~ciderpress.gpaw.cider_fft.CiderGGA` or
:class:`~ciderpress.gpaw.cider_fft.CiderMGGA` functional.  Its smooth-grid
calculation proceeds as follows:

1. :mod:`ciderpress.gpaw.cider_fft` collects the spin density, gradient
   contractions, and kinetic-energy density required by the model.
2. :class:`~ciderpress.dft.plans.NLDFSplinePlan` generates the Version-J source
   coefficients.  :class:`~ciderpress.gpaw.nldf_interface.LibCiderPW` performs
   the fixed-kernel FFT convolutions, and target interpolation produces the
   NLDF block.
3. :class:`~ciderpress.gpaw.cider_kernel.CiderKernel` assembles and normalizes
   the feature array, evaluates the mapped kernels, and adds their energy
   density directly to GPAW's ``e_g`` array.
4. The target and source interpolation adjoints and the backward convolution
   add the NLDF contribution to GPAW's smooth potential.  The same reciprocal
   kernels supply the cell derivative used for stress.

For PAW calculations, :mod:`ciderpress.gpaw.cider_paw` augments the convolution
source and projects the returned fields onto atomic support grids.  The
all-electron and pseudo on-site evaluations add their energy, potential,
force, and stress corrections to the smooth-grid result.  This PASDW
construction was introduced with the periodic CIDER23X implementation.
:footcite:p:`CIDER23X`

The PAW/PASDW equations, radial reconstruction, and implementation classes are
documented in :doc:`../ciderpress/gpaw/numerical`.  GPAW does not implement the
SDMX descriptors used by CIDER24X.

.. footbibliography::
