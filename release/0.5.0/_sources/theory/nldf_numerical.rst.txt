.. _nldf_numerical:

Numerical Evaluation of NLDF Features
=====================================

The packaged CIDER23X and CIDER26XC models use the Version-J algorithm.  Its
feature definition and exponent parameterization are given in
:doc:`../features/nldf`.  The numerical construction follows the kernel
expansion introduced for CIDER23X, which adapts the convolution method of
Román-Pérez and Soler to density-dependent CIDER kernels.
:footcite:p:`CIDER23X,Roman-Perez2009`

Kernel expansion
----------------

Suppressing the constant normalization of the feature, a Version-J NLDF is

.. math::

   G_i(\mathbf r)
   = \int \mathrm d^3\mathbf r'\,
     \Phi\!\left(a_0(\mathbf r'),a_i(\mathbf r),
     |\mathbf r-\mathbf r'|\right)n(\mathbf r'),

where

.. math::

   \Phi(a,b,R)=\exp[-(a+b)R^2].

The density-dependent exponents are represented on a finite set of positive
values :math:`\{q_\alpha\}`.  Source and target interpolation functions give

.. math::

   \Phi(a,b,R)
   \simeq \sum_{\alpha\beta}
   p^a_\alpha(a)\,\Phi_{\alpha\beta}(R)\,p^b_\beta(b),
   \qquad
   \Phi_{\alpha\beta}(R)
   = \exp[-(q_\alpha+q_\beta)R^2].

This is Equation 61 of the CIDER23X paper.  It converts the feature into three
operations:

.. math::

   \theta_\alpha(\mathbf r')
   &= p^a_\alpha(a_0(\mathbf r'))n(\mathbf r'), \\
   F_\beta(\mathbf r)
   &= \sum_\alpha \int \mathrm d^3\mathbf r'\,
      \Phi_{\alpha\beta}(|\mathbf r-\mathbf r'|)
      \theta_\alpha(\mathbf r'), \\
   G_i(\mathbf r)
   &\simeq \sum_\beta
      p^b_\beta(a_i(\mathbf r))F_\beta(\mathbf r).

Only the middle line is spatially nonlocal, and every kernel in that line is
independent of the density.  CiderPress therefore evaluates a finite set of
ordinary convolutions and performs local interpolation before and after them.

The potential applies the transpose of the same discrete operations.  It
first propagates :math:`\partial E_{\mathrm{xc}}/\partial G_i` through the
target coefficients, applies the adjoint convolution, and then differentiates
the source coefficients and source density.  Derivatives of
:math:`p_\alpha(a)` supply the response of the semilocal exponents
:math:`a_0[n]` and :math:`a_i[n]`.

Exponent coefficients and spatial convolutions
----------------------------------------------

Exponent interpolation and spatial convolution are separate numerical
choices.  Their implementation objects are:

.. list-table:: Numerical representations used in the NLDF evaluator
   :header-rows: 1
   :widths: 22 30 48

   * - Operation
     - Implementation
     - Representation
   * - Exponent coefficients
     - :class:`~ciderpress.dft.plans.NLDFGaussianPlan`
     - Projects :math:`\exp(-aR^2)` onto the Gaussian functions at
       :math:`q_\alpha` using their overlap matrix, as in CIDER23X
       Equations 63--65.
   * - Exponent coefficients
     - :class:`~ciderpress.dft.plans.NLDFSplinePlan`
     - Tabulates the projection coefficients as functions of the exponent
       coordinate and evaluates them with cubic splines.
   * - Molecular convolution
     - :class:`~ciderpress.pyscf.nldf_convolutions.PyscfNLDFGenerator`
     - Expands the spatial source and convolved fields in atom-centered
       Gaussian auxiliary bases.
   * - Periodic convolution
     - :class:`~ciderpress.gpaw.nldf_interface.LibCiderPW`
     - Applies the fixed kernels on the distributed reciprocal-space grid.

The current PySCF default uses :class:`~ciderpress.dft.plans.NLDFSplinePlan`
for the exponent coefficients.  Its spatial convolution still uses the
atom-centered Gaussian auxiliary bases described below.  GPAW also uses
:class:`~ciderpress.dft.plans.NLDFSplinePlan`, with an even-tempered exponent
grid :math:`q_\alpha=q_0\lambda^\alpha`.

Molecular auxiliary-basis algorithm
-----------------------------------

The molecular path implements the CIDER23X construction of Equations 67--78.
For each source channel, it performs the following sequence:

1. :class:`~ciderpress.pyscf.gen_cider_grid.CiderGrids` records the atomic
   radial and angular structure of PySCF's partitioned quadrature grid.
2. :class:`~ciderpress.dft.lcao_nldf_generator.LCAONLDFGenerator` forms
   :math:`\theta_\alpha` on that grid, resolves each atomic contribution into
   spherical harmonics, and projects the radial components onto an
   even-tempered Gaussian auxiliary basis.
3. :class:`~ciderpress.dft.lcao_convolutions.ConvolutionCollection` applies
   analytically evaluated Gaussian convolution integrals to obtain the
   coefficients of :math:`F_\beta` in a second auxiliary basis.
4. :class:`~ciderpress.dft.lcao_interpolation.LCAOInterpolatorDirect` evaluates
   the convolved fields on the molecular quadrature grid.  The target
   coefficients then produce :math:`G_i`.

The ``LCAONLDFGenerator.get_potential`` call in
:source:`ciderpress/dft/lcao_nldf_generator.py` reverses this sequence using
the stored forward intermediates.  Analytical nuclear gradients also
differentiate the atomic partition, auxiliary projection, and grid
interpolation; see
:doc:`../ciderpress/pyscf/numerical`.

Periodic FFT and PAW algorithm
------------------------------

For a periodic cell, the Fourier transform of a fixed Gaussian kernel is

.. math::

   \widetilde\Phi_{\alpha\beta}(\mathbf k)
   = \left(\frac{\pi}{q_\alpha+q_\beta}\right)^{3/2}
     \exp\!\left[-\frac{|\mathbf k|^2}
     {4(q_\alpha+q_\beta)}\right].

:mod:`ciderpress.gpaw.cider_fft` constructs the source interpolation
coefficients and passes the source channels to
:class:`~ciderpress.gpaw.nldf_interface.LibCiderPW`.  The compiled evaluator
transforms them to reciprocal space, multiplies by
:math:`\widetilde\Phi_{\alpha\beta}`, and transforms the convolved fields back
to the real-space grid.  Target interpolation produces the NLDFs.  The reverse
calls supply the smooth-grid potential and the reciprocal-kernel contribution
to stress.

The smooth density in PAW lacks the all-electron source inside augmentation
spheres.  PASDW augments the source before the FFT so that the convolved field
has the correct exterior behavior, then projects that field onto atomic
support grids for the all-electron-minus-pseudo energy and potential
correction.  This is the construction in CIDER23X Equations 82--98. :footcite:p:`CIDER23X`
Its equations and implementation objects are documented in
:doc:`../ciderpress/gpaw/numerical`.

Numerical parameters
--------------------

.. list-table:: Parameters that define the NLDF discretization
   :header-rows: 1
   :widths: 24 24 52

   * - Path
     - Parameter
     - Effect
   * - GPAW
     - ``qmax``
     - Largest interpolation exponent, in inverse bohr squared.
   * - GPAW
     - ``lambd``
     - Ratio between adjacent exponents.  With an automatically selected
       ``Nalpha``, a smaller ratio gives a denser exponent grid.
   * - GPAW
     - ``Nalpha``
     - Number of exponent channels.  The value is inferred from ``qmax``,
       ``lambd``, and the model exponents when omitted.
   * - PySCF
     - ``alpha_min``, ``alpha_max``, ``aux_lambd``
     - Exponent range and spacing used when constructing the molecular NLDF
       generator.
   * - PySCF
     - ``lmax``, ``aug_beta``
     - Angular truncation and spacing of the atom-centered auxiliary basis.

The host quadrature or real-space grid adds another discretization to these
kernel and auxiliary representations.  PAW calculations also contain the
finite PASDW projection bases described on the GPAW implementation page.

The features described above are labeled the Verion-J NLDF features.
Version-I and Version-K definitions are listed in :doc:`../features/nldf`.
They are experimental features and are not used in the packaged Version-J
functionals described here.

.. footbibliography::
