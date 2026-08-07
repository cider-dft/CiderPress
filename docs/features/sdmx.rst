.. _sdmx_feat:

Smoothed Density Matrix Exchange (SDMX)
=======================================

Smoothed density matrix exchange (SDMX) features are nonlocal
featurizations of the one-particle density matrix.  The density matrix is
smoothed around a real-space point and projected onto the
:math:`\ell=0` and :math:`\ell=1` angular channels.  Quadratic contractions
of the projected quantities provide rotationally invariant proxies for the
radial and angular structure of the exchange hole
:math:`|n_1(\mathbf r,\mathbf r')|^2`. :footcite:p:`CIDER24X`

The construction was inspired by the Rung 3.5 functionals of Janesko *et
al.* :footcite:p:`Janesko2010,Janesko2013,Janesko2014,Janesko2018`. SDMX
features are quadratic functionals of the density matrix.  The packaged
CIDER24X models use them to learn exchange; ``CIDER24Xe`` also uses
orbital-occupation derivative data during training.

Scalar features
---------------

The scalar component starts from a smoothed, spherically averaged density
matrix

.. math::

   \rho^0(R;\mathbf r)
   = \int \mathrm d^3\mathbf r'\,
     h(|\mathbf r'-\mathbf r|;R)n_1(\mathbf r',\mathbf r),

where :math:`R` is a smoothing length.  CIDER24X uses

.. math::

   h(u;R)
   = \left(\frac{2}{\pi}\right)^{3/2}
     \frac{4}{4-\sqrt{2}}\frac{e^{-2u^2/R^2}}{R^3}
     \left(1-e^{-2u^2/R^2}\right).

The kernel broadens with :math:`R`, so
:math:`\rho^0(R;\mathbf r)` is a smoothed approximation to the spherical
average of the density matrix at distance :math:`R`.  Integrating its square
over the smoothing length gives

.. math::

   H_j^0(\mathbf r)
   = 4\pi\int \mathrm dR\,R^{2-j}|\rho^0(R;\mathbf r)|^2.

The radial derivative supplies a second scalar family,

.. math::

   H_j^{0\mathrm d}(\mathbf r)
   = 4\pi\int \mathrm dR\,R^{4-j}
     \left|\frac{\partial\rho^0(R;\mathbf r)}{\partial R}\right|^2.

Angular features
----------------

The :math:`\ell=1` projection is represented by the vector

.. math::

   \boldsymbol{\rho}^1(R;\mathbf r)
   = \int \mathrm d^3\mathbf r'\,
     [\nabla h(|\mathbf r'-\mathbf r|;R)]
     n_1(\mathbf r',\mathbf r).

Its norm and radial derivative define two more scalar feature families,

.. math::

   H_j^1(\mathbf r)
   &= 4\pi\int \mathrm dR\,R^{4-j}
      |\boldsymbol{\rho}^1(R;\mathbf r)|^2, \\
   H_j^{1\mathrm d}(\mathbf r)
   &= 4\pi\int \mathrm dR\,R^{6-j}
      \left|\frac{\partial\boldsymbol{\rho}^1(R;\mathbf r)}
      {\partial R}\right|^2.

The radial-derivative terms reuse the principal contractions required by the
corresponding :math:`H_j^0` and :math:`H_j^1` features.  The vector channel
adds the :math:`\ell=1` angular information.

Uniform coordinate scaling
--------------------------

For the coordinate-scaled density matrix defined in
:doc:`../theory/uniform_scaling`, all four families obey

.. math::

   H_j[n_1^\lambda](\mathbf r)
   = \lambda^{3+j}H_j[n_1](\lambda\mathbf r).

The implemented uniform-electron-gas normalizations cover
:math:`j\in\{0,1,2\}`.  Model-specific normalizers convert these raw powers
into the scale-invariant coordinates used by the exchange regression.

Numerical representation
------------------------

CiderPress evaluates the smoothed density matrix at a discrete set of
lengths :math:`R_i` using Gaussian convolutions.  It then represents the
:math:`R` dependence in a Gaussian basis, allowing the integrals that define
the :math:`H_j` features to be contracted analytically.  The settings and
contraction plans are implemented by
:class:`~ciderpress.dft.settings.SDMXFullSettings` and
:class:`~ciderpress.dft.plans.SDMXFullPlan`; the molecular contraction
algorithm is described in :doc:`../ciderpress/pyscf/numerical`.

CIDER24X feature layout
-----------------------

Both packaged CIDER24X models contain the same 13 raw electronic features:
three semilocal meta-GGA ingredients followed by ten SDMX features.  The SDMX
block is ordered as

.. math::

   \left(
   H_1^0,H_2^0,
   H_1^{0\mathrm d},H_2^{0\mathrm d},
   H_1^1,H_2^1,H_0^1,
   H_1^{1\mathrm d},H_2^{1\mathrm d},H_0^{1\mathrm d}
   \right).

Feature order, normalization, bounded transforms, and mapped neural-network
weights are stored in each model file.  The optimized molecular
implementation is available through PySCF.  The periodic PySCF interface
supports the CIDER24X methodology with pseudopotentials and uniform grids;
GPAW evaluates the NLDF model families.  Backend compatibility is listed in
:doc:`../usage/production_models`.

.. footbibliography::
