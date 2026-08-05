.. _sdmx_feat:

Smoothed Density-Matrix Exchange Features
==========================================

Smoothed density-matrix exchange (SDMX) features describe the one-particle
density matrix near a real-space point.  They are quadratic in the density
matrix and provide rotationally invariant proxies for radial and angular
structure in the exchange hole.  CIDER24X introduced the feature family and
used it to learn exchange from total-energy and, for CIDER24Xe, orbital-energy
data. :footcite:p:`CIDER24X`

Smoothed density matrix
-----------------------

The scalar component begins with a smoothed, spherically averaged density
matrix

.. math::

   \rho^0(R;\mathbf r)
   = \int \mathrm d^3\mathbf r'\,
     h(|\mathbf r'-\mathbf r|;R)n_1(\mathbf r',\mathbf r),

where :math:`R` is a smoothing length. CIDER24X uses

.. math::

   h(u;R)
   = \left(\frac{2}{\pi}\right)^{3/2}
     \frac{4}{4-\sqrt{2}}\frac{e^{-2u^2/R^2}}{R^3}
     \left(1-e^{-2u^2/R^2}\right).

Integrating the squared smoothed quantity over :math:`R` gives the scalar
features

.. math::

   H_j^0(\mathbf r)
   = 4\pi\int \mathrm dR\,R^{2-j}|\rho^0(R;\mathbf r)|^2.

The radial derivative supplies a second scalar family,

.. math::

   H_j^{0\mathrm d}(\mathbf r)
   = 4\pi\int \mathrm dR\,R^{4-j}
     \left|\frac{\partial\rho^0(R;\mathbf r)}{\partial R}\right|^2.

Angular information
-------------------

The vector-smoothed density matrix is

.. math::

   \boldsymbol{\rho}^1(R;\mathbf r)
   = \int \mathrm d^3\mathbf r'\,
     [\nabla h(|\mathbf r'-\mathbf r|;R)]
     n_1(\mathbf r',\mathbf r).

Its rotationally invariant norm and radial derivative define

.. math::

   H_j^1(\mathbf r)
   &= 4\pi\int \mathrm dR\,R^{4-j}
      |\boldsymbol{\rho}^1(R;\mathbf r)|^2, \\
   H_j^{1\mathrm d}(\mathbf r)
   &= 4\pi\int \mathrm dR\,R^{6-j}
      \left|\frac{\partial\boldsymbol{\rho}^1(R;\mathbf r)}
      {\partial R}\right|^2.

All four families scale as

.. math::

   H_j[n_1^\lambda](\mathbf r)
   = \lambda^{3+j}H_j[n_1](\lambda\mathbf r)

under uniform coordinate scaling of the density matrix. The implemented
uniform-electron-gas normalizations cover :math:`j\in\{0,1,2\}`. The
normalizers stored with a model convert these raw powers into the coordinates
used by its exchange regression.

CIDER24X feature layout
-----------------------

Both packaged CIDER24X models contain the same 13 raw electronic features:
three semilocal meta-GGA ingredients followed by ten SDMX features. The SDMX
block is ordered as

.. math::

   \left(
   H_1^0,H_2^0,
   H_1^{0\mathrm d},H_2^{0\mathrm d},
   H_1^1,H_2^1,H_0^1,
   H_1^{1\mathrm d},H_2^{1\mathrm d},H_0^{1\mathrm d}
   \right).

This layout is represented by
:class:`~ciderpress.dft.settings.SDMXFullSettings`. Feature order,
normalization, bounded transformations, and mapped neural-network weights are
stored in each model file and are applied automatically during evaluation.

The optimized molecular implementation is available through PySCF.  The
periodic PySCF implementation supports the CIDER24X methodology.  The GPAW
interface evaluates NLDF models.  See
:doc:`../usage/production_models` for the supported calculation path and
:doc:`../ciderpress/pyscf/numerical` for the molecular contraction algorithm.

.. footbibliography::
