.. _unif_scaling:

Uniform Scaling
===============

Uniform coordinate scaling\ :footcite:p:`Martin2004` relates a density
:math:`n(\mathbf r)` to

.. math:: n_\lambda(\mathbf{r}) = \lambda^3 n(\lambda \mathbf{r})

The transformation compresses
(:math:`\lambda>1`) or expands (:math:`0<\lambda<1`) the density and preserves
its shape and particle number:

.. math:: \int \mathrm{d}^3\mathbf{r}\, n_\lambda(\mathbf{r}) = \int \mathrm{d}^3\mathbf{r}\, n(\mathbf{r}).

The corresponding coordinate-scaled one-particle density matrix is

.. math:: n_1^\lambda(\mathbf{r}, \mathbf{r}') = \lambda^3 n_1(\lambda \mathbf{r}, \lambda \mathbf{r}')

Density-matrix scaling
----------------------

For an orbital-dependent functional, scaling a density and scaling one
particular density matrix are distinct operations because multiple density
matrices can yield the same density.  Orbital relaxation under the scaled
potential can therefore make the minimizing density matrix at
:math:`n_\lambda` differ from :math:`n_1^\lambda`, the coordinate-scaled form
of the original minimizing density matrix.

This distinction enters statements that depend explicitly on orbitals or the
one-particle density matrix; see :footcite:t:`Gorling1995`.  The density notation
:math:`n_\lambda` is used below for the density-functional scaling relations.

The exact exchange functional :math:`E_\text{x}[n]` has a simple,
exact behavior under uniform scaling:\ :footcite:p:`Levy1985`

.. math:: E_\text{x}[n_\lambda] = \lambda E_\text{x}[n]

The correlation functional :math:`E_\text{c}[n]` obeys the low- and
high-density limits\ :footcite:p:`Kaplan2023`

.. math::

   E_\text{c}[n_\lambda] &\rightarrow \lambda\,\mathcal{C}_0[n]
   \quad (\lambda\rightarrow 0) \\
   \lim_{\lambda\rightarrow \infty} E_\text{c}[n_\lambda] &> -\infty

where :math:`\mathcal{C}_0[n]` is a density functional that does not depend
on :math:`\lambda`.

A scale-invariant feature vector satisfies

.. math:: \mathbf{x}[n_\lambda](\mathbf{r}) = \mathbf{x}[n](\lambda \mathbf{r})

An exchange functional of the form

.. math:: E_\text{x}[n] = \int \mathrm{d}^3\mathbf{r}\, e_\text{x}^\text{LDA}(n(\mathbf{r}))\, F_\text{x}^\text{ML}(\mathbf{x}(\mathbf{r}))

then obeys the exchange scaling rule
:math:`E_\text{x}[n_\lambda] = \lambda E_\text{x}[n]`.
The LDA exchange energy density :math:`e_\text{x}^\text{LDA}\propto n^{4/3}`
supplies the required :math:`\lambda` scaling.  The learned enhancement factor
:math:`F_\text{x}^\text{ML}` is scale invariant because its inputs are.

With a correlation baseline, the multiplicative energy form inherits the
baseline's scaling behavior.  Exact correlation has nonhomogeneous coordinate
scaling, so a full-XC model may include explicit density dependence alongside
scale-invariant descriptors.  The CIDER26XC construction follows this route,
as described in :doc:`full_xc`.

.. footbibliography::
