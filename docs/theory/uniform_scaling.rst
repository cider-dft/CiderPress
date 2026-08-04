.. _unif_scaling:

Uniform Scaling
===============

Uniform scaling is a critical concept for understanding density functional
design. Consider a given electron density distribution :math:`n(\mathbf{r})`.
We can consider a transformed density of the form

.. math:: n_\lambda(\mathbf{r}) = \lambda^3 n(\lambda \mathbf{r})

This transformation is called *uniform coordinate scaling*,\ :footcite:p:`Martin2004`
because it essentially involves redefining
:math:`\mathbf{r}\leftarrow\lambda\mathbf{r}`, which compresses
(:math:`\lambda>1`) or expands (:math:`0<\lambda<1`) the
density while maintaining its relative shape. The :math:`\lambda^3`
prefactor ensures that the particle number is conserved, i.e.

.. math:: \int \text{d}^3\mathbf{r} n_\lambda(\mathbf{r}) = \int \text{d}^3\mathbf{r} n(\mathbf{r})

One can also consider uniform scaling of the density *matrix*:

.. math:: n_1^\lambda(\mathbf{r}, \mathbf{r}') = \lambda^3 n_1(\lambda \mathbf{r}, \lambda \mathbf{r}')

Density-matrix scaling
----------------------

For an orbital-dependent functional, scaling a density and scaling one
particular density matrix are distinct operations because multiple density
matrices can yield the same density.  Orbital relaxation under the scaled
potential can therefore make the minimizing density matrix at
:math:`n_\lambda` differ from :math:`n_1^\lambda`, the coordinate-scaled form
of the original minimizing density matrix.  The distinction is important
when a statement depends explicitly on orbitals or the one-particle density
matrix; see :footcite:t:`Gorling1995`.  The density notation
:math:`n_\lambda` is used below for the density-functional scaling relations.

The exact exchange functional :math:`E_\text{x}[n]` has a simple,
exact behavior under uniform scaling:\ :footcite:p:`Levy1985`

.. math:: E_\text{x}[n_\lambda] = \lambda E_\text{x}[n]

The correlation functional :math:`E_\text{c}[n]` does not have such simple
behavior under uniform scaling, but it does obey the limits\ :footcite:p:`Kaplan2023`

.. math::

   \lim_{\lambda\rightarrow 0} E_\text{c}[n_\lambda] &= \lambda C_0[n] \\
   \lim_{\lambda\rightarrow \infty} E_\text{c}[n_\lambda] &> -\infty

Therefore, it can be helpful to design features with simple, well-understood
behavior under uniform scaling. In particular, if the feature vector :math:`\mathbf{x}`
for the ML model is *scale-invariant*, i.e. if

.. math:: \mathbf{x}[n_\lambda](\mathbf{r}) = \mathbf{x}[n](\lambda \mathbf{r})

then an exchange functional of the form

.. math:: E_\text{x}[n] = \int \text{d}^3\mathbf{r}\, e_\text{x}^\text{LDA}(n(\mathbf{r}))\, F_\text{x}^\text{ML}(\mathbf{x}(\mathbf{r}))

obeys the uniform scaling rule for exchange (:math:`E_\text{x}[n_\lambda] = \lambda E_\text{x}[n]`).
The LDA exchange energy density :math:`e_\text{x}^\text{LDA}\propto n^{4/3}`
supplies the required :math:`\lambda` scaling, while the learned enhancement
factor :math:`F_\text{x}^\text{ML}` is scale-invariant because its inputs are.
Scale-invariant inputs can also be combined with a correlation baseline, but
then their multiplicative energy form inherits the baseline's scaling
behavior.  Because exact correlation is not homogeneous under coordinate
scaling, a full-XC model may need explicit density dependence in addition to
scale-invariant descriptors.  The CIDER26XC construction follows this route,
as described in :doc:`full_xc`.

.. footbibliography::
