Machine Learning Framework for CIDER Functionals
================================================

CIDER functionals are trained with Gaussian-process regression (GPR). This
section describes the form used for local energy-density models and integrated
electronic-energy observations.

Gaussian Process Regression
---------------------------

Gaussian process regression is a nonparametric Bayesian regression method. A
general treatment is given by Rasmussen and Williams. :footcite:p:`Rasmussen2006`
The construction below follows its application to density functionals in
CIDER23X and CIDER24X. :footcite:p:`CIDER23X,CIDER24X`

Let :math:`f(\mathbf{x})` denote the learned function, with
:math:`\mathbf{x}` as its feature vector.
Let :math:`\mathbf{X}` be a feature matrix, where each row :math:`\mathbf{x}_i`
is the feature vector for training point :math:`i`, and let :math:`\mathbf{y}`
be a target vector, where each element :math:`y_i` is the observed value of
the target function for training point :math:`i`.

The covariance kernel
:math:`k(\mathbf{x}, \mathbf{x}')` defines the covariance between the
predictive function at two points:
:math:`k(\mathbf{x}, \mathbf{x}')=\operatorname{Cov}(f(\mathbf{x}), f(\mathbf{x}'))`.
The matrix of covariances between training points is denoted :math:`\mathbf{K}`
with matrix elements :math:`K_{ij}=k(\mathbf{x}_i, \mathbf{x}_j)`.
Observation standard deviations :math:`\sigma_i` specify the assumed noise or
prior uncertainty. Their variances form the diagonal matrix
:math:`\boldsymbol{\Sigma}_\mathrm{noise}`.  The predictive function for a
test point :math:`\mathbf{x}_*` is :footcite:p:`Rasmussen2006`

.. math:: f(\mathbf{x}_*) = \sum_i k(\mathbf{x}_*, \mathbf{x}_i) \alpha_i

with weight vector

.. math:: \boldsymbol{\alpha} = \left(\mathbf{K} + \boldsymbol{\Sigma}_\text{noise}\right)^{-1} \mathbf{y}

For molecular and solid-state energies, each observation is an integral of
the local predictive function.

Fitting Total Energy Data
-------------------------

Let :math:`F` be an exchange, correlation, or exchange-correlation
contribution to the total electronic energy.  CIDER represents it as a
baseline energy density multiplied by a learned enhancement factor:

.. math::

   F = \int \mathrm{d}^3\mathbf{r}\,
   e^{\mathrm{base}}(\mathbf{r})\,f\left(\mathbf{x}(\mathbf{r})\right)

Here :math:`f\left(\mathbf{x}(\mathbf{r})\right)` is the dimensionless
enhancement factor learned by the regression, and :math:`e^{\mathrm{base}}`
is the multiplicative baseline energy density: the LDA exchange energy
density for an exchange kernel, and the PBE correlation energy density for a
correlation kernel.  For a chemical system indexed by :math:`m`, numerical
quadrature gives

.. math::

   F^m = \sum_{g\in m} w_g^m e_g^{\mathrm{base}}
   f\left(\mathbf{x}_g^m\right)

where :math:`g` indexes quadrature points and :math:`w_g^m` are the
quadrature weights.
The covariances between the numerical integrals :math:`F^m` and :math:`F^n`
can be written as

.. math:: \text{Cov}(F^m, F^n) = \sum_{g \in m} \sum_{h \in n} w_g^m w_h^n e_g^{\mathrm{base}} e_h^{\mathrm{base}} k(\mathbf{x}_g^m, \mathbf{x}_h^n)

where :math:`k(\mathbf{x}, \mathbf{x}')` is the covariance kernel for
:math:`f(\mathbf{x})`. CIDER avoids the direct double sum by introducing a
set of control points :math:`\tilde{\mathbf{x}}_a` and applying a
Nyström/resolution-of-the-identity approximation:

.. math::

   \operatorname{Cov}(F^m, F^n) = K_{mn}
   &\approx \tilde{\mathbf{k}}_m^{\mathsf T}
   \tilde{\mathbf{K}}^{-1}\tilde{\mathbf{k}}_n, \\
   \left(\tilde{\mathbf{K}}\right)_{ab}
   &= k(\tilde{\mathbf{x}}_a, \tilde{\mathbf{x}}_b), \\
   \left(\tilde{\mathbf{k}}_m\right)_a
   &= \sum_{g\in m} w_g^m e_g^{\mathrm{base}}
   k(\mathbf{x}_g^m, \tilde{\mathbf{x}}_a).

Let :math:`\mathbf{K}_\mathrm{sys}` be the resulting covariance matrix between
the integrated system observations. The predictive mean follows from

.. math::

   \mathbf{K}_\mathrm{sys}
   &= [K_{mn}], \\
   \boldsymbol{\beta}
   &= \left(\mathbf{K}_\mathrm{sys}
      +\boldsymbol{\Sigma}_\mathrm{noise}\right)^{-1}\mathbf{y}, \\
   \boldsymbol{\alpha}
   &= \tilde{\mathbf{K}}^{-1}
      \sum_m \tilde{\mathbf{k}}_m\,\beta_m, \\
   f(\mathbf{x}_*)
   &= \sum_a k(\mathbf{x}_*,\tilde{\mathbf{x}}_a)\alpha_a.

The vector :math:`\mathbf{y}` contains the integrated training labels. In
:meth:`ciderpress.models.train.MOLGP.fit`, ``Kmm`` is
:math:`\tilde{\mathbf{K}}`, the columns of ``Kmn`` are
:math:`\tilde{\mathbf{k}}_m`, ``alpha_new`` is
:math:`\boldsymbol{\beta}`, and ``kernel.alpha`` contains
:math:`\boldsymbol{\alpha}`.

If several independent GP components contribute to one observation, their
system covariance matrices are added before the shared
:math:`\boldsymbol{\beta}` solve. Each component then obtains its own
control-point coefficients. CIDER26XC uses this construction for the learned
exchange and correlation terms. :footcite:p:`CIDER26XC`

Fitting Eigenvalues
-------------------

The Gaussian process scheme discussed above can be extended to fit
orbital-occupation derivatives. :footcite:p:`CIDER24X` At a stationary
generalized Kohn--Sham solution, the derivative of the total energy with
respect to the occupation :math:`f_i^m` gives the corresponding eigenvalue
:math:`\epsilon_i^m`: :footcite:p:`Janak1978`

.. math:: \epsilon_i^m = \frac{\partial E}{\partial f_i^m}

The exact particle-number derivatives at an integer electron number
:math:`N` are :math:`-I` from below and :math:`A` from above, where
:math:`I` and :math:`A` are the ionization potential and electron affinity.
The DFT derivative discontinuity enters through the difference between
a neutral system's lowest unoccupied orbital energy and
its electron affinity :math:`A` in pure Kohn--Sham DFT.

Occupation-derivative labels can provide both valence- and conduction-edge
information.  CIDER24X fits derivatives of the target energy contribution,
:math:`\partial F/\partial f_i^m`.

For exact exchange :math:`E_\mathrm{x}^\mathrm{exact}`, the occupation
derivative can be evaluated explicitly for a given set of orbitals.  A full
exchange-correlation reference derivative can be constructed from a
consistent ionization-potential, electron-affinity, or
band-edge target. The total energy in generalized Kohn--Sham DFT is a
functional of the one-particle density matrix :math:`n_1` (and therefore of
its orbitals and occupations):

.. math::
   E[n_1] &= T[n_1] + V[n] + U[n] + E_\mathrm{xc}[n_1] \\
   &= E_0[n_1] + E_\mathrm{xc}[n_1].

Here :math:`E_0[n_1]` is the sum of the kinetic (:math:`T`), external (:math:`V`),
and Hartree (:math:`U`) energies, all of which can be written explicitly in terms
of the generalized Kohn--Sham orbitals. Given a reference total occupation
derivative, the XC part follows by subtracting the explicit non-XC derivative:

.. math::

   \frac{\partial E_\mathrm{xc}[n_1]}{\partial f_i^m}
   = \epsilon_i^m - \frac{\partial E_0[n_1]}{\partial f_i^m}.

The density matrix, orbitals, reference derivative, and side of the integer
particle number must all be defined consistently when constructing such a
training label.

The occupation derivative of :math:`F` is related to the model energy density
:math:`f(\mathbf{x})` by

.. math:: \frac{\partial F^m}{\partial f_i^m} = \sum_{g \in m} w_g^m \frac{\partial f(\mathbf{x}_g^m)}{\partial\mathbf{x}_g^m} \cdot \frac{\partial \mathbf{x}_g^m}{\partial f_i^m}

Including this label requires its covariance with an integrated energy
:math:`F^n` or another occupation derivative.  These covariances are

.. math::
   \operatorname{Cov}\left(\frac{\partial F^m}{\partial f_i^m}, F^n\right) &= \tilde{\mathbf{d}}_{mi}^{\mathsf T} \tilde{\mathbf{K}}^{-1} \tilde{\mathbf{k}}_n \\
   \operatorname{Cov}\left(\frac{\partial F^m}{\partial f_i^m}, \frac{\partial F^n}{\partial f_j^n}\right) &= \tilde{\mathbf{d}}_{mi}^{\mathsf T} \tilde{\mathbf{K}}^{-1} \tilde{\mathbf{d}}_{nj} \\
   \left(\tilde{\mathbf{d}}_{mi}\right)_a &= \sum_{g\in m} w_g^m \frac{\partial \mathbf{x}_g^m}{\partial f_i^m} \cdot \frac{\partial}{\partial \mathbf{x}_g^m} k(\mathbf{x}_g^m, \tilde{\mathbf{x}}_a)

These covariance blocks place integrated energies and occupation derivatives
in the same Gaussian-process fit.  The complete occupation-derivative
derivation is given in :footcite:t:`CIDER24X`.

.. footbibliography::
