Machine Learning Framework for CIDER Functionals
================================================

This page covers the machine learning framework used to train exchange-correlation
functionals in the CIDER formalism, which is built on Gaussian process regression
(GPR).

Gaussian Process Regression
---------------------------

Gaussian process regression is a nonparametric, Bayesian statistical learning model.
The detailed theory of Gaussian processes can be found in the excellent textbook
by Rasmussen and Williams. :footcite:p:`Rasmussen2006` This section covers the application of Gaussian
process regression to functional design. More details on the formalism
can be found in :footcite:t:`CIDER23X` and :footcite:t:`CIDER24X`

Consider the problem of learning some function :math:`f(\mathbf{x})`, with
:math:`\mathbf{x}` being a feature vector of independent variables.
Let :math:`\mathbf{X}` be a feature matrix, where each row :math:`\mathbf{x}_i`
is the feature vector for training point :math:`i`, and let :math:`\mathbf{y}`
be a target vector, where each element :math:`y_i` is the observed value of
the target function for training point :math:`i`. Then, there are two key
components necessary to construct a Gaussian process regression for
:math:`f(\mathbf{x})`. The first is the covariance kernel
:math:`k(\mathbf{x}, \mathbf{x}')`, which defines the covariance between the
values of the predictive function for two points, i.e.
:math:`k(\mathbf{x}, \mathbf{x}')=\text{Cov}(f(\mathbf{x}), f(\mathbf{x}'))`.
The matrix of covariances between training points is denoted :math:`\mathbf{K}`
with matrix elements :math:`K_{ij}=k(\mathbf{x}_i, \mathbf{x}_j)`.
The second is the noise labels :math:`\sigma_i`, indicating the estimated
prior uncertainty on each training observation :math:`i`. We will denote the
diagonal matrix whose entries are the noise covariances :math:`{\sigma_i^{}}^2`
as :math:`\boldsymbol{\Sigma}_\text{noise}`. Using these components, the
predictive function for a test point :math:`\mathbf{x}_*`
takes the form :footcite:p:`Rasmussen2006`

.. math:: f(\mathbf{x}_*) = \sum_i k(\mathbf{x}_*, \mathbf{x}_i) \alpha_i

with the following definition for the weight vector :math:`\boldsymbol{\alpha}`:

.. math:: \boldsymbol{\alpha} = \left(\mathbf{K} + \boldsymbol{\Sigma}_\text{noise}\right)^{-1} \mathbf{y}

In the case of fitting energies of molecules and solids, we need to fit an
extensive quantity in which the training labels are integrals of the predictive
function :math:`f(\mathbf{x})` over real space. The next section covers
adjustments to the Gaussian process model necessary to perform this task.

Fitting Total Energy Data
-------------------------

Consider an extensive quantity :math:`F`, which is a contribution to the
total electronic energy of a chemical system. In the case of CiderPress,
this quantity will be the exchange, correlation, or exchange-correlation energy.
We can fit :math:`F` by learning a function that gets integrated over
real-space to yield :math:`F` for a given system:

.. math::

   F = \int \mathrm{d}^3\mathbf{r}\,f\left(\mathbf{x}(\mathbf{r})\right)

In the above equation, :math:`f\left(\mathbf{x}(\mathbf{r})\right)` is the predictive
function for the energy density to be learned by the Gaussian process.
In practice, the integral must be performed numerically. For a given chemical system
indexed by :math:`m`, we write
    
.. math::

   F^m = \sum_{g\in m} w_g^m f\left(\mathbf{x}_g^m\right)

where :math:`g` indexes quadrature points and :math:`w_g^m` are the respective quadrature weights.
The covariances between the numerical integrals :math:`F^m` and :math:`F^n`
can be written as

.. math:: \text{Cov}(F^m, F^n) = \sum_{g \in m} \sum_{h \in n} w_g^m w_h^n k(\mathbf{x}_g^m, \mathbf{x}_h^n)

where :math:`k(\mathbf{x}, \mathbf{x}')` is the covariance kernel for
:math:`f(\mathbf{x})`. Computing the above double numerical integral directly
would be expensive. To overcome this issue, we define a small set of "control points"
:math:`\tilde{\mathbf{x}}_a` and approximate :math:`\text{Cov}(F^m, F^n)`
using a resolution-of-the-identity approximation:

.. math::

   \operatorname{Cov}(F^m, F^n) = K_{mn}
   &\approx \tilde{\mathbf{k}}_m^{\mathsf T}
   \tilde{\mathbf{K}}^{-1}\tilde{\mathbf{k}}_n, \\
   \left(\tilde{\mathbf{K}}\right)_{ab}
   &= k(\tilde{\mathbf{x}}_a, \tilde{\mathbf{x}}_b), \\
   \left(\tilde{\mathbf{k}}_m\right)_a
   &= \sum_{g\in m} w_g^m
   k(\mathbf{x}_g^m, \tilde{\mathbf{x}}_a).

Using this definition of the covariance kernel, the predictive function can be expressed as

.. math::

   f(\mathbf{x}_*)
   &= \sum_a k(\mathbf{x}_*, \tilde{\mathbf{x}}_a)\alpha_a, \\
   \boldsymbol{\beta}
   &= \left(\mathbf{K}+\boldsymbol{\Sigma}_\mathrm{noise}\right)^{-1}
   \mathbf{y}, \\
   \tilde{\boldsymbol{\alpha}}
   &= \sum_m \tilde{\mathbf{k}}_m\,\beta_m, \\
   \boldsymbol{\alpha}
   &= \tilde{\mathbf{K}}^{-1}\tilde{\boldsymbol{\alpha}}.

with :math:`\mathbf{y}` being the vector of training labels :math:`F^m`.

The final multiplication by :math:`\tilde{\mathbf{K}}^{-1}` is required by
the resolution-of-the-identity construction. It was inadvertently omitted
from Eq. 43 of the CIDER23X paper and Eq. 29 of the CIDER24X paper; the
expressions above give the corrected predictive mean.

Fitting Eigenvalues
-------------------

The Gaussian process scheme discussed above can be extended to fit
orbital-occupation derivatives. :footcite:p:`CIDER24X` At a stationary
generalized Kohn--Sham solution, the derivative of the total energy with
respect to the occupation :math:`f_i^m` gives the corresponding eigenvalue
:math:`\epsilon_i^m`: :footcite:p:`Janak1978`

.. math:: \epsilon_i^m = \frac{\partial E}{\partial f_i^m}

Most individual eigenvalues do not have a direct quasiparticle
interpretation. The rigorous statements concern derivatives with respect to
particle number: at an integer electron number :math:`N`, the derivative from
below is :math:`-I` and the derivative from above is :math:`-A`, where
:math:`I` and :math:`A` are the ionization potential and electron affinity.
The discontinuity between these derivatives is important; in ordinary
Kohn--Sham DFT it must not be replaced by a blanket identification of the
neutral-system LUMO with :math:`-A`. Occupation-derivative data nevertheless
provide a useful way to train valence- and conduction-edge information. We
therefore fit derivatives of the target energy contribution,
:math:`\partial F/\partial f_i^m`.

For exact exchange :math:`E_\mathrm{x}^\mathrm{exact}`, the occupation
derivative can be evaluated explicitly for a given set of orbitals. For a
full exchange-correlation target, a reference derivative can instead be
constructed from a consistent ionization-potential, electron-affinity, or
band-edge target. The total energy in generalized Kohn--Sham DFT is a
functional of the one-particle density matrix :math:`n_1` (and therefore of
its orbitals and occupations):

.. math::
   E[n_1] &= T[n_1] + V[n] + U[n] + E_\mathrm{xc}[n_1] \\
   &= E_0[n_1] + E_\mathrm{xc}[n_1].

where :math:`E_0[n_1]` is the sum of the kinetic (:math:`T`), external (:math:`V`),
and Hartree (:math:`U`) energies, all of which can be written explicitly in terms
of the generalized Kohn--Sham orbitals. Given a reference total occupation
derivative, the XC part follows by subtracting the explicit non-XC derivative:

.. math::

   \frac{\partial E_\mathrm{xc}[n_1]}{\partial f_i^m}
   = \epsilon_i^m - \frac{\partial E_0[n_1]}{\partial f_i^m}.

The density matrix, orbitals, reference derivative, and side of the integer
particle number must all be defined consistently when constructing such a
training label.

Given such training data, we can relate the occupation derivative of :math:`F` to our model energy
density :math:`f(\mathbf{x})` as

.. math:: \frac{\partial F^m}{\partial f_i^m} = \sum_{g \in m} w_g^m \frac{\partial f(\mathbf{x}_g^m)}{\partial\mathbf{x}_g^m} \cdot \frac{\partial \mathbf{x}_g^m}{\partial f_i^m}

To fit our model to the above equation, we only need to know the covariance between
:math:`\frac{\partial F^m}{\partial f_i^m}` and :math:`\frac{\partial F^n}{\partial f_j^n}`
or :math:`F^n`. This relationship is given by

.. math::
   \operatorname{Cov}\left(\frac{\partial F^m}{\partial f_i^m}, F^n\right) &= \tilde{\mathbf{d}}_{mi}^{\mathsf T} \tilde{\mathbf{K}}^{-1} \tilde{\mathbf{k}}_n \\
   \operatorname{Cov}\left(\frac{\partial F^m}{\partial f_i^m}, \frac{\partial F^n}{\partial f_j^n}\right) &= \tilde{\mathbf{d}}_{mi}^{\mathsf T} \tilde{\mathbf{K}}^{-1} \tilde{\mathbf{d}}_{nj} \\
   \left(\tilde{\mathbf{d}}_{mi}\right)_a &= \sum_{g\in m} w_g^m \frac{\partial \mathbf{x}_g^m}{\partial f_i^m} \cdot \frac{\partial}{\partial \mathbf{x}_g^m} k(\mathbf{x}_g^m, \tilde{\mathbf{x}}_a)

which allows occupation derivative training data to be included in the Gaussian process.
For further details, see :footcite:t:`CIDER24X`

.. footbibliography::
