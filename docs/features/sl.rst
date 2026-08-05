.. _sl_feat:

Semilocal Features (SL)
=======================

Semilocal DFT uses quantities evaluated at each integration point.  The
electron density is

.. math:: n(\mathbf{r}) = \sum_i f_i |\phi_i(\mathbf{r})|^2

where :math:`\phi_i(\mathbf{r})` are Kohn--Sham orbitals and :math:`f_i` are
their occupations.  A local-density approximation (LDA) depends on
:math:`n(\mathbf r)`.  A generalized-gradient approximation (GGA) also depends
on :math:`\nabla n(\mathbf r)`.  A meta-GGA can additionally use the
kinetic-energy density

.. math:: \tau(\mathbf{r}) = \frac{1}{2} \sum_i f_i |\nabla\phi_i(\mathbf{r})|^2

Regularized combinations of these ingredients can encode physical constraints
such as uniform scaling.  They include the reduced gradient\ :footcite:p:`Perdew1996`

.. math:: p = s^2 = \frac{|\nabla n|^2}{4(3\pi^2)^{2/3}n^{8/3}}

and the iso-orbital indicator\ :footcite:p:`Sun2013`

.. math:: \alpha = \frac{\tau - \tau_W}{\tau_0}

where :math:`\tau_W=|\nabla n|^2/8n` is the single-electron kinetic energy
density, and :math:`\tau_0=\frac{3}{10}(3\pi^2)^{2/3}n^{5/3}` is the kinetic
energy density of the uniform electron gas.

These equations use the spin-unpolarized convention, with orbital occupations
:math:`f_i\in [0,2]`.

In CiderPress, :class:`~ciderpress.dft.settings.SemilocalSettings` specifies
the semilocal block.  Its ``mode`` selects one of four raw feature layouts:

* ``ns``: :math:`\mathbf{x}_\text{sl} = [n, |\nabla n|^2]`
* ``np``: :math:`\mathbf{x}_\text{sl} = [n, p]`
* ``nst``: :math:`\mathbf{x}_\text{sl} = [n, |\nabla n|^2, \tau]`
* ``npa``: :math:`\mathbf{x}_\text{sl} = [n, p, \alpha]`

The ``ns`` and ``np`` layouts omit the kinetic-energy density :math:`\tau`.
An associated :ref:`NLDF <nldf_feat>` block therefore uses
``sl_level="GGA"``.  Orbital-dependent features such as :ref:`SDMX
<sdmx_feat>` may still be present.  Modes ``nst`` and ``npa`` include
:math:`\tau` and are meta-GGA level.

The regularized features :math:`p` and :math:`\alpha` are
scale-invariant, meaning that
an exchange functional trained with these features and
a proper exchange functional baseline will obey the
uniform scaling rule (see :ref:`Uniform Scaling Constraints <unif_scaling>`).

The raw features :math:`n`, :math:`|\nabla n|^2`, and :math:`\tau` carry
nonzero scaling powers.  During model construction, normalizers and feature
maps supply the behavior required by the chosen functional form.
Those transformations are serialized with a trained model and are applied
automatically when a packaged functional is evaluated.

See :class:`~ciderpress.dft.settings.SemilocalSettings` in the
:ref:`Feature Settings <settings_module>`
documentation for more details on the API for setting up these features.

.. footbibliography::
