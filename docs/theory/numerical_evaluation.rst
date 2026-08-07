Numerical Evaluation in PySCF and GPAW
======================================

This page follows a mapped CIDER model from host-code density arrays to the
XC energy and potential.  The feature definitions are collected in
:doc:`../features/features`; the NLDF interpolation algorithm is derived in
:doc:`nldf_numerical`.  Module-level references for the two backends are given
in :doc:`../ciderpress/pyscf/numerical` and
:doc:`../ciderpress/gpaw/numerical`.

Forward evaluation and its adjoint
----------------------------------

Let :math:`x_i[n](\mathbf r)` denote the ordered raw features required by a
model, including their spin channels.  The numerical integral has the form

.. math::

   E_{\mathrm{xc}}[n]
   = \int \mathrm d^3\mathbf r\,
     \mathcal E_{\mathrm{map}}\!\left(
       x_1[n](\mathbf r),\ldots,x_m[n](\mathbf r)
     \right).

The mapped evaluator returns :math:`\mathcal E_{\mathrm{map}}` and
:math:`\partial\mathcal E_{\mathrm{map}}/\partial x_i`.  Each feature
generator then applies the adjoint of its discretized derivative,

.. math::

   \frac{\delta E_{\mathrm{xc}}}{\delta n_s}
   = \sum_i
     \left(\frac{\delta x_i}{\delta n_s}\right)^{\!*}
     \frac{\partial\mathcal E_{\mathrm{map}}}{\partial x_i},

to form the XC contribution expected by the host code.  For semilocal
features this produces density, gradient, and kinetic-energy-density
potentials.  An NLDF generator additionally reverses the exponent
interpolation and convolution; an SDMX generator returns a density-matrix
potential.  This forward/adjoint organization is the numerical form of the
CIDER23X NLDF method and the CIDER24X SDMX method, and is also used by the
CIDER26XC models. :footcite:p:`CIDER23X,CIDER24X,CIDER26XC`

The common mapped-model layer is implemented by
:class:`~ciderpress.dft.xc_evaluator.MappedXC` and
:class:`~ciderpress.dft.xc_evaluator2.MappedXC2`.  The backend applies the
serialized physical normalizers before calling these objects and reverses the
normalization afterward.  The mapped kernels apply their bounded feature
transforms, evaluate the regression, and return derivatives in normalized
feature order.

Backend map
-----------

.. list-table:: Main numerical objects used during an XC evaluation
   :header-rows: 1
   :widths: 22 39 39

   * - Stage
     - PySCF molecular path
     - GPAW plane-wave/PAW path
   * - Host-code entry
     - :func:`~ciderpress.pyscf.dft.make_cider_calc` installs
       :class:`~ciderpress.pyscf.numint.CiderNumInt` on an RKS or UKS object.
     - :func:`~ciderpress.gpaw.calculator.get_cider_functional` constructs a
       :class:`~ciderpress.gpaw.cider_fft.CiderGGA` or
       :class:`~ciderpress.gpaw.cider_fft.CiderMGGA` functional.
   * - Density integration
     - :func:`~ciderpress.pyscf.numint.nr_rks` and
       :func:`~ciderpress.pyscf.numint.nr_uks` process atom-centered grid
       blocks and assemble an AO-matrix potential.
     - :mod:`ciderpress.gpaw.cider_fft` processes the distributed smooth grid;
       :mod:`ciderpress.gpaw.cider_kernel` evaluates the mapped energy and
       feature derivatives.
   * - NLDF forward/adjoint
     - :class:`~ciderpress.pyscf.nldf_convolutions.PyscfNLDFGenerator` uses
       atom-centered auxiliary Gaussian expansions and interpolation.
     - :class:`~ciderpress.gpaw.nldf_interface.LibCiderPW` evaluates fixed
       kernels by reciprocal-space multiplication and FFTs.
   * - SDMX forward/adjoint
     - :class:`~ciderpress.pyscf.sdmx.EXXSphGenerator` contracts smoothed
       orbital quantities with the molecular density matrix.
     - SDMX models are not implemented in the GPAW interface.
   * - Atom-centered response
     - :mod:`ciderpress.pyscf.rks_grad` and
       :mod:`ciderpress.pyscf.uks_grad` differentiate the molecular grid and
       NLDF auxiliary representation for nuclear gradients.
     - :mod:`ciderpress.gpaw.cider_paw` and
       :mod:`ciderpress.gpaw.atom_utils` apply PAW/PASDW energy, potential,
       force, and stress contractions.

Molecular Gaussian-basis path
-----------------------------

PySCF supplies the density ingredients on blocks of its atom-centered
quadrature grid.  :class:`~ciderpress.pyscf.numint.CiderNumInt` selects the
feature generator from the settings stored in the model and retains the same
serialized feature order for restricted and unrestricted calculations.  The
unrestricted path keeps the alpha and beta arrays separate until the model's
spin-combination rule is applied.

For an NLDF model,
:class:`~ciderpress.pyscf.gen_cider_grid.CiderGrids` records the radial and
angular indexing of the molecular quadrature points.  The forward call to
:class:`~ciderpress.pyscf.nldf_convolutions.PyscfNLDFGenerator` produces the
NLDF feature block; its backward call returns contributions to the density
and to the density-dependent exponents.  For an SDMX model,
:class:`~ciderpress.pyscf.sdmx.EXXSphGenerator` performs the corresponding
density-matrix contractions.  The full blockwise sequence is implemented in
:source:`ciderpress/pyscf/numint.py`.

The final return from :func:`~ciderpress.pyscf.numint.nr_rks` or
:func:`~ciderpress.pyscf.numint.nr_uks` contains the integrated XC energy and
the AO-matrix potential.  Analytical NLDF nuclear gradients add the response
of the atom-centered grid, auxiliary basis, and interpolation operations; the
implementation is described in :doc:`../ciderpress/pyscf/numerical`.

Periodic FFT and PAW path
-------------------------

:mod:`ciderpress.gpaw.cider_fft` receives GPAW's smooth density, gradient,
and, for meta-GGA models, kinetic-energy-density arrays.  Version-J NLDF
source channels are interpolated over kernel exponents by
:class:`~ciderpress.gpaw.nldf_interface.LibCiderPW`.  Each fixed-kernel
convolution is evaluated on the distributed reciprocal grid, after which
:class:`~ciderpress.gpaw.cider_kernel.CiderKernel` evaluates the mapped model.
The backward FFT and interpolation calls add the NLDF contribution to GPAW's
smooth XC potential and stress.

The smooth grid omits the all-electron density variation inside PAW
augmentation spheres.  :class:`~ciderpress.gpaw.cider_paw.CiderGGAPASDW` and
:class:`~ciderpress.gpaw.cider_paw.CiderMGGAPASDW` transfer NLDF sources and
potentials between the uniform grid and atom-centered support grids.  The
PAW/PASDW equations and their implementation classes are given in
:doc:`../ciderpress/gpaw/numerical`.

:class:`~ciderpress.gpaw.interp_paw.DiffPAWXCCorrection` supplies the radial
all-electron and pseudo quantities used by the on-site correction.  When
``build_kinetic=True``, the pair densities and their radial derivatives are
formed from products of the interpolated partial waves, alongside the
kinetic-energy-density terms required by the meta-GGA correction.  With
``build_kinetic=False``, the tabulated GPAW pair-density arrays and their
derivatives are interpolated directly.  The implementation is in
:source:`ciderpress/gpaw/interp_paw.py`.

The radial reconstruction also applies two finite-grid safeguards.  Mapping
beyond a radial endpoint is accepted only within floating-point roundoff, and
the all-electron and pseudo frozen-core kinetic-energy densities are bounded
below by the radial-grid form of
:math:`\tau_{\mathrm W}=|\nabla n|^2/(8n)`.  Elements above argon retain the
setup's denser all-electron radial grid inside the required cutoff.

Numerical state
---------------

The molecular NLDF representation is determined by the model settings, the
PySCF quadrature grid, and the auxiliary expansion constructed for that
molecule.  A molecular restart recreates the CIDER-decorated mean-field object
and supplies orbitals or a density matrix loaded from the PySCF checkpoint, as
shown in :doc:`../usage/pyscf`.

For GPAW, ``qmax``, ``lambd``, and ``Nalpha`` determine the exponent range and
interpolation resolution of the periodic NLDF.  ``pasdw_ovlp_fit`` selects the
atomic overlap fit, while ``pasdw_store_funcs`` controls caching of atomic
projector values.  :class:`~ciderpress.gpaw.calculator.CiderGPAW` records the
mapped model and NLDF interpolation state in its checkpoint.  Meta-GGA
restarts are written with ``mode="all"`` so that GPAW retains the
wavefunctions needed to reconstruct the kinetic-energy density; see
:doc:`../usage/gpaw`.

.. footbibliography::
