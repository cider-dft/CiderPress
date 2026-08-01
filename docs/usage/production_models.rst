Production Models
=================

CiderPress 0.5.0 includes three full exchange-correlation (XC) models.  The
names below are package resources: no separate model download or absolute file
path is required.

.. list-table:: Packaged CIDER26XC models
   :header-rows: 1
   :widths: 24 28 18 30

   * - Model
     - Recommended domain
     - Backend
     - Dispersion treatment
   * - ``CIDER26XCCHEM``
     - Molecular chemistry
     - PySCF
     - No explicit long-range correction
   * - ``CIDER26XCCHEMD4``
     - Molecular chemistry where D4 is appropriate
     - PySCF only
     - Geometry-dependent D4, evaluated after the SCF
   * - ``CIDER26XCSURFSCI``
     - Solids, surfaces, adsorption, and combined-domain applications
     - PySCF and classic GPAW
     - Learned XC model without a separate D4 term

The model contents can be identified independently of their installed path:

.. list-table:: Model SHA-256 checksums
   :header-rows: 1
   :widths: 30 70

   * - Model
     - SHA-256
   * - ``CIDER26XCCHEM``
     - ``fd2e0b5cd7408cd4b0ff09495bb026b109b5066bc2f1267c011ce5f0408bdf1d``
   * - ``CIDER26XCCHEMD4``
     - ``ee6824e258625246180efb75fdec7c1e23310a6fe2fcdf2ab836ec90d29bb00c``
   * - ``CIDER26XCSURFSCI``
     - ``e141a998359da9a64f3c5d06b4804e06762ab2e53dc6409979ff3eb0eacd793e``

Loading a packaged model
------------------------

Pass the short name anywhere that CiderPress accepts an ``mlfunc`` path.  An
optional ``.yaml`` suffix is accepted as well::

   from pyscf import dft, gto
   from ciderpress.pyscf.dft import make_cider_calc

   mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-tzvp")
   mf = make_cider_calc(dft.RKS(mol), "CIDER26XCCHEM")
   energy = mf.kernel()

Explicit paths to other YAML or joblib models remain supported.  A real file
at the supplied path takes precedence over a packaged short name.

Full-XC settings
----------------

The CIDER26XC models predict the full XC energy.  They must not be combined
with the PBE exchange/correlation baselines used by older CIDER exchange-only
models.

``make_cider_calc`` has the correct full-XC defaults.  In the GPAW interface,
state the full-XC settings explicitly::

   from ciderpress.gpaw.calculator import get_cider_functional

   xc = get_cider_functional(
       "CIDER26XCSURFSCI",
       xmix=1.0,
       xkernel=None,
       ckernel=None,
       pasdw_store_funcs=False,
   )

``pasdw_store_funcs=False`` is the memory-saving default.  The templates retain
the default ``pasdw_ovlp_fit=True`` for improved numerical precision of the PAW
projections.  Disabling overlap fitting can reduce its cost, but treat that
choice as part of the numerical setup when comparing tightly converged results.

.. warning::

   Omitting ``xkernel=None`` and ``ckernel=None`` in
   ``get_cider_functional`` adds the function's default PBE baselines.  That is
   appropriate for some older exchange-only models, but not for CIDER26XC.

D4 energy contract
------------------

Install the optional dependency before using ``CIDER26XCCHEMD4``::

   pip install 'ciderpress[d4]'

D4 is a geometry-dependent, post-density energy correction.  It does not
enter the XC potential or change the converged density.  CiderPress applies the
correction exactly once and exposes the accounting after ``kernel()``::

   total = mf.e_tot
   base = mf.e_tot_base
   d4 = mf.e_vdw_expected
   adjustment = mf.e_vdw_delta
   assert abs(total - (base + adjustment)) < 1e-10

Do not add a second PySCF D4 wrapper to the CIDER total.  If an incoming
mean-field object already has a supported dispersion wrapper, CiderPress
reconciles it with the model contract rather than silently double counting it.
GPAW raises an error for ``CIDER26XCCHEMD4`` because this post-density D4
contract is implemented only for PySCF in version 0.5.0.

Choosing and reporting a model
------------------------------

Use ``CIDER26XCCHEM`` or ``CIDER26XCCHEMD4`` for molecular calculations and
``CIDER26XCSURFSCI`` for periodic solids, surfaces, and adsorption studies.
The model name does not replace ordinary numerical convergence: basis sets,
integration grids, PAW setups, plane-wave cutoffs, cells, and k-point meshes
must still be converged for the requested property.

For reproducibility, report the CiderPress version, model name and checksum,
backend version, numerical representation, charge and spin state, SCF
settings, and whether a calculation was restarted from another functional.

The CIDER26XC models are described in the forthcoming manuscript
:footcite:t:`CIDER26XC`.  The CIDER representation and earlier molecular/solid
implementations are described in :footcite:t:`CIDER22X` and
:footcite:t:`CIDER23X`.

.. footbibliography::
