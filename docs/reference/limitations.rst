Supported Use and Limitations
=============================

The following conditions apply to the documented backend, feature, and property
interfaces in this release.

* CiderPress supports the classic GPAW calculator.  The ``gpaw.new`` interface
  is not implemented in this release.
* The supported periodic CIDER26XC route is plane-wave mode with PAW setups.
  PAW supplies the all-electron information needed by its nonlocal features.
* GPAW implements version-j NLDF models.  CIDER24X SDMX models use PySCF.
* CIDER24X requires PyTorch and is evaluated through PySCF.  Periodic PySCF
  SDMX support is intended primarily for methodological reproduction
  of the CIDER24X work. :footcite:p:`CIDER24X` Periodic PySCF calculations
  with SDMX use pseudopotentials and uniform grids. Do not use
  all-electron setups or atom-centered grids for periodic SDMX calculations
  in PySCF.
* ``CIDER26XCCHEMD4`` uses the PySCF interface.  Its total energy and analytical
  molecular gradient both include the D4 correction.  CiderPress raises an
  error for gradients when a model requests the unsupported D3 or nonlocal-
  correlation post-density modes.
* The PySCF CIDER decorator provides SCF energies and the gradients listed in
  :doc:`../usage/properties`.  Hessians, NMR, polarizability,
  post-Hartree--Fock methods, and related response interfaces are not
  implemented currently.
* Newton/SOSCF is not implemented for CIDER PySCF calculations for all
  packaged families: the CIDER ``NumInt`` leaves the ``fxc`` and
  ``cache_xc_kernel`` entry points that second-order SCF requires
  unimplemented.  Use the DIIS methods in :doc:`../usage/convergence`.
* Nonlocal features add memory and communication overhead.  GPAW exposes
  augmented-grid parallelization through
  ``parallel={"augment_grids": True}``.
* The documented feature interfaces cover semilocal, NLDF, and SDMX models.
  Other feature classes in the source tree are experimental.

Concrete restart controls for open-shell, near-degenerate, metallic, and
magnetic calculations are listed in :doc:`../usage/convergence`.

.. footbibliography::
