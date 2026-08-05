Supported Paths and Limitations
===============================

The following boundaries define the documented backend, feature, and property
interfaces in this release.

* CiderPress 0.5.0 supports the classic GPAW calculator.  ``gpaw.new`` is
  outside this release interface.
* The supported periodic CIDER26XC route is plane-wave mode with PAW setups.
  PAW supplies the all-electron information needed by its nonlocal features.
* GPAW implements version-j NLDF models.  CIDER24X SDMX models use PySCF.
* CIDER24X requires PyTorch and is evaluated through PySCF.  Periodic PySCF
  SDMX support covers methodological reproduction of the CIDER24X work.
* ``CIDER26XCCHEMD4`` uses the PySCF interface.  Its D4 contribution enters the
  energy; molecular gradients contain the electronic CIDER contribution.
* The PySCF CIDER decorator provides SCF energies and the gradients listed in
  :doc:`../usage/properties`.  Hessians, NMR, polarizability,
  post-Hartree--Fock methods, and related response interfaces are outside its
  scope.
* Newton/SOSCF is outside the CIDER26XC PySCF numerical
  integration path.
* Nonlocal features add memory and communication overhead.  GPAW exposes
  augmented-grid parallelization through
  ``parallel={"augment_grids": True}``.
* Mapped YAML and joblib files reconstruct Python objects and therefore
  require a trusted source.
* The documented feature interfaces cover semilocal, NLDF, and SDMX models.
  Other feature classes in the source tree are experimental.  Release
  compatibility guarantees apply to the documented families.

Concrete restart controls for open-shell, near-degenerate, metallic, and
magnetic calculations are listed in :doc:`../usage/convergence`.
