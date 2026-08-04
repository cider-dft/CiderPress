Supported Paths and Limitations
===============================

This page collects implementation boundaries that otherwise become easy to
miss across backend and feature documentation.

* CiderPress 0.5.0 supports classic GPAW, not ``gpaw.new``.
* The supported periodic CIDER26XC route is plane-wave mode with PAW setups.
  Norm-conserving pseudopotentials omit all-electron information needed by
  nonlocal features and are not recommended for production.
* GPAW implements version-j NLDF models.  SDMX and nonlocal-orbital feature
  models are not supported there.
* CIDER24X requires PyTorch and is evaluated through PySCF.  Periodic PySCF
  SDMX support is retained primarily for reproducing the published work.
* ``CIDER26XCCHEMD4`` is PySCF-only.  Its D4 contribution is energy-only in the
  current CiderPress interface and is absent from molecular gradients.
* PySCF CIDER Hessians, NMR, polarizability, post-Hartree--Fock methods, and
  related response interfaces are not implemented by the decorated object.
* Newton/SOSCF is not supported by the production CIDER26XC PySCF numerical
  integration path.
* Nonlocal features add memory and communication overhead.  GPAW jobs should
  use augmented-grid parallelization and a consistent MPI/FFT/OpenMP stack.
* Mapped YAML and joblib files must be trusted because loading reconstructs
  Python objects.

Numerical robustness is system dependent.  Difficult open-shell,
near-degenerate, metallic, or magnetic systems may require the controlled
restart sequences in :doc:`../usage/convergence`.  A workaround that reaches
a low residual does not remove the need to validate the intended electronic
state and final numerical settings.
