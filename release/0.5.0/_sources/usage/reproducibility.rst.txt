Recording and Checking a Calculation
====================================

A converged SCF flag is necessary but does not by itself identify the
functional, numerical representation, or electronic state.  Preserve enough
information to reconstruct all three.

Functional identity
-------------------

Record the CiderPress version, model short name, model SHA-256, and complete
energy composition.  For exchange-only models this includes ``xmix``,
``xkernel``, and ``ckernel``.  For CIDER+D4, record the base energy, expected
D4 energy, and final adjustment.

Electronic and numerical state
------------------------------

For PySCF, record the PySCF version, geometry, charge, spin, restricted or
unrestricted method, basis/ECP, grid specification, density-fitting and
auxiliary basis, SCF tolerances, occupations, and any restart or fallback
settings.

For GPAW, record the GPAW and ASE versions, atomic structure and cell, PAW
setup versions, plane-wave cutoff, k-points, bands, symmetry, charge, magnetic
moments, occupations and smearing, convergence criteria, mixer, eigensolver,
parallel layout, and CIDER interpolation/PASDW settings.  State whether the
calculation began from PBE or a CIDER checkpoint.

Recommended checks for a final calculation
------------------------------------------

* Confirm energy, density, and orbital/eigenstate criteria rather than relying
  on the final energy alone.
* Inspect occupations, gaps, spin expectation values, and total/local magnetic
  moments where relevant.
* Verify that a restarted calculation retains the intended model and numerical
  controls.
* Repeat a final calculation without temporary level shifting, damping, or
  deliberately relaxed tolerances unless those settings define the reported
  method.
* Converge the numerical parameters that affect the requested difference,
  force, or stress.
* Preserve the input, text output, checkpoint, environment/package versions,
  and model hash together.

When a calculation follows a geometry or parameter series, compare adjacent
states for discontinuous occupation or magnetic changes.  A previous density
is a useful starting point only while the intended electronic state remains
continuous.
