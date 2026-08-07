PySCF Electronic Analyzers
==========================

An analyzer binds a molecule, density matrix, molecular orbitals, occupations,
orbital energies, and atom-centered grid.  It evaluates and caches quantities
used in model construction, including the density, kinetic-energy density,
Hartree and exchange energy densities, and orbital-resolved responses.

:class:`ciderpress.pyscf.analyzers.RHFAnalyzer` stores a restricted state, and
:class:`ciderpress.pyscf.analyzers.UHFAnalyzer` stores separate alpha and beta
states.  ``from_calc`` creates an analyzer from a completed PySCF calculation.
``dump`` and ``load`` preserve the molecule, grid specification, orbitals,
density matrix, and cached arrays in HDF5.

The descriptor workflow is described in
:doc:`../../workflows/descriptors`.

.. automodule:: ciderpress.pyscf.analyzers
   :members:
