PySCF Descriptor Extraction
===========================

:func:`ciderpress.pyscf.descriptors.get_descriptors` evaluates one feature
settings component on the fixed density and orbitals held by an analyzer.
Semilocal, NLDF, and SDMX settings select their corresponding molecular
generators.  Passing the component settings from a loaded model preserves its
feature order and parameters.

Leaving the orbital selectors unset gives an array with shape
``(nspin, nfeature, ngrid)``.  Orbital selectors request fixed-orbital
occupation derivatives and return the descriptor array, a nested derivative
dictionary, and the selected orbital energies.  Selectors can count from the
bottom, highest occupied orbital, or lowest unoccupied orbital as documented
by the function.

See :doc:`../../workflows/descriptors` for an example and data-provenance
guidance.

.. automodule:: ciderpress.pyscf.descriptors
   :members:
