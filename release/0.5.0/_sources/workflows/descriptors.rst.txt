Inspecting Densities and Descriptors
====================================

The PySCF analyzer and descriptor modules expose the same electronic
ingredients used to construct training data and diagnose a mapped model.
These tools operate on a fixed density; they do not run an SCF calculation by
themselves.

Analyzer objects
----------------

:class:`~ciderpress.pyscf.analyzers.RHFAnalyzer` and
:class:`~ciderpress.pyscf.analyzers.UHFAnalyzer` bind a molecule, density
matrix, molecular orbitals, and atom-centered grid.  They can evaluate and
cache densities, kinetic-energy densities, exchange energy densities, and
related quantities.  Their HDF5 representation is useful for separating an
electronic-structure calculation from later descriptor generation.

.. literalinclude:: ../../examples/pyscf/descriptors.py
   :language: python
   :linenos:

Descriptor arrays
-----------------

:func:`ciderpress.pyscf.descriptors.get_descriptors` evaluates one settings
component at a time.  The descriptor array has shape
``(nspin, nfeature, ngrid)``.  When orbital selectors are supplied, the
routine also returns feature derivatives with respect to selected occupation
numbers and their corresponding orbital energies.

Use the settings stored in the model being analyzed.  Constructing a new
``NLDFSettings`` object with similar-looking parameters does not guarantee the
same descriptors because normalizers, exponent parameters, and feature order
are part of the model definition.

Fixed-density versus self-consistent use
----------------------------------------

Descriptor extraction evaluates the chosen density.  It is appropriate for
model analysis, training-data construction, and comparing feature
representations.  It is not equivalent to evaluating a total energy with the
model self-consistently.  When reporting a fixed-density result, identify the
functional and numerical settings that generated the density.

The GPAW descriptor module provides analogous access to plane-wave and PAW
quantities, but its interfaces are more closely coupled to a live GPAW
calculator.  For backend implementation work, begin with
:doc:`../theory/numerical_evaluation` and the module source rather than
assuming the PySCF array layout is portable.
