Inspecting Densities and Descriptors
====================================

The PySCF analyzer and descriptor modules expose the electronic ingredients
used to construct training data and inspect a mapped model.  They evaluate the
fixed density and orbitals supplied by a completed calculation.

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

Use the settings stored in the model being analyzed.  They preserve its exact
normalizers, exponent parameters, and feature order.

Fixed-density versus self-consistent use
----------------------------------------

Descriptor extraction evaluates a chosen, fixed density for model analysis,
training-data construction, and comparison of feature representations.  A
self-consistent energy calculation also updates that density through the
model potential.  The density-generating calculation and the descriptor
settings are separate inputs to a fixed-density analysis.

The GPAW descriptor interface obtains plane-wave and PAW quantities from a
live, completed GPAW calculator.  See
:doc:`../ciderpress/gpaw/descriptors` for its supported settings, array
shapes, PAW behavior, and occupation selectors.  Use each backend's documented
point layout and orbital-index convention.
