PySCF Interface and Numerical Modules
=====================================

The PySCF interface is the molecular all-electron route for all packaged
families.  :func:`ciderpress.pyscf.dft.make_cider_calc` decorates a normal
restricted or unrestricted Kohn--Sham object and selects its numerical
integration path from the settings stored in the mapped model.

This documentation assumes you are familiar with the PySCF code and
have a working installation of the software.
For PySCF documentation, please see the `PySCF <https://pyscf.org/>`_
website.

The molecular implementation is divided by responsibility:

* :mod:`ciderpress.pyscf.dft` constructs the decorated SCF object and defines
  total-energy, dispersion, checkpoint, and gradient behavior.
* :mod:`ciderpress.pyscf.numint` connects semilocal and NLDF feature
  evaluation to PySCF's blockwise atom-centered quadrature.
* :mod:`ciderpress.pyscf.gen_cider_grid` and
  :mod:`ciderpress.pyscf.nldf_convolutions` build and evaluate the auxiliary
  nonlocal-density representation.
* :mod:`ciderpress.pyscf.sdmx` evaluates the density-matrix descriptors used
  by CIDER24X.
* :mod:`ciderpress.pyscf.rks_grad` and :mod:`ciderpress.pyscf.uks_grad`
  propagate feature and grid response into nuclear gradients.
* :mod:`ciderpress.pyscf.analyzers` and
  :mod:`ciderpress.pyscf.descriptors` expose fixed-density ingredients for
  inspection and training-data construction.

For production model selection, complete examples, and restart guidance, see
:doc:`../../usage/production_models`, :doc:`../../usage/pyscf`, and
:doc:`../../usage/convergence`.

The periodic PySCF SDMX path requires pseudopotentials and a uniform XC grid.
It is retained mainly for methodological reproduction; classic GPAW with PAW
is the documented periodic route for NLDF production models.

.. toctree::
   :maxdepth: 2

   dft
   pbc/dft
   analyzers
   descriptors
   numerical
