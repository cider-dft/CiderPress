pbc.dft
=======

The :py:mod:`ciderpress.pyscf.pbc.dft` module serves the same
purpose as :py:mod:`ciderpress.pyscf.dft` but for periodic
systems. This module's version of
:py:func:`~ciderpress.pyscf.pbc.dft.make_cider_calc`
modifies a periodic Kohn--Sham DFT object from ``pyscf.pbc.dft``
to evaluate a CIDER functional. Currently only semilocal
and SDMX features are supported.

.. caution::

   This periodic PySCF module is retained to reproduce the CIDER24X work
   (:footcite:t:`CIDER24X`).  It is not the documented production periodic
   route and currently supports only semilocal and SDMX features.

.. automodule:: ciderpress.pyscf.pbc.dft
    :members:

.. footbibliography::
