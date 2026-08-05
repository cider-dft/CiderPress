Periodic PySCF Interface
========================

The :py:mod:`ciderpress.pyscf.pbc.dft` module decorates periodic Kohn--Sham
objects from ``pyscf.pbc.dft``.  Its
:py:func:`~ciderpress.pyscf.pbc.dft.make_cider_calc` function supports the
semilocal and SDMX features used for methodological reproduction of the
CIDER24X work. :footcite:p:`CIDER24X`  Packaged periodic NLDF calculations use
the classic GPAW/PAW interface.

Numerical implementation
------------------------

.. py:module:: ciderpress.pyscf.pbc.numint

:mod:`ciderpress.pyscf.pbc.numint` connects the mapped evaluator to PySCF's
periodic atom-centered and uniform-grid integration paths.  It evaluates the
semilocal and SDMX blocks, applies their adjoint matrix contributions, and
supports the k-point layouts accepted by the periodic Kohn--Sham object.

.. py:module:: ciderpress.pyscf.pbc.sdmx_fft

:mod:`ciderpress.pyscf.pbc.sdmx_fft` constructs periodic SDMX quantities from
real- and reciprocal-space orbital representations.  Its Gaussian smoothing
and k-point conventions implement the CIDER24X periodic descriptor path.

.. py:module:: ciderpress.pyscf.pbc.util

:mod:`ciderpress.pyscf.pbc.util` supplies the FFT interpolation used when the
XC evaluation mesh is denser than the cell's base mesh.  Forward feature
evaluation and the returned matrix potential use the corresponding pair of
mesh transfers.

.. automodule:: ciderpress.pyscf.pbc.dft
    :members:

.. footbibliography::
