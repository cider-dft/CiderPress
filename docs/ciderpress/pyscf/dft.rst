dft
===

The :py:mod:`ciderpress.pyscf.dft` module provides the
function :py:func:`make_cider_calc`, which takes
a Pyscf :py:class:`KohnShamDFT` object and a CIDER
functional object (:py:class:`~ciderpress.dft.xc_evaluator.MappedXC` or
:py:class:`~ciderpress.dft.xc_evaluator2.MappedXC2`)
and returns an instance of a :py:class:`KohnShamDFT`
subclass that uses the CIDER functional. The function
is similar to native PySCF routines like :py:func:`density_fit`,
in which the input SCF object is "decorated" with the
necessary routines to evaluate the CIDER functional.

The basic full-XC use case is:

.. code-block:: python

    from pyscf import gto
    from pyscf.dft import RKS

    from ciderpress.pyscf.dft import make_cider_calc

    mol = gto.M(...)
    ks = RKS(mol)
    ks = make_cider_calc(ks, "CIDER26XCCHEM")
    etot = ks.kernel()

For a complete example, please see :source:`examples/pyscf/production_calc.py`
and :doc:`../../usage/pyscf`.

.. automodule:: ciderpress.pyscf.dft
    :members:
