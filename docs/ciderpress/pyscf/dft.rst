PySCF Calculator Decoration
===========================

:func:`~ciderpress.pyscf.dft.make_cider_calc` decorates a restricted or
unrestricted PySCF Kohn--Sham object with a mapped CIDER functional.  The
``mlfunc`` argument accepts a packaged name, trusted model path, or loaded
:class:`~ciderpress.dft.xc_evaluator.MappedXC`/
:class:`~ciderpress.dft.xc_evaluator2.MappedXC2` object.  The decorator selects
its numerical integrator from the stored feature settings and supplies CIDER
energy, potential, checkpoint, density-fitting, and supported gradient
behavior.

Exchange models use ``xmix``, ``xkernel``, and ``ckernel`` to define their
surrogate-hybrid composition.  Full-XC models use their serialized energy
composition.  D4 metadata is reconciled after the SCF energy as described in
:doc:`../../usage/production_models`.

The basic full-XC use case is:

.. code-block:: python

    from pyscf import gto
    from pyscf.dft import RKS

    from ciderpress.pyscf.dft import make_cider_calc

    mol = gto.M(...)
    ks = RKS(mol)
    ks = make_cider_calc(ks, "CIDER26XCCHEM")
    etot = ks.kernel()

See :source:`examples/pyscf/production_calc.py` and
:doc:`../../usage/pyscf` for a complete example.

.. automodule:: ciderpress.pyscf.dft
    :members:
