.. _GPAW Calculator Interface:

GPAW Calculator Interface
=========================

:func:`ciderpress.gpaw.calculator.get_cider_functional` loads a mapped model,
validates its feature support and XC composition, and constructs the matching
GGA or meta-GGA smooth-grid/PAW object.  Exchange models receive their
``xmix``, ``xkernel``, and ``ckernel`` composition.  Full-XC models receive the
composition stored in the model.

:class:`ciderpress.gpaw.calculator.CiderGPAW` extends the classic ``GPAW``
calculator with CIDER checkpoint state.  Its dictionary stores the mapped
model text, XC composition, NLDF interpolation parameters, and PASDW options
needed to reconstruct the functional during restart.

.. code-block:: python

    from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional

    xc = get_cider_functional(
        "CIDER26XCSURFSCI", xmix=1.0, xkernel=None, ckernel=None)
    atoms.calc = CiderGPAW(xc=xc, ...)
    atoms.get_potential_energy()

For a full example, see :source:`examples/gpaw/production_calc.py`
and :doc:`../../usage/gpaw`.

.. automodule:: ciderpress.gpaw.calculator
    :members:
