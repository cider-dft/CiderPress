GPAW Interface and PAW Implementation
=====================================

The GPAW interface evaluates supported CIDER models in classic GPAW
plane-wave mode.  The supported path uses PAW setups: FFTs evaluate smooth
nonlocal features over the cell, and PAW/PASDW reconstructs the missing
all-electron contribution inside augmentation spheres.

This documentation assumes that you are familiar with the GPAW code and
have a working installation of the software. For GPAW documentation,
see the `GPAW website <https://gpaw.readthedocs.io/>`_.

The modules separate the host-code interface from the numerical layers:

* :mod:`ciderpress.gpaw.calculator` constructs a CIDER XC object and provides
  the checkpoint-aware ``CiderGPAW`` calculator.
* :mod:`ciderpress.gpaw.cider_fft` and
  :mod:`ciderpress.gpaw.nldf_interface` evaluate the smooth-grid NLDF forward
  and adjoint paths.
* :mod:`ciderpress.gpaw.cider_paw` and :mod:`ciderpress.gpaw.atom_utils`
  assemble atom-centered PAW/PASDW corrections and their derivatives.
* :mod:`ciderpress.gpaw.interp_paw` extends the PAW potential code
  used in GPAW to support the nonlocal features in CIDER models.
* :mod:`ciderpress.gpaw.descriptors` exposes descriptor evaluation from a
  live GPAW calculation.

For packaged model selection, complete examples, and suggested fallback
settings, see :doc:`../../usage/production_models`,
:doc:`../../usage/gpaw`, and :doc:`../../usage/convergence`.

The documented periodic NLDF path uses PAW setups for its all-electron
information.  Version 0.5.0 supports the classic GPAW calculator;
``gpaw.new`` is outside its interface.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   calculator
   descriptors
   numerical
