GPAW Interface and PAW Implementation
=====================================

The GPAW interface evaluates supported CIDER models in classic GPAW
plane-wave mode.  The production path uses PAW setups: FFTs evaluate smooth
nonlocal features over the cell, while PAW/PASDW reconstructs the missing
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
* :mod:`ciderpress.gpaw.interp_paw` reconstructs differentiable all-electron
  and pseudo partial-wave XC quantities on radial grids.
* :mod:`ciderpress.gpaw.descriptors` exposes descriptor evaluation from a
  live GPAW calculation.

For production model selection, complete examples, and suggested fallback
settings, see :doc:`../../usage/production_models`,
:doc:`../../usage/gpaw`, and :doc:`../../usage/convergence`.

Norm-conserving pseudopotentials do not contain the all-electron information
required by the documented NLDF path.  CiderPress does not support
``gpaw.new`` in version 0.5.0.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   calculator
   descriptors
   numerical
