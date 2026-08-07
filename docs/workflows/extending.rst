Extending CiderPress
====================

A new feature touches five layers: a settings class in
:mod:`ciderpress.dft.settings`, a numerical plan in
:mod:`ciderpress.dft.plans`, a normalizer in
:mod:`ciderpress.dft.feat_normalizer`, a bounded map in
:mod:`ciderpress.dft.transform_data`, and a backend adapter under
``ciderpress.pyscf`` or ``ciderpress.gpaw``.  All five must agree on one raw
feature order, because that order is serialized with the model and replayed
whenever it is loaded.

.. important::

   Open a
   `CiderPress GitHub issue <https://github.com/mir-group/CiderPress/issues/new>`_
   before changing a feature family, numerical backend, PAW/PASDW route,
   mapped-model format, or checkpoint schema.  These are cross-backend
   interfaces, and a model file written by one version must remain
   interpretable by the code that loads it.

Adding or modifying a feature
-----------------------------

1. Subclass :class:`~ciderpress.dft.settings.BaseSettings` and implement
   ``nfeat``, ``get_feat_usps`` (the uniform scaling power of each raw
   feature), and ``ueg_vector`` (its uniform electron gas value).
   ``get_reasonable_normalizer`` supplies the default normalization.
2. Implement a plan in :mod:`ciderpress.dft.plans` that evaluates the raw
   feature and its adjoint derivative with respect to the electronic inputs.
   The adjoint is what becomes the XC potential, so it must be the exact
   transpose of the forward operation rather than an approximation to it.
3. Add the matching normalizer and bounded transform, keeping the same
   feature order used during training.
4. Connect the plan to each intended backend, and raise an error at construction time
   in the backends that cannot evaluate it.  :func:`~ciderpress.gpaw.calculator.get_cider_functional`
   shows the pattern: it rejects SDMX and non-version-j NLDF models rather
   than failing later in the SCF.
5. Check feature values, finite difference derivatives, spin exchange symmetry,
   low-density behavior, and a serialization round trip before fitting a
   model.

Backend data flow
-----------------

PySCF selects its numerical integrator from the feature settings stored in the
model.  A new molecular feature must work for restricted and unrestricted
density layouts and, if gradients are claimed, through the grid and orbital
response path.

In GPAW, which uses the PAW formalism, nonlocal features contain contributions
from the smooth pseudo-density, handled on a uniform grid, and the compact atomic
core density, handled on radial support grids around the atoms. Therefore, introducing a
new nonlocal feature requires matching all-electron and pseudo forward
terms, PASDW transfer, potential back-propagation, and any claimed force or
stress derivative.  See :doc:`../ciderpress/gpaw/numerical` for details.

PAW and interpolation invariants
--------------------------------

Changes to partial-wave interpolation or full-potential/core handling can
alter energies and descriptors across many elements.  Preserve these
invariants:

* Values at radial-grid endpoints are accepted within floating-point
  tolerance; genuine extrapolation raises an error.
* Reconstructed all-electron and pseudo quantities use the same partial-wave
  convention as their derivatives.
* Kinetic-energy-density terms remain finite at the origin and obey their
  physical core lower bound.
* Energy, potential, force, and stress paths use the same discretized feature.
* Changing a stored reference value means the functional itself changed;
  establish which numerical difference caused it before updating the value.

Model and checkpoint compatibility
----------------------------------

Settings, mapped evaluators, and checkpoint dictionaries are persistent
interfaces: a model file or checkpoint written today must still load after
the code changes.  A new field therefore needs a default that reproduces the
behavior of objects serialized before it existed, and a changed functional
composition needs explicit serialized metadata so that the difference is
visible to whatever loads it.

See :doc:`../reference/limitations` for the feature and property
combinations the release currently supports.
