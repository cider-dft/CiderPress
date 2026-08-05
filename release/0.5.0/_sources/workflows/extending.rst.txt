Extending CiderPress
====================

A complete density-functional feature includes its settings, numerical
forward operation, adjoint derivative, backend connections, serialization,
and validation.  These parts share one feature order and mathematical
definition across every supported backend.

.. important::

   We strongly recommend opening a
   `CiderPress GitHub issue <https://github.com/mir-group/CiderPress/issues/new>`_
   and collaborating with the MIR developers before implementing or changing
   a feature family, numerical backend, PAW/PASDW route, mapped-model format,
   or checkpoint schema. Substantial feature work should receive joint design
   and code review: settings, feature order, forward and adjoint evaluation,
   analytical derivatives, serialization, and regression tests must evolve
   together for a functional to remain scientifically well defined.

Adding or modifying a feature
-----------------------------

1. Define the feature and its parameters in the settings layer, including its
   feature count, spin behavior, uniform-scaling behavior, and uniform-electron
   gas limit where applicable.
2. Implement a numerical plan that evaluates the raw feature and its adjoint
   derivative with respect to its electronic inputs.
3. Define normalization and bounded transforms with the same feature order in
   training and evaluation.
4. Connect the plan to each intended backend and give unsupported backends a
   clear error at construction time.
5. Test feature values, finite differences, spin exchange symmetry, low-density
   behavior, and serialization before fitting a model with it.

Backend data flow
-----------------

PySCF selects its numerical integrator from the feature settings stored in the
model.  A new molecular feature must work for restricted and unrestricted
density layouts and, if gradients are claimed, through the grid and orbital
response path.

GPAW separates smooth-grid evaluation from atom-centered PAW corrections.  A
new nonlocal PAW feature requires matching all-electron and pseudo forward
terms, PASDW transfer, potential back-propagation, and any claimed force or
stress derivative.  Separate smooth-grid and PAW tests localize a disagreement
to the relevant layer.

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
* A reference-value update records the finite-difference and independent
  numerical-setting comparisons that justify the change.

Model and checkpoint compatibility
----------------------------------

Settings, mapped evaluators, and checkpoint dictionaries are persistent
interfaces.  A new field needs a default that preserves the established
composition of older objects and coverage in round-trip tests.  A changed
functional composition requires explicit serialized metadata and a
compatibility decision.

Before release, exercise serial and MPI layouts, restricted and unrestricted
molecular cases, PAW elements spanning light and transition-metal regimes,
and a full checkpoint restart.  See :doc:`../reference/limitations` before
advertising a new feature/property combination.

Building the documentation
--------------------------

From the ``docs`` directory, build the strict HTML target and check the
generated internal links and mathematics with:

.. code-block:: bash

   sphinx-build -E -n -W --keep-going -b html . _build/html
   python tools/check_html_links.py _build/html
   python tools/check_mathjax.py _build/html
