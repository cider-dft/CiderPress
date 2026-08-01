Using CIDER26XC with PySCF
==========================

PySCF is the recommended backend for molecular calculations with
``CIDER26XCCHEM`` and ``CIDER26XCCHEMD4``.  ``CIDER26XCSURFSCI`` can also be
used in PySCF when a combined-domain model is desired.

Closed-shell molecules
----------------------

Start from a normal PySCF ``RKS`` object and decorate it with
:func:`ciderpress.pyscf.dft.make_cider_calc`.  The following complete example
also shows the optional D4 energy accounting:

.. literalinclude:: ../../examples/pyscf/production_calc.py
   :language: python
   :linenos:

The basis and grid in an example are starting points, not universal
production settings.  Converge them for the requested energy difference,
force, or response property.  Density fitting affects the Coulomb evaluation;
use a compatible auxiliary basis and keep the treatment consistent across all
members of an energy difference.

Open-shell molecules and atoms
------------------------------

Use ``UKS`` when the target state is open shell.  Set ``mol.spin`` to the
number of alpha electrons minus beta electrons and verify the final
``spin_square()`` result, occupations, and orbital character.  A converged
energy alone does not prove that the intended electronic state was obtained.

For a difficult state, first converge a conventional functional using the
same molecule, basis, grid, charge, and spin, then pass its density explicitly
to CIDER:

.. literalinclude:: ../../examples/pyscf/restart_calc.py
   :language: python
   :linenos:

Store the baseline and CIDER checkpoints separately.  If a fallback setting
produces a stable density, use that density to rerun the intended tight
calculation rather than reporting a deliberately loosened preconditioning
step as the final result.

Recommended numerical practice
------------------------------

* Use the same basis, grid, density-fitting treatment, charge, and spin for
  all systems entering an energy difference.
* A target such as ``conv_tol = 1e-9`` Hartree is reasonable for molecular
  energy differences, but the required tolerance is property dependent.
* Warm-start related geometries or a sequence of CIDER calculations from the
  previous converged density when the physical state is continuous.
* Preserve each checkpoint and the settings that produced it.  Do not replace
  a failed calculation with the energy of an earlier state.
* Validate energy stationarity before relaxing ``conv_tol_grad``.  Persistent
  orbital-gradient oscillation can indicate a different occupation, spin
  state, or unstable SCF solution.

See :doc:`convergence` for suggested fallback settings by molecular
calculation class and :doc:`production_models` for model selection and D4
semantics.
