Energies and Derivative Properties
==================================

CIDER models participate in a self-consistent calculation through their
energy density and functional derivatives.  Property support depends on the
feature family, backend, and any additive correction.

.. list-table:: Property support in the documented calculation paths
   :header-rows: 1
   :widths: 28 24 24 24

   * - Property
     - Molecular PySCF NLDF
     - Molecular PySCF SDMX
     - GPAW NLDF with PAW
   * - Total energy
     - Supported
     - Supported
     - Supported
   * - Self-consistent potential
     - Supported
     - Supported
     - Supported
   * - Nuclear forces/gradients
     - Supported, subject to the restrictions below
     - Not supported
     - Supported
   * - Cell stress
     - Not applicable
     - Not applicable
     - Supported
   * - Hessians and response properties
     - Not implemented by the CIDER decorator
     - Not implemented
     - Outside the documented interface

Molecular gradients
-------------------

For an NLDF CIDER calculation, obtain a PySCF gradient object from the
converged mean-field object:

.. literalinclude:: ../../examples/pyscf/gradient_calc.py
   :language: python
   :linenos:

Use ``grid_response=True`` when the desired derivative must include the
response of the atom-centered integration grid.  Converge the electronic SCF,
basis, quadrature grid, and density-fitting treatment before interpreting a
gradient.

``CIDER26XCCHEMD4`` adds D4 to the final energy.  ``nuc_grad_method()`` returns
the electronic CIDER gradient.  A geometry optimization of the composite
CIDER+D4 energy requires a separate implementation of the D4 force.

Periodic forces and stress
--------------------------

With classic GPAW and PAW setups, use the standard ASE calls after the CIDER
SCF has converged:

.. literalinclude:: ../../examples/gpaw/forces_stress.py
   :language: python
   :linenos:

The analytical contribution includes the FFT feature response and PAW/PASDW
terms.  The supported meta-GGA force and stress route uses PAW setups.
Converge the plane-wave cutoff, k-point mesh, cell, PAW setups, and electronic
criteria to the accuracy required by the derivative.

Energy differences
------------------

Every member of an energy difference must use the same model composition and
compatible numerical settings.  In particular:

* Use the same full-XC or surrogate-hybrid initialization convention for every
  member.
* Apply D4 exactly once to every applicable structure.
* Use consistent charge, spin, basis/grid or cutoff, setups, k-points, and
  smearing conventions.
* Converge isolated references in their own representation to the accuracy
  required by the energy difference.

See :doc:`reproducibility` for the associated reporting checklist.
