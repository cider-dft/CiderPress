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

Use ``grid_response=True`` to include the response of the atom-centered
integration grid.

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

Energy composition
------------------

The selected model determines how CIDER enters the total energy:

* CIDER23X and CIDER24X use the explicit surrogate-hybrid composition given by
  ``xmix``, ``xkernel``, and ``ckernel``.
* CIDER26XC contains the complete exchange-correlation model and uses the
  full-XC initialization shown in :doc:`production_models`.
* ``CIDER26XCCHEMD4`` adds its expected D4 contribution once through the
  ``e_vdw_delta`` accounting described in :doc:`production_models`.

The same model construction applies to every term in an energy difference.
