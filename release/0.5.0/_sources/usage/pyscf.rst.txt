Molecular Calculations with PySCF
=================================

The PySCF interface decorates an existing restricted or unrestricted
Kohn--Sham object.  Molecular geometry, charge, spin, basis, grids, density
fitting, occupations, and SCF controls remain PySCF concepts; CiderPress
replaces the numerical XC evaluation and adds any model-specific energy terms.

Choose the functional composition first
---------------------------------------

CIDER26XC models contain full exchange and correlation.  Their PySCF
initializer defaults are correct:

.. code-block:: python

   mf = make_cider_calc(dft.RKS(mol), "CIDER26XCCHEM")

CIDER23X and CIDER24X contain exchange.  Specify the surrogate-hybrid
composition explicitly:

.. code-block:: python

   mf = make_cider_calc(
       dft.RKS(mol),
       "CIDER23X_NL_MGGA_DTR",
       xmix=0.25,
       xkernel="GGA_X_PBE",
       ckernel="GGA_C_PBE",
   )

The compositions associated with each packaged model are listed in
:doc:`production_models`.

Closed-shell workflow
---------------------

A complete closed-shell example is:

.. literalinclude:: ../../examples/pyscf/production_calc.py
   :language: python
   :linenos:

The example uses a moderate basis and grid so it can be run directly.  For a
quantitative calculation, converge the basis, integration grid, SCF thresholds,
and density-fitting approximation for the requested energy difference or
derivative.  For a small end-to-end energetics workflow,
:source:`examples/pyscf/compute_ae.py` computes a molecular atomization energy
with a chosen libxc or packaged CIDER functional.

Density fitting
---------------

Calling ``density_fit`` after ``make_cider_calc`` accelerates the Coulomb
problem.  CIDER26XC has a zero exact-exchange fraction.  Select a compatible
auxiliary basis and keep it consistent across an energy difference:

.. code-block:: python

   mf = make_cider_calc(dft.RKS(mol), "CIDER26XCCHEM")
   mf = mf.density_fit(auxbasis="def2-universal-jfit")

Density fitting approximates the Coulomb contribution.  The selected CIDER
model and its D4 behavior remain unchanged.

Open-shell systems
------------------

Use ``UKS`` and set ``mol.spin`` to :math:`N_\alpha-N_\beta`.  Supply a
physically motivated initial state and verify the converged occupations,
orbital character, and ``spin_square()`` result.  Multiple stationary
solutions can satisfy the SCF thresholds.

For a difficult system, first converge a conventional functional with the
same molecule, basis, grid, charge, and spin, then provide its density to the
CIDER calculation:

.. literalinclude:: ../../examples/pyscf/restart_calc.py
   :language: python
   :linenos:

The fallback ladder records each control set separately.  A relaxed or
level-shifted rung supplies a new starting density, followed in this example
by a calculation with the listed standard CDIIS controls.

Checkpoints and interrupted calculations
----------------------------------------

Set a different ``chkfile`` for the baseline, each fallback rung, and the
final CIDER calculation.  A checkpoint can recover molecular orbitals and a
density after interruption:

.. code-block:: python

   from pyscf.scf import chkfile

   mol_from_chk, scf_data = chkfile.load_scf("cider.chk")
   mo = scf_data["mo_coeff"]
   occ = scf_data["mo_occ"]
   dm0 = mf.make_rdm1(mo, occ)
   energy = mf.kernel(dm0=dm0)

Recreate the selected CIDER model, then load the orbitals and occupations from
the PySCF checkpoint.  Confirm that its molecule, basis, charge, spin, and
orbital dimensions match the new calculation.

D4-corrected energy
-------------------

``CIDER26XCCHEMD4`` evaluates its electronic full-XC contribution during the
SCF and D4 from the geometry afterward.  Every CIDER26XC calculation exposes
``e_tot_base``, ``e_vdw_present``, ``e_vdw_expected``, and ``e_vdw_delta``.
For ``CIDER26XCCHEM`` and ``CIDER26XCSURFSCI``, the expected dispersion is zero
and a supported ``with_dftd3``/``with_dftd4`` wrapper is disabled.  For
``CIDER26XCCHEMD4``, CiderPress measures an attached wrapper contribution and
adjusts it to the model's expected D4 value.  The returned ``kernel()`` value
and ``e_tot`` contain the final energy.  See :doc:`production_models` for the
attribute definitions and accounting formula.

The current molecular gradient contains the electronic CIDER contribution.
See :doc:`properties` for composite CIDER+D4 derivatives.

CIDER24X
--------

Install ``ciderpress[cider24]`` and use the same exchange-only composition as
CIDER23X.  These SDMX models use the density matrix and PyTorch-backed mapped
evaluator.  Keep the model on the device selected by its evaluator and avoid
mixing CUDA and CPU PyTorch installations within one environment.  A minimal
SDMX calculation is shown in :source:`examples/pyscf/simple_sdmx.py`.

Gradients and other methods
---------------------------

Analytical restricted and unrestricted nuclear gradients are available for
the molecular NLDF path, including density-fitted calculations.  Use
``mf.nuc_grad_method()`` as shown in :doc:`properties`.

The CIDER mean-field object provides the energies and derivatives listed in
:doc:`properties`.  Hessians, NMR, polarizability, coupled-cluster,
multireference, and similar response or post-SCF methods are outside that
interface.  The documented analytical-gradient implementation covers the
molecular NLDF path.

SCF practice
------------

* Start routine closed-shell calculations with normal CDIIS and the accuracy
  required by the final energy difference.
* Use a baseline density for open-shell, transition-metal, stretched-bond, or
  near-degenerate systems.
* Compare baseline-seeded and atomic-guess solutions when multiple electronic
  states are plausible.
* Confirm that a converged solution has the intended occupations, spin state,
  and orbital character.
* Diagnose occupation switching before applying finite-temperature smearing.

The complete symptom-based ladder is in :doc:`convergence`, and the record to
retain with a result is in :doc:`reproducibility`.
