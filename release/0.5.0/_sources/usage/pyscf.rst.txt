Molecular Calculations with PySCF
=================================

The PySCF interface decorates an existing restricted or unrestricted
Kohn--Sham object.  Molecular geometry, charge, spin, basis, grids, density
fitting, occupations, and SCF controls remain PySCF concepts; CiderPress
replaces the numerical XC evaluation and adds any model-specific energy terms.

NOTE: the decorated object sets ``ks.xc`` to a placeholder semilocal name
(``"PBE"`` for GGA-level models, ``"R2SCAN"`` for meta-GGA) so that PySCF
routes the calculation through its semilocal machinery instead of treating it
as a hybrid.  The functional actually evaluated is the CIDER model, so
``ks.xc`` should not be read as the functional in use.

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

The example uses the ``def2-tzvp`` basis and PySCF grid level 3.  For a small
end-to-end energetics workflow,
:source:`examples/pyscf/compute_ae.py` computes a molecular atomization energy
with a chosen libxc or packaged CIDER functional.

Density fitting
---------------

Calling ``density_fit`` after ``make_cider_calc`` accelerates the Coulomb
problem.  CIDER26XC has a zero exact-exchange fraction.  The auxiliary basis
is supplied through the normal PySCF interface:

.. code-block:: python

   mf = make_cider_calc(dft.RKS(mol), "CIDER26XCCHEM")
   mf = mf.density_fit(auxbasis="def2-universal-jfit")

Density fitting approximates the Coulomb contribution.  The selected CIDER
model and its D4 behavior remain unchanged.

Open-shell systems
------------------

Use ``UKS`` and set ``mol.spin`` to :math:`N_\alpha-N_\beta`.
``spin_square()`` returns :math:`\langle S^2\rangle` and the corresponding
multiplicity for the converged unrestricted solution.

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
the PySCF checkpoint.  ``load_scf`` also returns the checkpoint's ``Mole``
object; the stored orbital dimensions must match the new mean-field object.

D4-corrected energy
-------------------

``CIDER26XCCHEMD4`` evaluates its electronic full-XC contribution during the
SCF and D4 from the geometry afterward, so the dispersion term changes the
total energy but not the density or the CIDER potential.

Because a user may already have attached a dispersion wrapper, CiderPress
reconciles what is present with what the model expects rather than simply
adding a term.  After ``kernel()``, every CIDER26XC calculation exposes:

``mf.e_tot_base``
   Energy returned by the underlying SCF object, before the dispersion
   adjustment.

``mf.e_vdw_present``
   Dispersion already contained in that energy, if a supported
   ``with_dftd3``/``with_dftd4`` wrapper was attached.

``mf.e_vdw_expected``
   Dispersion the selected model was trained with.

``mf.e_vdw_delta``
   The correction ``e_vdw_expected - e_vdw_present``.

The returned ``mf.e_tot`` is ``mf.e_tot_base + mf.e_vdw_delta``, which
contains the expected term exactly once.  For ``CIDER26XCCHEM`` and
``CIDER26XCSURFSCI`` the expected dispersion is zero, so an attached wrapper
is disabled and does not contribute to the returned energy.

The current molecular gradient contains the electronic CIDER contribution
only.  See :doc:`properties` for composite CIDER+D4 derivatives.

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
``mf.nuc_grad_method()`` as shown in :doc:`properties`.  The gradient code
raises ``NotImplementedError`` for models carrying SDMX features, so CIDER24X
calculations give energies but not forces.

The CIDER mean-field object provides the energies and derivatives listed in
:doc:`properties`.  Hessians, NMR, polarizability, coupled-cluster,
multireference, and similar response or post-SCF methods are outside that
interface.  The documented analytical-gradient implementation covers the
molecular NLDF path.

The PBE-seeded CDIIS, ADIIS, EDIIS, level-shift, and damping settings used by
the restart example are listed in :doc:`convergence`.
