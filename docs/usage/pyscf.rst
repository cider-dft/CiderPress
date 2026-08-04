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

See :doc:`production_models` before changing the model or mixing fraction.

Closed-shell workflow
---------------------

The complete first calculation is:

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
problem; CIDER26XC evaluates no exact exchange.  Select a compatible auxiliary
basis and keep it consistent across an energy difference:

.. code-block:: python

   mf = make_cider_calc(dft.RKS(mol), "CIDER26XCCHEM")
   mf = mf.density_fit(auxbasis="def2-universal-jfit")

Density fitting leaves the selected CIDER model and its D4 behavior unchanged.
It is a numerical approximation, not a change to the CIDER functional form.

Open-shell systems
------------------

Use ``UKS`` and set ``mol.spin`` to :math:`N_\alpha-N_\beta`.  Supply a
physically motivated initial state and verify the converged occupations,
orbital character, and ``spin_square()`` result.  A converged total energy can
belong to an unintended electronic basin.

For a difficult system, first converge a conventional functional with the
same molecule, basis, grid, charge, and spin, then provide its density to the
CIDER calculation:

.. literalinclude:: ../../examples/pyscf/restart_calc.py
   :language: python
   :linenos:

The fallback ladder changes one control family at a time.  A relaxed or
level-shifted rung supplies a new starting density; the example reruns that
density with the intended tight controls before returning a result.

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

Recreate the CIDER calculation before loading the checkpoint orbitals; a PySCF
checkpoint does not recreate the selected CIDER model automatically.  Confirm
that its molecule, basis, charge, spin, and orbital dimensions match the new
calculation.

D4-corrected energy
-------------------

``CIDER26XCCHEMD4`` uses the same SCF density path as its mapped full-XC model
and evaluates D4 from the geometry afterward.  Inspect
``e_tot_base``, ``e_vdw_present``, ``e_vdw_expected``, and ``e_vdw_delta``
when combining wrappers or restarting an existing object.  The returned
``kernel()`` value and ``e_tot`` include the final adjustment exactly once.

This dispersion accounting is active for every CIDER26XC model, not only
``CIDER26XCCHEMD4``.  If a supported ``with_dftd3``/``with_dftd4`` wrapper is
attached to the incoming object, CiderPress removes it and restores the
dispersion behavior the model was trained with: the D4 term for
``CIDER26XCCHEMD4``, and no dispersion term for ``CIDER26XCCHEM`` and
``CIDER26XCSURFSCI`` (their ``e_vdw_expected`` is zero).  An external D4
wrapper added on top of these models therefore does not contribute to the
returned energy.

D4 does not contribute to the current molecular gradient implementation.  See
:doc:`properties` before using derivative-based workflows.

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

Hessians, NMR, polarizability, coupled-cluster, multireference, and similar
methods are not available from a CIDER mean-field object.  SDMX and
nonlocal-orbital molecular gradients are also unavailable.  The complete list
of currently supported properties is given in :doc:`properties`.

SCF practice
------------

* Start routine closed-shell calculations with normal CDIIS and the accuracy
  required by the final energy difference.
* Use a baseline density for open-shell, transition-metal, stretched-bond, or
  near-degenerate systems.
* Preserve a fresh-atomic-guess route; a baseline density can select the wrong
  basin.
* Confirm that a converged solution has the intended occupations, spin state,
  and orbital character.
* Diagnose occupation switching before applying finite-temperature smearing.

The complete symptom-based ladder is in :doc:`convergence`, and the record to
retain with a result is in :doc:`reproducibility`.
