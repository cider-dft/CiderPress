Using CIDER26XC with GPAW
=========================

Use ``CIDER26XCSURFSCI`` with the classic GPAW calculator for periodic solids,
surfaces, adsorption systems, and isolated references in periodic boxes.
CiderPress currently targets plane-wave mode with PAW setups.  Norm-conserving
pseudopotentials are not recommended because the nonlocal features depend on
all-electron information, and ``gpaw.new`` is not supported in version 0.5.0.

Periodic calculation
--------------------

A robust workflow is to converge PBE first, save a ``.gpw`` checkpoint, and
start CIDER from that density while preserving the numerical representation.
For the production meta-GGA models, write the PBE checkpoint with
``mode="all"`` so that the wavefunctions needed to reconstruct the
kinetic-energy density are present.

.. literalinclude:: ../../examples/gpaw/production_calc.py
   :language: python
   :linenos:

Always construct CIDER26XC as a full-XC functional::

   xc = get_cider_functional(
       "CIDER26XCSURFSCI",
       xmix=1.0,
       xkernel=None,
       ckernel=None,
       pasdw_store_funcs=False,
   )

Use :class:`ciderpress.gpaw.calculator.CiderGPAW` for calculations that must be
written or restarted.  Write CIDER checkpoints with ``mode="all"`` as well as
the PBE seed checkpoints.  When resuming a CIDER-written checkpoint, construct
a fresh functional and pass it explicitly to ``CiderGPAW``.  The ordinary
``gpaw.restart`` reader does not reconstruct every CIDER functional state.

The examples use the memory-saving default ``pasdw_store_funcs=False`` and
retain the default PASDW overlap fit for improved PAW-projection precision.
Disabling ``pasdw_ovlp_fit`` can reduce its cost; keep the choice fixed in
comparative calculations and include it in numerical sensitivity checks when
small energy differences matter.

Isolated systems in periodic boxes
----------------------------------

An isolated atom or molecule requires a sufficiently large cell, Gamma-point
sampling, and the correct charge and spin state:

.. literalinclude:: ../../examples/gpaw/isolated_calc.py
   :language: python
   :linenos:

A cubic box of approximately 12--15 Angstrom is a useful initial choice for
small neutral systems.  It is not a substitute for a box-size convergence
study.  Charged, diffuse, strongly polar, or response-property calculations
can require larger cells and electrostatic finite-size treatment.

For open-shell atoms and molecules, set the initial magnetic moments and
preserve them through the PBE and CIDER stages.  Check the final total and
local moments rather than relying only on the input state.

The baseline PBE stage is a density preconditioner rather than the reported
result.  A density target around ``1e-2`` can be sufficient for that stage;
apply the final energy, density, and eigenstate criteria to the subsequent
CIDER calculation.  The example's CIDER criteria are general starting values,
not universal tolerances: tighten them and verify convergence for the requested
energy difference, force, stress, or response property.

Recommended numerical practice
------------------------------

* Keep PAW setups, plane-wave cutoff, grid spacing, k-points, number of bands,
  symmetry, charge, and spin consistent between the baseline and CIDER stages.
* Converge cutoff, k-point mesh, vacuum, slab thickness, and cell dimensions
  for the final energy difference or force.
* Use ``parallel={"augment_grids": True}`` to distribute CIDER's grid work.
* Save separate PBE and CIDER checkpoints.  For large calculations, running
  one fallback rung per process can reduce retained calculator memory.
* ``CIDER26XCCHEMD4`` is unavailable in GPAW; use it only through PySCF.

See :doc:`convergence` for suggested mixers, eigensolvers, smearing widths,
and restart ladders for different periodic calculation classes.
