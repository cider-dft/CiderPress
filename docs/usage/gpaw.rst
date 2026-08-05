Periodic and Isolated Calculations with GPAW
============================================

CiderPress integrates with the classic GPAW calculator in plane-wave mode.
The supported route uses PAW setups so the nonlocal descriptors include the
all-electron core-region information described in
:doc:`../theory/numerical_evaluation`.

Functional construction
-----------------------

CIDER26XC is a full-XC model.  Construct it with the composition stored in the
model:

.. code-block:: python

   xc = get_cider_functional(
       "CIDER26XCSURFSCI",
       xmix=1.0,
       xkernel=None,
       ckernel=None,
       pasdw_store_funcs=False,
   )

For an exchange-only CIDER23X model, retain an explicit PBE0/CIDER
composition:

.. code-block:: python

   xc = get_cider_functional(
       "CIDER23X_NL_MGGA_DTR",
       xmix=0.25,
       xkernel="GGA_X_PBE",
       ckernel="GGA_C_PBE",
   )

The GPAW interface evaluates NLDF models.  CIDER24X uses the PySCF SDMX path,
and CIDER26XCCHEMD4 uses the PySCF D4 energy interface.

PBE-seeded periodic workflow
----------------------------

Converge a PBE calculation using the final cell, PAW setups, cutoff, k-points,
bands, symmetry, charge, spin, and occupations.  Write wavefunctions with
``mode="all"`` and restart with a newly constructed CIDER functional:

.. literalinclude:: ../../examples/gpaw/production_calc.py
   :language: python
   :linenos:

Use :class:`~ciderpress.gpaw.calculator.CiderGPAW` for checkpointed
calculations; it saves the CIDER-specific restart information.  A normal
``GPAW`` calculator can evaluate an in-memory CIDER object within the current
process.

Surfaces and adsorption systems
-------------------------------

A surface calculation adds convergence dimensions beyond a bulk calculation:
slab thickness, vacuum, lateral cell, k-point sampling, dipole treatment,
adsorbate coverage, and magnetic state.  The workflow template preserves
separate PBE and CIDER checkpoints:

.. literalinclude:: ../../examples/gpaw/surface_calc.py
   :language: python
   :linenos:

Evaluate the slab, isolated adsorbate/reference, and combined system with a
consistent functional composition.  Each system can use the representation
appropriate to its boundary conditions; for example, a molecule may use an
isolated box.  Converge each representation to the accuracy required by the
energy difference.

Isolated systems in periodic boxes
----------------------------------

Use Gamma-point sampling, an explicit cell, the correct charge and spin, and
enough vacuum for the property of interest:

.. literalinclude:: ../../examples/gpaw/isolated_calc.py
   :language: python
   :linenos:

A cubic box of approximately 12--15 Angstrom is a useful initial choice for a
small neutral compact system.  Charged, diffuse, polar, or response-property
calculations may require a larger cell and finite-size electrostatic
treatment.  Use the intended magnetic state in both PBE and CIDER stages.

Restarting a CIDER checkpoint
-----------------------------

Write both baseline and CIDER meta-GGA calculations with ``mode="all"``.  To
continue a CIDER checkpoint, construct a fresh functional and pass it when the
calculator is recreated:

.. literalinclude:: ../../examples/gpaw/restart_calc.py
   :language: python
   :linenos:

The checkpoint stores the mapped model text and nonlocal interpolation
parameters, including ``Nalpha``, ``lambd``, and the plane-wave cutoff
``encut`` derived from the ``qmax`` argument.  Opening the checkpoint with its
saved XC setting makes :class:`~ciderpress.gpaw.calculator.CiderGPAW`
reconstruct the saved CIDER functional.  An explicit ``xc=`` argument selects
a replacement functional.  Verify the model name, composition, and requested
numerical controls in the resumed output.

Forces and stress
-----------------

ASE's normal ``get_forces()`` and ``get_stress()`` calls include the CIDER FFT
and PAW derivative contributions for the supported PAW path.  See
:doc:`properties` and the complete force/stress template.  The supported
meta-GGA force/stress implementation uses PAW setups.

PASDW and interpolation controls
--------------------------------

``pasdw_store_funcs=False`` is the memory-saving default.  Setting it to
``True`` caches atomic projector functions and can reduce repeated cost at
substantial memory expense.  ``pasdw_ovlp_fit=True`` selects overlap fitting
for the projection.  Use the same value throughout a numerical comparison.

``qmax`` and ``lambd`` control the expansion of the nonlocal kernel.  Defaults
are intended as general settings; smaller ``lambd`` gives denser interpolation
and higher cost.  Treat a change to these values as a numerical convergence
choice and keep it consistent across an energy difference.  ``Nalpha`` is
normally determined automatically.

Parallel execution and memory
-----------------------------

Use ``parallel={"augment_grids": True}`` to distribute XC grid work.  The
nonlocal FFT buffers and PAW data add memory beyond a comparable PBE
calculation.  For memory-heavy systems, run each restart attempt as a separate
job so memory from the previous calculator is released.

The CiderPress extension and GPAW must use compatible MPI, FFT, BLAS, and
OpenMP runtimes.  Validate the exact launcher and rank layout on a small job
before scaling to multiple nodes.

SCF strategy
------------

Start from a converged PBE checkpoint.  If the direct CIDER restart fails,
preconverge PBE with the mixer intended for the next CIDER attempt.  Metals
and magnetic systems often need smaller density and magnetization mixing,
occupation smearing, and explicit monitoring of local moments.

Use :doc:`convergence` for calculation-class ladders and
:doc:`reproducibility` for the settings and checks to retain with a calculation.
