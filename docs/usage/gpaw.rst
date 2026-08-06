Periodic and Isolated Calculations with GPAW
============================================

CiderPress integrates with the classic GPAW :footcite:p:`mortensenGPAWOpenPython2024`
calculator in plane-wave mode.
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

The periodic example evaluates PBE and writes its wavefunctions with
``mode="all"`` before constructing the CIDER functional and restarting:

.. literalinclude:: ../../examples/gpaw/production_calc.py
   :language: python
   :linenos:

Use :class:`~ciderpress.gpaw.calculator.CiderGPAW` for checkpointed
calculations; it saves the CIDER-specific restart information.  A normal
``GPAW`` calculator can evaluate an in-memory CIDER object within the current
process.

Surfaces and adsorption systems
-------------------------------

The surface template uses a dipole correction, a slab-adapted k-point mesh,
and separate PBE and CIDER checkpoints:

.. literalinclude:: ../../examples/gpaw/surface_calc.py
   :language: python
   :linenos:

Isolated systems in periodic boxes
----------------------------------

The isolated-system example places an atom in a 12 Angstrom cubic cell and
uses Gamma-point sampling:

.. literalinclude:: ../../examples/gpaw/isolated_calc.py
   :language: python
   :linenos:

For small neutral compact systems, 12--15 Angstrom is a practical initial box
size.  Charged or diffuse systems generally require a larger cell and an
appropriate finite-size electrostatic treatment.

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
the functional for the resumed calculation.

Forces and stress
-----------------

ASE's normal ``get_forces()`` and ``get_stress()`` calls include the CIDER FFT
and PAW derivative contributions for the supported PAW path.  See
:doc:`properties`, which includes the complete
:source:`examples/gpaw/forces_stress.py` template.

PASDW and interpolation controls
--------------------------------

``pasdw_store_funcs=False`` is the memory-saving default.  Setting it to
``True`` caches atomic projector functions and can reduce repeated cost at
substantial memory expense.  ``pasdw_ovlp_fit=True`` selects overlap fitting
for the projection.  Both options are part of the PAW numerical
representation.

``qmax`` and ``lambd`` control the expansion of the nonlocal kernel.  Smaller
``lambd`` gives denser interpolation and higher cost.  ``Nalpha`` is normally
determined automatically.

Parallel execution and memory
-----------------------------

Use ``parallel={"augment_grids": True}`` to distribute XC grid work.  The
nonlocal FFT buffers and PAW data add memory beyond a comparable PBE
calculation.  For memory-heavy systems, run each restart attempt as a separate
job so memory from the previous calculator is released.

The CiderPress extension and GPAW must use compatible MPI, FFT, BLAS, and
OpenMP runtimes.

SCF strategy
------------

:doc:`convergence` lists the mixer, eigensolver, occupation, and restart
settings used by the bulk, surface, isolated-system, and magnetic fallback
examples.
