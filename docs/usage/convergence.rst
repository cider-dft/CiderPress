SCF Convergence Recommendations
===============================

SCF behavior depends on the spectrum, occupations, spin state, and numerical
representation of the system.  The settings on this page are starting points
for diagnosing common convergence patterns.  Numerical thresholds and any
stabilizing controls form part of the reported calculation and should be
converged for the requested property.

Match the first change to the observed behavior:

.. list-table:: Failure pattern and first response
   :header-rows: 1
   :widths: 35 35 30

   * - Observed pattern
     - Control to examine
     - State to inspect
   * - Density residual decays, then stalls
     - Mixing amplitude or DIIS/Pulay history
     - Occupations and spin
   * - Energy and occupations alternate
     - Smearing or eigensolver
     - Frontier occupations
   * - Magnetic moment drifts or flips
     - Separate charge and magnetization mixing
     - Local and total moments
   * - Restart moves far from its checkpoint
     - Baseline and restart representations
     - Cell, basis/setup, bands, spin, and charge
   * - A relaxed calculation converges and a tighter one oscillates
     - Restart from the relaxed density and test the tighter controls
     - Energy, density, and electronic state

Save a checkpoint before changing the SCF method.  Comparing one control at a
time makes the source of an improvement easier to identify.

Molecular calculations in PySCF
-------------------------------

CDIIS with a history of about eight vectors is a useful initial method for a
routine closed-shell molecule.  Open-shell, stretched-bond,
transition-metal, and near-degenerate systems often benefit from an initial
PBE or PBE0 density evaluated with the same molecule, basis, grid, charge, and
spin.

The following settings illustrate a progression from ordinary CDIIS to more
conservative orbital updates:

.. list-table:: PySCF starting and fallback settings
   :header-rows: 1
   :widths: 18 22 15 15 15 15

   * - Setting
     - DIIS method
     - Space
     - ``conv_tol``
     - Level shift
     - Damping
   * - Standard CDIIS
     - CDIIS
     - 8
     - ``1e-9``
     - 0
     - 0
   * - Larger history
     - CDIIS
     - 12
     - ``1e-8``
     - 0
     - 0
   * - Energy-based mixing
     - ADIIS or EDIIS
     - 12
     - ``1e-7``
     - 0
     - 0
   * - Shifted orbitals
     - CDIIS
     - 8
     - ``1e-7``
     - ``0.2 Ha``
     - 0
   * - Stronger stabilization
     - CDIIS
     - 8
     - ``1e-6``
     - ``0.5 Ha``
     - ``0.3``

A density from a relaxed, shifted, or damped calculation can initialize a
tighter calculation.  Repeating the calculation with reduced stabilization
measures the sensitivity of the energy and state to those controls.  A
reported calculation may retain level shifting or damping when the method is
stated and the target quantity is converged with respect to it.

Baseline-seeded and atomic-guess calculations can converge to different
stationary solutions.  Comparing their energies, occupations, spin
expectation values, and orbital character helps identify the state of
interest.

Finite-temperature smearing in the range ``0.01--0.05 Ha`` can stabilize
genuinely near-degenerate frontier orbitals.  Report the occupation model and
distinguish a finite-temperature free energy from an extrapolated
zero-temperature energy when smearing is retained.

The CIDER26XC PySCF integration path uses the conventional DIIS methods listed
above; Newton/SOSCF is outside this path.  Deliberate non-Aufbau occupations
should be recorded together with the procedure used to select them.

Insulating and semiconducting GPAW calculations
-----------------------------------------------

A converged PBE checkpoint provides a useful starting density when it uses
the CIDER calculation's cell, PAW setups, cutoff, k-points, bands, symmetry,
charge, spin, and occupations.  The baseline calculation can also be repeated
with the mixer planned for a difficult CIDER restart.

.. list-table:: GPAW periodic starting and fallback settings
   :header-rows: 1
   :widths: 20 35 20 25

   * - Setting
     - Mixer
     - Fermi width
     - Eigensolver
   * - Moderate Pulay
     - ``Mixer(0.05, 5, 50)``
     - ``0.1 eV``
     - Inherited/default
   * - Conservative Pulay
     - ``Mixer(0.02, 2, 100)``
     - ``0.1 eV``
     - Inherited/default
   * - Near-linear mixing
     - ``Mixer(0.02, 1, 50)``
     - ``0.1 eV``
     - Inherited/default
   * - RMM-DIIS option
     - ``Mixer(0.02, 1, 50)``
     - ``0.1 eV``
     - ``rmm-diis``

Persistent orbital-step oscillation can be tested with bounded RMM-DIIS
steps:

.. code-block:: python

   from gpaw.eigensolvers import RMMDIIS

   eigensolver = RMMDIIS(
       niter=3,
       limit_lambda={"absolute": False, "lower": 0.01, "upper": 0.1},
       trial_step=0.01,
   )

This eigensolver can be paired with ``Mixer(0.02, 1, 50)``.  Density and
eigenstate criteria should then be chosen for the requested energy, force, or
stress accuracy.

Metallic and magnetic GPAW calculations
---------------------------------------

Metals commonly require occupation smearing and smaller mixing amplitudes.
For a magnetic system whose charge and spin densities converge on different
scales, GPAW provides separate mixing parameters:

.. code-block:: python

   from gpaw import MixerDif

   mixer = MixerDif(
       0.005, 5, 100,
       beta_m=0.005,
       nmaxold_m=3,
       weight_m=100,
   )

Track total and local magnetic moments throughout the restart sequence.
Davidson with a small number of inner iterations is another useful choice
when RMM-DIIS produces large orbital steps or repeated occupation changes.

For an occupation-driven cycle, the Fermi width can be reduced in stages,
for example ``0.10 -> 0.05 -> 0.02 eV``.  Each stage should start from the
preceding checkpoint, and the final width should represent the intended
electronic temperature or converged zero-temperature limit.

Isolated atoms and molecules in GPAW
------------------------------------

Use Gamma-point sampling, an explicit charge and spin, and a cell converged
for the property of interest.  For a gapped, closed-shell system, start from
``Mixer(0.05, 5, 50)`` with Davidson.  An open-shell
or diffuse system may benefit from ``Mixer(0.02, 2, 100)`` or the near-linear
``Mixer(0.02, 1, 50)`` setting.

Fixed total magnetization during a preconvergence stage can stabilize an
intended open-shell state:

.. code-block:: python

   mixer = Mixer(0.02, 1, 50)
   occupations = {
       "name": "fermi-dirac",
       "width": 0.01,
       "fixmagmom": True,
   }

The isolated-system example supplies illustrative energy, density, and
eigenstate criteria.  Cell size, plane-wave cutoff, PAW setup, occupations,
and all electronic criteria require convergence for the target property.

Checks after convergence
------------------------

Record the functional and numerical representation together with the SCF
controls that produced the result.  Inspect occupations and magnetic moments
where relevant, and compare members of an energy difference using consistent
functional composition and accuracy.  A restarted density is an initial
condition; the resulting calculation supplies the energy and state to report.
