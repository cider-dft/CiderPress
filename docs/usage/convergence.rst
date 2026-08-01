SCF Convergence Recommendations
===============================

There is no single optimal SCF configuration for every system.  The settings
below are conservative starting points and fallback sequences.  Change one
class of setting at a time, save checkpoints for each rung, and verify that a
converged calculation represents the intended charge, spin, occupations, and
electronic basin.

Molecular calculations in PySCF
-------------------------------

For a routine closed-shell molecule, begin with CDIIS, a DIIS space of 8, and
the accuracy required by the final energy difference.  If the calculation is
difficult, first create a baseline density with PBE or PBE0 using identical
numerical settings.

The following ladder is suggested for open-shell, stretched-bond,
transition-metal, or near-degenerate molecular calculations:

.. list-table:: Suggested PySCF fallback ladder
   :header-rows: 1
   :widths: 18 22 15 15 15 15

   * - Rung
     - DIIS method
     - Space
     - ``conv_tol``
     - Level shift
     - Damping
   * - Tight default
     - CDIIS
     - 8
     - ``1e-9``
     - 0
     - 0
   * - Larger subspace
     - CDIIS
     - 12
     - ``1e-8``
     - 0
     - 0
   * - Energy-based mixing
     - ADIIS, then EDIIS
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
   * - Conservative shifted
     - CDIIS
     - 8
     - ``1e-6``
     - ``0.5 Ha``
     - ``0.3``

Treat relaxed tolerances as a way to obtain a stable density.  Restart that
density with the desired final tolerance and, where possible, reduce or remove
the level shift and damping.  If all warm-start rungs fail, repeat the ladder
from a fresh atomic guess; a baseline density can occasionally select the
wrong electronic basin.

Small finite-temperature smearing, approximately ``0.01--0.05 Ha``, can help
when frontier orbitals are genuinely near degenerate.  Use it only after
diagnosing occupation switching, and document whether the reported energy is
the finite-temperature or zero-temperature quantity.  Inspect occupations,
spin, orbital character, and SCF stability after convergence.

Newton/SOSCF is not supported for the production CIDER26XC numerical
integration paths.  Disabling PySCF's convergence checks or pinning a
non-Aufbau state can hide an occupation change and is not a general fallback
recommendation.

Insulating and semiconducting GPAW calculations
-----------------------------------------------

Start from a converged PBE checkpoint with the intended PAW setups, cutoff,
k-points, bands, symmetry, and spin.  A short attempt using inherited settings
is reasonable.  If it fails, preconverge PBE with each new mixer before
starting the corresponding CIDER rung.

.. list-table:: Suggested GPAW periodic fallback ladder
   :header-rows: 1
   :widths: 20 35 20 25

   * - Rung
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
   * - Near-linear
     - ``Mixer(0.02, 1, 50)``
     - ``0.1 eV``
     - Inherited/default
   * - RMM fallback
     - ``Mixer(0.02, 1, 50)``
     - ``0.1 eV``
     - ``rmm-diis``

For persistent orbital-step oscillation, a bounded-step solver is a further
fallback::

   from gpaw.eigensolvers import RMMDIIS

   eigensolver = RMMDIIS(
       niter=3,
       limit_lambda={"absolute": False, "lower": 0.01, "upper": 0.1},
       trial_step=0.01,
   )

Pair it with ``Mixer(0.02, 1, 50)`` and consider a tighter density criterion
such as ``1e-6``.  If no rung is stable, rebuild the starting point from a
fresh PBE calculation rather than repeatedly recycling a poor CIDER density.

Metallic and magnetic GPAW calculations
---------------------------------------

Metallic systems often need occupation smearing and smaller mixing steps.
Magnetic systems can additionally benefit from mixing the total and spin
densities separately::

   from gpaw import MixerDif

   mixer = MixerDif(
       0.005, 5, 100,
       beta_m=0.005,
       nmaxold_m=3,
       weight_m=100,
   )

Use this as a conservative option when the total density and magnetization
converge on different scales.  Monitor total and local magnetic moments.  If
RMM-DIIS produces large orbital steps or occupation reordering, Davidson with
a small number of inner iterations is a reasonable alternative.

For occupation-driven limit cycles, anneal the Fermi width through a sequence
such as ``0.10 -> 0.05 -> 0.02 eV``, writing a checkpoint at every stage.  A
wider value such as ``0.2 eV`` is a last-resort diagnostic, not automatically
an appropriate final electronic temperature.

Isolated atoms and molecules in GPAW
------------------------------------

Use Gamma-point sampling, explicit spin polarization and magnetic moments,
and a box large enough for the property of interest.  For a gapped closed-shell
atom or molecule, the moderate ``Mixer(0.05, 5, 50)`` used in the isolated
example is a reasonable starting point.  Davidson with a small number of inner
iterations is often robust in finite boxes.

For difficult open-shell references, first reduce the mixer to
``Mixer(0.02, 2, 100)``.  If Pulay-history oscillations persist, use the
near-linear fallback::

   mixer = Mixer(0.02, 1, 50)
   occupations = {
       "name": "fermi-dirac",
       "width": 0.01,
       "fixmagmom": True,
   }

For the PBE preconditioning stage, a density criterion around ``1e-2`` is
often sufficient, but tighten it if the CIDER restart begins far from a stable
density.  ``5e-4`` eV/electron for energy, ``1e-4`` for density, and ``5e-3``
eV :sup:`2`/electron for eigenstates are reasonable general CIDER starting
criteria.  Tighten and independently converge them as required for the
reported property.

Then try ``rmm-diis`` or the bounded-step ``RMMDIIS`` above.  Direct
minimization with ``FDPWETDM`` is an expert, GPAW-version-sensitive fallback;
it must use the no-mixing backend and fixed occupations.  Do not combine a
direct-minimization eigensolver with an ordinary density mixer.

Diagnosing the failure mode
---------------------------

* A smoothly decaying density residual that stalls suggests smaller mixing or
  a larger history.
* Alternating energies and occupations suggest smearing or an eigensolver
  change rather than still smaller density mixing.
* Drifting or flipping magnetic moments suggest separate spin mixing, fixed
  total moment during preconvergence, or a different initial magnetic state.
* A stable energy with an oscillating orbital gradient requires inspection of
  occupations and stability before any gradient tolerance is relaxed.
* Large changes after a restart can indicate inconsistent numerical settings
  or convergence into a different electronic basin.

For every final result, confirm the requested energy and density criteria,
physical state, numerical convergence, and consistency of all members of an
energy difference.  A copied earlier checkpoint is a restart candidate, not a
newly converged result.
