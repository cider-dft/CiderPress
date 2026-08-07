Handling SCF Convergence Issues
===============================

The CIDER interfaces use the SCF algorithms supplied by PySCF and GPAW.  This
page collects settings for calculations that do not converge with the usual
SCF controls.  The runnable examples keep the selected CIDER model and feature
representation fixed while changing the SCF algorithm and initial density.

PySCF restart ladder
--------------------

:source:`examples/pyscf/restart_calc.py` implements a molecular restart
ladder for restricted or unrestricted calculations.  It first evaluates a
PBE density for the same PySCF ``Mole`` object and then tries the following
CIDER controls with both the PBE density and PySCF's atomic initial guess:

.. list-table:: Controls in the PySCF restart example
   :header-rows: 1
   :widths: 23 18 15 15 15 14

   * - Label
     - DIIS method
     - ``diis_space``
     - ``conv_tol``
     - ``level_shift``
     - ``damp``
   * - ``cdiis8``
     - CDIIS
     - 8
     - ``1e-9``
     - 0
     - 0
   * - ``cdiis12``
     - CDIIS
     - 12
     - ``1e-8``
     - 0
     - 0
   * - ``adiis``
     - ADIIS
     - 12
     - ``1e-7``
     - 0
     - 0
   * - ``ediis``
     - EDIIS
     - 12
     - ``1e-7``
     - 0
     - 0
   * - ``cdiis_shift02``
     - CDIIS
     - 8
     - ``1e-7``
     - ``0.2 Ha``
     - 0
   * - ``cdiis_shift05``
     - CDIIS
     - 8
     - ``1e-6``
     - ``0.5 Ha``
     - ``0.3``

When a relaxed rung converges, the example starts one final ``cdiis8``
calculation from its density.  Each attempt writes a separate checkpoint.
The PySCF CIDER26XC integration supports the conventional DIIS methods used
here; Newton/SOSCF is not currently implemented for CIDER functionals.

GPAW reference settings
-----------------------

The GPAW examples use a PBE checkpoint containing wavefunctions
(``write(..., mode="all")``), followed by a CIDER restart.  Saving
wavefunctions is required because the packaged CIDER26XC models use the
meta-GGA kinetic-energy density.

.. list-table:: Settings in the GPAW examples
   :header-rows: 1
   :widths: 25 26 19 30

   * - Calculation
     - Mixer
     - Fermi width
     - Source
   * - Bulk Si
     - ``Mixer(0.05, 5, 50)``
     - ``0.1 eV``
     - :source:`examples/gpaw/production_calc.py`
   * - CO/Pt(111)
     - ``Mixer(0.03, 5, 100)``
     - ``0.1 eV``
     - :source:`examples/gpaw/surface_calc.py`
   * - Isolated He
     - ``Mixer(0.05, 5, 50)``
     - ``0.01 eV``
     - :source:`examples/gpaw/isolated_calc.py`

Two conservative Pulay settings used as fallbacks for difficult periodic,
surface, or isolated calculations are ``Mixer(0.02, 2, 100)`` and
``Mixer(0.02, 1, 50)``.  The latter has a single stored density and therefore
approaches linear mixing.

Bounded RMM-DIIS steps
~~~~~~~~~~~~~~~~~~~~~~

For repeated large orbital steps, GPAW's RMM-DIIS eigensolver accepts an
explicit step interval:

.. code-block:: python

   from gpaw.eigensolvers import RMMDIIS

   eigensolver = RMMDIIS(
       niter=3,
       limit_lambda={"absolute": False, "lower": 0.01, "upper": 0.1},
       trial_step=0.01,
   )

This setting can be paired with ``Mixer(0.02, 1, 50)``.  Davidson with a
small number of inner iterations is the corresponding choice in the
isolated-system example.

Magnetic density mixing
~~~~~~~~~~~~~~~~~~~~~~~

GPAW's ``MixerDif`` assigns separate histories and weights to charge and
magnetization densities.  A conservative magnetic setting is:

.. code-block:: python

   from gpaw import MixerDif

   mixer = MixerDif(
       0.005, 5, 100,
       beta_m=0.005,
       nmaxold_m=3,
       weight_m=100,
   )

For an open-shell isolated system, a fixed-total-magnetization preconvergence
stage can be specified as:

.. code-block:: python

   from gpaw import Mixer

   mixer = Mixer(0.02, 1, 50)
   occupations = {
       "name": "fermi-dirac",
       "width": 0.01,
       "fixmagmom": True,
   }

Occupation-driven cycles can also be restarted through successively smaller
Fermi widths, for example ``0.10``, ``0.05``, and ``0.02 eV``.  The width in
the final calculation defines its electronic temperature.

Restart compatibility
---------------------

A GPAW PBE checkpoint used to initialize CIDER must provide a compatible
cell, PAW setups, plane-wave cutoff, k-points, bands, symmetry, charge, spin,
and occupations.  A CIDER checkpoint additionally stores the mapped model
and its NLDF interpolation parameters.  See :doc:`gpaw` for checkpoint
construction and restart syntax.
