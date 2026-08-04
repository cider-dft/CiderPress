First Calculations
==================

This page provides the shortest supported path from an installed environment
to a CIDER result.  The complete backend guides explain numerical choices,
properties, and restarts.

Molecule with PySCF
-------------------

The packaged CIDER26XC molecular model uses the normal PySCF molecular and SCF
objects:

.. literalinclude:: ../../examples/pyscf/production_calc.py
   :language: python
   :linenos:

Run it from the repository root with:

.. code-block:: bash

   python examples/pyscf/production_calc.py --model CIDER26XCCHEM

The calculation must report ``mf.converged`` before its energy is used.  For
``CIDER26XCCHEMD4``, the printed base, expected dispersion, and adjustment
terms explain how the final total was assembled.

Exchange-only model with PySCF
------------------------------

Published CIDER23X and CIDER24X models contain exchange rather than full XC.
The following example uses the PBE0/CIDER surrogate composition:

.. literalinclude:: ../../examples/pyscf/exchange_model.py
   :language: python
   :linenos:

The explicit ``xmix``, ``xkernel``, and ``ckernel`` arguments are essential.
For CIDER24X, install ``ciderpress[cider24]`` and select the corresponding name.

Periodic solid with GPAW
------------------------

The periodic example first writes a full PBE checkpoint and then restarts it
with CIDER26XCSURFSCI:

.. literalinclude:: ../../examples/gpaw/production_calc.py
   :language: python
   :linenos:

Run GPAW examples with the MPI launcher and rank count appropriate for the
installed GPAW build, for example:

.. code-block:: bash

   mpirun -np 4 python examples/gpaw/production_calc.py

Before scaling up, verify a small calculation with the same MPI, FFT, BLAS,
OpenMP, and GPAW setup stack intended for production.  Continue with
:doc:`pyscf` or :doc:`gpaw`, and use :doc:`convergence` if the direct SCF path
does not converge.
