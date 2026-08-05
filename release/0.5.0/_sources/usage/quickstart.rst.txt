First Calculations
==================

The examples below run one molecular and one periodic CIDER calculation.  The
complete :doc:`PySCF <pyscf>` and :doc:`GPAW <gpaw>` guides cover numerical
choices, properties, and restarts.

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

The example raises ``RuntimeError`` when the SCF does not converge.  It prints
the base, expected dispersion, and adjustment terms for every CIDER26XC
model.  With ``CIDER26XCCHEMD4`` they show how the final total was assembled;
``CIDER26XCCHEM`` and ``CIDER26XCSURFSCI`` print zero for the dispersion
entries.

Exchange-only model with PySCF
------------------------------

Packaged CIDER23X and CIDER24X are exchange functionals.  The following
example supplies the PBE0/CIDER surrogate composition:

.. literalinclude:: ../../examples/pyscf/exchange_model.py
   :language: python
   :linenos:

The explicit ``xmix``, ``xkernel``, and ``ckernel`` arguments define the
surrogate-hybrid composition.
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

Continue with :doc:`pyscf` or :doc:`gpaw`; :doc:`convergence` gives concrete
restart settings for difficult SCF calculations.
