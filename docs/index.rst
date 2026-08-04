CiderPress: Machine-Learned Exchange-Correlation Functionals
=============================================================

CiderPress implements the CIDER framework for constructing and evaluating
machine-learned density functionals.  It connects physically constrained
electronic descriptors and mapped Gaussian-process models to self-consistent
calculations in PySCF and GPAW.

CIDER stands for *Compressed scale-Invariant DEnsity Representation*.  The
name originally described the scale-invariant nonlocal density features used
to learn exchange.  The framework now also includes semilocal and smoothed
density-matrix features, full exchange-correlation models, and molecular and
periodic numerical implementations.

Choose a path
-------------

* To run a calculation, begin with :doc:`installation/installation`, then use
  :doc:`usage/production_models` and :doc:`usage/quickstart`.
* To understand the functional, begin with :doc:`theory/framework` and
  :doc:`features/features`.
* To inspect models, generate descriptors, or work on the implementation, use
  :doc:`workflows/workflows` and the API reference.

The calculation guides state which combinations are supported.  In brief,
PySCF is the molecular all-electron backend, while classic GPAW provides the
periodic plane-wave PAW implementation.  The packaged functional families
have different energy forms; in particular, the exchange-only CIDER23X and
CIDER24X models must not be initialized with the full-XC settings used for
CIDER26XC.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation/installation
   usage/production_models
   usage/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Running calculations

   usage/pyscf
   usage/gpaw
   usage/properties
   usage/convergence
   usage/reproducibility

.. toctree::
   :maxdepth: 2
   :caption: The CIDER framework

   theory/theory
   features/features

.. toctree::
   :maxdepth: 2
   :caption: Model and descriptor workflows

   workflows/workflows

.. toctree::
   :maxdepth: 2
   :caption: API and implementation reference

   ciderpress/dft/dft
   ciderpress/models/models
   ciderpress/pyscf/pyscf
   ciderpress/gpaw/gpaw
   c_extensions/c_extensions
   reference/limitations
   reference/citing
