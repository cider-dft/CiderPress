CiderPress: Machine-Learned Exchange-Correlation Functionals
=============================================================

CiderPress implements the CIDER framework for constructing and evaluating
machine-learned density functionals.  The code serves several purposes:

* Specify the physically constrained electronic descriptors used as model
  inputs for CIDER functionals, along with the numerical settings for
  evaluating these features.
* Train, evaluate, store, and load Gaussian process regression models
  representing CIDER functionals. Tools are also included to map
  Gaussian processes to more efficient inference-time models.
* Through interfaces to existing DFT backends (PySCF and GPAW), compute
  the electronic input descriptors, evaluate the XC energy and potential,
  and perform full self-consistent field calculations with CIDER functionals.

CIDER stands for *Compressed scale-Invariant DEnsity Representation*.  The
name originally described the scale-invariant density features used
to learn exchange.  The framework now also includes smoothed
density-matrix features, full exchange-correlation models, and molecular and
periodic numerical implementations.

Getting started
---------------

* To run a calculation, begin with :doc:`installation/installation`, then use
  :doc:`usage/production_models` and :doc:`usage/quickstart`.
* To understand the functional forms used in CiderPress, begin with
  :doc:`theory/framework` and :doc:`features/features`.
* To inspect models, generate descriptors, or work on the implementation, use
  :doc:`workflows/workflows` and the API reference.

The calculation guides state which combinations of models, settings, and DFT
backends are supported.  In brief, PySCF is the molecular all-electron backend,
while classic GPAW provides the periodic plane-wave PAW implementation.
The packaged functional families compute different parts of the XC energy.
In particular, the CIDER23X and CIDER24X models compute only the exchange
energy, and therefore they must not be initialized with the full-XC
settings used for CIDER26XC models.

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

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/limitations
   reference/citing
