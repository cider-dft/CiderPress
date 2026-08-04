The CIDER Framework
===================

CIDER is a framework for learning a grid-resolved exchange or
exchange-correlation energy density from electronic structure.  CiderPress is
the software implementation of that framework.  It supplies feature
definitions, regression and mapping tools, numerical algorithms, and the
interfaces that insert a mapped model into a Kohn--Sham calculation.

From density to self-consistent functional
------------------------------------------

At each integration point, a backend supplies the density and the ingredients
required by the model, such as its gradient, kinetic-energy density, nonlocal
density convolutions, or smoothed density-matrix quantities.  CiderPress then
applies the following sequence:

.. code-block:: text

   density or density matrix
       -> raw semilocal/nonlocal features
       -> physical normalization
       -> bounded feature transforms
       -> mapped regression model
       -> XC energy density and feature derivatives
       -> XC potential, forces, and stress

The same model object therefore contains more than regression coefficients.
Its :class:`~ciderpress.dft.settings.FeatureSettings` identify the required
electronic ingredients, its feature transforms define the coordinates seen by
the regression, and its mapped evaluators return both the energy density and
the derivatives needed for self-consistency.

Why nonlocal electronic features?
---------------------------------

A semilocal functional sees only the density and a small set of derivatives at
one point.  CIDER augments those quantities with descriptors constructed from
the density or density matrix over a finite neighborhood.  These descriptors
are geometry independent: they are defined from the electronic state rather
than atom types, bonds, or a graph of the structure.  The same functional can
therefore be evaluated for molecules, solids, and surfaces.

The original CIDER representation was designed so that its exchange features
obey simple uniform-coordinate-scaling rules.  This allows an exact exchange
constraint to be enforced by construction instead of learned from data.  See
:doc:`uniform_scaling` and :doc:`../features/features` for the corresponding
feature definitions.

Published functional families
-----------------------------

The published CIDER developments use the same broad framework but expose
different information and learn different quantities:

``CIDER22X``
   Introduced nonlocal, scale-invariant density features and learned exchange
   energy densities under exact constraints.  This is the conceptual origin
   of the framework; CiderPress 0.5.0 does not ship a CIDER22X inference
   file. :footcite:t:`CIDER22X`

``CIDER23X``
   Redesigned the nonlocal density features for efficient molecular and
   plane-wave evaluation, trained exchange from total-energy data, and added
   analytical molecular and periodic derivatives.  Both semilocal reference
   models and nonlocal models are packaged. :footcite:t:`CIDER23X`

``CIDER24X``
   Introduced smoothed density-matrix exchange (SDMX) descriptors and extended
   Gaussian-process training to orbital-energy labels.  ``CIDER24Xe`` includes
   eigenvalue information; ``CIDER24Xne`` does not.  These models require
   PyTorch and are evaluated through PySCF. :footcite:t:`CIDER24X`

``CIDER26XC``
   Extends the framework from exchange to the full XC energy.  Separate
   exchange and correlation corrections are learned, full-XC models are
   trained on self-consistent densities, and molecular and combined
   molecular--surface-science variants are provided.  See :doc:`full_xc`.
   :footcite:t:`CIDER26XC`

The model families are not interchangeable names for the same functional.
Their feature requirements, learned energy contribution, external baseline,
and supported backends differ.  The executable selection matrix is given in
:doc:`../usage/production_models`.

Relationship to the code
------------------------

The core modules mirror the scientific decomposition:

* :mod:`ciderpress.dft.settings` declares which features a model needs.
* :mod:`ciderpress.dft.plans` turns those declarations into numerical plans.
* :mod:`ciderpress.dft.feat_normalizer` and
  :mod:`ciderpress.dft.transform_data` prepare regression inputs.
* ``ciderpress.models`` constructs Gaussian-process models and maps them to
  efficient evaluators.
* :mod:`ciderpress.dft.xc_evaluator` and
  :mod:`ciderpress.dft.xc_evaluator2` evaluate mapped models.
* ``ciderpress.pyscf`` and ``ciderpress.gpaw`` provide densities and
  propagate model derivatives into their host codes.

The steps required to connect a new feature to these layers are described in
:doc:`../workflows/extending`.

.. footbibliography::
