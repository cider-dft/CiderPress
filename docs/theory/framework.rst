The CIDER Framework
===================

CIDER :footcide:p:`CIDER22X,CIDER23X,CIDER24X,CIDER26XC`
is a framework for learning a model for the exchange or
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

The model object contains regression coefficients together with
:class:`~ciderpress.dft.settings.FeatureSettings`, feature transforms, and
mapped evaluators.  These objects identify the required electronic
ingredients, define the regression coordinates, and return the energy density
and derivatives needed for self-consistency.

Why nonlocal electronic features?
---------------------------------

A semilocal functional uses the density and a small set of derivatives at one
point.  CIDER augments those quantities with descriptors constructed from the
density or density matrix distribution in the neighborhood around that point.
The descriptors derive from the electronic state and are independent of atom
types, bonds, or a structural graph.  The same functional can therefore be
evaluated for molecules, solids, surfaces, and even model systems like
the uniform electron gas.

The original CIDER representation was designed so that its exchange features
obey simple uniform coordinate scaling rules.  This construction encodes the
exact exchange scaling constraint.  See
:doc:`uniform_scaling` and :doc:`../features/features` for the corresponding
feature definitions.

Packaged functional families
----------------------------

The packaged functional families differ in their descriptors, training
labels, and learned energy contributions:

``CIDER23X``
   Introduced nonlocal density features for efficient molecular and
   plane-wave evaluation, trained exchange from total-energy data, and added
   support for analytical force and stress calculations in molecular
   and periodic systems.  Both semilocal reference
   models and nonlocal models are packaged. :footcite:t:`CIDER23X`

``CIDER24X``
   Introduced smoothed density-matrix exchange (SDMX) descriptors and extended
   Gaussian-process training to orbital-energy labels.  ``CIDER24Xe`` uses
   energy and eigenvalue information; ``CIDER24Xne`` uses energy information.
   These models require PyTorch and are evaluated through PySCF.
   :footcite:t:`CIDER24X`

``CIDER26XC``
   Extends the framework from exchange to the full XC energy.  Separate
   exchange and correlation corrections are learned, full-XC models are
   trained on self-consistent densities, and molecular and combined
   molecular--surface-science variants are provided.  See :doc:`full_xc`.
   :footcite:t:`CIDER26XC`

The :doc:`model guide <../usage/production_models>` lists the feature
requirements, learned energy contribution, external baseline, and supported
backends for each family.

Relationship to the code
------------------------

The core modules mirror the scientific framework:

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
