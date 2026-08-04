Core DFT Representation
=======================

The ``ciderpress.dft`` package defines the backend-independent contract
between an electronic feature representation and a mapped XC model.  A
backend reads the stored settings, computes the requested raw features, calls
the evaluator, and propagates the returned feature derivatives back to its
density or density matrix.  Start with :doc:`../../theory/framework` for the
scientific data flow and use the pages below for individual objects.

* The :mod:`ciderpress.dft.settings` module consists of classes
  for specifying the types of features to be computed
  for an ML model along with the parametrizations of those features.
* The :mod:`ciderpress.dft.plans` module provides classes
  that specify *how* a given set of features is to be computed.
  For example, an instance of
  :class:`~ciderpress.dft.settings.NLDFSettingsVJ` specifies that version-j
  :ref:`NLDF <nldf_feat>` features are required, and an instance of
  :class:`~ciderpress.dft.plans.NLDFSplinePlan` specifies their spline-based
  evaluation (see
  :ref:`NLDF Numerical Evaluation <nldf_numerical>`).
* The :mod:`ciderpress.dft.feat_normalizer` module provides
  utilities to transform "raw" features (which might not be scale-invariant)
  to scale-invariant "normalized features". Note it is not necessary to make
  every feature scale-invariant unless you want to enforce the uniform
  scaling rule for exchange.
* The :mod:`ciderpress.dft.transform_data` module provides
  utilities to transform "normalized" features (which do not necessarily fall
  in a finite interval, making them unwieldy for ML models) into
  "transformed" features suitable for direct input into Gaussian process
  regression.
* The :mod:`ciderpress.dft.xc_evaluator` and
  :mod:`ciderpress.dft.xc_evaluator2`
  modules, which provide tools to efficiently evaluate trained CIDER models.
* The :mod:`ciderpress.dft.model_utils` module resolves packaged model names
  and trusted YAML/joblib paths.

Settings are serialized scientific state.  Feature order, normalization,
spin treatment, energy baselines, and the distinction between exchange-only
and full-XC evaluators must be preserved when a model is mapped or extended.
See :doc:`../../workflows/extending` before changing these interfaces.

.. toctree::
   :maxdepth: 1

   settings
   plans
   feat_normalizer
   transform_data
   xc_evaluator
   model_utils
