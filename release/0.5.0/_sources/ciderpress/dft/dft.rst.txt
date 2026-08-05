Core DFT Representation
=======================

The ``ciderpress.dft`` package defines the backend-independent contract
between an electronic feature representation and a mapped XC model.  A
backend reads the stored settings, computes the requested raw features, calls
the evaluator, and propagates the returned feature derivatives to its density
or density matrix.  :doc:`../../theory/framework` introduces the scientific
data flow; the pages below document the corresponding objects.

Settings and numerical plans
----------------------------

:mod:`ciderpress.dft.settings` declares the ordered semilocal, NLDF, and SDMX
features and their parameters.  :mod:`ciderpress.dft.plans` implements their
backend-neutral forward and adjoint operations.  For example,
:class:`~ciderpress.dft.settings.NLDFSettingsVJ` defines version-j NLDF
features, and :class:`~ciderpress.dft.plans.NLDFSplinePlan` defines one
interpolation representation for them.  The numerical algorithm is described
in :ref:`nldf_numerical`.

Regression coordinates
----------------------

:mod:`ciderpress.dft.feat_normalizer` converts raw electronic quantities into
descriptors with the scaling behavior chosen for the functional.
:mod:`ciderpress.dft.transform_data` applies the bounded maps used by the
regression.  Scale-invariant coordinates support the exchange scaling
construction in :doc:`../../theory/uniform_scaling`; full-XC correlation
models can retain explicit density dependence.

Energy composition and evaluation
---------------------------------

:mod:`ciderpress.dft.baselines` provides additive and multiplicative
energy-density factors for model construction.  The
:mod:`ciderpress.dft.xc_evaluator` and
:mod:`ciderpress.dft.xc_evaluator2` modules combine those factors with mapped
spline, neural, or direct RBF predictions and return feature derivatives.
:mod:`ciderpress.dft.model_utils` resolves packaged model names and trusted
YAML/joblib paths.

Settings, feature order, normalization, spin treatment, energy baselines, and
evaluator type form part of the functional definition.  Preserve them together
when mapping or extending a model.  The extension contract is described in
:doc:`../../workflows/extending`.

.. toctree::
   :maxdepth: 1

   settings
   plans
   lcao_numerical
   feat_normalizer
   transform_data
   baselines
   xc_evaluator
   model_utils
