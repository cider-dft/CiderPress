Core DFT Representation
=======================

The ``ciderpress.dft`` package defines the backend-independent contract
between an electronic feature representation and a mapped XC model.  A DFT
backend (PySCF or GPAW) reads the stored settings, computes the requested
raw features, calls the evaluator, and propagates the returned feature
derivatives into the Kohn-Sham effective potential.
:doc:`../../theory/framework` introduces the scientific
data flow; the pages below document the corresponding objects.

Settings and numerical plans
----------------------------

:mod:`ciderpress.dft.settings` declares the semilocal, NLDF, and SDMX
features and their parameters.  An instance of the :class:`~ciderpress.dft.settings.FeatureSettings`
object defines the set of descriptors to compute for a given ML model.
:mod:`ciderpress.dft.plans` implements the backend-independent parts
of the forward (feature evaluation) and adjoint (XC potential evaluation) operations.
For example, :class:`~ciderpress.dft.settings.NLDFSettingsVJ` defines version-j NLDF
features, and :class:`~ciderpress.dft.plans.NLDFSplinePlan` defines one
interpolation representation for them.  The numerical algorithm is described
in :ref:`nldf_numerical`.

Normalization and Transformation of Features
--------------------------------------------

:mod:`ciderpress.dft.feat_normalizer` converts raw features into
descriptors with the uniform scaling behavior chosen for the functional.
The most common use of the feature normalizers is to transform ``raw`` features
(which might not be scale-invariant) into scale-invariant ``normalized``
features, though it is not necessary to make every feature scale-invariant unless
you want to enforce the uniform scaling rule for exchange. See
:doc:`../../theory/uniform_scaling` for details. Models for the correlation
energy typically retain at least one feature that is not scale-invariant,
usually the density.

Even after normalization, features might remain unbounded, i.e. they can take
arbitrarily large values. :mod:`ciderpress.dft.transform_data` applies bounded
maps to the input features before passing them to the Gaussian process regression
model, thereby guaranteeing that the final features used in regression fall
inside finite intervals that are convenient for machine learning.

Model Baselines and Model Evaluation
------------------------------------

:mod:`ciderpress.dft.baselines` provides additive and multiplicative
energy density baselines for model construction.  Baselines are used
because it is typically easier for a model to learn the difference
between the training data and an existing functional than to learn
the full functional from scratch. Details on how the baselines
are used can be found in :doc:`../../workflows/training`.

The :mod:`ciderpress.dft.xc_evaluator` and :mod:`ciderpress.dft.xc_evaluator2`
modules evaluate the model baselines and the trained ML model,
and they return the predicted XC energy density
along with the derivatives with respect to input features.
The role of both XC evaluator modules is to wrap trained models into
computationally efficient, black-box interfaces for evaluating the
functionals within self-consistent DFT calculations. Depending on how
the model was mapped after training, the ML part is evaluated one of
three ways by the XC evaluator:

* Via a cubic spline interpolation over the possible values of the features,
  with spline coefficients calculated after training and stored in the model.
* Via a neural network trained to reproduce the Gaussian process and stored
  in the model.
* Directly via the Gaussian process predictive mean formula, with
  kernel weights precomputed for efficiency and parallelized within the C backend.
  All CIDER models use a set of sparse control points to accelerate training
  and model evaluation, making this direct approach performant.

CIDER23X models use the cubic spline approach, CIDER24X models use the neural
network approach, and CIDER26XC models use the direct evaluation approach.

Finally, :mod:`ciderpress.dft.model_utils` resolves packaged model names and trusted
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
