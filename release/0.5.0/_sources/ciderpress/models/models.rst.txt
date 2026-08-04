.. _models_module:

Regression, Training, and Mapping APIs
======================================

The ``ciderpress.models`` package constructs Gaussian-process density
functionals and maps them to compact inference evaluators.  The training
objects operate on integrated system and reaction observations rather than
ordinary pointwise scalar targets.

:mod:`ciderpress.models.train` provides ``MOLGP`` and ``MOLGP2`` containers.
The original representation maps to ``MappedXC``; the multi-component
representation maps to ``MappedXC2`` and supports the separate exchange and
correlation kernels used by full-XC models.  Covariance kernels build on
scikit-learn primitives, while DFT kernels combine them with feature
transforms, energy-density baselines, and sparse control points.

Mapping plans under ``ciderpress.models.kernel_plans`` select an efficient
evaluator for a trained kernel.  Mapping is part of model publication: the
mapped model must preserve predictions, feature derivatives, settings,
functional composition, and correction metadata.  See
:doc:`../../workflows/models` and :doc:`../../workflows/training` before using
the individual APIs below.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   train
   dft_kernel
   kernels
   kernel_tools
