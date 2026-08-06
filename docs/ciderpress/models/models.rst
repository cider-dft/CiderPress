.. _models_module:

Regression, Training, and Mapping APIs
======================================

The ``ciderpress.models`` package constructs Gaussian-process density
functionals and maps them to compact inference evaluators.  The training
objects operate on integrated system and reaction observations.

:mod:`ciderpress.models.train` provides ``MOLGP`` and ``MOLGP2`` containers.
The original ``MOLGP`` representation maps to
:class:`~ciderpress.dft.xc_evaluator.MappedXC`; the newer ``MOLGP2``
representation maps to :class:`~ciderpress.dft.xc_evaluator2.MappedXC2`. The
primary difference is that ``MOLGP2`` and ``MappedXC2`` use libxc as a backend
for baseline functionals, making it easier to construct full-XC functionals.
Covariance kernels build on
scikit-learn primitives.  DFT kernels combine those covariance functions with
feature transforms, energy-density baselines, and sparse control points.

Mapping plans under ``ciderpress.models.kernel_plans`` select an inference
evaluator for a trained kernel.  A validated mapped model preserves
predictions, feature derivatives, settings, functional composition, and
correction metadata.  See
:doc:`../../workflows/models` and :doc:`../../workflows/training` before using
the individual APIs below.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   train
   dft_kernel
   kernels
   kernel_tools
