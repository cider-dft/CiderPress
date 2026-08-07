.. _dftkernel:

DFT Kernels
===========

A DFT kernel binds a differentiable covariance kernel to feature transforms,
control points, spin/component mode, and energy-density baselines.  It
accumulates integrated covariances during fitting and maps the fitted
coefficients to an inference evaluator.

``DFTKernel`` uses Python baseline callables.
``DFTKernel2`` uses libxc baseline identifiers and supplies the full-XC
component interface described in :doc:`../../theory/full_xc`.

.. automodule:: ciderpress.models.dft_kernel
   :members: DFTKernel, DFTKernel2
