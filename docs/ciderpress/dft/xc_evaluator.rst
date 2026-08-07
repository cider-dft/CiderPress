.. _xc_evaluator_module:

Mapped XC Evaluators
====================

Function evaluators operate on transformed descriptors and return
both a value and its derivatives with respect to the descriptors.  Mapped DFT kernels apply the
stored feature transforms and energy baselines.  The top-level mapped XC
object combines all kernel components with their shared feature settings.

Original mapped interface
-------------------------

.. automodule:: ciderpress.dft.xc_evaluator
   :members: XCEvalSerializable, FuncEvaluator, KernelEvaluator, RBFEvaluator, AntisymRBFEvaluator, SpinRBFEvaluator, SplineSetEvaluator, NNEvaluator, GlobalLinearEvaluator, MappedDFTKernel, ModelWithNormalizer, MappedXC
   :undoc-members:

Libxc-baseline/full-XC interface
--------------------------------

.. automodule:: ciderpress.dft.xc_evaluator2
   :members: MappedDFTKernel2, MappedXC2
