Differentiable Covariance Kernels
=================================

These kernels extend scikit-learn covariance objects with derivatives with
respect to descriptor coordinates.  DFT kernels use those derivatives to
construct XC potentials and occupation-derivative observations.  Subset and
spin-symmetry wrappers apply a base kernel to selected coordinates and
preserve the corresponding derivative layout.

.. automodule:: ciderpress.models.kernels
   :members: DiffKernelMixin, DiffTransform, DiffWhiteKernel, DiffConstantKernel, DiffRBF, DiffAntisymRBF, DiffLinearKernel, DiffPolyKernel, PartialRBF, DiffARBF, DiffARBFV2, DiffAddLLRBF, DiffAddRQ, PartialARBF, SpinSymRBF, SpinSymARBF, SpinSymPoly, DensityNoise, ExponentialDensityNoise, FittedDensityNoise
