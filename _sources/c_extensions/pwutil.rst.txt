Plane-wave Utilities (pwutil)
=============================

``pwutil`` implements the inner loops used to convolve version-j NLDF
quantities on a uniform plane-wave grid and to propagate feature derivatives.
The public calculation interface is
:func:`ciderpress.gpaw.calculator.get_cider_functional`; this header reference
is intended for maintaining the Python/C boundary.

Grid dimensions, interpolation-point count, feature count, spin layout, and
array contiguity are part of that boundary.  When modifying the implementation,
compare the forward and adjoint operations and run the GPAW energy,
finite-difference force, and stress tests.

.. c:type:: fft_plan_t

   Opaque serial FFT-plan handle owned by the internal FFT wrapper.

.. doxygenfile:: nldf_fft_core.h
   :project: CiderPress
