C Extension Libraries
=====================

CiderPress moves the repeated convolution, interpolation, and spline-adjoint
operations used by nonlocal descriptors into compiled libraries.  Python owns
the model settings, backend orchestration, and array layout.  The extension
layer consumes arrays with the shapes and memory conventions established by
that interface.

The plane-wave library supplies FFT-grid NLDF kernels and their derivative
operations.  Other compiled sources support atom-centered convolution and
mapping operations used by PySCF.  Extension-interface tests run through the
Python caller so that spin, feature, and grid axes are checked together with
the C signature.

.. toctree::
   :maxdepth: 2

   pwutil
