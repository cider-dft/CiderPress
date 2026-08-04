C Extension Libraries
=====================

CiderPress moves the repeated convolution, interpolation, and spline-adjoint
operations used by nonlocal descriptors into compiled libraries.  Python owns
the model settings, backend orchestration, and array layout; the extension
layer assumes those array shapes and memory conventions have already been
validated.

The plane-wave library supplies FFT-grid NLDF kernels and their derivative
operations.  Other compiled sources support atom-centered convolution and
mapping operations used by PySCF.  A change to an extension interface must be
tested through its Python caller because matching a C signature alone does
not establish that spin, feature, or grid axes have the intended meaning.

.. toctree::
   :maxdepth: 2

   pwutil
