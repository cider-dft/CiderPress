Energy-Density Baselines
========================

A CIDER kernel combines a learned scalar function with baseline energy-density
models.  For the original ``DFTKernel`` interface, Python callables evaluate

.. math::

   e(\mathbf X)=a(\mathbf X)+m(\mathbf X)f_\mathrm{ML}(\mathbf X)

and return the corresponding derivatives with respect to the raw features.
In the above equation, :math:`a(\mathbf X)` and :math:`m(\mathbf X)` are
additive and multiplicative baselines, respectively, and :math:`f(\mathbf X)`
is the function learned via Gaussian process regression.
``zero_xc`` and ``one_xc`` provide constant additive or multiplicative terms;
the exchange and correlation helpers provide LDA/GGA factors.  These
callables are selected during model construction and serialized into the
mapped kernel.

``DFTKernel2`` stores libxc identifiers for :math:`a` and :math:`m`.
``MappedDFTKernel2`` evaluates their energy, density, gradient, and
kinetic-energy-density derivatives through ``get_libxc_baseline``.  This is
the baseline path used by the separate exchange and correlation kernels in
CIDER26XC.

.. automodule:: ciderpress.dft.baselines
   :members: zero_xc, one_xc, lda_x, nlda_x_damp, gga_x_chachiyo, gga_x_pbe, gga_c_pbe, get_libxc_baseline
