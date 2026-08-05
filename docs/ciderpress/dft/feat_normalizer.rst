Physical Feature Normalization
==============================

Raw NLDF and SDMX quantities carry density and length-scale powers.  Feature
normalizers combine a raw quantity :math:`x` with the spin density and a
dimensionless inhomogeneity variable to produce the coordinates expected by a
model.  ``DensityNormalizer``, ``InhomogeneityNormalizer``, and
``GeneralNormalizer`` implement factors of the form

.. math::

   x_\mathrm n
   = c_1 x\,n^p(1+c_2 I)^q.

Their backward operations propagate a mapped-model derivative to
:math:`x`, :math:`n`, and :math:`I`.  ``get_usp`` reports the uniform-scaling
power of the normalized quantity, and ``get_ueg`` evaluates its
uniform-electron-gas value.

``FeatNormalizerList`` applies one normalizer per raw feature.  Its semilocal
mode determines how density, reduced-gradient, and kinetic-energy-density
inputs form the inhomogeneity variable.  The list and its cutoff are serialized
with the feature settings.

.. automodule:: ciderpress.dft.feat_normalizer
   :members:
