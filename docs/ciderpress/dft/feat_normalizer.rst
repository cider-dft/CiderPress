Feature Normalization
=====================

As explained in the :doc:`../../theory/uniform_scaling` section, it is often useful
for model inputs to be invariant under uniform scaling. However, many semilocal
and nonlocal descriptors are not scale-invariant from the outset. To
create scale-invariant features, "raw" features must be multiplied by
an appropriate power of the density, and they can optionally be multiplied
by other scale-invariant factors to improve their numerical behavior.

This is the role of the feature normalizers module.  Feature
normalizers combine a raw quantity :math:`x` with the spin density and a
dimensionless inhomogeneity variable to produce the uniform scaling behavior
expected by a model.  ``DensityNormalizer``, ``InhomogeneityNormalizer``, and
``GeneralNormalizer`` implement factors of the form

.. math::

   x_\mathrm n
   = c_1 x\,n^p(1+c_2 I)^q.

These objects' backward operations propagate the model derivatives to
:math:`x`, :math:`n`, and :math:`I`.  ``get_usp`` reports the uniform scaling
power of the normalizer term, and ``get_ueg`` returns its
value for the uniform electron gas.

``FeatNormalizerList`` applies one normalizer per raw feature.  Its semilocal
mode (``slmode``) determines how density, reduced-gradient, and kinetic-energy-density
inputs form the inhomogeneity variable.  The list and its cutoff are serialized
with the feature settings when the model is stored.

.. autoclass:: ciderpress.dft.feat_normalizer.FeatNormalizerList

.. automodule:: ciderpress.dft.feat_normalizer
   :members:
