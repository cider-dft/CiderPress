From Exchange Models to Full XC
===============================

Exchange-only CIDER
-------------------

The CIDER23X and CIDER24X files represent an exchange functional.  A common
use is a surrogate-hybrid construction

.. math::

   E_{\mathrm{xc}} = a E_{\mathrm{x}}^{\mathrm{CIDER}}
   + (1-a)E_{\mathrm{x}}^{\mathrm{PBE}}
   + E_{\mathrm{c}}^{\mathrm{PBE}},

with :math:`a=0.25` for the PBE0-surrogate form used in the corresponding
work.  The mixing fraction and external semilocal terms are part of the
functional definition.  They must be supplied explicitly when an
exchange-only model is initialized. :footcite:t:`CIDER23X,CIDER24X`

CIDER26XC functional form
-------------------------

The CIDER26XC models instead return the full exchange-correlation energy.  In
the notation of the CIDER26XC work, the energy density contains an additive
PBE baseline and learned exchange and correlation corrections,

.. math::

   e_{\mathrm{xc}}^{\mathrm{CIDER}} =
   \sum_\sigma e_{\mathrm{x}}^{\mathrm{PBE}}[n_\sigma]
   + e_{\mathrm{c}}^{\mathrm{PBE}}[n_\uparrow,n_\downarrow]
   + \sum_\sigma e_{\mathrm{x}}^{\mathrm{LDA}}[n_\sigma]
     f_{\mathrm{x}}(\mathbf{X}^{\mathrm{x}}_\sigma)
   + e_{\mathrm{c}}^{\mathrm{PBE}}
     f_{\mathrm{c}}\!\left(\tfrac{1}{2}\sum_\sigma
     \mathbf{X}^{\mathrm{c}}_\sigma\right).

Exchange is evaluated separately for the two spin channels.  Correlation uses
a spin-coupled PBE multiplier and the spin-averaged CIDER feature vector.  The
exchange descriptors are scale invariant; correlation additionally receives
density information because correlation does not obey the same homogeneous
scaling rule.

Because the YAML model already contains its additive PBE contributions, no
external exchange or correlation kernel may be added during evaluation.  This
is why the GPAW initializer must be called with ``xmix=1.0``,
``xkernel=None``, and ``ckernel=None`` for CIDER26XC.  CiderPress validates
these arguments and rejects an additional semilocal contribution.  See
:doc:`../usage/production_models` for the required call.

CIDER26XC feature vectors
-------------------------

The exchange feature vector :math:`\mathbf{X}^{\mathrm{x}}_\sigma` contains
the scale-invariant semilocal descriptors and the three version-j nonlocal
density features used by the CIDER23X nonlocal meta-GGA models
(:doc:`../features/sl` and :doc:`../features/nldf`), with one substitution:
the SCAN-style iso-orbital indicator :math:`\alpha` is replaced by the
bounded indicator

.. math:: t = \frac{\tau-\tau_0}{\tau+\tau_0},

where :math:`\tau` is the kinetic-energy density and :math:`\tau_0` its
uniform-electron-gas value as defined in :doc:`../features/sl`.  Like
:math:`\alpha`, :math:`t` is scale invariant and, in combination with the
reduced-gradient descriptor, carries the same iso-orbital information, but it
is bounded and numerically better behaved in the trained models.

The correlation feature vector :math:`\mathbf{X}^{\mathrm{c}}_\sigma` appends
one additional descriptor, the density itself, :math:`X_6 = n`.  The other
features are scale invariant, which enforces the exchange scaling constraint
but is too restrictive for correlation, so the correlation model receives the
density explicitly.  Before entering the Gaussian-process kernel, every
feature is mapped onto a finite interval by the bounded transforms stored in
the serialized model (implemented in :mod:`ciderpress.dft.transform_data`;
the :math:`t` indicator is handled by the ``SLTMap`` transform).

Self-consistent training
------------------------

For an exchange-only target, exact exchange can be evaluated on a chosen
density matrix.  For a reaction-energy-trained full-XC functional, the density
is part of the prediction: once the model is placed in an SCF loop, orbital
relaxation changes both the features and the remaining Kohn--Sham energy
contributions.

CIDER26XC addresses this distribution shift by iterating the training density.
At iteration :math:`k`, descriptors and residual labels are constructed from
the current densities, the model is refit, and self-consistent calculations
with that model generate the next densities.  The process is converged when
fixed-density and self-consistent errors agree at the required resolution.
:footcite:t:`CIDER26XC`

CiderPress provides the descriptor, model, kernel, and mapping primitives used
by this process.  Dataset construction and orchestration of a complete
self-consistent training campaign are not part of the installed package; see
:doc:`../workflows/training` for the supported boundary.

Optional long-range correction
------------------------------

``CIDER26XCCHEMD4`` was trained with a D4 term removed from each target and
therefore adds the same term back when the model is evaluated.  D4 is evaluated
from nuclear geometry after the density SCF.  It does not alter the CIDER
potential or density.  The PySCF interface records the base energy, expected D4
energy, any correction already present on the incoming object, and the final
adjustment so that the correction is included exactly once.

This construction restores an explicit long-range asymptote; it should not be
interpreted as evidence that every noncovalent application requires D4.  The
dispersion-free model can learn nonlocal interaction physics within the range
represented by its electronic descriptors.

.. footbibliography::
