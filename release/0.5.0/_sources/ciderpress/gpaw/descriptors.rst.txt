GPAW Descriptor Interface
=========================

.. py:module:: ciderpress.gpaw.descriptors

The GPAW descriptor interface evaluates semilocal or version-J NLDF features
on the fixed density and orbitals of a completed classic GPAW plane-wave
calculation. It combines uniform-grid values with the matching PAW
all-electron-minus-pseudo contributions and uses the electronic state retained
by the calculator.

.. py:function:: get_descriptors(calc, settings, p_i=None, use_paw=True, screen_dens=True, **kwargs)

   Evaluate one feature-settings component on a converged GPAW calculator.

   ``calc`` must retain its density, wavefunctions, PAW setups, and grid
   distribution. ``settings`` is a semilocal or NLDF settings object, or the
   string ``"l"`` for the raw density/gradient/kinetic-energy-density vector.
   The documented GPAW descriptor settings are semilocal and version-J NLDF.

   With ``p_i=None``, the return value is ``(features, weights)``. ``features``
   has shape ``(nspin, nfeature, npoint)`` and ``weights`` has shape
   ``(npoint,)``. With selected orbitals, the return value is
   ``(features, feature_derivatives, weights)``;
   ``feature_derivatives`` has shape ``(norbital, nfeature, npoint)``.
   Each orbital selector is a zero-based ``(spin, kpoint, band)`` tuple.

   ``use_paw=True`` includes the atomic all-electron correction used by the
   supported PAW calculation path.
   ``screen_dens=True`` removes very-low-density uniform-grid points before
   returning arrays. Keyword arguments such as ``qmax`` and ``lambd`` select
   the same NLDF numerical representation used by a CIDER calculation.

Using settings from a packaged model
------------------------------------

Feature order and normalization belong to the model.  Load its serialized
component settings to preserve the exact feature definition:

.. code-block:: python

   from ciderpress.dft.model_utils import load_cider_model
   from ciderpress.gpaw.descriptors import get_descriptors

   model = load_cider_model("CIDER26XCSURFSCI")

   sl_features, sl_weights = get_descriptors(
       calc,
       model.settings.sl_settings,
       use_paw=True,
   )
   nldf_features, nldf_weights = get_descriptors(
       calc,
       model.settings.nldf_settings,
       use_paw=True,
       qmax=300,
       lambd=1.8,
   )

The two calls describe the same retained electronic state and return
independent point lists.  A workflow that combines them must align the
weights, screening convention, and model feature order.

Occupation derivatives
----------------------

Supplying ``p_i`` evaluates derivatives with respect to the selected orbital
occupations. The routine constructs both the smooth-grid density response and
the PAW atomic density-matrix response, then applies the descriptor
forward/adjoint machinery at fixed orbitals.  The orbital indices,
k-point distribution, spin convention, occupations, and numerical settings
must be recorded with the arrays.

This interface serves inspection and training-data construction.  Its
occupation derivatives hold the orbitals fixed; a self-consistent energy is
obtained through the calculator interface.

See :doc:`../../workflows/descriptors` for the cross-backend workflow and
:doc:`numerical` for the GPAW/PASDW implementation.
