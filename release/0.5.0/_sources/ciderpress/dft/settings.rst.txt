.. _settings_module:

Feature Settings
================

Settings objects are the serialized declarations of a model's raw electronic
features.  Their parameter order, spin convention, scaling powers, and
uniform electron gas limits define the inputs expected by the normalizers and
mapped evaluator.  Calculation interfaces read these objects from the model;
model construction and descriptor workflows can create them explicitly.

Composite settings
------------------

.. py:module:: ciderpress.dft.settings

.. py:class:: FeatureSettings

   Composite feature declaration stored by a trainable or mapped model.  The
   documented model families use ``sl_settings`` together with an optional
   ``nldf_settings`` or ``sdmx_settings`` component.  ``normalizers`` records
   the transformation of the concatenated raw blocks, and ``nfeat`` gives the
   resulting raw feature count.

Semilocal settings
------------------

.. autoclass:: EmptySettings
   :members:

.. autoclass:: SemilocalSettings
   :members:

NLDF settings
-------------

.. autoclass:: NLDFSettings
   :members:

.. autoclass:: NLDFSettingsVI
   :members:

.. autoclass:: NLDFSettingsVJ
   :members:

.. autoclass:: NLDFSettingsVIJ
   :members:

.. autoclass:: NLDFSettingsVK
   :members:

SDMX settings
-------------

.. autoclass:: SDMXSettings
   :members:

.. autoclass:: SDMXGSettings
   :members:

.. autoclass:: SDMX1Settings
   :members:

.. autoclass:: SDMXG1Settings
   :members:

.. autoclass:: SDMXFullSettings
   :members:

Semilocal helper functions
--------------------------

.. autofunction:: get_cider_exponent
.. autofunction:: get_cider_exponent_gga
.. autofunction:: get_uniform_tau
.. autofunction:: get_single_orbital_tau
.. autofunction:: get_s2
.. autofunction:: get_alpha
