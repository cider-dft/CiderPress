Model Objects and Mapping
=========================

Model contents
--------------

A mapped CIDER functional contains four coupled layers:

``FeatureSettings``
   Declares semilocal, NLDF, SDMX, nonlocal-orbital, and other electronic
   ingredients.  Backends inspect these settings to choose their numerical
   integrator.

Normalizers and transforms
   Convert raw density-dependent quantities into physically normalized and
   bounded regression coordinates.  Normalization can carry exact scaling
   behavior; bounded transforms improve regression and extrapolation behavior.

Mapped kernels
   Evaluate the learned enhancement or correction from control points and
   return derivatives with respect to every transformed input.

Energy baselines and contracts
   Define additive and multiplicative energy-density factors, optional libxc
   contributions, and metadata such as a fitted D4 correction.

Loading and inspection
----------------------

Use :func:`ciderpress.dft.model_utils.load_cider_model` for a packaged name or
trusted explicit path:

.. code-block:: python

   from ciderpress.dft.model_utils import load_cider_model

   model = load_cider_model("CIDER26XCSURFSCI")
   print(type(model).__name__)
   print(model.nfeat)
   print(model.settings.sl_settings.level)
   print(model.settings.nldf_settings.version)

The backend is selected from the stored settings, not from the filename.  A
filename is nevertheless part of the public scientific identity and should
not be repurposed for a different model.

Trainable and mapped representations
------------------------------------

The :mod:`ciderpress.models.train` classes assemble covariances between
integrated electronic systems and the model's control points.  After fitting,
``map(mapping_plans)`` converts each trainable kernel into a mapped kernel.
Mapping plans choose an evaluator representation appropriate to the kernel;
the resulting object can be serialized and evaluated without the training
database.

``MOLGP`` and :class:`~ciderpress.dft.xc_evaluator.MappedXC` provide the
original single-output evaluation interface.  ``MOLGP2`` and
:class:`~ciderpress.dft.xc_evaluator2.MappedXC2` support separate exchange
and correlation components and the additional density inputs required by the
full-XC form.  CIDER26XC uses the latter representation.

Model files
-----------

Joblib files normally store a trainable Python object, including training and
mapping state.  Mapped YAML stores an inference object and is the preferred
artifact for calculations and release packaging.  Model-specific metadata,
including the D4 fit/evaluation behavior, must survive the joblib-to-YAML
mapping step.

Both formats reconstruct Python objects and are unsafe when obtained from an
untrusted source.  A release artifact should have a stable filename and
checksum and should be tested through the same backend entry points users will
call.

See :doc:`training` for the data boundary and the API reference under
:doc:`../ciderpress/models/models` for individual classes.
