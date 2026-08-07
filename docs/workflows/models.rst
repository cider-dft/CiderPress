Model Objects and Mapping
=========================

Contents of a mapped functional
-------------------------------

A mapped CIDER functional combines the scientific choices needed to evaluate
one energy form:

Feature settings
   :class:`~ciderpress.dft.settings.FeatureSettings` records the ordered
   semilocal, NLDF, and SDMX blocks required by the model.  A backend uses
   these settings to construct its numerical integrator.

Normalizers and transforms
   Physical normalizers convert raw electronic quantities into descriptors
   with the scaling behavior chosen for the functional.  Bounded transforms
   produce the coordinates used by the regression model.

Mapped kernels and baselines
   Each kernel combines a learned function with additive and multiplicative
   energy-density baselines.  It returns the energy contribution and
   derivatives with respect to every raw input.

Evaluation representation
   A mapped kernel can evaluate the Gaussian process predictive mean directly or use a fitted
   spline or neural network representation.  The evaluator type is stored in the
   model file.

Functional metadata
   The top-level object records feature settings and fitted correction
   metadata such as the D4 term associated with ``CIDER26XCCHEMD4``.
   Full-XC models also record the additive libxc baseline used to combine
   their exchange and correlation components.

Loading and inspection
----------------------

Use :func:`ciderpress.dft.model_utils.load_cider_model` with a packaged name
or trusted explicit path:

.. code-block:: python

   from ciderpress.dft.model_utils import load_cider_model

   model = load_cider_model("CIDER26XCSURFSCI")
   print(type(model).__name__)
   print(model.nfeat)
   print(model.settings.sl_settings.level)
   print(model.settings.nldf_settings.version)

The stored settings determine the backend requirements.  A model's short
name and checksum identify the released scientific artifact; the checksum
table is in :doc:`../usage/production_models`.

Trainable and mapped representations
------------------------------------

:class:`~ciderpress.models.dft_kernel.DFTKernel` and
:class:`~ciderpress.models.dft_kernel.DFTKernel2` hold a covariance kernel,
feature transforms, control points, and energy baselines.  A
:class:`~ciderpress.models.train.MOLGP` or
:class:`~ciderpress.models.train.MOLGP2` combines one or more such kernels
with feature settings, integrated system observations, reaction definitions,
and noise assignments.

After fitting, ``MOLGP.map(mapping_plans)`` or
``MOLGP2.map(mapping_plans)`` performs the following conversion:

1. Each mapping plan constructs a
   :class:`~ciderpress.dft.xc_evaluator.FuncEvaluator` for its trained kernel.
2. The evaluator, transforms, mode, and baselines form a mapped DFT kernel.
3. The mapped kernels and shared settings form
   :class:`~ciderpress.dft.xc_evaluator.MappedXC` or
   :class:`~ciderpress.dft.xc_evaluator2.MappedXC2`.
4. Correction metadata is copied to the mapped object before YAML
   serialization.

The packaged families illustrate three evaluator choices.  CIDER23X uses
mapped spline evaluators.  CIDER24X uses a neural evaluator trained to
reproduce its GP.  CIDER26XC stores the sparse control-point prediction in a
radial-basis-function (RBF) evaluator.  Here RBF denotes the
squared-exponential covariance kernel evaluated directly between each feature
vector and the stored GP control points.  The implementation is
:class:`~ciderpress.dft.xc_evaluator.RBFEvaluator`.

Baselines and functional composition
------------------------------------

For the original evaluator interface, ``DFTKernel`` accepts Python baseline
callables from :mod:`ciderpress.dft.baselines`.  Its local energy contribution
has the form

.. math::

   e(\mathbf X)=a(\mathbf X)+m(\mathbf X)f_\mathrm{ML}(\mathbf X).

``DFTKernel2`` identifies additive and multiplicative baselines by libxc
string and evaluates their energy and derivative terms through
``MappedDFTKernel2``.  This interface supplies the separate exchange and
correlation components in CIDER26XC.  See :doc:`../ciderpress/dft/baselines`
for the baseline API and :doc:`../theory/full_xc` for the CIDER26XC energy
form.

Serialization and packaged models
---------------------------------

Joblib files store trainable Python objects and their fitting state.  Mapped
YAML stores the inference object used by the calculation interfaces.  Both
formats reconstruct Python objects and should be loaded from trusted sources.

The package installs the mapped YAML models published with the CIDER23X,
CIDER24X, and CIDER26XC work under ``ciderpress/data/functionals``.
``MANIFEST.in`` includes these files as package data, and
:mod:`ciderpress.dft.model_utils` resolves their short names.
:doc:`../usage/production_models` lists every name with its checksum.  The
CIDER23X and CIDER24X functional sets are also archived on Zenodo and can be
fetched with ``scripts/download_functionals.py``.

See :doc:`training` for the expected training-data boundary and
:doc:`../ciderpress/models/models` for the regression APIs.
