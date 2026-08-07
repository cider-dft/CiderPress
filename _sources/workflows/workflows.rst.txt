Model and Descriptor Workflows
==============================

This section describes how CiderPress models are constructed, inspected, and
extended.  A trainable model and its mapped evaluator serve different stages
of that workflow.

A trainable :class:`~ciderpress.models.train.MOLGP` or
:class:`~ciderpress.models.train.MOLGP2` retains the reaction observations,
covariances, noise models, and control-point fitting state.  Mapping converts
that object into :class:`~ciderpress.dft.xc_evaluator.MappedXC` or
:class:`~ciderpress.dft.xc_evaluator2.MappedXC2`, which contains the objects
required for efficient DFT evaluation.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   models
   descriptors
   training
   extending
