Model and Descriptor Workflows
==============================

The calculation guides treat a packaged functional as one object.  This
section opens that object and follows the code paths used to construct,
inspect, and extend it.

The central distinction is between a trainable model and a mapped evaluator.
A trainable :class:`~ciderpress.models.train.MOLGP` or
:class:`~ciderpress.models.train.MOLGP2` retains reaction observations,
covariances, noise models, and control-point fitting state.  Mapping converts
that object into ``MappedXC`` or ``MappedXC2``, which contains only the
objects required for efficient DFT evaluation.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   models
   descriptors
   training
   extending
