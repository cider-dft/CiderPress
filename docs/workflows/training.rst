Training and Mapping Boundary
=============================

CiderPress contains the model, kernel, feature, covariance, fitting, and
mapping primitives used to train CIDER functionals.  It does not contain a
complete public workflow for constructing reference databases or scheduling
self-consistent training campaigns.

Data expected by the model layer
--------------------------------

A training workflow must provide, for every electronic system used by a
reaction or other observation:

* the density, orbitals, grid, and feature settings;
* reference exchange, XC, total-energy, or orbital-energy information;
* the explicitly evaluated baseline contributions needed to form the target;
* reaction membership and stoichiometric coefficients;
* observation-specific noise assignments; and
* consistent metadata for optional corrections such as D4.

The analyzer and descriptor tools described in :doc:`descriptors` generate
the electronic arrays consumed by this layer.  Dataset schemas and reference
calculation provenance remain the responsibility of the surrounding workflow.

Gaussian-process construction
-----------------------------

A :class:`~ciderpress.models.dft_kernel.DFTKernel` couples a feature transform,
covariance kernel, energy-density baseline, and control-point set.  One or
more kernels are assembled into :class:`~ciderpress.models.train.MOLGP` or
:class:`~ciderpress.models.train.MOLGP2`.

The model accumulates integrated covariances for systems and reactions, forms
the noisy covariance problem, and solves for weights.  The control-point
approximation makes evaluation depend on a compact representation rather than
all training grid points.  Separate kernels can represent exchange and
correlation while sharing reaction-level labels.

Mapping and validation
----------------------

After fitting, mapping plans convert trainable kernels into efficient
evaluators.  A mapped artifact should be validated at four levels:

1. Kernel predictions and feature derivatives agree with the trainable model.
2. Integrated fixed-density energies agree on representative systems.
3. Self-consistent calculations preserve the intended functional composition.
4. Backend energies and analytical derivatives agree within the discretization
   error appropriate to each representation.

For full-XC self-consistent training, regenerate descriptors and residual
labels on the model's own densities, refit with controlled hyperparameters,
and compare fixed-density with self-consistent errors.  The scheduling,
database replacement, and stopping policy for that outer loop are external to
CiderPress.

Use the API documentation for :mod:`ciderpress.models.train`,
:mod:`ciderpress.models.dft_kernel`, and ``ciderpress.models.kernel_plans``
when implementing a supported workflow.
The CIDER papers define the scientific training objectives and should be
treated as the source for family-specific model forms.
