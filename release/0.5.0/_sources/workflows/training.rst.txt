Training and Mapping Workflow
=============================

CiderPress supplies feature, covariance, fitting, and mapping primitives for
constructing CIDER models.  A surrounding workflow supplies reference data,
electronic-structure calculations, dataset provenance, and job scheduling.

Electronic data expected by the model layer
--------------------------------------------

For each electronic system, the training layer consumes arrays generated with
one consistent feature definition:

* quadrature weights and spin-resolved semilocal density ingredients;
* normalized descriptor blocks in their serialized order;
* total-energy and XC baseline terms used to form the target;
* exact-exchange values when exchange is an explicit target;
* orbital-occupation derivatives when eigenvalue data are included; and
* optional correction values and metadata associated with the reference.

:class:`~ciderpress.pyscf.analyzers.ElectronAnalyzer` and the descriptor tools
in :doc:`descriptors` provide these electronic arrays.  Their stored geometry,
basis or PAW representation, density-generating functional, grid, spin, and
orbital conventions establish the provenance of a system record.

Integrated observations
-----------------------

CIDER fits integrated energies and their linear combinations.  A reaction
record gives the member identifiers, stoichiometric coefficients, reference
energy, unit conversion, and observation noise.  The model evaluates the
explicit Kohn--Sham and additive baseline contributions on the same systems,
then fits the residual assigned to the learned component.

A :class:`~ciderpress.models.dft_kernel.DFTKernel` couples transformed
descriptors to a differentiable covariance kernel, an energy multiplier, an
additive baseline, and a control-point set.  ``DFTKernel2`` supplies the
libxc-based baseline interface used by full-XC models.  Multiple kernels can
represent separate exchange and correlation terms and contribute to the same
reaction labels.

Sparse Gaussian-process fit
---------------------------

For each kernel, integrated covariances with the control points form the
columns of ``Kmn``.  ``Kmm`` is the covariance matrix among control points.
The system covariance is

.. math::

   \mathbf K_\mathrm{sys}
   = \mathbf K_{mn}^{\mathsf T}\mathbf K_{mm}^{-1}\mathbf K_{mn}.

Independent learned components contribute additive system covariance
matrices.  :meth:`ciderpress.models.train.MOLGP.fit` solves their shared noisy
system problem and converts the resulting system weights into one set of
control-point coefficients per kernel.  The complete predictive-mean
derivation is in :doc:`../theory/gp`.

A typical construction sequence is:

1. Create feature settings, normalizers, transforms, and differentiable
   kernels.
2. Select control points that cover the transformed feature distribution.
3. Store each system's integrated covariance, explicit baseline, and any
   requested occupation derivatives.
4. Assemble reaction or constraint observations with their units and noise.
5. Fit the shared system covariance problem.
6. Map each kernel to its selected inference evaluator and validate the
   mapped object.

Self-consistent full-XC training
--------------------------------

For a full-XC reaction-energy model, orbital relaxation changes the density
and therefore the descriptors, baselines, and residual labels.  An iterative
workflow can regenerate those quantities on densities produced by the current
model and refit the GP.  Comparisons between fixed-density and
self-consistent predictions measure convergence of that outer procedure.

The CIDER26XC work uses this approach together with reaction-specific noise
and uniform-electron-gas constraints. :footcite:p:`CIDER26XC`  Dataset
selection, weighting policy, hyperparameter selection, and stopping criteria
are scientific choices of the training workflow and should be stored with the
resulting model provenance.

Mapping and validation
----------------------

The mapping step creates the inference evaluator described in
:doc:`models`.  Validate pointwise energies and derivatives, integrated
fixed-density values, self-consistent composition, serialized metadata, and
backend derivatives at matching numerical settings.  Model publication also
assigns the stable filename and checksum used to identify the inference
artifact.

The relevant APIs are :mod:`ciderpress.models.train`,
:mod:`ciderpress.models.dft_kernel`,
:mod:`ciderpress.models.kernels`, and
:mod:`ciderpress.models.kernel_plans.kernel_tools`.

.. footbibliography::
