PySCF Numerical and Derivative Implementation
=============================================

This page is an implementation reference for the molecular backend. The
objects documented here support the public calculation and descriptor
interfaces, but are not themselves compatibility-stable user APIs. Code that
only runs calculations should use
:func:`~ciderpress.pyscf.dft.make_cider_calc`; code that extracts fixed-density
data should use :func:`~ciderpress.pyscf.descriptors.get_descriptors`.

Blockwise XC integration
------------------------

.. py:module:: ciderpress.pyscf.numint

:mod:`ciderpress.pyscf.numint` connects PySCF's atom-centered quadrature to
the feature settings stored in a mapped model. For each grid block it:

1. evaluates the density, gradient, and, for meta-GGA models,
   kinetic-energy density from the density matrix;
2. evaluates semilocal, NLDF, SDMX, and any enabled hybrid feature blocks in
   their serialized order;
3. applies the mapped model to obtain an energy density and feature
   derivatives; and
4. applies the adjoint of every feature operation to assemble the matrix XC
   potential returned to PySCF.

The restricted and unrestricted routes use the same feature plans, while the
unrestricted route retains separate alpha- and beta-spin arrays until the
model's spin-combination rule is applied.

.. py:function:: nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None)

   Internal restricted numerical-integration entry point. It returns the
   electron count, XC energy, and matrix potential for one or more density
   matrices.

.. py:function:: nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None)

   Internal spin-polarized counterpart of :func:`nr_rks`.

.. py:class:: CiderNumInt

   PySCF ``NumInt`` implementation that owns the mapped evaluator and the
   feature generators selected from its settings. Specialized subclasses add
   NLDF or fractional-Laplacian behavior; callers should let
   :func:`~ciderpress.pyscf.dft.make_cider_calc` select the appropriate class.

NLDF grid and convolution
-------------------------

.. py:module:: ciderpress.pyscf.gen_cider_grid

:mod:`ciderpress.pyscf.gen_cider_grid` extends a PySCF molecular grid with
radial and spherical-harmonic indexing needed by the nonlocal evaluator. The
indexer records how PySCF's sorted quadrature points map to atom-centered
radial/angular shells; feature generation and its adjoint must use the same
mapping.

.. py:class:: CiderGrids

   Atom-centered integration grid carrying the CIDER radial/angular indexer.
   It is constructed by the decorated calculation when its model requires
   NLDFs.

.. py:module:: ciderpress.pyscf.nldf_convolutions

:mod:`ciderpress.pyscf.nldf_convolutions` builds the auxiliary Gaussian
representation of the density-dependent kernel and evaluates its forward and
backward contractions. The exponent grid, angular cutoff, interpolation
scheme, and low-density cutoffs are numerical approximations to the feature
definition and must remain identical between energy and potential paths.

.. py:class:: PySCFNLDFInitializer

   Stores the serialized :class:`~ciderpress.dft.settings.NLDFSettings` and
   numerical options until the molecule and CIDER grid are available.

.. py:class:: PyscfNLDFGenerator

   Molecular wrapper around the common LCAO NLDF generator. Its forward
   operation produces grid features; its backward operation returns the
   corresponding density and exponent derivatives.

SDMX feature evaluation
-----------------------

.. py:module:: ciderpress.pyscf.sdmx

:mod:`ciderpress.pyscf.sdmx` evaluates the optimized smoothed density-matrix
features used by CIDER24X. Rather than depending only on local density
ingredients, these features contract the one-particle density matrix with
smoothed atom-centered orbital quantities.

.. py:class:: PySCFSDMXInitializer

   Defers construction of the SDMX generator until the molecule and spin
   layout are known.

.. py:class:: EXXSphGenerator

   Evaluates SDMX features and applies their adjoint contribution to the
   density-matrix potential.

.. py:module:: ciderpress.pyscf.sdmx_slow

:mod:`ciderpress.pyscf.sdmx_slow` is the slower reference formulation used to
check the optimized contractions. It is useful for implementation validation,
not production calculation setup. Analytical SDMX nuclear gradients are not a
supported release-0.5.0 property.

Nuclear-gradient response
-------------------------

.. py:module:: ciderpress.pyscf.rks_grad

:mod:`ciderpress.pyscf.rks_grad` supplies restricted analytical nuclear
gradients. In addition to the usual AO and quadrature response, an NLDF
gradient includes motion of the atom-centered CIDER grid, auxiliary-basis
response, interpolation response, and the adjoint nonlocal potential.

.. py:class:: Gradients

   Restricted CIDER gradient implementation selected by the decorated SCF
   object.

.. py:class:: DFGradients

   Restricted gradient implementation including the response terms required
   by a density-fitted calculation.

.. py:module:: ciderpress.pyscf.uks_grad

:mod:`ciderpress.pyscf.uks_grad` carries the same response terms for separate
alpha and beta densities.

.. py:class:: Gradients
   :no-index:

   Unrestricted CIDER gradient implementation.

.. py:class:: DFGradients
   :no-index:

   Unrestricted density-fitted CIDER gradient implementation.

Consistency requirements
------------------------

Finite-difference comparisons must use the same basis, atom-centered and
CIDER grids, feature settings, density-fitting treatment, occupations, and
electronic state as the analytical calculation. Agreement of total energies
alone does not validate a new feature: its raw values, mapped derivatives,
matrix potential, and nuclear response must agree through the complete
forward/adjoint path.

See :doc:`../../theory/nldf_numerical` for the molecular NLDF algorithm,
:doc:`../../theory/numerical_evaluation` for the backend comparison, and
:doc:`../../workflows/extending` for the extension contract.
