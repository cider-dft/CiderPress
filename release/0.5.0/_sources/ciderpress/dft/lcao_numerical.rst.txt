Common LCAO NLDF Infrastructure
===============================

The molecular NLDF implementation is divided between PySCF adapters and a
host-independent linear-combination-of-atomic-orbitals (LCAO) layer.  The
classes on this page define the atom-centered grid indexing, auxiliary
Gaussian convolutions, and grid/basis interpolation used by the PySCF forward
and adjoint paths.

Atom-centered grid indexing
---------------------------

.. py:module:: ciderpress.dft.grids_indexer

.. py:class:: AtomicGridsIndexer

``AtomicGridsIndexer`` relates PySCF's sorted and screened quadrature points
to atom, radial shell, angular point, and spherical-harmonic indices.  Energy,
potential, and grid-response operations share this mapping.

Auxiliary Gaussian convolutions
-------------------------------

.. py:module:: ciderpress.dft.lcao_convolutions

.. py:class:: ATCBasis

.. py:class:: ConvolutionCollection

.. py:class:: ConvolutionCollectionK

``ATCBasis`` stores the atom-centered auxiliary Gaussian basis and its angular
layout.  ``ConvolutionCollection`` builds the matrices that convolve the
source expansion for version-I/J features; ``ConvolutionCollectionK`` supplies
the corresponding version-K operations.

Grid/basis interpolation
------------------------

.. py:module:: ciderpress.dft.lcao_interpolation

.. py:class:: LCAOInterpolator

.. py:class:: LCAOInterpolatorDirect

The interpolators transfer fields between the atom-centered auxiliary basis
and molecular quadrature points.  Their forward and backward methods form the
pair used for NLDF values and potentials.  The direct variant treats the
on-site contribution explicitly and is used by the packaged molecular NLDF
path.

NLDF orchestration
------------------

.. py:module:: ciderpress.dft.lcao_nldf_generator

.. py:class:: LCAONLDFGenerator

``LCAONLDFGenerator`` combines an exponent-interpolation plan, convolution
collection, and LCAO interpolator.  The PySCF-specific
:class:`~ciderpress.pyscf.nldf_convolutions.PyscfNLDFGenerator` constructs
these objects from a molecule and its CIDER grid.

See :doc:`../../theory/nldf_numerical` for the mathematical description and
:doc:`../pyscf/numerical` for the PySCF numerical implementation details.
