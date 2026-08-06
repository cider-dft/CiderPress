Choosing a CIDER Functional
===========================

CiderPress 0.5.0 packages CIDER23X, CIDER24X, and CIDER26XC model files. Pass
the model's short name to the PySCF or GPAW constructor. The families learn
different parts of the functional, and the learned quantity determines the
initializer arguments described below.

Family selection
----------------

.. list-table:: Packaged functional families
   :header-rows: 1
   :widths: 15 23 20 22 20

   * - Family
     - Learned quantity
     - Electronic features
     - Main use
     - Backend
   * - CIDER23X
     - Exchange
     - Semilocal or NLDF
     - Surrogate-hybrid molecular and solid calculations
     - PySCF and classic GPAW
   * - CIDER24X
     - Exchange
     - Semilocal and SDMX
     - Molecular energetics and orbital-energy-sensitive applications
     - PySCF; PyTorch required
   * - CIDER26XC
     - Full exchange-correlation
     - Semilocal and NLDF
     - Molecular chemistry, solids, surfaces, and adsorption
     - PySCF and classic GPAW, subject to the model-specific restrictions below

CIDER23X exchange models
------------------------

All CIDER23X files learn exchange and are normally evaluated in the
PBE0/CIDER surrogate-hybrid form.  The semilocal models are useful references
for separating the value of nonlocality from the regression method; the
nonlocal meta-GGA variants are the main application models from that work.

.. list-table:: CIDER23X model files
   :header-rows: 1
   :widths: 30 22 48

   * - Name
     - Feature level
     - Distinguishing role
   * - ``CIDER23X_SL_GGA``
     - Semilocal GGA
     - Learned semilocal exchange reference using the reduced gradient
   * - ``CIDER23X_SL_MGGA``
     - Semilocal meta-GGA
     - Adds kinetic-energy-density information
   * - ``CIDER23X_NL_GGA``
     - Nonlocal GGA
     - Adds three version-j NLDF descriptors
   * - ``CIDER23X_NL_MGGA``
     - Nonlocal meta-GGA
     - Combines meta-GGA and version-j NLDF descriptors
   * - ``CIDER23X_NL_MGGA_PBE``
     - Nonlocal meta-GGA
     - Uses a PBE exchange baseline in the learned exchange model
   * - ``CIDER23X_NL_MGGA_DTR``
     - Nonlocal meta-GGA
     - The PBE-baseline model retrained on the diverse-training (DTR) partition of GMTKN55 for improved numerical stability

Initialize one of these models with the exchange fraction and external
semilocal terms stated explicitly:

.. code-block:: python

   mf = make_cider_calc(
       dft.RKS(mol),
       "CIDER23X_NL_MGGA_DTR",
       xmix=0.25,
       xkernel="GGA_X_PBE",
       ckernel="GGA_C_PBE",
   )

The corresponding GPAW construction uses the same three composition
arguments.  Changing ``xmix`` defines a different surrogate-hybrid
functional.

CIDER24X SDMX exchange models
-----------------------------

The CIDER24X models use smoothed density-matrix exchange descriptors and a
mapped neural evaluator.  Install their optional runtime before loading them:

.. code-block:: bash

   pip install 'ciderpress[cider24]'

For systems requiring a platform-specific CPU or CUDA build, install PyTorch
using its platform instructions before installing CiderPress.

``CIDER24Xne``
   Trained on energy labels.

``CIDER24Xe``
   Trained on total-energy and orbital-energy labels.

Both are exchange models.  Use the explicit PBE0/CIDER composition shown for
CIDER23X.  They are supported through the molecular PySCF interface.  The
periodic PySCF SDMX implementation supports methodological reproduction of
the CIDER24X work.  Packaged periodic NLDF calculations use GPAW.

CIDER26XC full-XC models
------------------------

.. list-table:: CIDER26XC model files
   :header-rows: 1
   :widths: 28 30 22 20

   * - Name
     - Intended domain
     - Dispersion
     - Recommended backend
   * - ``CIDER26XCCHEM``
     - Molecular chemistry
     - Electronic model
     - PySCF
   * - ``CIDER26XCCHEMD4``
     - Molecular chemistry where an explicit D4 asymptote is desired
     - Post-density D4
     - PySCF only
   * - ``CIDER26XCSURFSCI``
     - Molecules, solids, surfaces, and adsorption
     - Electronic model
     - PySCF and classic GPAW

These files already contain the full XC energy form and feature vectors
described in :doc:`../theory/full_xc`.  PySCF's defaults are correct:

.. code-block:: python

   mf = make_cider_calc(dft.RKS(mol), "CIDER26XCCHEM")

GPAW's initializer serves both exchange and full-XC models.  Its defaults
are ``xmix=1.0`` with the PBE kernel strings ``"GGA_X_PBE"`` and
``"GGA_C_PBE"``; leaving ``ckernel`` set would add a second PBE correlation
term to the full XC already stored in CIDER26XC.  Supply the full-XC
composition explicitly:

.. code-block:: python

   xc = get_cider_functional(
       "CIDER26XCSURFSCI",
       xmix=1.0,
       xkernel=None,
       ckernel=None,
   )

``get_cider_functional`` validates these arguments against the stored model
composition.  Pass all three explicitly for GPAW full-XC models.

PySCF is the intended molecular representation for ``CIDER26XCCHEM``; the
electronic model can also be evaluated in an isolated GPAW PAW box.
``CIDER26XCCHEMD4`` uses the PySCF D4 energy interface, and selecting it in
GPAW raises an error.

D4 with ``CIDER26XCCHEMD4``
---------------------------

D4 is evaluated from the geometry after the density SCF, so it changes the
total energy but not the CIDER potential or density.  CiderPress adds the
term the model was trained with exactly once, whether or not a dispersion
wrapper is already attached to the SCF object.  The runtime attributes that
record this are documented in :doc:`pyscf`.

The D4 term is absent from the current nuclear gradient; see
:doc:`properties` before using derivative-based workflows.

Loading rules and model trust
-----------------------------

All packaged names accept an optional ``.yaml`` suffix.  Explicit paths to
other YAML or joblib models remain supported, and a real file at the supplied
path takes precedence over a packaged name.

Mapped CIDER YAML and joblib files reconstruct Python model objects and
therefore require a trusted source.  Packaged models have the fixed checksums
listed below.  The checksum provides an exact identifier for an external
model artifact.

Checksums
---------

.. list-table:: SHA-256 of packaged models
   :header-rows: 1
   :widths: 34 66

   * - Model
     - SHA-256
   * - ``CIDER23X_SL_GGA``
     - ``2e518727b836cd806c4f27c5566e9401b8c15db77182e34380728efdea24f0ce``
   * - ``CIDER23X_SL_MGGA``
     - ``fdd60d58ae0f981dcdf64ec24437454bdc334e65b162e379e3630bb99dc82bba``
   * - ``CIDER23X_NL_GGA``
     - ``b4e29d9530eaa7c94a3c61cc5e942a0cf7cfd0dd3c4a265dc35395123ae26c86``
   * - ``CIDER23X_NL_MGGA``
     - ``f49060978575ffeb8b18c64df8b9eb917fddf2e0cb8b4cc120e4a11f8a01a537``
   * - ``CIDER23X_NL_MGGA_PBE``
     - ``9ce3303986f80aee859c00f8973c7947f02bf3837dfc356c57d7d82fa36660f4``
   * - ``CIDER23X_NL_MGGA_DTR``
     - ``93312ddde97a8b8e88a9f875df221a44f65a1a1b7c955a55d9214bd449e363ea``
   * - ``CIDER24Xne``
     - ``ab76620ad504ae2934f2f1d599c9c5457c3b882315cbbca710358d641b3e505a``
   * - ``CIDER24Xe``
     - ``f2650a9416ca13e0967d932e3b8d43e232fe87bd7bd2b27eb46fe6c82fd3aae1``
   * - ``CIDER26XCCHEM``
     - ``fd2e0b5cd7408cd4b0ff09495bb026b109b5066bc2f1267c011ce5f0408bdf1d``
   * - ``CIDER26XCCHEMD4``
     - ``ee6824e258625246180efb75fdec7c1e23310a6fe2fcdf2ab836ec90d29bb00c``
   * - ``CIDER26XCSURFSCI``
     - ``e141a998359da9a64f3c5d06b4804e06762ab2e53dc6409979ff3eb0eacd793e``

See :doc:`../reference/citing` for family-specific citations.
