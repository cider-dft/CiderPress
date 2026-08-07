Installation
============

CiderPress combines Python model code with compiled C/C++ numerical kernels.
PySCF is installed as a Python dependency.  GPAW is an optional host code that
must be installed separately and built consistently with CiderPress for
parallel plane-wave calculations.

Requirements
------------

* Python 3.10--3.12
* A C and C++ compiler with OpenMP support
* BLAS and LAPACK
* CMake
* FFTW or Intel MKL for FFT operations. By default the non-MKL build downloads
  and compiles FFTW and libxc, which requires network access. Set
  ``BUILD_FFTW=OFF`` and ``BUILD_LIBXC=OFF`` to use discoverable installations
  supplied by the environment.
* MPI and ``mpicc`` for parallel GPAW calculations

Install the core package
------------------------

Install the released source distribution with:

.. code-block:: bash

   pip install ciderpress

CIDER23X, CIDER24X, and CIDER26XC model files are included.  CIDER24X loads
PyTorch with its mapped neural evaluator.  CIDER23X and CIDER26XC use the
dependencies installed with the core package.

Optional model dependencies
---------------------------

Install the D4 dependencies for ``CIDER26XCCHEMD4`` with:

.. code-block:: bash

   pip install 'ciderpress[d4]'

This installs ``pyscf-dispersion``, which evaluates the model's D4 term, and
the ``dftd4`` Python package, which provides interoperability with externally
attached dispersion wrappers.

Install PyTorch support for CIDER24X with:

.. code-block:: bash

   pip install 'ciderpress[cider24]'

For a platform-specific CPU or CUDA build, install PyTorch using the official
PyTorch instructions first.  Both extras can be requested together:

.. code-block:: bash

   pip install 'ciderpress[cider24,d4]'

Build configuration
-------------------

The source distribution invokes CMake.  Pass configuration options through
``CMAKE_CONFIGURE_ARGS``:

.. list-table:: CMake options
   :header-rows: 1
   :widths: 28 16 56

   * - Option
     - Default
     - Meaning
   * - ``BUILD_WITH_MKL``
     - ``OFF``
     - Use MKL for BLAS/LAPACK and FFT operations
   * - ``BUILD_LIBXC``
     - ``ON``
     - Download and build libxc; ``OFF`` selects a discoverable installation
   * - ``BUILD_FFTW``
     - ``ON``
     - Build FFTW for a non-MKL configuration
   * - ``BUILD_MARCH_NATIVE``
     - ``OFF``
     - Compile for the instruction set of the build host
   * - ``BUILD_WITH_MPI``
     - ``ON``
     - Build the MPI plane-wave interface when MPI is discoverable

For example, to use MKL and build the MPI interface:

.. code-block:: bash

   export CMAKE_CONFIGURE_ARGS="-DBUILD_WITH_MKL=ON -DBUILD_WITH_MPI=ON"
   pip install ciderpress

If CiderPress resolves BLAS through MKL, set ``BUILD_WITH_MKL=ON`` and omit a
separately embedded FFTW.  MKL exports FFTW-compatible symbols; loading both
FFT implementations can produce symbol collisions and runtime crashes.

Install from a source checkout
------------------------------

From the repository root:

.. code-block:: bash

   pip install .

For an editable development installation:

.. code-block:: bash

   pip install -e .

The same ``CMAKE_CONFIGURE_ARGS`` options apply.  A direct CMake build under
``ciderpress/lib`` builds the extension libraries. For example: ::

    cd ciderpress/lib
    mkdir build
    cd build
    cmake <CMAKE_ARGS> ..
    make

Use ``pip install`` to install the Python package and model resources.

PySCF environment
-----------------

PySCF is a core dependency.  Source and optimized PySCF builds use the same
Python interface and can provide different performance.  Confirm the basic
imports and a packaged model before running a calculation:

.. code-block:: bash

   python -c "import pyscf, ciderpress; print(pyscf.__version__, ciderpress.__version__)"
   python -c "from ciderpress.dft.model_utils import load_cider_model; print(load_cider_model('CIDER26XCCHEM').nfeat)"

Continue with :doc:`../usage/quickstart` and :doc:`../usage/pyscf`.

GPAW environment
----------------

Install classic GPAW with libxc, an FFT backend, and MPI as required by the
target machine.  CiderPress and GPAW must resolve compatible versions of:

* the MPI implementation and ABI;
* FFTW or MKL FFT symbols;
* the OpenMP runtime; and
* BLAS/LAPACK.

An MPI GPAW calculation requires an MPI-enabled CiderPress build.  The
periodic smoke-test calculation is :source:`examples/gpaw/production_calc.py`.

The repository's GPAW site-configuration template
(``.github/workflows/gpaw_siteconfig.py``) illustrates an MKL/MPI build.
Adapt its compiler and launcher settings to the local environment.  CiderPress
0.5.0 supports the classic GPAW calculator; ``gpaw.new`` is outside this
release interface.

Continue with :doc:`../usage/gpaw`.  See :doc:`../reference/limitations` for
the supported calculation and property scope.
