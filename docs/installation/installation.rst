Installation
============

CiderPress combines Python model code with compiled C/C++ numerical kernels.
PySCF is installed as a Python dependency.  GPAW is an optional host code that
must be installed separately and built consistently with CiderPress for
parallel plane-wave calculations.

Requirements
------------

* Python 3.9--3.12
* A C and C++ compiler with OpenMP support
* BLAS and LAPACK
* CMake
* FFTW or Intel MKL for FFT operations
* MPI and ``mpicc`` for parallel GPAW calculations

Install the core package
------------------------

Install the released source distribution with:

.. code-block:: bash

   pip install ciderpress

All published CIDER23X, CIDER24X, and CIDER26XC model files are included.
CIDER24X needs PyTorch only when one of those models is loaded; CIDER26XC and
CIDER23X do not.

Optional model dependencies
---------------------------

Install the D4 runtime for ``CIDER26XCCHEMD4`` with:

.. code-block:: bash

   pip install 'ciderpress[d4]'

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
     - Download and build libxc instead of using a discoverable installation
   * - ``BUILD_FFTW``
     - ``ON``
     - Build FFTW when MKL is not selected
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

If CiderPress resolves BLAS through MKL, set ``BUILD_WITH_MKL=ON`` rather than
also embedding a separate FFTW.  MKL exports FFTW-compatible symbols, and
loading it beside a separately embedded FFTW can produce symbol collisions and
runtime crashes.

Install from a source checkout
------------------------------

From the repository root:

.. code-block:: bash

   pip install .

For an editable development installation:

.. code-block:: bash

   pip install -e .

The same ``CMAKE_CONFIGURE_ARGS`` options apply.  A direct CMake build is also
possible under ``ciderpress/lib`` but does not install the Python package or
model resources.

PySCF environment
-----------------

PySCF is a core dependency.  A source or optimized PySCF build can improve
performance, but the Python interface is the same.  Confirm the basic imports
and a packaged model before running a calculation:

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

A successful import does not prove that an MPI/FFT combination is safe.
Before a large job, run a small CIDER plane-wave calculation with the same MPI
launcher and rank layout intended for production.  A serial CiderPress build
cannot be used inside an MPI GPAW calculation.

The repository's GPAW site-configuration template illustrates an MKL/MPI
build, but cluster compiler and launcher settings must match the local
environment.  CiderPress 0.5.0 supports the classic GPAW calculator, not
``gpaw.new``.

Continue with :doc:`../usage/gpaw`.  The central limitations are collected in
:doc:`../reference/limitations`.

Documentation build
-------------------

From the ``docs`` directory, build the same strict HTML target used by CI:

.. code-block:: bash

   sphinx-build -n -W --keep-going -b html . _build/html
