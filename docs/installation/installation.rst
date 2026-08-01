.. highlight:: bash

Overview and Installation Instructions
======================================

CiderPress is a Python/C package that provides tools for training and evaluating CIDER Exchange-Correlation (XC) functionals for use in Density Functional Theory calculations. Interfaces to the `GPAW <https://gpaw.readthedocs.io/>`_ and `PySCF <https://pyscf.org/>`_ codes are included.

What is the CIDER formalism?
----------------------------

Machine Learning (ML) has recently gained attention as a means to fit more accurate Exchange-Correlation (XC) functionals for use in Density Functional Theory (DFT). We have developed CIDER, a set of features, models, and training techniques for efficiently learning the exchange and correlation functionals. **CIDER** stands for **C**\ ompressed scale-\ **I**\ nvariant **DE**\ nsity **R**\ epresentation, which refers to the fact that the descriptors are invariant under squishing or expanding of the density while maintaining its shape. This property makes it efficient for learning the XC functional, especially the exchange energy.

**WARNING**: The CiderPress Code Base is Experimental
-----------------------------------------------------

Both the code and the functionals themselves are experimental. The code base will likely change significantly in the next few years. Therefore, please read the installation guidance, usage instructions, examples, and known issues thoroughly before using CiderPress.

Installation
------------

Installation of CiderPress requires the following:

* Python 3.9-3.12
* BLAS and LAPACK
* C and C++ compilers with OpenMP support.

CiderPress uses cmake to build its C backend. If you use ``pip``, cmake is automatically installed as a dependency to enable the build process. The C compiler and linear algebra libraries must be findable by cmake.

The production CIDER26XC models do not require PyTorch.  PyTorch remains an
optional requirement for legacy ``CIDER24X`` models.  To install PyTorch with
CUDA 11.8, for example, use::

    pip3 install torch --index-url https://download.pytorch.org/whl/cu118

If you want to run plane-wave DFT calculations, you must also install GPAW with LibXC and FFTW. GPAW uses a ``siteconfig.py`` file to customize the libraries it links to. This repo's ``.github/workflows/gpaw_siteconfig.py`` could be useful for compiling GPAW with MKL, LibXC, and FFTW.

If you wish to run parallel calculations with GPAW, you should also have an MPI installation on your system along with an ``mpicc`` compiler. CiderPress has only been tested with OpenMPI but in principle should be compatible with MPICH and Intel MPI as well. The CiderPress build will automatically detect whether MPI is available, and if so it will build an MPI-parallel version of the GPAW interface.

The rest of the code is parallelized with OpenMP only for now.

Installation from PyPI
~~~~~~~~~~~~~~~~~~~~~~

You can install with the usual::

    pip install ciderpress

For the D4-corrected molecular model, install the optional D4 dependency::

    pip install 'ciderpress[d4]'

Currently only the sdist is available (no wheels yet), so it will take some time to build. The C backend of CiderPress is built using cmake, so you can customize the installation by setting the ``CMAKE_CONFIGURE_ARGS`` environment variable. For example, by default, CiderPress builds its own FFTW and searches for a (non-MKL) BLAS/LAPACK installation to link to. To use the Intel Math Kernel Library (MKL) as the FFT and linear algebra backend instead, use the following: ::

    export CMAKE_CONFIGURE_ARGS="-DBUILD_WITH_MKL=ON"
    pip install ciderpress

NOTE: If CiderPress will link to MKL as its linear algebra backend, make sure that you set ``-DBUILD_WITH_MKL=ON``. Otherwise, CiderPress might link to MKL and FFTW, which can cause runtime crashes because MKL contains an FFTW wrapper with identical function names.

For GPAW calculations, build CiderPress and GPAW against one consistent FFT,
MPI, and OpenMP stack.  In particular, do not combine a CiderPress extension
that embeds its own FFTW with a GPAW build that loads MKL's FFTW compatibility
symbols in the same process.  Either use MKL for both, or link both programs to
the same external FFTW installation.  For parallel calculations, also verify
that both builds use the same MPI implementation; a serial CiderPress extension
cannot be used from an MPI GPAW calculation.

Here is a list of cmake build options with their default values:

* ``BUILD_WITH_MKL (OFF)``: If ON, use Intel Math Kernel Library as the linear algebra and FFT backends. If OFF, link to whatever BLAS/LAPACK version is found by cmake and link to FFTW as the FFT backend.
* ``BUILD_LIBXC (ON)``: If ON, libxc is downloaded, compiled, and linked to CiderPress during the compilation process. If OFF, a libxc installation must be available to link to and findable by cmake at compile time.
* ``BUILD_FFTW (ON)``: Ignored if ``BUILD_WITH_MKL=ON``. Otherwise, if ON, FFTW is built and linked to by cmake during compilation. If OFF, an FFTW installation must be available to link to and findable by cmake at compile time.
* ``BUILD_MARCH_NATIVE (OFF)``: If ON, use the ``-march=native`` C compiler flag, which enables instruction sets on the CPU used for compilation, potentially resulting in higher performance.
* ``BUILD_WITH_MPI (ON)``: If ON, use the MPI installation found by cmake.  Set
  this explicitly and inspect the cmake summary when preparing a parallel GPAW
  environment.

To further customize the installation, you can build from source and edit the ``CMakeLists.txt`` files in ``ciderpress/lib`` and its subdirectories.

Build from Source
~~~~~~~~~~~~~~~~~

You can also build from source. If you clone the CiderPress repository, you can enter the repository directory and simply type::

    pip install .

Alternatively, to build the C extensions "in-place," you can use cmake directly as follows: ::

    cd ciderpress/lib
    mkdir build
    cd build
    cmake <CMAKE_ARGS> ..
    make

You can also use the cmake configuration arguments listed above with these approaches. Note that if you use ``pip``, ``cmake`` will be installed as a dependency. If you build from source directly, you must have ``cmake`` installed on your system.

Installation in a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation in a conda environment follows the same procedure as above, but with the added benefit that non-Python dependencies can be installed using conda. For example, you can install the MKL using::

    conda install mkl"<=2024.0" mkl-devel"<=2024.0" mkl-service"<=2024.0" mkl_fft mkl_random

The ``<=2024.0`` is to fix a compatibility issue with PyTorch and MKL, so you can remove it if you don't need PyTorch (i.e. if you don't want to use CIDER24X models). In principle, it is also possible to pip install the MKL dependencies, but we have had trouble getting the libraries to link. Then you install CiderPress using MKL::

    CMAKE_CONFIGURE_ARGS="-DBUILD_WITH_MKL=ON" pip install .

Step-by-step Installation with Conda, Micromamba, etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section covers how to install CiderPress and its dependencies from a fresh conda environment. Micromamba is also supported; you will just need to replace the ``conda`` commands with ``micromamba`` below.

1. Make sure you have a C compiler installed. (Or you can install one through conda after creating your environment in step 1.)

2. Create a new conda environment.::

    conda create -n <my_env> python=3.11
    conda activate <my_env>

   Python 3.9-3.12 are supported.

3. Install dependencies. The scripts ``.github/workflows/mm_install_torch.sh`` and
   ``.github/workflows/mm_install_mpi.sh`` can both be used to set up an environment
   for running CIDER calculations. ``mm_install_torch.sh`` installs MKL, libxc, FFTW,
   and pytorch, so it is useful if you want to run calculations with CIDER24X
   functionals, which require pytorch. ``mm_install_mpi.sh`` installs MKL, libxc,
   FFTW, OpenMPI, and mpicc, so it is useful if you want to run GPAW calculations.
   Note that the conda MPI installation might not work well for multi-node jobs on
   clusters, so you might want to use your own MPI/mpicc instead if that
   is your use case. Single-node jobs should work fine with conda's MPI.

4. Build C extensions and install CiderPress.::

    pip install .

5. (If using GPAW) Install GPAW from source. We recommend using our ``gpaw_siteconfig.py`` to link gpaw to
   MPI and MKL for simplicity and speed. (You can download GPAW at gitlab.com/gpaw/gpaw.)::

    cd <place you want to save the GPAW source>
    git clone https://gitlab.com/gpaw/gpaw.git
    cd gpaw
    cp <CiderPress>/.github/workflows/gpaw_siteconfig.py siteconfig.py
    python setup.py build install

   **Note**: Currently CiderPress does not support the new GPAW version (``gpaw.new``), but we plan to support it in the future.

Notes on External Code Performance and Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CiderPress automatically installs PySCF as a dependency, and GPAW can be installed simply by ``pip install gpaw``. However, both codes will in general have better performance if compiled from source. See the `PySCF installation instructions <https://pyscf.org/install.html>`_ and the `GPAW installation instructions <https://gpaw.readthedocs.io/install.html>`_ for details.
* GPAW uses MPI for parallelization, and the CiderPress extensions must also link to MPI to run parallel GPAW calculations. Make sure cmake can find OpenMPI or an equivalent installation and that you have a working ``mpicc`` compiler before building CiderPress and GPAW together.
* CiderPress and GPAW must resolve the same FFT and MPI implementations at
  runtime.  A successful import is not a sufficient compatibility check; run a
  small CIDER plane-wave calculation with the same launcher and rank layout
  intended for production before submitting large jobs.
* The CiderPress C extensions must use the same OpenMP as PySCF and GPAW, otherwise you will run into parallelization issues and code crashes. The ``gpaw_siteconfig.py`` provided in the CiderPress repository (see Step 5 of step-by-step instructions above) assumes Intel's ``iomp5`` as the OpenMP library by default. If you are using GNU OpenMP, you should change ``iomp5`` to ``gomp`` in ``gpaw_siteconfig.py``. CiderPress will find OpenMP automatically using cmake, so please make sure this version is the one used to build PySCF and GPAW.
* To run the CIDER24X functionals, you also need to install Pytorch.
* To run ``CIDER26XCCHEMD4``, install ``dftd4`` through
  ``pip install 'ciderpress[d4]'``.  The other production models do not require
  this extra.

How can I run a CIDER calculation?
----------------------------------

CiderPress 0.5.0 packages three production models, so no separate model
download is required:

* ``CIDER26XCCHEM`` for molecular chemistry without explicit dispersion.
* ``CIDER26XCCHEMD4`` for molecular chemistry with post-density D4.
* ``CIDER26XCSURFSCI`` for solids, surfaces, adsorption, and combined-domain
  applications.

Molecular calculations use
:func:`ciderpress.pyscf.dft.make_cider_calc`; periodic plane-wave PAW
calculations use :func:`ciderpress.gpaw.calculator.get_cider_functional` and
:class:`ciderpress.gpaw.calculator.CiderGPAW`.  See
:doc:`../usage/production_models` for model selection,
:doc:`../usage/pyscf` and :doc:`../usage/gpaw` for complete examples, and
:doc:`../usage/convergence` for suggested fallback settings.

Older downloaded YAML/joblib models and explicit model paths remain supported.
The legacy CIDER23X and CIDER24X models have different intended energy forms
and dependencies; do not copy their exchange-only mixing settings into a
CIDER26XC full-XC calculation.

How can I train a CIDER functional?
-----------------------------------

The basic ML training framework for CiderPress is stored in ``ciderpress.models``. CiderPress currently only contains the ML model classes themselves, but not the various training tools needed to set up the training databases. If you are interested in training your own CIDER model, we suggest reaching out to us to discuss (email kylebystrom@gmail.com).

Known Issues
------------

CiderPress has a few known issues that we are currently investigating. Please be aware of these when attempting calculations with CIDER functionals. We will make a note and publish a new release when we fix these issues. If you run into any other problems, please post an issue on the Github repository.

* Nonlocal CIDER features add memory overhead in GPAW.  Allocate more memory
  than for a comparable PBE calculation, use augmented-grid parallelization,
  and consider running fallback rungs in separate processes for large systems.
* Difficult molecular, metallic, magnetic, and near-degenerate systems can
  require conservative mixing, smearing, level shifting, or a baseline-density
  restart.  See :doc:`../usage/convergence`; never infer convergence from the
  total energy alone.
* The GPAW interface in version 0.5.0 supports the classic calculator, not
  ``gpaw.new``.  Plane-wave mode with PAW setups is recommended.
* ``CIDER26XCCHEMD4`` is supported only in PySCF.  GPAW rejects the model rather
  than omitting its fitted D4 contribution.
* D4 is an energy-only post-density correction.  Forces from the D4 component
  are not included in the CIDER PySCF nuclear-gradient interface.

Questions and Comments
----------------------

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the `Github page <https://github.com/mir-group/CiderPress>`_.

Citing
------

The CIDER26XC production models are described in the forthcoming manuscript
``Machine-Learned Exchange-Correlation Functionals in the CIDER Framework and
Application to Chemistry and Surface Science`` by Mohamed Samy Abdallah,
Zhuotao Jin, Boris Kozinsky, and Kyle Bystrom.  Its public identifier is
pending and will be added when available.

If you find CiderPress or CIDER functionals useful in your research, please cite the following article::

 @article{PhysRevB.110.075130,
  title = {Nonlocal machine-learned exchange functional for molecules and solids},
  author = {Bystrom, Kyle and Kozinsky, Boris},
  journal = {Phys. Rev. B},
  volume = {110},
  issue = {7},
  pages = {075130},
  numpages = {30},
  year = {2024},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.110.075130},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.110.075130}
 }

The above article introduces the CIDER23X functionals and much of the algorithms in CiderPress. If you use the CIDER24X functionals, please also cite::

 @article{doi:10.1021/acs.jctc.4c00999,
  author = {Bystrom, Kyle and Falletta, Stefano and Kozinsky, Boris},
  title = {Training Machine-Learned Density Functionals on Band Gaps},
  journal = {Journal of Chemical Theory and Computation},
  volume = {20},
  number = {17},
  pages = {7516-7532},
  year = {2024},
  doi = {10.1021/acs.jctc.4c00999},
  note ={PMID: 39178337},
  URL = {https://doi.org/10.1021/acs.jctc.4c00999},
  eprint = {https://doi.org/10.1021/acs.jctc.4c00999}
 }
