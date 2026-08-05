#!/usr/bin/env bash
set -euo pipefail

./.github/workflows/apt_deps.sh
./.github/workflows/mpi_apt_deps.sh

if [ "$RUNNER_OS" == "macOS" ]; then
    export CC=gcc-14
    export CXX=g++-14
fi

if [ "$RUNNER_OS" == "Linux" ]; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=ON -DBUILD_FFTW=ON -DBUILD_WITH_MKL=OFF -DBUILD_WITH_MPI=ON -DBUILD_MARCH_NATIVE=OFF" \
    python -m pip install '.[test,cider24]'

CIDER_DEPS="$(python -c "import ciderpress, os; print(os.path.dirname(ciderpress.__file__))")/lib/deps"
export CIDER_DEPS
export LIBRARY_PATH="$CIDER_DEPS/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CIDER_DEPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="$CIDER_DEPS/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export GPAW_CONFIG="$PWD/.github/workflows/gpaw_blas_siteconfig.py"
python -m pip install --no-cache-dir 'gpaw==25.7.0'
./.github/workflows/install_gpaw_data.sh

./.github/workflows/run_gpaw_tests.sh
