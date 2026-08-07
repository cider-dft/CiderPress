#!/usr/bin/env bash
set -euo pipefail

./.github/workflows/apt_deps.sh
if [ "$RUNNER_OS" == "Linux" ]; then
    python -m pip install -c .github/workflows/test-constraints.txt \
        torch --index-url https://download.pytorch.org/whl/cpu
elif [ "$RUNNER_OS" == "macOS" ]; then
    python -m pip install -c .github/workflows/test-constraints.txt torch
fi
if [ "$RUNNER_OS" == "macOS" ]; then
    export CC=gcc-14
    export CXX=g++-14
    HOMEBREW_PREFIX="$(brew --prefix)"
    export HOMEBREW_PREFIX
    export C_INCLUDE_PATH="$HOMEBREW_PREFIX/include"
    export LIBRARY_PATH="$HOMEBREW_PREFIX/lib"
    export LD_LIBRARY_PATH="$HOMEBREW_PREFIX/lib"
fi
export CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=1 -DBUILD_FFTW=1 -DBUILD_WITH_MKL=0 -DBUILD_WITH_MPI=0 -DBUILD_MARCH_NATIVE=0"
python -m pip install -c .github/workflows/test-constraints.txt '.[test,cider24]'

CIDER_DEPS="$(python -c "import ciderpress, os; print(os.path.dirname(ciderpress.__file__))")/lib/deps"
export CIDER_DEPS
export LIBRARY_PATH="$CIDER_DEPS/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CIDER_DEPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="$CIDER_DEPS/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
if [ "$RUNNER_OS" == "macOS" ]; then
    export DYLD_LIBRARY_PATH="$CIDER_DEPS/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export GPAW_CONFIG="$PWD/.github/workflows/gpaw_openblas_nompi_siteconfig.py"
python -m pip install --no-cache-dir \
    -c .github/workflows/test-constraints.txt 'gpaw==25.7.0'
./.github/workflows/install_gpaw_data.sh

./.github/workflows/run_gpaw_tests.sh
