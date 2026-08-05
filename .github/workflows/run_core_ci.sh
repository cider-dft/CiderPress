#!/usr/bin/env bash
set -euo pipefail

./.github/workflows/apt_deps.sh
if [ "$RUNNER_OS" == "Linux" ]
then
    python -m pip install -c .github/workflows/test-constraints.txt \
        torch --index-url https://download.pytorch.org/whl/cpu
elif [ "$RUNNER_OS" == "macOS" ]
then
    python -m pip install -c .github/workflows/test-constraints.txt torch
else
    echo "$RUNNER_OS not supported"
    exit 1
fi
if [ "$RUNNER_OS" == "macOS" ]; then
    export CC=gcc-14
    export CXX=g++-14
    brew_prefix="$(brew --prefix)"
    export C_INCLUDE_PATH="$brew_prefix/include"
    export LIBRARY_PATH="$brew_prefix/lib"
    export LD_LIBRARY_PATH="$brew_prefix/lib"
fi
export CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=1 -DBUILD_FFTW=1 -DBUILD_WITH_MKL=0 -DBUILD_WITH_MPI=0 -DBUILD_MARCH_NATIVE=0"
python -m pip install -c .github/workflows/test-constraints.txt '.[test,d4,cider24]'
./.github/workflows/run_tests.sh
