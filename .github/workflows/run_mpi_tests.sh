#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
ulimit -s 20000

# GPAW links against the libxc bundled in the installed CiderPress wheel.  The
# setup script runs in a child shell, so reconstruct the runtime search path
# here before launching fresh MPI processes.
CIDER_DEPS="$(python -c 'from importlib.metadata import distribution; print(distribution("ciderpress").locate_file("ciderpress/lib/deps").resolve())')"
export LD_LIBRARY_PATH="$CIDER_DEPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export PYTHONHASHSEED=0
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_basic_calc
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_si_force
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_si_stress
