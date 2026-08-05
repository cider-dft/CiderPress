#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
ulimit -s 20000

export PYTHONHASHSEED=0
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_basic_calc
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_si_force
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_si_stress
