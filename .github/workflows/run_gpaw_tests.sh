#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
ulimit -s 20000

PYTHONHASHSEED=0 python -m pytest -s -c pytest_mpi.ini ciderpress/gpaw
