#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=2
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
ulimit -s 20000

# See https://github.com/pytest-dev/pytest/issues/1075
PYTHONHASHSEED=0 python -m pytest -s -c pytest.ini ciderpress
