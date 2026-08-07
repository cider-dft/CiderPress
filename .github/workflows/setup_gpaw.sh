#!/usr/bin/env bash
set -euo pipefail

export GPAW_CONFIG="$PWD/.github/workflows/gpaw_siteconfig.py"
python -m pip install --no-cache-dir \
    -c .github/workflows/test-constraints.txt 'gpaw==25.7.0'
./.github/workflows/install_gpaw_data.sh
