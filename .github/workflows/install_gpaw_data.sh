#!/usr/bin/env bash
set -euo pipefail

data_root="${RUNNER_TEMP:-$PWD/.gpaw-data}"
archive_root="$data_root/archives"
mkdir -p "$archive_root"

paw_archive_name="gpaw-setups-24.1.0.tar.gz"
sg15_archive_name="sg15_oncv_upf_2020-02-06.tar.gz"
paw_archive="$archive_root/$paw_archive_name"
sg15_archive="$archive_root/$sg15_archive_name"

curl --fail --location --retry 3 \
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-24.1.0.tar.gz \
    --output "$paw_archive"
curl --fail --location --retry 3 \
    http://www.quantum-simulation.org/potentials/sg15_oncv/sg15_oncv_upf_2020-02-06.tar.gz \
    --output "$sg15_archive"

python - "$paw_archive" \
    314d43168f7b57a2d942855d3d5ad21da9ef74e772d37343d416305113a95c23 \
    "$sg15_archive" \
    3f3bd74aa5d6e0b038218a6051bb99ed9469dc03d0f05b3ec8a523f0f7a7dff0 <<'PY'
import hashlib
import pathlib
import sys

for filename, expected in zip(sys.argv[1::2], sys.argv[2::2]):
    digest = hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"Checksum mismatch for {filename}: {digest}")
PY

cd "$archive_root"
gpaw install-data --tarball "$paw_archive_name" --register "$data_root"
gpaw install-data --tarball "$sg15_archive_name" --register "$data_root"
