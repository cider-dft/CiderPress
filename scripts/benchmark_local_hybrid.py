#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#

"""Benchmark the local-hybrid XC build (nr_rks/nr_uks with a trained
hybrid model) in incore and blockwise A-tensor modes.

Times repeated XC builds at a fixed density (the dominant per-SCF-
iteration hybrid cost), reports per-call wall time and peak RSS, and
emits JSON for before/after comparisons across commits.

Usage:
    python scripts/benchmark_local_hybrid.py [--uks] [--repeats N]
        [--basis def2-svp] [--grid-level 2] [--out results.json]
"""

import argparse
import json
import resource
import subprocess
import time

from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc
from ciderpress.pyscf.tests.lh_test_utils import get_hyb_model_path

GLYCINE = """
N  -0.0304  1.2949  0.1237
C   1.2273  0.5556  0.0025
C   1.0361 -0.9525 -0.0510
O  -0.0708 -1.4757  0.0071
O   2.1567 -1.6949 -0.1660
H   2.0031 -2.6301 -0.1934
H  -0.7910  0.7127 -0.1899
H  -0.1873  2.0663 -0.5087
H   1.7864  0.8875 -0.8800
H   1.8236  0.7969  0.8894
"""


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench(args):
    model = get_hyb_model_path("tiny")
    assert model is not None, "tiny hybrid model not found"
    mol = gto.M(
        atom=GLYCINE,
        basis=args.basis,
        charge=1 if args.uks else 0,
        spin=1 if args.uks else 0,
        verbose=0,
        output="/dev/null",
    )
    mf = dft.UKS(mol) if args.uks else dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = args.grid_level
    mf.xc = "PBE"
    e = mf.kernel()
    dm = mf.make_rdm1()

    mfh = dft.UKS(mol) if args.uks else dft.RKS(mol)
    mfh.xc = "PBE"
    mfh.grids.level = args.grid_level
    mfh = make_cider_calc(mfh, model, xkernel=None, ckernel=None, xmix=1.0)
    mfh.grids.build()
    ni = mfh._numint
    ni.build(mol)

    results = {
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "system": "glycine" + ("+ (UKS)" if args.uks else " (RKS)"),
        "basis": args.basis,
        "grid_level": args.grid_level,
        "nao": int(mol.nao),
        "ngrids": int(mfh.grids.coords.shape[0]),
        "modes": {},
    }
    for mode in ("incore", "blockwise"):
        ni.hyb_atensor_mode = mode
        ni._hyb_cache = {}
        times = []
        for rep in range(args.repeats + 1):
            t0 = time.perf_counter()
            if args.uks:
                out = ni.nr_uks(mol, mfh.grids, "PBE", dm)
            else:
                out = ni.nr_rks(mol, mfh.grids, "PBE", dm)
            dt = time.perf_counter() - t0
            if rep > 0:  # first call warms SGX caches
                times.append(dt)
        results["modes"][mode] = {
            "exc": float(out[1]),
            "warm_time_per_call_s": sum(times) / len(times),
            "times_s": times,
            "peak_rss_mb": peak_rss_mb(),
        }
        print(
            f"{mode:10s}: {sum(times)/len(times):8.3f} s/call "
            f"(exc={out[1]:.8f}, peak RSS {peak_rss_mb():.0f} MB)"
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uks", action="store_true")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--grid-level", type=int, default=2)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = bench(args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", args.out)
