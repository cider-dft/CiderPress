#!/usr/bin/env python
"""Run a packaged CIDER23X or CIDER24X exchange model with PySCF."""

import argparse

from pyscf import dft, gto

from ciderpress.dft.model_utils import CIDER23X_MODELS, CIDER24X_MODELS
from ciderpress.pyscf.dft import make_cider_calc


MODELS = CIDER23X_MODELS + CIDER24X_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="CIDER23X_NL_MGGA_DTR")
    args = parser.parse_args()

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="def2-svp",
        charge=0,
        spin=0,
    )

    base = dft.RKS(mol)
    base.grids.level = 3
    mf = make_cider_calc(
        base,
        args.model,
        xmix=0.25,
        xkernel="GGA_X_PBE",
        ckernel="GGA_C_PBE",
    )
    mf = mf.density_fit(auxbasis="def2-universal-jfit")
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    energy = mf.kernel()

    if not mf.converged:
        raise RuntimeError("CIDER SCF did not converge")
    print(f"model = {args.model}")
    print(f"total energy = {energy:.12f} Ha")


if __name__ == "__main__":
    main()
