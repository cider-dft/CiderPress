#!/usr/bin/env python
"""Closed-shell molecular CIDER26XC calculation in PySCF."""

import argparse

from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc

MODELS = ("CIDER26XCCHEM", "CIDER26XCCHEMD4", "CIDER26XCSURFSCI")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="CIDER26XCCHEM")
    args = parser.parse_args()

    mol = gto.M(
        atom="""
        O  0.000000  0.000000  0.117790
        H  0.000000  0.755453 -0.471161
        H  0.000000 -0.755453 -0.471161
        """,
        basis="def2-tzvp",
        charge=0,
        spin=0,
    )

    base = dft.RKS(mol)
    base.grids.level = 3
    mf = make_cider_calc(base, args.model)
    mf = mf.density_fit(auxbasis="def2-universal-jfit")
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    energy = mf.kernel()

    if not mf.converged:
        raise RuntimeError("CIDER SCF did not converge")

    print(f"model = {args.model}")
    print(f"total energy = {energy:.12f} Ha")
    if hasattr(mf, "e_vdw_expected"):
        print(f"SCF/base energy = {mf.e_tot_base:.12f} Ha")
        print(f"expected dispersion = {mf.e_vdw_expected:.12f} Ha")
        print(f"dispersion adjustment = {mf.e_vdw_delta:.12f} Ha")


if __name__ == "__main__":
    main()
