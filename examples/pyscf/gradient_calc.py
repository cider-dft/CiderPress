#!/usr/bin/env python
"""Evaluate a molecular CIDER26XC energy and analytical gradient."""

from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc


def main():
    mol = gto.M(
        atom="""
        O  0.000000  0.000000  0.117790
        H  0.000000  0.755453 -0.471161
        H  0.000000 -0.755453 -0.471161
        """,
        basis="def2-svp",
        charge=0,
        spin=0,
    )

    base = dft.RKS(mol)
    base.grids.level = 3
    mf = make_cider_calc(base, "CIDER26XCCHEM")
    mf = mf.density_fit(auxbasis="def2-universal-jfit")
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("CIDER SCF did not converge")

    gradient = mf.nuc_grad_method().set(grid_response=True).kernel()
    print(f"total energy = {energy:.12f} Ha")
    print("gradient (Ha/Bohr):")
    print(gradient)


if __name__ == "__main__":
    main()
