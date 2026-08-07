#!/usr/bin/env python
"""Demonstrate a reusable restart ladder on a compact open-shell molecule.

O2 keeps the example inexpensive and normally converges on an early rung. The
same sequence can be applied when a larger open-shell calculation needs a PBE
warm start or more conservative SCF controls.
"""

from pathlib import Path

from pyscf import dft, gto
from pyscf.scf import diis as pyscf_diis

from ciderpress.pyscf.dft import make_cider_calc


LADDER = (
    {"label": "cdiis8", "DIIS": pyscf_diis.CDIIS, "diis_space": 8,
     "conv_tol": 1e-9, "level_shift": 0.0, "damp": 0.0},
    {"label": "cdiis12", "DIIS": pyscf_diis.CDIIS, "diis_space": 12,
     "conv_tol": 1e-8, "level_shift": 0.0, "damp": 0.0},
    {"label": "adiis", "DIIS": pyscf_diis.ADIIS, "diis_space": 12,
     "conv_tol": 1e-7, "level_shift": 0.0, "damp": 0.0},
    {"label": "ediis", "DIIS": pyscf_diis.EDIIS, "diis_space": 12,
     "conv_tol": 1e-7, "level_shift": 0.0, "damp": 0.0},
    {"label": "cdiis_shift02", "DIIS": pyscf_diis.CDIIS, "diis_space": 8,
     "conv_tol": 1e-7, "level_shift": 0.2, "damp": 0.0},
    {"label": "cdiis_shift05", "DIIS": pyscf_diis.CDIIS, "diis_space": 8,
     "conv_tol": 1e-6, "level_shift": 0.5, "damp": 0.3},
)


def configure(mf, rung):
    mf.DIIS = rung["DIIS"]
    mf.diis_space = rung["diis_space"]
    mf.conv_tol = rung["conv_tol"]
    mf.level_shift = rung["level_shift"]
    mf.damp = rung["damp"]
    mf.max_cycle = 500


def run_ladder(mf, warm_density):
    guesses = (("warm", warm_density), ("atom", mf.get_init_guess(key="atom")))
    for guess_name, density in guesses:
        for rung in LADDER:
            configure(mf, rung)
            mf.chkfile = str(Path(f"o2_cider_{rung['label']}_{guess_name}.chk"))
            energy = mf.kernel(dm0=density)
            if not mf.converged:
                continue

            # A relaxed rung supplies a preconditioned density.  This example
            # follows it with the standard CDIIS control set.
            if rung["label"] != "cdiis8":
                tight_density = mf.make_rdm1()
                configure(mf, LADDER[0])
                mf.chkfile = "o2_cider_final.chk"
                energy = mf.kernel(dm0=tight_density)
                if not mf.converged:
                    continue
            return energy, rung["label"], guess_name
    raise RuntimeError("CIDER SCF did not converge with the suggested ladder")


def main():
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.21",
        basis="def2-svp",
        charge=0,
        spin=2,
    )

    baseline = dft.UKS(mol)
    baseline.xc = "PBE"
    baseline.grids.level = 3
    baseline.conv_tol = 1e-9
    baseline.max_cycle = 200
    baseline.chkfile = "o2_pbe.chk"
    baseline.kernel()
    if not baseline.converged:
        raise RuntimeError("Baseline PBE calculation did not converge")
    warm_density = baseline.make_rdm1()

    base = dft.UKS(mol)
    base.grids.level = 3
    mf = make_cider_calc(base, "CIDER26XCCHEM")
    energy, rung, guess = run_ladder(mf, warm_density)

    print(f"total energy = {energy:.12f} Ha")
    print(f"initial density = {guess}")
    print(f"successful preconditioning rung = {rung}")
    print(f"<S^2>, multiplicity = {mf.spin_square()}")


if __name__ == "__main__":
    main()
