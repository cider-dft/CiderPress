#!/usr/bin/env python
"""CIDER23X pseudopotential calculation for methodological reproduction."""

from ase.build import bulk
from ase.parallel import parprint
from gpaw import PW

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional


def main():
    atoms = bulk("Si")
    xc = get_cider_functional(
        "CIDER23X_NL_MGGA_DTR",
        xmix=0.25,
        xkernel="GGA_X_PBE",
        ckernel="GGA_C_PBE",
        use_paw=False,
    )
    atoms.calc = CiderGPAW(
        mode=PW(520),
        xc=xc,
        setups="sg15",
        h=0.13,
        kpts={"size": (12, 12, 12), "gamma": False},
        occupations={"name": "fermi-dirac", "width": 0.01},
        convergence={"energy": 1e-5},
        parallel={"augment_grids": True},
        txt="si_cider23_pp.txt",
    )
    energy = atoms.get_potential_energy()
    parprint(f"CIDER23X pseudopotential energy = {energy:.12f} eV")


if __name__ == "__main__":
    main()
