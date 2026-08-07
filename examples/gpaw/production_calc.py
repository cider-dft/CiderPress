#!/usr/bin/env python
"""PBE-seeded periodic CIDER26XCSURFSCI calculation in classic GPAW."""

from ase.build import bulk
from ase.parallel import parprint
from gpaw import GPAW, PW, Mixer

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional


def main():
    atoms = bulk("Si")
    atoms.calc = GPAW(
        xc="PBE",
        mode=PW(400),
        kpts={"size": (4, 4, 4), "gamma": False},
        occupations={"name": "fermi-dirac", "width": 0.1},
        mixer=Mixer(0.05, 5, 50),
        convergence={
            "energy": 1e-5,
            "density": 1e-5,
            "eigenstates": 1e-3,
            "bands": "occupied",
        },
        parallel={"augment_grids": True},
        txt="si_pbe.txt",
    )
    atoms.get_potential_energy()
    # The production models are meta-GGAs and need the wavefunctions to
    # reconstruct the kinetic-energy density after restart.
    atoms.calc.write("si_pbe.gpw", mode="all")

    xc = get_cider_functional(
        "CIDER26XCSURFSCI",
        xmix=1.0,
        xkernel=None,
        ckernel=None,
        pasdw_store_funcs=False,
    )
    calc = CiderGPAW(
        restart="si_pbe.gpw",
        xc=xc,
        mixer=Mixer(0.05, 5, 50),
        occupations={"name": "fermi-dirac", "width": 0.1},
        maxiter=500,
        parallel={"augment_grids": True},
        txt="si_cider.txt",
    )
    atoms = calc.get_atoms()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    calc.write("si_cider.gpw", mode="all")
    parprint(f"CIDER26XCSURFSCI energy = {energy:.12f} eV")


if __name__ == "__main__":
    main()
