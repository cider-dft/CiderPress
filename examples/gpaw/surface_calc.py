#!/usr/bin/env python
"""PBE-seeded CIDER26XCSURFSCI template for a CO/Pt(111) slab."""

from ase.build import add_adsorbate, fcc111, molecule
from ase.parallel import parprint
from gpaw import GPAW, Mixer, PW

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional


def main():
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0)
    add_adsorbate(slab, molecule("CO"), height=1.85, position="ontop")

    mixer = Mixer(0.03, 5, 100)
    occupations = {"name": "fermi-dirac", "width": 0.1}
    convergence = {
        "energy": 1e-5,
        "density": 1e-5,
        "eigenstates": 1e-3,
        "bands": "occupied",
    }
    slab.calc = GPAW(
        xc="PBE",
        mode=PW(450),
        kpts={"size": (4, 4, 1), "gamma": True},
        occupations=occupations,
        mixer=mixer,
        convergence=convergence,
        maxiter=500,
        poissonsolver={"dipolelayer": "xy"},
        parallel={"augment_grids": True},
        txt="co_pt_pbe.txt",
    )
    slab.get_potential_energy()
    slab.calc.write("co_pt_pbe.gpw", mode="all")

    xc = get_cider_functional(
        "CIDER26XCSURFSCI",
        xmix=1.0,
        xkernel=None,
        ckernel=None,
        pasdw_store_funcs=False,
    )
    calc = CiderGPAW(
        restart="co_pt_pbe.gpw",
        xc=xc,
        occupations=occupations,
        mixer=Mixer(0.03, 5, 100),
        convergence=convergence,
        maxiter=500,
        parallel={"augment_grids": True},
        txt="co_pt_cider.txt",
    )
    slab = calc.get_atoms()
    slab.calc = calc
    energy = slab.get_potential_energy()
    calc.write("co_pt_cider.gpw", mode="all")
    parprint(f"CIDER26XCSURFSCI energy = {energy:.12f} eV")


if __name__ == "__main__":
    main()
