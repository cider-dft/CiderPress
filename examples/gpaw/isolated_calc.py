#!/usr/bin/env python
"""Isolated molecule in a periodic box with classic GPAW and CIDER."""

from ase import Atoms
from ase.parallel import parprint
from gpaw import GPAW, PW, Mixer
from gpaw.eigensolvers.davidson import Davidson

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional


def main():
    atoms = Atoms("He")
    atoms.set_cell((12.0, 12.0, 12.0))
    atoms.center()
    atoms.pbc = True

    mixer = Mixer(0.05, 5, 50)
    occupations = {"name": "fermi-dirac", "width": 0.01}
    convergence = {
        "energy": 5e-4,
        "density": 1e-4,
        "eigenstates": 5e-3,
        "bands": "occupied",
    }
    # This example uses a relaxed baseline density criterion before the CIDER
    # restart.  Converge both stages for the target property.
    pbe_convergence = dict(convergence)
    pbe_convergence["density"] = 1e-2
    atoms.calc = GPAW(
        xc="PBE",
        mode=PW(400),
        kpts={"size": (1, 1, 1), "gamma": True},
        symmetry="off",
        mixer=mixer,
        eigensolver=Davidson(niter=3),
        occupations=occupations,
        convergence=pbe_convergence,
        maxiter=500,
        parallel={"augment_grids": True},
        txt="he_pbe.txt",
    )
    atoms.get_potential_energy()
    # Preserve wavefunctions for the meta-GGA kinetic-energy density.
    atoms.calc.write("he_pbe.gpw", mode="all")

    xc = get_cider_functional(
        "CIDER26XCSURFSCI",
        xmix=1.0,
        xkernel=None,
        ckernel=None,
        pasdw_store_funcs=False,
    )
    calc = CiderGPAW(
        restart="he_pbe.gpw",
        xc=xc,
        mixer=Mixer(0.05, 5, 50),
        eigensolver=Davidson(niter=3),
        occupations=occupations,
        convergence=convergence,
        maxiter=500,
        parallel={"augment_grids": True},
        txt="he_cider.txt",
    )
    atoms = calc.get_atoms()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    # Keep wavefunctions so a later meta-GGA restart can reconstruct the
    # kinetic-energy density.
    calc.write("he_cider.gpw", mode="all")
    parprint(f"CIDER26XCSURFSCI energy = {energy:.12f} eV")


if __name__ == "__main__":
    main()
