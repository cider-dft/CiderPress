#!/usr/bin/env python
"""Evaluate forces and stress from a converged CIDER GPAW checkpoint."""

from ase.parallel import parprint

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional


def main():
    xc = get_cider_functional(
        "CIDER26XCSURFSCI",
        xmix=1.0,
        xkernel=None,
        ckernel=None,
        pasdw_store_funcs=False,
    )
    calc = CiderGPAW(
        restart="si_cider.gpw",
        xc=xc,
        parallel={"augment_grids": True},
        txt="si_cider_properties.txt",
    )
    atoms = calc.get_atoms()
    atoms.calc = calc

    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    parprint("forces (eV/Angstrom):")
    parprint(forces)
    parprint("stress (eV/Angstrom^3):")
    parprint(stress)


if __name__ == "__main__":
    main()
