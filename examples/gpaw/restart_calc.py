#!/usr/bin/env python
"""Continue the CIDER checkpoint produced by production_calc.py."""

from ase.parallel import parprint
from gpaw import Mixer

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
        mixer=Mixer(0.05, 5, 50),
        maxiter=500,
        parallel={"augment_grids": True},
        txt="si_cider_restart.txt",
    )
    atoms = calc.get_atoms()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    calc.write("si_cider_restarted.gpw", mode="all")
    parprint(f"restarted CIDER26XCSURFSCI energy = {energy:.12f} eV")


if __name__ == "__main__":
    main()
