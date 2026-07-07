#!/usr/bin/env python
"""Quantify the nr_uks NLDF spin-cache bug's effect on converged
energies of open-shell species, using the production campaign model
(nod4 iter2 scaetm) at campaign-like settings.

Runs each species twice with the FIXED code: once normally and once
with a runtime shim forcing spin=0 in get_potential (reproducing the
bug exactly). Reports delta = E_buggy - E_fixed per species and derived
atomization-energy corrections for sample molecules.

Output: JSON table + stdout summary.
"""

import contextlib
import json
import sys

import numpy as np

sys.path.insert(
    0, "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF"
)

from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc

MODEL = (
    "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CIDER_gpaw/"
    "self_consistent_training_pyscf_final/runs/nod4_14164596_G400_scaetm/"
    "model_nldf_totalreaction_XCkernel_nod4_14164596_G400_scaetm_"
    "vb2_SEP_NPOL_0.05_iter2_mapped.yaml"
)
BASIS = "def2-qzvppd"
GRIDS_LEVEL = 3
HA2KCAL = 627.5094740631

# species: (atom string, spin, charge)
SPECIES = {
    "H": ("H 0 0 0", 1, 0),
    "Li": ("Li 0 0 0", 1, 0),
    "B": ("B 0 0 0", 1, 0),
    "C": ("C 0 0 0", 2, 0),
    "N": ("N 0 0 0", 3, 0),
    "O": ("O 0 0 0", 2, 0),
    "F": ("F 0 0 0", 1, 0),
    "Na": ("Na 0 0 0", 1, 0),
    "Al": ("Al 0 0 0", 1, 0),
    "Si": ("Si 0 0 0", 2, 0),
    "P": ("P 0 0 0", 3, 0),
    "S": ("S 0 0 0", 2, 0),
    "Cl": ("Cl 0 0 0", 1, 0),
    "Cr": ("Cr 0 0 0", 6, 0),
    "Mn": ("Mn 0 0 0", 5, 0),
    "OH": ("O 0 0 0; H 0 0 0.97", 1, 0),
    "NO": ("N 0 0 0; O 0 0 1.15", 1, 0),
    "CH3": (
        "C 0 0 0; H 0 1.079 0; H 0.934 -0.54 0; H -0.934 -0.54 0",
        1,
        0,
    ),
    "O2": ("O 0 0 0; O 0 0 1.208", 2, 0),
    "H2O_closed_ctrl": (
        "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        0,
        0,
    ),
}

# sample atomization corrections: molecule -> atom counts
AE_SAMPLES = {
    "H2O": {"O": 1, "H": 2},
    "CH4": {"C": 1, "H": 4},
    "NH3": {"N": 1, "H": 3},
    "C6H6": {"C": 6, "H": 6},
    "SiO2-like (SiO2)": {"Si": 1, "O": 2},
    "Cr2-like (2Cr)": {"Cr": 2},
}


@contextlib.contextmanager
def buggy_behavior(gen_cls):
    """Force spin=0 in get_potential -- exact reproduction of the bug."""
    orig = gen_cls.get_potential

    def wrapped(self, vfeat, spin=0, **kwargs):
        return orig(self, vfeat, spin=0, **kwargs)

    gen_cls.get_potential = wrapped
    try:
        yield
    finally:
        gen_cls.get_potential = orig


def run_one(name, atom, spin, charge, gen_cls, buggy):
    mol = gto.M(
        atom=atom,
        basis=BASIS,
        spin=spin,
        charge=charge,
        verbose=0,
        output="/dev/null",
    )
    mf = dft.UKS(mol) if spin != 0 else dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = GRIDS_LEVEL
    mf.max_cycle = 150
    mf.conv_tol = 1e-9
    mf = make_cider_calc(mf, MODEL, xkernel=None, ckernel=None, xmix=1.0)
    if buggy:
        with buggy_behavior(gen_cls):
            e = mf.kernel()
    else:
        e = mf.kernel()
    return e, bool(mf.converged)


def main():
    # grab the generator class from a throwaway build
    mol0 = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    mf0 = dft.UKS(mol0)
    mf0.xc = "PBE"
    mf0.grids.level = 1
    mf0 = make_cider_calc(mf0, MODEL, xkernel=None, ckernel=None, xmix=1.0)
    mf0.grids.build()
    mf0._numint.build(mf0.mol)
    mf0._numint.initialize_feature_generators(mf0.mol, mf0.grids, 2)
    gen_cls = type(mf0._numint.nldfgen)
    print(f"generator class: {gen_cls.__name__}")

    table = {}
    for name, (atom, spin, charge) in SPECIES.items():
        row = {"spin": spin}
        try:
            for label, buggy in (("fixed", False), ("buggy", True)):
                e, conv = run_one(name, atom, spin, charge, gen_cls, buggy)
                row[label] = e
                row[f"{label}_converged"] = conv
            row["delta_ha"] = row["buggy"] - row["fixed"]
            row["delta_kcal"] = row["delta_ha"] * HA2KCAL
            print(
                f"{name:18s} spin={spin} E_fixed={row['fixed']:.8f} "
                f"E_buggy={row['buggy']:.8f} "
                f"delta={row['delta_ha']:+.3e} Ha "
                f"({row['delta_kcal']:+.4f} kcal/mol) "
                f"conv={row['fixed_converged']}/{row['buggy_converged']}"
            )
        except Exception as e:
            row["error"] = str(e)
            print(f"{name:18s} FAILED: {e}")
        table[name] = row

    print("\n=== Sample atomization-energy corrections ===")
    print("(buggy AE - true AE = sum of atom deltas; positive = campaign")
    print(" AEs overestimated by this amount)")
    ae = {}
    for molname, counts in AE_SAMPLES.items():
        try:
            corr = sum(table[a]["delta_ha"] * n for a, n in counts.items())
            ae[molname] = corr * HA2KCAL
            print(f"{molname:22s}: {corr * HA2KCAL:+.4f} kcal/mol")
        except KeyError:
            pass

    out = {
        "model": MODEL,
        "basis": BASIS,
        "grids_level": GRIDS_LEVEL,
        "species": table,
        "ae_corrections_kcal": ae,
    }
    path = (
        "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/"
        "soscf_test_logs/nldf_uks_impact.json"
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
