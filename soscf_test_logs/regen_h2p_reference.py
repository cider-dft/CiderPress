#!/usr/bin/env python
"""Regenerate the H2+ dissociation regression reference after the
exx-feature sign-convention fix (commit 2142488).

Runs the canonical h2p_dissociation.py recipe (sdmx1_diff model,
def2-qzvppd, grid level 6, UKS, DF, EDIIS, energy-only convergence) at
the same exact geometries as the original campaign CSV, and writes
ciderpress/pyscf/tests/data/h2p_reference.json with the new energies
plus the documented deltas vs the pre-fix campaign values.
"""

import json
import os
import subprocess
import sys

sys.path.insert(
    0, "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF"
)

from ciderpress.pyscf.tests.test_local_hybrid_slow import (  # noqa: E402
    SDMX1_DIFF,
    TestH2PlusRegression,
    _load_h2p_reference,
)

OUT = (
    "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/"
    "ciderpress/pyscf/tests/data/h2p_reference.json"
)


def main():
    assert SDMX1_DIFF is not None
    csv_ref = _load_h2p_reference()
    assert csv_ref, "campaign CSV not found"
    tester = TestH2PlusRegression("test_h2p_dissociation_subset")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(OUT),
        capture_output=True,
        text=True,
    ).stdout.strip()

    entries = {}
    for r_target in TestH2PlusRegression.DISTANCES:
        r_string = min(csv_ref, key=lambda k: abs(float(k) - r_target))
        e = tester._run_point(r_string)
        entries[r_string] = {
            "energy_ha": e,
            "campaign_csv_energy_ha": csv_ref[r_string],
            "delta_vs_campaign_ha": e - csv_ref[r_string],
        }
        print(
            f"R={r_string}: E={e:.9f} csv={csv_ref[r_string]:.9f} "
            f"delta={e - csv_ref[r_string]:+.3e}"
        )

    payload = {
        "description": (
            "H2+ dissociation regression reference, regenerated after the "
            "exx-feature train/eval sign-convention fix. The pre-fix "
            "campaign (dissociation_data.csv) evaluated alpha(r) via "
            "out-of-domain spline extrapolation; deltas quantify the "
            "effect for the sdmx1_diff model."
        ),
        "model": os.path.basename(SDMX1_DIFF),
        "recipe": (
            "UKS PBE base, make_cider_calc(xmix=1.0), density_fit "
            "def2-universal-jfit, EDIIS damp=0.5, energy-only conv 1e-9, "
            "def2-qzvppd, grids level 6, dm0 from PBE"
        ),
        "generated_at_commit": commit,
        "points": entries,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
