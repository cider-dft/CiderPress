#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#

"""
Slow local-hybrid regression suite using trained HYB_EXX models
(vb2 NLDF, sdmx1, and the H2+ dissociation reference).

Gated by CIDER_SLOW_TESTS=1 and the availability of the model files
(CIDER_TEST_HYB_MODEL_DIR, default: the test_localhybrid_H2+ share).
Run via soscf_test_logs/run_hybrid_tests.sbatch on the SLURM test partition.
"""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from pyscf import dft, gto, scf

from ciderpress.pyscf.dft import make_cider_calc
from ciderpress.pyscf.tests.lh_test_utils import (
    assert_fd_consistent,
    assert_fd_two_delta,
    converged_dm,
    fd_energy_potential_check,
    forced_hyb_mode,
    get_hyb_model_path,
    get_mol,
    prep_cider_ni,
)

SLOW = os.environ.get("CIDER_SLOW_TESTS", "0") == "1"
SKIP_MSG = "slow local-hybrid tests disabled (set CIDER_SLOW_TESTS=1)"

VB2 = get_hyb_model_path("vb2")
SDMX1 = get_hyb_model_path("sdmx1")
# The model that produced dissociation_data.csv (verified via the
# "# Model YAML:" headers in the H2+_LOCAL_*.txt logs)
SDMX1_DIFF = get_hyb_model_path("sdmx1_diff")
TINY = get_hyb_model_path("tiny")
VB2_NONHYB = get_hyb_model_path("vb2_nonhyb")

# H2+ dissociation references are read from the original campaign CSV
# (dissociation_data.csv: sdmx1_diff HYB_EXX model, def2-qzvppd, grid
# level 6, xmix=1.0). NOTE: regenerate + document if Phase-1 fixes shift
# the numbers.
CSV_PATH = os.path.join(
    os.environ.get(
        "CIDER_TEST_HYB_MODEL_DIR",
        "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CIDER_gpaw/"
        "test_localhybrid_H2+",
    ),
    "dissociation_data.csv",
)


def _load_h2p_reference():
    """Load LOCAL-hybrid references from the campaign CSV.

    Returns {exact_distance_string: energy}, preserving the CSV's
    full-precision distances so test geometries match the campaign's
    geometries exactly.
    """
    import csv

    ref = {}
    if not os.path.exists(CSV_PATH):
        return ref
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        dist_col = cols[0]
        local_col = None
        for c in cols:
            if "local" in c.lower():
                local_col = c
        if local_col is None:
            return ref
        for row in reader:
            try:
                float(row[dist_col])
                ref[row[dist_col]] = float(row[local_col])
            except (ValueError, TypeError):
                continue
    return ref


def _make_hyb_calc(mol, model_path, grids_level=1, uks=False):
    mf = dft.UKS(mol) if (uks or mol.spin != 0) else dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = grids_level
    mf = make_cider_calc(mf, model_path, xkernel=None, ckernel=None, xmix=1.0)
    mf.conv_tol = 1e-10
    return mf


@unittest.skipUnless(SLOW, SKIP_MSG)
@unittest.skipUnless(VB2 is not None, "vb2 hybrid model not found")
class TestVB2Hybrid(unittest.TestCase):
    """S1/S2: FD + spin/mode identities with the production vb2 NLDF model."""

    def test_fd_rks(self):
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=1)
        mf = _make_hyb_calc(mol, VB2)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode("incore"):
            results = fd_energy_potential_check(
                ni, mol, grids, dm, n_dirs=2, deltas=[1e-3, 2.5e-4]
            )
        assert_fd_two_delta(results, rtol=1e-5, label="vb2-rks")

    @unittest.expectedFailure
    def test_fd_uks_openshell(self):
        # KNOWN ISSUE (pre-existing, NOT hybrid-specific): spin-polarized
        # FD energy-potential consistency fails at ~3e-3 rel for NLDF
        # models. Evidence: the non-hybrid vb2 control fails identically
        # (test_fd_control_nonhybrid_uks), the evaluator-level chain is
        # clean for spin-asymmetric inputs (~1e-7), and the NLDF-free
        # tiny hybrid model passes UKS FD -- localizing the inconsistency
        # to the NLDF UKS feature/potential chain in the pyscf layer.
        # Likely related to the pre-existing
        # test_nldf.py::TestNLDFGaussian::test_same_spin_issue failure.
        mol = get_mol("h2o+")
        dm = converged_dm(mol, grids_level=1)
        mf = _make_hyb_calc(mol, VB2)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode("incore"):
            results = fd_energy_potential_check(
                ni, mol, grids, dm, n_dirs=2, deltas=[1e-3, 2.5e-4]
            )
        assert_fd_two_delta(results, rtol=1e-5, label="vb2-uks")

    def test_rks_vs_uks_closed_shell(self):
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=1)
        mf_r = _make_hyb_calc(mol, VB2)
        ni_r, grids_r = prep_cider_ni(mf_r)
        with forced_hyb_mode("incore"):
            _, exc_r, vmat_r = ni_r.nr_rks(mol, grids_r, "PBE", dm)
        mol2 = get_mol("h2o")
        mf_u = _make_hyb_calc(mol2, VB2, uks=True)
        ni_u, grids_u = prep_cider_ni(mf_u)
        dmu = np.stack([dm / 2, dm / 2])
        with forced_hyb_mode("incore"):
            _, exc_u, vmat_u = ni_u.nr_uks(mol2, grids_u, "PBE", dmu)
        assert_almost_equal(exc_r, exc_u, 9)
        assert_allclose(vmat_r, vmat_u[0], atol=1e-8, rtol=0)

    def test_incore_vs_blockwise(self):
        # Also adjudicates NLDF two-pass block alignment (CiderGrids)
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=1)
        mf = _make_hyb_calc(mol, VB2)
        ni, grids = prep_cider_ni(mf)
        out = {}
        for mode in ("incore", "blockwise"):
            with forced_hyb_mode(mode):
                out[mode] = ni.nr_rks(mol, grids, "PBE", dm)[1:]
        assert_almost_equal(out["incore"][0], out["blockwise"][0], 10)
        assert_allclose(
            out["incore"][1], out["blockwise"][1], atol=1e-10, rtol=0
        )

    @unittest.skipUnless(VB2_NONHYB is not None, "non-hybrid vb2 model absent")
    def test_fd_control_nonhybrid(self):
        # Control: non-hybrid vb2 model must pass FD => any vb2-hybrid FD
        # failure localizes to the hybrid path.
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=1)
        mf = dft.RKS(mol)
        mf.xc = "PBE"
        mf.grids.level = 1
        mf = make_cider_calc(mf, VB2_NONHYB, xkernel=None, ckernel=None, xmix=1.0)
        ni, grids = prep_cider_ni(mf)
        results = fd_energy_potential_check(
                ni, mol, grids, dm, n_dirs=2, deltas=[1e-3, 2.5e-4]
            )
        assert_fd_two_delta(results, rtol=1e-5, label="vb2-nonhyb-control")

    @unittest.expectedFailure
    @unittest.skipUnless(VB2_NONHYB is not None, "non-hybrid vb2 model absent")
    def test_fd_control_nonhybrid_uks(self):
        # Documents the KNOWN pre-existing NLDF UKS chain inconsistency
        # (see test_fd_uks_openshell): fails WITHOUT any hybrid features,
        # proving the open-shell FD failure is not in the hybrid path.
        mol = get_mol("h2o+")
        dm = converged_dm(mol, grids_level=1)
        mf = dft.UKS(mol)
        mf.xc = "PBE"
        mf.grids.level = 1
        mf = make_cider_calc(mf, VB2_NONHYB, xkernel=None, ckernel=None, xmix=1.0)
        ni, grids = prep_cider_ni(mf)
        results = fd_energy_potential_check(
            ni, mol, grids, dm, n_dirs=2, deltas=[1e-3, 2.5e-4]
        )
        assert_fd_two_delta(results, rtol=1e-5, label="vb2-nonhyb-uks")


@unittest.skipUnless(SLOW, SKIP_MSG)
@unittest.skipUnless(SDMX1_DIFF is not None, "sdmx1_diff hybrid model not found")
class TestH2PlusRegression(unittest.TestCase):
    """S3: H2+ dissociation subset vs the original campaign results.

    Canonical recipe from h2p_dissociation.py: UKS + PBE base, xmix=1.0,
    density_fit def2-universal-jfit, EDIIS, damp 0.5, energy-only
    convergence, def2-qzvppd, grid level 6, dm0 from PBE. Model:
    sdmx1_diff (the one that generated the CSV, per the run logs).
    Geometries: the CSV's exact full-precision distances.
    """

    # approximate targets; the nearest exact CSV distance is used
    DISTANCES = [1.448, 2.017, 3.155]
    CSV_TOL = 5e-6  # Ha, pre-fix; references regenerated if Phase 1 shifts

    def _run_point(self, r_string):
        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {r_string}",
            basis="def2-qzvppd",
            charge=1,
            spin=1,
            verbose=0,
            output="/dev/null",
        )
        ks = dft.UKS(mol)
        ks.xc = "PBE"
        ks.grids.level = 6
        ks.conv_tol = 1e-9
        ks.kernel()
        dm0 = ks.make_rdm1()

        hyb = dft.UKS(mol)
        hyb.xc = "PBE"
        hyb.max_cycle = 150
        hyb.conv_tol = 1e-9
        hyb.grids.level = 6
        hyb = make_cider_calc(
            hyb, SDMX1_DIFF, xkernel=None, ckernel=None, xmix=1.0
        )
        hyb = hyb.density_fit()
        hyb.with_df.auxbasis = "def2-universal-jfit"
        hyb.damp = 0.5
        hyb.diis = scf.diis.EDIIS()
        hyb.diis_space = 4
        hyb.diis_start_cycle = 0
        hyb.conv_tol_grad = float("inf")
        hyb.conv_check = False

        def _energy_only_convergence(envs):
            current = envs.get("e_tot")
            previous = envs.get("last_hf_e")
            if current is None or previous is None:
                return False
            return abs(current - previous) < hyb.conv_tol

        hyb.check_convergence = _energy_only_convergence
        e = hyb.kernel(dm0=dm0)
        assert hyb.converged, f"H2+ R={r_string} did not converge"
        return e

    def test_h2p_dissociation_subset(self):
        # Post-fix reference (regenerated after the exx sign-convention
        # fix; soscf_test_logs/regen_h2p_reference.py). The pre-fix
        # campaign CSV deltas are recorded in the JSON metadata.
        ref_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "h2p_reference.json",
        )
        if os.path.exists(ref_json):
            import json

            with open(ref_json) as f:
                ref = {
                    k: v["energy_ha"] for k, v in json.load(f)["points"].items()
                }
        else:
            ref = _load_h2p_reference()
        if not ref:
            self.skipTest("no H2+ reference found")
        for r_target in self.DISTANCES:
            # exact reference distance nearest the target
            r_string = min(ref, key=lambda k: abs(float(k) - r_target))
            e = self._run_point(r_string)
            diff = abs(e - ref[r_string])
            assert diff < self.CSV_TOL, (
                f"H2+ R={r_string}: E={e:.9f} ref={ref[r_string]:.9f} "
                f"diff={diff:.2e}"
            )


@unittest.skipUnless(SLOW, SKIP_MSG)
@unittest.skipUnless(SDMX1 is not None, "sdmx1 hybrid model not found")
class TestSDMXHybrid(unittest.TestCase):
    """S4: SDMX + hybrid combination smoke + FD."""

    def test_scf_and_fd(self):
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=1)
        mf = _make_hyb_calc(mol, SDMX1)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode("incore"):
            _, exc, vmat = ni.nr_rks(mol, grids, "PBE", dm)
        assert np.all(np.isfinite(vmat)) and np.isfinite(exc)
        with forced_hyb_mode("incore"):
            results = fd_energy_potential_check(
                ni, mol, grids, dm, n_dirs=2, deltas=[1e-3, 2.5e-4]
            )
        assert_fd_two_delta(results, rtol=1e-5, label="sdmx1")


@unittest.skipUnless(SLOW, SKIP_MSG)
@unittest.skipUnless(VB2 is not None, "vb2 hybrid model not found")
class TestOpenShellTrained(unittest.TestCase):
    """S5: UKS open-shell FD with a trained model on H2+ (tiny nao)."""

    @unittest.expectedFailure
    def test_fd_h2plus(self):
        # spin_mask keeps the EMPTY beta channel unperturbed (negative
        # densities leave the functional's domain); the remaining failure
        # is the KNOWN pre-existing NLDF UKS chain inconsistency (see
        # TestVB2Hybrid.test_fd_uks_openshell).
        mol = get_mol("h2+", basis="def2-svp")
        dm = converged_dm(mol, grids_level=1)
        mf = _make_hyb_calc(mol, VB2)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode("incore"):
            results = fd_energy_potential_check(
                ni, mol, grids, dm, n_dirs=2, deltas=[1e-3, 2.5e-4],
                spin_mask=(True, False),
            )
        assert_fd_two_delta(results, rtol=1e-5, label="vb2-h2plus")


@unittest.skipUnless(SLOW, SKIP_MSG)
@unittest.skipUnless(TINY is not None, "tiny only_sl_exx_feat model not found")
class TestTinyTrainedModel(unittest.TestCase):
    """FD with the 2.4MB only_sl_exx_feat trained model (no NLDF/SDMX):
    a fast trained-model probe of the hybrid path in isolation."""

    def test_fd_rks_and_uks(self):
        for molname in ("h2o", "h2o+"):
            mol = get_mol(molname)
            dm = converged_dm(mol, grids_level=1)
            mf = _make_hyb_calc(mol, TINY)
            ni, grids = prep_cider_ni(mf)
            with forced_hyb_mode("incore"):
                results = fd_energy_potential_check(
                ni, mol, grids, dm, n_dirs=2, deltas=[1e-3, 2.5e-4]
            )
            assert_fd_two_delta(results, rtol=1e-5, label=f"tiny-{molname}")


if __name__ == "__main__":
    unittest.main()
