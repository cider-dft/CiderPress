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
Fast validation suite for the CIDER local-hybrid implementation.

Uses an in-memory fixed-alpha stub model (no trained-model files needed for
most tests), so correctness of the hybrid SCF machinery is validated at
machine precision independently of ML details:

- T1: explicit A-tensor oracle identity for vmat and exc (RKS + UKS)
- T2: PBE0 anchors (matched-quadrature identity, SCF vs pyscf PBE0,
      xmix semantics)
- T3: finite-difference energy-potential consistency (the arbiter for all
      factor/convention questions), incl. HYB_PBE_DIFF baseline probes
- T4: RKS == UKS closed-shell identity (H2 and H2O)
- T5: incore == blockwise A-tensor identity
- H-atom: one-electron system with alpha=1 must reproduce UHF (SIC-free)
"""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from pyscf import dft, scf

from ciderpress.dft.baselines import exx_energy_baseline, exx_pbe_diff_baseline
from ciderpress.pyscf.dft import make_cider_calc
from ciderpress.pyscf.tests.lh_test_utils import (
    S66_TEST_MODEL,
    assert_fd_consistent,
    build_hybrid_oracle,
    converged_dm,
    fd_energy_potential_check,
    forced_hyb_mode,
    get_mol,
    make_fixed_alpha_mlfunc,
    prep_cider_ni,
)

GRIDS_LEVEL = 2


def make_stub_calc(
    mol, alpha=1.0, xmix=0.25, baseline="HYB_EXX", with_kernels=True, rhocut=None
):
    mf = dft.UKS(mol) if mol.spin != 0 else dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = GRIDS_LEVEL
    kwargs = {}
    if with_kernels:
        kwargs = {"xkernel": "GGA_X_PBE", "ckernel": "GGA_C_PBE"}
    mf = make_cider_calc(
        mf, make_fixed_alpha_mlfunc(alpha, baseline=baseline), xmix=xmix, **kwargs
    )
    mf.conv_tol = 1e-11
    if rhocut is not None:
        mf._numint.rhocut = rhocut
    return mf


class TestHybridOracle(unittest.TestCase):
    """T1: production hybrid vmat/exc vs explicit A-tensor construction."""

    def _run_oracle(self, spin):
        mol = get_mol("h2o+" if spin else "h2o")
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)
        alpha, xmix = 1.0, 0.25
        mf = make_stub_calc(mol, alpha=alpha, xmix=xmix, rhocut=0.0)
        ni, grids = prep_cider_ni(mf)
        nspin = 2 if spin else 1

        with forced_hyb_mode("incore"):
            if nspin == 1:
                _, exc, vmat = ni.nr_rks(mol, grids, "PBE", dm)
            else:
                _, exc, vmat = ni.nr_uks(mol, grids, "PBE", dm)

        # Reference: plain semilocal part with the same slxc string
        pni = dft.numint.NumInt()
        slxc = ni.slxc
        if nspin == 1:
            _, exc_sl, vmat_sl = pni.nr_rks(mol, grids, slxc, dm)
        else:
            _, exc_sl, vmat_sl = pni.nr_uks(mol, grids, slxc, dm)

        # Hybrid part from the explicit oracle. dm_eff = (2/nspin) * P_s.
        w = grids.weights
        hyb_settings = ni.settings.hyb_settings
        if nspin == 1:
            feat_or, feat_code, vk_unit = build_hybrid_oracle(
                mol, grids, dm, hyb_settings
            )
            # Convention check: the code's feature is the quadratic form
            assert_allclose(feat_code, feat_or, rtol=1e-9, atol=1e-12)
            e_hyb = xmix * alpha * np.dot(w, feat_code)
            # dE/dP = xmix*alpha * sum_g w_g dfeat/d(dm_eff) (dm_eff == P here)
            v_hyb = xmix * alpha * vk_unit
            vmat_ref = vmat_sl + v_hyb
            exc_ref = exc_sl + e_hyb
            assert_allclose(vmat, vmat_ref, atol=2e-9, rtol=0)
            assert_almost_equal(exc, exc_ref, 9)
        else:
            vmat_ref = np.array(vmat_sl, copy=True)
            e_hyb = 0.0
            for s in range(2):
                dm_eff = 2 * dm[s]
                feat_or, feat_code, vk_unit = build_hybrid_oracle(
                    mol, grids, dm_eff, hyb_settings
                )
                assert_allclose(feat_code, feat_or, rtol=1e-9, atol=1e-12)
                # E contribution: (1/nspin) * xmix * alpha * int w feat_s
                e_hyb += 0.5 * xmix * alpha * np.dot(w, feat_code)
                # dE/dP_s = 0.5 * xmix*alpha * dfeat/d(dm_eff) * d(dm_eff)/dP_s
                #         = 0.5 * 2 * xmix*alpha * vk_unit = xmix*alpha*vk_unit
                vmat_ref[s] += xmix * alpha * vk_unit
            exc_ref = exc_sl + e_hyb
            assert_allclose(vmat, vmat_ref, atol=2e-9, rtol=0)
            assert_almost_equal(exc, exc_ref, 9)

    def test_oracle_rks(self):
        self._run_oracle(spin=0)

    def test_oracle_uks_openshell(self):
        self._run_oracle(spin=1)


class TestPBE0Anchors(unittest.TestCase):
    """T2: fixed-alpha stub against PBE0."""

    def test_matched_quadrature_energy_identity(self):
        # E_xc^cider(dm) == E_slxc(dm) + xmix * int w . feat  (same quadrature)
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)
        mf = make_stub_calc(mol, rhocut=0.0)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode("incore"):
            _, exc, _ = ni.nr_rks(mol, grids, "PBE", dm)
        pni = dft.numint.NumInt()
        _, exc_sl, _ = pni.nr_rks(mol, grids, ni.slxc, dm)
        from ciderpress.pyscf.descriptors import _hyb_desc_getter

        feat = _hyb_desc_getter(mol, grids, dm, ni.settings.hyb_settings)[0]
        e_x_sgx = np.dot(grids.weights, feat)
        assert_almost_equal(exc, exc_sl + 0.25 * e_x_sgx, 9)

    def test_scf_vs_pbe0(self):
        # Full SCF: documents the SGX-quadrature-vs-analytic-K gap.
        mol = get_mol("h2o")
        mf_ref = dft.RKS(mol)
        mf_ref.xc = "PBE0"
        mf_ref.grids.level = GRIDS_LEVEL
        mf_ref.conv_tol = 1e-11
        e_ref = mf_ref.kernel()
        assert mf_ref.converged

        mf = make_stub_calc(mol)
        e = mf.kernel()
        assert mf.converged
        diff = abs(e - e_ref)
        # SGX quadrature error bound; tighten once measured
        assert diff < 1e-4, f"stub-PBE0 vs PBE0: diff={diff:.3e} Ha"

    def test_xmix_semantics(self):
        # (alpha=1, xmix=0.25, xkernel/ckernel) == (alpha=0.25, xmix=1.0,
        #  xc="0.75*GGA_X_PBE + GGA_C_PBE") at fixed dm
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)

        mf1 = make_stub_calc(mol, alpha=1.0, xmix=0.25, rhocut=0.0)
        ni1, grids1 = prep_cider_ni(mf1)
        with forced_hyb_mode("incore"):
            _, exc1, vmat1 = ni1.nr_rks(mol, grids1, "PBE", dm)

        mf2 = dft.RKS(mol)
        mf2.xc = "PBE"
        mf2.grids.level = GRIDS_LEVEL
        mf2 = make_cider_calc(
            mf2,
            make_fixed_alpha_mlfunc(0.25),
            xmix=1.0,
            xc="0.75*GGA_X_PBE + GGA_C_PBE",
        )
        mf2._numint.rhocut = 0.0
        ni2, grids2 = prep_cider_ni(mf2)
        with forced_hyb_mode("incore"):
            _, exc2, vmat2 = ni2.nr_rks(mol, grids2, "PBE", dm)

        assert_almost_equal(exc1, exc2, 11)
        assert_allclose(vmat1, vmat2, atol=1e-11, rtol=0)


class TestFDConsistency(unittest.TestCase):
    """T3: FD energy-potential consistency -- the factor arbiter."""

    def _fd_case(self, molname, baseline, mode, rtol=3e-6, alpha=0.7):
        mol = get_mol(molname)
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)
        # Non-unit alpha and xmix to catch factor errors that cancel at 1
        mf = make_stub_calc(mol, alpha=alpha, xmix=0.5, baseline=baseline)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode(mode):
            results = fd_energy_potential_check(ni, mol, grids, dm)
        assert_fd_consistent(
            results, rtol=rtol, label=f"{molname}/{baseline}/{mode}"
        )

    def test_fd_rks_incore_hyb_exx(self):
        self._fd_case("h2o", "HYB_EXX", "incore")

    def test_fd_rks_blockwise_hyb_exx(self):
        self._fd_case("h2o", "HYB_EXX", "blockwise")

    def test_fd_uks_openshell_hyb_exx(self):
        self._fd_case("h2o+", "HYB_EXX", "incore")

    def test_fd_uks_blockwise_hyb_exx(self):
        self._fd_case("h2o+", "HYB_EXX", "blockwise")

    def test_fd_rks_hyb_pbe_diff(self):
        # np slmode: exx_pbe_diff_baseline's s^2 assumption is satisfied
        self._fd_case("h2o", "HYB_PBE_DIFF", "incore")

    @unittest.skipUnless(os.path.exists(S66_TEST_MODEL), "S66 model not found")
    def test_fd_s66_model_nst(self):
        # Trained HYB_PBE_DIFF model with nst slmode: probes the sigma-vs-s^2
        # baseline bug (expected to FAIL pre-fix; fixed in Phase 1).
        mol = get_mol("h2o")
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)
        mf = dft.RKS(mol)
        mf.xc = "PBE"
        mf.grids.level = GRIDS_LEVEL
        mf = make_cider_calc(mf, S66_TEST_MODEL, xmix=1.0)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode("incore"):
            results = fd_energy_potential_check(ni, mol, grids, dm)
        assert_fd_consistent(results, rtol=1e-5, label="s66-nst")

    def test_evaluator_baseline_fd(self):
        # Pure-evaluator FD: dedx of the EXX baselines vs finite differences.
        rng = np.random.RandomState(1)
        nsamp = 50
        for nspin in (1, 2):
            X0T = np.empty((nspin, 3, nsamp))
            X0T[:, 0] = rng.uniform(0.05, 2.0, size=(nspin, nsamp))  # rho
            X0T[:, 1] = rng.uniform(0.0, 3.0, size=(nspin, nsamp))  # s^2 (np)
            X0T[:, 2] = rng.uniform(0.3, 1.5, size=(nspin, nsamp))  # norm. exx
            for base in (exx_energy_baseline, exx_pbe_diff_baseline):
                e0, dedx = base(X0T)
                delta = 1e-6
                for s in range(nspin):
                    for i in range(3):
                        Xp = X0T.copy()
                        Xp[s, i] += delta
                        Xm = X0T.copy()
                        Xm[s, i] -= delta
                        fd = (base(Xp)[0] - base(Xm)[0]) / (2 * delta)
                        assert_allclose(
                            fd,
                            dedx[s, i],
                            rtol=1e-6,
                            atol=1e-7,
                            err_msg=f"{base.__name__} nspin={nspin} s={s} i={i}",
                        )


class TestSpinConsistency(unittest.TestCase):
    """T4: RKS == UKS for closed-shell systems (historical H2 test)."""

    def _rks_vs_uks(self, molname):
        mol = get_mol(molname)
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)

        mf_r = make_stub_calc(mol, alpha=0.7, xmix=0.5, rhocut=0.0)
        ni_r, grids_r = prep_cider_ni(mf_r)
        with forced_hyb_mode("incore"):
            _, exc_r, vmat_r = ni_r.nr_rks(mol, grids_r, "PBE", dm)

        mol_u = get_mol(molname)
        mf_u = dft.UKS(mol_u)
        mf_u.xc = "PBE"
        mf_u.grids.level = GRIDS_LEVEL
        mf_u = make_cider_calc(
            mf_u,
            make_fixed_alpha_mlfunc(0.7),
            xmix=0.5,
            xkernel="GGA_X_PBE",
            ckernel="GGA_C_PBE",
        )
        mf_u._numint.rhocut = 0.0
        ni_u, grids_u = prep_cider_ni(mf_u)
        dmu = np.stack([dm / 2, dm / 2])
        with forced_hyb_mode("incore"):
            _, exc_u, vmat_u = ni_u.nr_uks(mol_u, grids_u, "PBE", dmu)

        assert_almost_equal(exc_r, exc_u, 10)
        assert_allclose(vmat_u[0], vmat_u[1], atol=1e-12, rtol=0)
        assert_allclose(vmat_r, vmat_u[0], atol=1e-9, rtol=0)

    def test_h2_rks_vs_uks(self):
        self._rks_vs_uks("h2")

    def test_h2o_rks_vs_uks(self):
        self._rks_vs_uks("h2o")

    def test_h2_scf_rks_vs_uks(self):
        mol = get_mol("h2")
        mf_r = make_stub_calc(mol)
        e_r = mf_r.kernel()
        assert mf_r.converged
        mol_u = get_mol("h2")
        mf_u = dft.UKS(mol_u)
        mf_u.xc = "PBE"
        mf_u.grids.level = GRIDS_LEVEL
        mf_u = make_cider_calc(
            mf_u,
            make_fixed_alpha_mlfunc(1.0),
            xmix=0.25,
            xkernel="GGA_X_PBE",
            ckernel="GGA_C_PBE",
        )
        mf_u.conv_tol = 1e-11
        e_u = mf_u.kernel()
        assert mf_u.converged
        assert_almost_equal(e_r, e_u, 9)


class TestModeConsistency(unittest.TestCase):
    """T5: incore == blockwise A-tensor identity."""

    def _incore_vs_blockwise(self, molname):
        mol = get_mol(molname)
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)
        nspin = 1 if mol.spin == 0 else 2
        mf = make_stub_calc(mol, alpha=0.7, xmix=0.5)
        ni, grids = prep_cider_ni(mf)

        out = {}
        for mode in ("incore", "blockwise"):
            with forced_hyb_mode(mode):
                if nspin == 1:
                    _, exc, vmat = ni.nr_rks(mol, grids, "PBE", dm)
                else:
                    _, exc, vmat = ni.nr_uks(mol, grids, "PBE", dm)
            out[mode] = (exc, vmat)
        assert_almost_equal(out["incore"][0], out["blockwise"][0], 11)
        assert_allclose(
            out["incore"][1], out["blockwise"][1], atol=1e-11, rtol=0
        )

    def test_incore_vs_blockwise_rks(self):
        self._incore_vs_blockwise("h2o")

    def test_incore_vs_blockwise_uks(self):
        self._incore_vs_blockwise("h2o+")


class TestOneElectron(unittest.TestCase):
    """H atom (historical test): alpha=1, xmix=1, no semilocal baseline
    => E_xc = E_x^exact => total energy must match UHF (SIC-free)."""

    def test_h_atom_vs_uhf(self):
        mol = get_mol("h")
        mf_hf = scf.UHF(mol)
        mf_hf.conv_tol = 1e-11
        e_hf = mf_hf.kernel()
        assert mf_hf.converged

        mf = dft.UKS(mol)
        mf.xc = "PBE"
        mf.grids.level = GRIDS_LEVEL
        mf = make_cider_calc(mf, make_fixed_alpha_mlfunc(1.0), xmix=1.0)
        mf.conv_tol = 1e-11
        e = mf.kernel()
        assert mf.converged
        diff = abs(e - e_hf)
        # bound = SGX quadrature error for the exchange integral
        assert diff < 2e-4, f"H atom stub-EXX vs UHF: diff={diff:.3e} Ha"


class TestContracts(unittest.TestCase):
    """T6: interface contracts (documenting current behavior pre-Phase-1)."""

    def test_uks_vxc_hyb_shape(self):
        mol = get_mol("h2o+")
        dm = converged_dm(mol, grids_level=GRIDS_LEVEL)
        mf = make_stub_calc(mol)
        ni, grids = prep_cider_ni(mf)
        with forced_hyb_mode("incore"):
            _, exc, vmat = ni.nr_uks(mol, grids, "PBE", dm)
        assert np.all(np.isfinite(vmat))
        assert vmat.shape == (2, mol.nao, mol.nao)


if __name__ == "__main__":
    unittest.main()
