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
Tests for second-order SCF (newton / SOSCF) support for CIDER functionals.

The CIDER ML model used for testing is resolved in this order:
1. The CIDER_TEST_MLFUNC environment variable (path to a mapped yaml or
   joblib model).
2. The vb2 NLDF AE/TM self-consistently-trained model (nod4 iter2 scaetm)
   on the kozinsky_lab holystore share.
3. functionals/CIDER23X_NL_MGGA_DTR.yaml relative to the cwd (the path
   convention of test_functionals.py; populated by
   scripts/download_functionals.py).

All tests are skipped if no model file is found.
"""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc

_VB2_DEFAULT = (
    "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CIDER_gpaw/"
    "self_consistent_training_pyscf_final/runs/nod4_14164596_G400_scaetm/"
    "model_nldf_totalreaction_XCkernel_nod4_14164596_G400_scaetm_"
    "vb2_SEP_NPOL_0.05_iter2_mapped.yaml"
)


def _resolve_mlfunc():
    cand = os.environ.get("CIDER_TEST_MLFUNC")
    if cand and os.path.exists(cand):
        return cand
    if os.path.exists(_VB2_DEFAULT):
        return _VB2_DEFAULT
    cand = "functionals/CIDER23X_NL_MGGA_DTR.yaml"
    if os.path.exists(cand):
        return cand
    return None


MLFUNC = _resolve_mlfunc()
SKIP_MSG = (
    "No CIDER ML model found. Set CIDER_TEST_MLFUNC to a mapped model "
    "yaml/joblib, or run scripts/download_functionals.py from the repo "
    "root and run the tests from there."
)

CONV_TOL = 1e-11
GRIDS_LEVEL = 1


def _get_mol(charge=0, spin=0, basis="6-31g"):
    return gto.M(
        atom="O 0.0 0.0 0.0; H 0.0 0.757 0.587; H 0.0 -0.757 0.587",
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0,
        output="/dev/null",
    )


def _get_cider_ks(mol, xmix=1.0, xkernel=None, ckernel=None, df=False):
    if mol.spin == 0:
        ks = dft.RKS(mol)
    else:
        ks = dft.UKS(mol)
    ks.grids.level = GRIDS_LEVEL
    ks = make_cider_calc(ks, MLFUNC, xmix=xmix, xkernel=xkernel, ckernel=ckernel)
    if df:
        ks = ks.density_fit(auxbasis="def2-universal-jfit")
    ks.conv_tol = CONV_TOL
    return ks


@unittest.skipUnless(MLFUNC is not None, SKIP_MSG)
class TestNewton(unittest.TestCase):
    def _check_newton_vs_diis(self, spin, fd_response=False, df=False, decimal=8):
        mol = _get_mol(charge=spin, spin=spin)
        mf_ref = _get_cider_ks(mol, df=df)
        e_ref = mf_ref.kernel()
        assert mf_ref.converged

        mf = _get_cider_ks(mol, df=df)
        if fd_response:
            mf.fd_response = True
        mf_newton = mf.newton()
        e_newton = mf_newton.kernel()
        assert mf_newton.converged
        assert_almost_equal(e_newton, e_ref, decimal)
        mol.stdout.close()
        return mf_ref, mf_newton

    def test_rks_newton_proxy(self):
        self._check_newton_vs_diis(spin=0)

    def test_uks_newton_proxy(self):
        self._check_newton_vs_diis(spin=1)

    def test_rks_newton_fd_response(self):
        self._check_newton_vs_diis(spin=0, fd_response=True, decimal=7)

    def test_newton_with_density_fit(self):
        mf_ref, mf_newton = self._check_newton_vs_diis(spin=0, df=True, decimal=7)
        # undo_soscf must strip both the SOSCF wrapper and the _CiderSOSCF
        # glue mixin
        from ciderpress.pyscf.soscf import _CiderSOSCF

        mf_undone = mf_newton.undo_soscf()
        assert not isinstance(mf_undone, _CiderSOSCF)

    def test_fd_matches_analytic_kernel_when_xmix0(self):
        # With xmix=0 and a PBE baseline, the CIDER XC potential is exactly
        # PBE and the proxy kernel is exactly the PBE kernel, so the FD
        # response must match the analytic response to O(delta^2). This
        # validates the FD machinery against an analytic reference.
        mol = _get_mol()
        mf = _get_cider_ks(mol, xmix=0.0, xkernel="GGA_X_PBE", ckernel="GGA_C_PBE")
        mf.kernel()
        assert mf.converged

        from ciderpress.pyscf.soscf import gen_fd_response

        vind_proxy = mf.gen_response(hermi=1)
        vind_fd = gen_fd_response(mf, hermi=1)

        nao = mol.nao
        rng = np.random.RandomState(42)
        for _ in range(2):
            d = rng.uniform(-0.1, 0.1, size=(nao, nao))
            dm1 = d + d.T
            v_proxy = vind_proxy(dm1)
            v_fd = vind_fd(dm1)
            assert_allclose(v_fd, v_proxy, atol=1e-5, rtol=0)
        mol.stdout.close()

    def test_stability_smoke(self):
        mol = _get_mol()
        mf = _get_cider_ks(mol)
        mf.kernel()
        assert mf.converged
        # Internal stability exercises gen_g_hop with the proxy kernel
        # (hermi=1) and the hermi=2 path.
        mf.stability(internal=True, external=False)
        mol.stdout.close()


if __name__ == "__main__":
    unittest.main()
