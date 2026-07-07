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

"""SCF-level (numint) energy-potential consistency tests for NLDF
functionals, closing the coverage gap that let the nr_uks
get_potential spin-cache bug survive: the pre-existing suite validated
descriptors and the generator-level potential (which passes spin
explicitly) but never FD-tested nr_uks itself for open shells.

Uses a smooth linear evaluator on a trained vb2 model's settings and
transforms, so any FD failure indicts the numint/generator chain, not
the ML model (no spline domains involved).
"""

import os
import unittest

import numpy as np
from pyscf import dft

from ciderpress.dft.model_utils import load_cider_model
from ciderpress.dft.xc_evaluator import FuncEvaluator
from ciderpress.pyscf.dft import make_cider_calc
from ciderpress.pyscf.tests.lh_test_utils import (
    assert_fd_two_delta,
    converged_dm,
    fd_energy_potential_check,
    get_hyb_model_path,
    get_mol,
    prep_cider_ni,
)

VB2_NONHYB = get_hyb_model_path("vb2_nonhyb")


class LinearEvaluator(FuncEvaluator):
    """f(X1) = w . X1 + const; dres = w. Smooth everywhere."""

    def __init__(self, weights):
        self.weights = np.asarray(weights)

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        if dres is None:
            dres = np.zeros(X1.shape)
        res[:] += X1 @ self.weights + 0.01
        dres[:] += self.weights
        return res, dres


def _linear_probe_model(seed=0, scale=0.05):
    ml = load_cider_model(VB2_NONHYB, "yaml")
    rng = np.random.RandomState(seed)
    for k in ml.kernels:
        k.fevals = [LinearEvaluator(scale * rng.uniform(-1, 1, size=k.N1))]
    return ml


def _fd_case(molname, rtol=1e-6):
    mol = get_mol(molname)
    dm = converged_dm(mol, grids_level=1)
    mf = dft.UKS(mol) if mol.spin != 0 else dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = 1
    mf = make_cider_calc(
        mf, _linear_probe_model(), xkernel=None, ckernel=None, xmix=1.0
    )
    ni, grids = prep_cider_ni(mf)
    # two-delta protocol: rel error must either meet rtol or shrink ~
    # delta^2 (separates truncation from systematic chain errors)
    results = fd_energy_potential_check(
        ni, mol, grids, dm, n_dirs=3, deltas=[1e-3, 2.5e-4]
    )
    assert_fd_two_delta(results, rtol=rtol, label=f"nldf-scf-{molname}")


@unittest.skipUnless(VB2_NONHYB is not None, "vb2 non-hybrid model not found")
class TestNLDFSCFConsistency(unittest.TestCase):
    def test_fd_rks_closed_shell(self):
        _fd_case("h2o")

    def test_fd_uks_open_shell(self):
        # Regression guard for the nr_uks get_potential spin-cache bug
        # (fixed; history in soscf_test_logs/NLDF_UKS_DIAGNOSIS.md): the
        # beta-channel NLDF potential was built from the ALPHA channel's
        # cached forward data because the spin argument was omitted.
        _fd_case("h2o+")


if __name__ == "__main__":
    unittest.main()
