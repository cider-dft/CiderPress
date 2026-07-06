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
Shared utilities for the local-hybrid validation suite
(test_local_hybrid.py / test_local_hybrid_slow.py).

Provides:
- ConstEvaluator / make_fixed_alpha_mlfunc: an in-memory MappedXC stub whose
  ML output is a constant mixing fraction alpha(r) = const. With
  alpha=1, xmix=0.25, xkernel="GGA_X_PBE", ckernel="GGA_C_PBE" this
  reproduces PBE0 (up to SGX quadrature error in the exchange term).
- fd_energy_potential_check: central-finite-difference consistency test
  between the XC energy and potential returned by the CIDER numint --
  the arbiter for all factor/convention questions in the hybrid path.
- build_hybrid_oracle_vmat: an independent, explicit A-tensor construction
  of the local-hybrid K contribution for constant alpha, used to validate
  the production code's block accumulation at machine precision.
- forced_hyb_mode: context manager forcing incore/blockwise A-tensor mode.
- Model-file resolution for the trained HYB_EXX models used in slow tests.
"""

import contextlib
import os

import numpy as np
from pyscf import dft, gto
from pyscf.dft import numint as pyscf_numint

from ciderpress.dft.baselines import (
    exx_energy_baseline,
    exx_pbe_diff_baseline,
    zero_xc,
)
from ciderpress.dft.plans import HybridPlan
from ciderpress.dft.settings import (
    FeatureSettings,
    HybridSettings,
    SemilocalSettings,
)
from ciderpress.dft.transform_data import FeatureList, UMap
from ciderpress.dft.xc_evaluator import FuncEvaluator, MappedDFTKernel, MappedXC

# ---------------------------------------------------------------------------
# Trained-model resolution (slow tests). Override dir with
# CIDER_TEST_HYB_MODEL_DIR; individual tests skip if a file is absent.
# ---------------------------------------------------------------------------

DEFAULT_HYB_MODEL_DIR = (
    "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CIDER_gpaw/"
    "test_localhybrid_H2+"
)

HYB_MODEL_FILES = {
    "vb2": "model_hybrid_GMTK_26055374_HYB_EXX_vb2_kplan_hybrid_mapped.yaml",
    "sdmx1": "model_hybrid_GMTK_26055379_HYB_EXX_sdmx1_kplan_hybrid_sdmx_mapped.yaml",
    "tiny": (
        "model_hybrid_GMTK_26307055_HYB_EXX_sdmx1_kplan_only_sl_exx_feat_"
        "threekernels_mapped.yaml"
    ),
    "vb2_nonhyb": "model_GMTK_25970616_vb2_mapped.yaml",
}

# Small HYB_PBE_DIFF model shipped with the test suite (nst slmode; the
# baseline-type probe). Copied from the historical debug_hybrid_tests work.
S66_TEST_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "model_hybrid_S66_test_mapped.yaml",
)


def get_hyb_model_path(key):
    """Return path to a trained hybrid model, or None if unavailable."""
    dirname = os.environ.get("CIDER_TEST_HYB_MODEL_DIR", DEFAULT_HYB_MODEL_DIR)
    path = os.path.join(dirname, HYB_MODEL_FILES[key])
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Fixed-alpha stub model
# ---------------------------------------------------------------------------


class ConstEvaluator(FuncEvaluator):
    """ML function stub: f(X1) = const, df/dX1 = 0, so alpha(r) = const.

    MappedDFTKernel calls evaluators with preallocated (res, dres) buffers
    and expects in-place accumulation.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        if dres is None:
            dres = np.zeros(X1.shape)
        res[:] += self.value
        return res, dres


def make_fixed_alpha_mlfunc(alpha=1.0, baseline="HYB_EXX", slmode="np"):
    """Build an in-memory MappedXC with constant mixing fraction alpha.

    The kernel's multiplicative baseline supplies the exact-exchange energy
    density (HYB_EXX) or its PBE-difference (HYB_PBE_DIFF), so the model's
    XC energy contribution is alpha * eps_x^exact (or alpha * (eps_x^exact -
    eps_x^PBE)). load_cider_model passes MappedXC objects through unchanged,
    so this can be handed directly to make_cider_calc.
    """
    settings = FeatureSettings(
        sl_settings=SemilocalSettings(slmode),
        hyb_settings=HybridSettings(),
    )
    settings.assign_reasonable_normalizer()
    if baseline == "HYB_EXX":
        mul = exx_energy_baseline
    elif baseline == "HYB_PBE_DIFF":
        mul = exx_pbe_diff_baseline
    else:
        raise ValueError(baseline)
    kernel = MappedDFTKernel(
        fevals=[ConstEvaluator(alpha)],
        # Content is irrelevant (df/dX1 = 0) but must be well-formed;
        # feature 1 exists in every slmode.
        feature_list=FeatureList([UMap(1, 1.0)]),
        mode="SEP",
        multiplicative_baseline=mul,
        # NOTE: must be passed explicitly. apply_baseline's None-check tests
        # the always-truthy bound method `self.additive_baseline`, not
        # `self._add_basefunc`, so a None additive baseline would crash.
        additive_baseline=zero_xc,
    )
    return MappedXC([kernel], settings)


# ---------------------------------------------------------------------------
# Molecule / calculation helpers
# ---------------------------------------------------------------------------

H2O_GEOM = "O 0.0 0.0 0.0; H 0.0 0.757 0.587; H 0.0 -0.757 0.587"


def get_mol(name="h2o", basis="def2-svp", verbose=0):
    if name == "h2o":
        atom, charge, spin = H2O_GEOM, 0, 0
    elif name == "h2o+":
        atom, charge, spin = H2O_GEOM, 1, 1
    elif name == "h2":
        atom, charge, spin = "H 0 0 0; H 0 0 0.74", 0, 0
    elif name == "h":
        atom, charge, spin = "H 0 0 0", 0, 1
    elif name == "h2+":
        atom, charge, spin = "H 0 0 0; H 0 0 2.0", 1, 1
    else:
        raise ValueError(name)
    return gto.M(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=verbose,
        output="/dev/null" if verbose == 0 else None,
    )


def converged_dm(mol, xc="PBE", grids_level=2):
    """Converged (non-CIDER) density matrix as the FD/oracle reference point."""
    mf = dft.UKS(mol) if mol.spin != 0 else dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grids_level
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return np.asarray(mf.make_rdm1())


def prep_cider_ni(mf):
    """Prepare a make_cider_calc'd mean-field object for direct numint calls."""
    mf.grids.build()
    mf._numint.build(mf.mol)
    return mf._numint, mf.grids


@contextlib.contextmanager
def forced_hyb_mode(mode):
    """Force the hybrid A-tensor mode: 'incore' or 'blockwise'.

    Works by monkeypatching HybridPlan.estimate_memory_gb, which nr_rks/nr_uks
    compare against max_memory/1000. Replace with the hyb_atensor_mode knob
    once Phase 2 lands.
    """
    orig = HybridPlan.estimate_memory_gb
    if mode == "incore":
        HybridPlan.estimate_memory_gb = lambda self, ngrids, nao: 0.0
    elif mode == "blockwise":
        HybridPlan.estimate_memory_gb = lambda self, ngrids, nao: 1e9
    else:
        raise ValueError(mode)
    try:
        yield
    finally:
        HybridPlan.estimate_memory_gb = orig


# ---------------------------------------------------------------------------
# FD energy-potential consistency driver (THE arbiter)
# ---------------------------------------------------------------------------


def _xc_energy_and_vmat(ni, mol, grids, dm, nspin):
    # xc_code is ignored by eval_xc_cider (slxc is used), but nr_rks/nr_uks
    # require a string (xc_code.upper() is called on the NLDF path).
    if nspin == 1:
        _, exc, vmat = ni.nr_rks(mol, grids, "PBE", dm)
    else:
        _, exc, vmat = ni.nr_uks(mol, grids, "PBE", dm)
    return exc, vmat


def fd_energy_potential_check(ni, mol, grids, dm0, n_dirs=3, delta=1e-4, seed=0):
    """Central-FD directional derivative of E_xc vs Tr(vxc . u).

    dm0: (nao, nao) for RKS, (2, nao, nao) for UKS. Directions u are random
    symmetric matrices (plus one rank-1 direction), applied jointly to both
    spin channels for UKS with independent spin components.

    Returns list of (lhs, rhs) tuples, one per direction.
    """
    dm0 = np.asarray(dm0)
    nspin = 1 if dm0.ndim == 2 else 2
    nao = dm0.shape[-1]
    rng = np.random.RandomState(seed)

    _, vmat0 = None, None
    exc0, vmat0 = _xc_energy_and_vmat(ni, mol, grids, dm0, nspin)

    results = []
    for idir in range(n_dirs):
        if idir == n_dirs - 1:
            # rank-1 direction resembling an orbital rotation
            c = rng.uniform(-1, 1, size=nao)
            u1 = np.outer(c, c)
        else:
            d = rng.uniform(-1, 1, size=(nao, nao))
            u1 = 0.5 * (d + d.T)
        u1 /= np.linalg.norm(u1)
        if nspin == 1:
            u = 0.01 * u1
            rhs = np.einsum("ij,ij->", vmat0, u)
        else:
            # independent spin components
            d2 = rng.uniform(-1, 1, size=(nao, nao))
            u2 = 0.5 * (d2 + d2.T)
            u2 /= np.linalg.norm(u2)
            u = 0.01 * np.stack([u1, u2])
            rhs = np.einsum("sij,sij->", vmat0, u)
        ep, _ = _xc_energy_and_vmat(ni, mol, grids, dm0 + delta * u, nspin)
        em, _ = _xc_energy_and_vmat(ni, mol, grids, dm0 - delta * u, nspin)
        lhs = (ep - em) / (2 * delta)
        results.append((lhs, rhs))
    return results


def assert_fd_consistent(results, rtol=3e-6, atol=1e-9, label=""):
    for i, (lhs, rhs) in enumerate(results):
        denom = max(abs(lhs), abs(rhs), atol)
        rel = abs(lhs - rhs) / denom
        assert rel < rtol or abs(lhs - rhs) < atol, (
            f"{label} FD mismatch dir {i}: dE/dlam={lhs:.10e} "
            f"Tr(V.u)={rhs:.10e} rel={rel:.3e}"
        )


# ---------------------------------------------------------------------------
# Explicit A-tensor oracle for constant alpha (T1)
# ---------------------------------------------------------------------------


def build_hybrid_oracle(mol, grids, dm_eff, hyb_settings):
    """Explicit construction of the hybrid feature and K contribution.

    Conventions (matching descriptors._hyb_desc_getter and the formalism):
        feat_g   = -1/4 chi_g^T dm_eff A_g dm_eff chi_g     (checked below)
        dfeat/d(dm_eff)_{mu nu} = -1/4 (chi (X) G + G (X) chi),
            G_g = A_g dm_eff chi_g
    Returns (feat_oracle, feat_code, vk_oracle_unit) where vk_oracle_unit is
    Sum_g w_g dfeat_g/d(dm_eff) -- i.e. the K contribution PER UNIT WEIGHT
    (multiply by xmix*alpha and the dm_eff->dm chain factor outside).
    """
    from ciderpress.pyscf.descriptors import _hyb_desc_getter

    feat_code, a_tensor = _hyb_desc_getter(
        mol, grids, dm_eff, hyb_settings, return_a_tensor=True
    )
    feat_code = feat_code[0]
    ao = pyscf_numint.eval_ao(mol, grids.coords)  # (ngrids, nao)
    F = ao @ dm_eff  # (ngrids, nao)
    G = np.einsum("gnl,gl->gn", a_tensor, F)  # (ngrids, nao)
    # feat = -1/4 F^T A F (both indices dm-contracted; ek = -1/2 F^T A F
    # in sgx_tools, feat = 0.5*ek)
    feat_oracle = -0.25 * np.einsum("gn,gn->g", F, G)
    w = grids.weights
    # Sum_g w_g * (-1/4) * (chi (X) G + G (X) chi)
    half = np.einsum("g,gm,gn->mn", w, ao, -0.25 * G)
    vk_unit = half + half.T
    return feat_oracle, feat_code, vk_unit
