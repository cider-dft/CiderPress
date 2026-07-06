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
Local-hybrid SCF machinery for CIDER functionals (PySCF layer).

DEFINITIVE CONVENTIONS AND DERIVATION
=====================================

Feature (training/SCF contract, see also descriptors._hyb_desc_getter):

    dm_eff   = (2/nspin) * P_s        (P_total for RKS; 2*P_s per spin, UKS)
    F_g      = dm_eff . chi(r_g)
    ek_g     = -1/2 F_g^T A_g F_g     (SGX seminumerical exchange density)
    feat_g   = -1/2 ek_g = +1/4 F_g^T A_g F_g  >= 0   (raw hybrid feature)

    A_{nu lambda}(r_g) = int chi_nu(r') chi_lambda(r') / |r' - r_g| dr'
        (density-independent 3-center integrals, pyscf.sgx batch_nuc)

The LDA-normalized feature feat/(LDA_FACTOR * rho^(4/3)) is NEGATIVE
(UEG value -1), matching the trained models' transform design
(VMap(gamma<0), "pole far right").

Energy: the ML evaluator (baselines.exx_energy_baseline) recovers the
physical exchange energy density e_x = -feat and multiplies it by the
learned mixing function alpha(features); with the model's spin-sum
convention (1/nspin per spin channel in SEP mode):

    E_hyb = xmix * sum_s (1/nspin) * sum_g w_g alpha_s(g) e_x[s](g)

Potential (two pieces):
1. Multiplicative semilocal terms from d(alpha)/d(features) flow through
   the standard CIDER feature chains (SL/NLDF/SDMX plans + normalizers).
2. The nonlocal K-matrix term from d(feat)/dP. The evaluator chain
   returns the per-spin K weight

    W_s(g) = dE/d(feat_s(g))   ("dalpha_deps"; alpha_eff is implicit:
                                for E = ... + alpha(feat)*e_x(feat),
                                W = d/dfeat[alpha * e_x] carries both
                                the alpha and the alpha' * e_x terms)

   and the unsymmetrized feature derivative is
    d(feat_g)/d(dm_eff)_{mu nu} = +1/2 chi_mu(g) (A_g F_g)_nu ,
   so each grid block contributes (build_k_matrix_block)
    K_{mu nu} += +1/2 sum_g w_g W(g) chi_mu(g) (A_g F_g)_nu .

Finalization factors: the block accumulation above is unsymmetrized,
and the caller (nr_rks/nr_uks) applies vmat = lib.hermi_sum(vmat)
AFTER the K contribution is added, which doubles it. Writing
sym(K) = (K + K^T)/2 and using d(dm_eff)/dP_s = 2/nspin, the physical
contribution must be dE/dP_s = (2/nspin) * (2/nspin)... -- rather than
re-deriving symbol-by-symbol here, the operative identity (validated at
~1e-10 by the A-tensor oracle, FD-consistency, RKS==UKS and
incore==blockwise tests in tests/test_local_hybrid.py) is:

    pre-hermi_sum addition = 0.25 * nspin * (K_s + K_s^T)
      RKS  (nspin=1): 0.5 * sym(K)   -> net sym(K) after hermi_sum
      UKS  (nspin=2): 1.0 * sym(K_s) -> net 2*sym(K_s) after hermi_sum

  The evaluator's 1/nspin spin-sum in W_s combines with these factors to
  make the closed-shell RKS == UKS identity exact.

A-tensor modes:
- "incore": one batch_nuc pass computes eps_exx AND stores the full A
  tensor (ngrids*nao^2 doubles), which is REUSED for both spins (A is
  density-independent) and for the per-block K accumulation.
- "blockwise": eps_exx is computed without storing A; the K matrix is
  assembled in a second lean batch_nuc sweep after the ML weights are
  known. Peak memory O(blk*nao^2) instead of O(ngrids*nao^2).
- "auto": incore if the A tensor fits in max_memory.
Select via CiderNumIntMixin.hyb_atensor_mode.
"""

import numpy as np
from pyscf import lib

__all__ = ["HybridSCFHelper"]


class HybridSCFHelper:
    """All local-hybrid state and logic for one nr_rks/nr_uks call.

    nspin-parameterized so the RKS and UKS code paths in numint.py share
    one implementation. Usage sequence:

        hyb = HybridSCFHelper(ni, mol, grids, dm_effs, nset, max_memory)
        hyb.compute_eps_exx()
        ... per block:
            feat = hyb.feat_block(iset, ip0, ip1)      # -> eval_xc_cider
            hyb.store_weight(iset, ip0, ip1, vxc_hyb)  # K weight
            hyb.accumulate_k_block(iset, ip0, ip1, ao, weight)  # incore
        hyb.finalize_into(vmat)
    """

    def __init__(self, ni, mol, grids, dm_effs, nset, max_memory):
        """
        Args:
            ni: CIDER numint (provides hyb_plan, settings, timer,
                hyb_atensor_mode, _hyb_cache)
            dm_effs: list over spins of per-set effective density
                matrices, dm_effs[s][iset] = (2/nspin) * P_s[iset].
                RKS: [[P_total, ...]]; UKS: [[2*Pa,...], [2*Pb,...]].
        """
        self.ni = ni
        self.mol = mol
        self.grids = grids
        self.dm_effs = dm_effs
        self.nspin = len(dm_effs)
        self.nset = nset
        self.max_memory = max_memory
        self.ngrids = grids.coords.shape[0]
        self.nao = dm_effs[0][0].shape[-1]
        self.plan = ni.hyb_plan
        self.settings = ni.settings.hyb_settings
        self.sgx_cache = getattr(ni, "_hyb_cache", None)
        if self.sgx_cache is None:
            self.sgx_cache = {}

        mode = getattr(ni, "hyb_atensor_mode", "auto")
        if mode == "auto":
            mem_gb = self.plan.estimate_memory_gb(self.ngrids, self.nao)
            self.use_blockwise = mem_gb > max_memory / 1000.0
            if self.use_blockwise:
                lib.logger.debug(
                    mol,
                    "Using blockwise hybrid mode: estimated A-tensor "
                    f"memory {mem_gb:.2f} GB > {max_memory / 1000.0:.2f} GB",
                )
        elif mode == "incore":
            self.use_blockwise = False
        elif mode == "blockwise":
            self.use_blockwise = True
        else:
            raise ValueError(f"invalid hyb_atensor_mode: {mode}")

        self.eps_exx = np.zeros((self.nspin, nset, self.ngrids))
        self.weights = np.zeros((self.nspin, nset, self.ngrids))
        self.a_tensor = None
        self.k_contrib = [
            [np.zeros((self.nao, self.nao)) for _ in range(nset)]
            for _ in range(self.nspin)
        ]

    def _timer(self, action, name):
        timer = getattr(self.ni, "timer", None)
        if timer is not None:
            getattr(timer, action)(name)

    def _compute_feat(self, dm_eff, return_a=False):
        """Raw hybrid feature (and optionally the A tensor) via the shared
        training/SCF descriptor getter."""
        from ciderpress.pyscf.descriptors import _hyb_desc_getter

        result = _hyb_desc_getter(
            self.mol,
            self.grids,
            dm_eff,
            self.settings,
            sgx_cache=self.sgx_cache,
            return_a_tensor=return_a,
        )
        if return_a:
            return result[0][0], result[1]
        return result[0], None

    def compute_eps_exx(self):
        """Compute the raw hybrid feature on the full grid for all spins
        and sets. In incore mode the A tensor is computed ONCE (it is
        density-independent) and reused for every spin/set K build."""
        self._timer("start", "hybrid exx")
        for s in range(self.nspin):
            for i in range(self.nset):
                need_a = not self.use_blockwise and self.a_tensor is None
                feat, a = self._compute_feat(self.dm_effs[s][i], return_a=need_a)
                if need_a:
                    self.a_tensor = a
                self.eps_exx[s, i] = feat
        self._timer("stop", "hybrid exx")

    def feat_block(self, iset, ip0, ip1):
        """Hybrid feature slice for eval_xc_cider.

        Returns (1, nblk) for RKS (closed-shell squeeze inside
        eval_xc_cider) and (nspin, 1, nblk) for UKS.
        """
        if self.nspin == 1:
            return self.eps_exx[0, iset, ip0:ip1][None, :]
        return self.eps_exx[:, iset, ip0:ip1][:, None, :]

    def store_weight(self, iset, ip0, ip1, vxc_hyb):
        """Store the per-spin K weight W = dE/d(feat) for this block.

        Contract: vxc_hyb has shape (1, nblk) for RKS (post-squeeze) and
        (nspin, 1, nblk) for UKS.
        """
        nblk = ip1 - ip0
        if self.nspin == 1:
            assert vxc_hyb.shape == (1, nblk), (
                f"vxc_hyb shape {vxc_hyb.shape} != (1, {nblk})"
            )
            self.weights[0, iset, ip0:ip1] = vxc_hyb[0]
        else:
            assert vxc_hyb.shape == (self.nspin, 1, nblk), (
                f"vxc_hyb shape {vxc_hyb.shape} != ({self.nspin}, 1, {nblk})"
            )
            for s in range(self.nspin):
                self.weights[s, iset, ip0:ip1] = vxc_hyb[s, 0]

    def accumulate_k_block(self, iset, ip0, ip1, ao, weight):
        """Accumulate the K contribution for this block (incore mode).
        No-op in blockwise mode (K is assembled in finalize_into)."""
        if self.use_blockwise or self.a_tensor is None:
            return
        self._timer("start", "hybrid K build")
        for s in range(self.nspin):
            w_blk = self.weights[s, iset, ip0:ip1]
            self.k_contrib[s][iset] += self.plan.build_k_matrix_block(
                ao,
                self.dm_effs[s][iset],
                self.a_tensor[ip0:ip1],
                w_blk,
                weight,
            )
        self._timer("stop", "hybrid K build")

    def finalize_into(self, vmat):
        """Assemble blockwise K if needed, then symmetrize and add the K
        contribution to vmat with the derivation-backed prefactors.
        Called BEFORE the caller's lib.hermi_sum(vmat) (which doubles the
        contribution; see module docstring)."""
        if self.use_blockwise:
            from ciderpress.external.sgx_tools import build_k_matrix_blockwise

            self._timer("start", "hybrid K blockwise")
            sgx_obj = self.sgx_cache.get("sgx_blockwise")
            if sgx_obj is None:
                from pyscf.sgx import sgx

                sgx_obj = sgx.SGX(self.mol)
                sgx_obj.grids = self.grids
                self.sgx_cache["sgx_blockwise"] = sgx_obj
            for s in range(self.nspin):
                for i in range(self.nset):
                    if self.weights[s, i].any():
                        self.k_contrib[s][i] = build_k_matrix_blockwise(
                            sgx_obj,
                            self.dm_effs[s][i],
                            self.weights[s, i],
                            grids_weights=self.grids.weights,
                            max_memory=self.max_memory,
                        )
            self._timer("stop", "hybrid K blockwise")

        # pre-hermi_sum addition = 0.25 * nspin * (K + K^T); see module
        # docstring for the derivation and the tests that enforce it
        for s in range(self.nspin):
            for i in range(self.nset):
                k = self.k_contrib[s][i]
                k += k.T
                k *= 0.25 * self.nspin
                if self.nspin == 1:
                    vmat[i] += k
                else:
                    vmat[s, i] += k
        return vmat
