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
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

"""
=====================================================================
CIDER Numerical Integration Module - Architecture Overview
=====================================================================

This module implements numerical integration for CIDER functionals with support for:
- Semi-local (SL) functionals
- SDMX (Semi-local Density Matrix eXpansion)
- Hybrid functionals with local mixing (Laqua et al. 2018)
- NLDF (Non-Local Density Functionals)
- All combinations of the above

COMPUTATIONAL FLOW
==================

The computation follows different strategies depending on features:

1. WITHOUT NLDF (Standard Path):
   - Single pass through grid blocks
   - For each block:
     a) Evaluate AO basis functions
     b) Compute electron density ρ(r)
     c) Compute SDMX features (if needed)
     d) Compute hybrid features ε_x^ex (if needed)
     e) Evaluate XC functional → exc, vxc
     f) Build K matrix contribution for hybrids (if needed)
     g) Contract vxc with AO functions → vmat
   - Finalize K matrix and add to vmat

2. WITH NLDF (Two-Pass Strategy):
   First Pass:
   - Loop through all grid blocks
   - Compute and store full density ρ(r) for entire grid
   - Compute NLDF features from full density

   Second Pass:
   - Use extra_block_loop (includes extra AOs for NLDF)
   - For each block:
     a) Retrieve pre-computed density
     b) Get NLDF features for block
     c) Compute SDMX/hybrid features
     d) Evaluate XC functional
     e) Store weighted potentials
   - After all blocks: Apply NLDF potential
   - Contract to vmat with K matrix contributions

BLOCK LOOPS
===========
- block_loop: Standard iteration over grid blocks
  * Returns: (idm, ip0, ip1, ao, mask, weight, coords)
  * ip0, ip1: indices for block in full grid
  * Used for density computation and non-NLDF evaluation

- extra_block_loop: Special loop for NLDF
  * Includes extra AO functions needed for NLDF
  * Returns: (mask, weight, coords)
  * Used in second pass of NLDF evaluation

RKS vs UKS DIFFERENCES
======================
nr_rks (Restricted):
- Single set of orbitals (spin-paired)
- Density matrix shape: (nset, nao, nao)
- Features computed once for total density
- Hybrid: Uses total density P_total = 2*P_α
- K matrix: Single contribution, factor 0.5

nr_uks (Unrestricted):
- Separate α and β spin orbitals
- Density matrices: dma (alpha), dmb (beta)
- Features computed separately per spin
- Hybrid: Uses 2*P_α and 2*P_β separately
- K matrix: Separate α/β contributions, factor 1.0

HYBRID FUNCTIONAL IMPLEMENTATION
================================
Following Laqua et al., J. Chem. Theory Comput. 2018, 14, 3451-3458:

1. Exact Exchange Energy Density:
   ε_X^ex(r) = -1/2 Σ_μνλσ χ_μ(r) P_μν A_νλ(r) P_λσ χ_σ(r)
   where A_νλ(r) = ∫ χ_ν(r') χ_λ(r') / |r - r'| dr'

2. Local Mixing Function:
   α(r) = f(ε_X^ex(r)) - ML model output

3. K Matrix (nonlocal contribution to Fock matrix):
   K_μν = ∫ (∂α/∂ε_X^ex)(r) A_νλ(r) P_λσ χ_σ(r) χ_μ(r) dr

The implementation:
- Pre-computes ε_X^ex and A tensor on full grid
- During block loop: evaluates ML model → α(r), ∂α/∂ε_X^ex
- Builds K matrix contributions block by block
- Finalizes with symmetrization: K → (K + K^T)/2

FEATURE TYPES
=============
1. Semi-local (SL): ρ, ∇ρ, τ (kinetic energy density)
2. SDMX: Density matrix expansion features
3. Hybrid: ε_X^ex (exact exchange energy density)
4. NLDF: Non-local density features via convolutions

Each feature type has:
- Forward pass: compute features from density/orbitals
- Backward pass: apply potential (derivative of E_xc w.r.t. features)
"""

import numpy as np
from ase.utils.timing import Timer
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import ALIGNMENT_UNIT, BLKSIZE, NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse, _scale_ao_sparse, eval_ao

import ciderpress.pyscf.frac_lapl as nlof
from ciderpress.dft.plans import (
    FracLaplPlan,
    HybridPlan,
    SemilocalPlan,
    get_rho_tuple_with_grad_cross,
    vxc_tuple_to_array,
)
from ciderpress.dft.settings import FeatureSettings
from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2
from ciderpress.pyscf.nldf_convolutions import PyscfNLDFGenerator
from ciderpress.pyscf.sdmx import EXXSphGenerator

DEFAULT_RHOCUT = 1e-9


def _tau_dot_sparse(
    bra, ket, wv, nbins, screen_index, pair_mask, ao_loc, out=None, aow=None
):
    """Similar to _tau_dot, while sparsity is explicitly considered. Note the
    return may have ~1e-13 difference to _tau_dot.
    """
    nao = bra.shape[1]
    if out is None:
        out = np.zeros((nao, nao), dtype=bra.dtype)
    hermi = 1
    aow = _scale_ao_sparse(ket[1], wv, screen_index, ao_loc, out=aow)
    _dot_ao_ao_sparse(
        bra[1], aow, None, nbins, screen_index, pair_mask, ao_loc, hermi, out
    )
    aow = _scale_ao_sparse(ket[2], wv, screen_index, ao_loc, out=aow)
    _dot_ao_ao_sparse(
        bra[2], aow, None, nbins, screen_index, pair_mask, ao_loc, hermi, out
    )
    aow = _scale_ao_sparse(ket[3], wv, screen_index, ao_loc, out=aow)
    _dot_ao_ao_sparse(
        bra[3], aow, None, nbins, screen_index, pair_mask, ao_loc, hermi, out
    )
    return out


def _get_sdmx_orbs(ni, mol, coords, ao):
    if hasattr(ni, 'timer'):
        ni.timer.start("sdmx ao")
    if ni.has_sdmx and ni.sdmxgen.fast:
        if ao is None:
            sdmx_ao = eval_ao(mol, coords, deriv=0)
        else:
            sdmx_ao = ao[0] if ao.ndim == 3 else ao
        sdmx_cao = ni.sdmxgen.get_cao(mol, coords, save_buf=True)
    elif ni.has_sdmx:
        sdmx_ao, sdmx_cao = ni.sdmxgen.get_orb_vals(mol, coords, save_buf=True)
    else:
        sdmx_ao, sdmx_cao = None, None
    if hasattr(ni, 'timer'):
        ni.timer.stop("sdmx ao")
    return sdmx_ao, sdmx_cao


def nr_rks(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    """
    Unified nr_rks implementation that supports all features:
    - Semi-local (SL)
    - SDMX
    - Hybrid
    - NLDF
    - All combinations of the above

    Args:
        ni (CiderNumInt):
        mol (pyscf.gto.Mole):
        grids (pyscf.dft.gen_grid.Grids):
        xc_code:
        dms:
        relativity:
        hermi:
        max_memory:
        verbose:

    Returns:

    """
    if hasattr(ni, 'timer'):
        ni.timer.start("nr_rks")
    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 1)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

    if dms.ndim == 2:
        dms = dms[None, :, :]

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))

    # Initialize for NLDF if needed
    rho_full = None
    wv_full = None
    vxc_nldf_full = None
    if ni.has_nldf:
        if not hasattr(grids, "grids_indexer"):
            raise ValueError("Grids object must have indexer for NLDF evaluation")

        nrho = 5 if xctype == "MGGA" else 4
        if not ni.settings.nlof_settings.is_empty:
            nrho += ni.settings.nlof_settings.nrho

        rho_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")
        wv_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")

        num_nldf = ni.settings.nldf_settings.nfeat
        vxc_nldf_full = np.zeros(
            (nset, num_nldf, grids.weights.size), dtype=np.float64, order="C"
        )

    # Compute exact exchange data for hybrid functionals if needed
    eps_exx, a_tensor = None, None
    alpha_ml = None
    dalpha_deps = None
    K_contrib = None
    use_blockwise_hybrid = False
    sgx_cache = {}  # Cache for SGX data

    if ni.hyb_plan is not None:
        if hasattr(ni, 'timer'):
            ni.timer.start("hybrid exx")

        # Check memory requirement for A tensor
        memory_gb = ni.hyb_plan.estimate_memory_gb(grids.coords.shape[0], nao)
        memory_threshold_gb = max_memory / 1000  # Convert MB to GB

        if memory_gb > memory_threshold_gb:
            # Use blockwise mode to avoid OOM
            use_blockwise_hybrid = True
            # Use logger.debug instead of print to reduce verbosity
            lib.logger.debug(mol, f"Using blockwise hybrid mode: estimated A-tensor memory {memory_gb:.2f} GB > threshold {memory_threshold_gb:.2f} GB")

            # Compute exact exchange energy density on the full grid (but no A tensor)
            eps_exx_all = np.zeros((nset, grids.coords.shape[0]))

            # Get only exchange energy density for all density matrices
            for i in range(nset):
                eps_exx_all[i] = ni.hyb_plan.compute_eps_exx_only(
                    mol, grids, dms[i], ni.settings.hyb_settings,
                    sgx_cache=sgx_cache
                )

            a_tensor = None  # Will compute blockwise later
        else:
            # Use existing full computation mode
            use_blockwise_hybrid = False

            # Compute exact exchange energy density on the full grid
            # Following equation (2) from Laqua et al., J. Chem. Theory Comput. 2018:
            # ε_X^ex(r_g) = -1/2 * Σ_μνλσ χ_μ(r_g) P_μν [∫ χ_ν(r') χ_λ(r') / |r' - r_g| dr'] P_λσ χ_σ(r_g)
            eps_exx_all = np.zeros((nset, grids.coords.shape[0]))

            # Get A tensor and exact exchange for first density matrix
            eps_exx_all[0], a_tensor = ni.hyb_plan.compute_a_tensor_and_exx(
                mol, grids, dms[0], ni.settings.hyb_settings,
                sgx_cache=sgx_cache, return_a_tensor=True
            )

            # For additional density matrices, reuse cached SGX object
            for i in range(1, nset):
                eps_exx_all[i], _ = ni.hyb_plan.compute_a_tensor_and_exx(
                    mol, grids, dms[i], ni.settings.hyb_settings,
                    sgx_cache=sgx_cache, return_a_tensor=False
                )

        eps_exx = eps_exx_all

        # Initialize arrays to store ML model outputs
        alpha_ml = np.zeros((nset, grids.coords.shape[0]))
        dalpha_deps = np.zeros((nset, grids.coords.shape[0]))

        # Initialize K contributions (accumulated during grid loop)
        K_contrib = [np.zeros((nao, nao)) for _ in range(nset)]

        if hasattr(ni, 'timer'):
            ni.timer.stop("hybrid exx")

    def block_loop(ao_deriv, include_extra_ao=True):
        ip0 = 0  # Initialize ip0 for hybrid functional block indexing
        extra_ao = 0
        if include_extra_ao:
            if ni.has_sdmx:
                extra_ao += ni.sdmxgen.get_extra_ao(mol)
            if ni.has_nldf:
                extra_ao += ni.nldfgen.get_extra_ao()
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size  # Calculate end index for this block
            for i in range(nset):
                yield i, ip0, ip1, ao, mask, weight, coords
            ip0 = ip1

    # First pass: compute rho for all blocks (needed for NLDF)
    if ni.has_nldf:
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")
        ao_deriv = 1
        for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv, include_extra_ao=False):
            rho_full[i, :, ip0:ip1] = make_rho(i, ao, mask, xctype)

        # Compute NLDF features
        nldf_feat = []
        for idm in range(nset):
            nldf_feat.append(ni.nldfgen.get_features(rho_full[idm]))
    else:
        nldf_feat = None

    # Second pass: compute XC and accumulate
    # For NLDF case, use extra_block_loop; for non-NLDF, use regular block_loop
    if ni.has_nldf:
        # NLDF evaluation using extra_block_loop
        ip0 = 0
        extra_ao = 0
        if ni.has_sdmx:
            extra_ao += ni.sdmxgen.get_extra_ao(mol)
        if ni.has_nldf:
            extra_ao += ni.nldfgen.get_extra_ao()

        for mask, weight, coords in ni.extra_block_loop(
            mol, grids, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, None)

            for idm in range(nset):
                # Get density for this block
                rho = rho_full[idm, :, ip0:ip1]

                # Get SDMX features if needed
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dms[idm], mol, coords, ao=sdmx_ao, cao=sdmx_cao
                    )
                else:
                    sdmx_feat = None

                # Get hybrid features if needed
                if ni.hyb_plan is not None:
                    hyb_feat = eps_exx[idm][ip0:ip1]
                    hyb_feat = hyb_feat[None, :]  # Shape: (1, ngrids)
                else:
                    hyb_feat = None

                # Get NLDF features for this block
                nldf_feat_block = nldf_feat[idm][:, ip0:ip1]
                # Evaluate XC
                result = ni.eval_xc_cider(
                    xc_code,
                    rho,
                    nldf_feat_block,
                    sdmx_feat,
                    hyb_feat=hyb_feat,
                    deriv=1,
                    xctype=xctype,
                )
                exc = result[0]
                vxc, vxc_nldf, vxc_sdmx, vxc_hyb = result[1][:4]
                f_raw = result[4] if len(result) > 4 else None

                # Handle SDMX
                if ni.has_sdmx and vxc_sdmx is not None:
                    ni.sdmxgen.get_vxc_(vmat[idm], vxc_sdmx[0] * weight)

                # Handle NLDF potential
                if vxc_nldf is not None:
                    vxc_nldf_full[idm, :, ip0:ip1] = vxc_nldf * weight

                # Store weighted vxc for later contraction
                wv_full[idm, :, ip0:ip1] = weight * vxc

                # Accumulate energy and density
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += np.dot(den, exc)

                # Handle hybrid ML outputs and K matrix
                if ni.hyb_plan is not None and vxc_hyb is not None:
                    # Store derivative
                    dalpha_deps[idm, ip0:ip1] = vxc_hyb[0]

                    # Store ML output
                    if f_raw is not None:
                        if isinstance(f_raw, np.ndarray):
                            if f_raw.ndim == 2:
                                alpha_ml[idm, ip0:ip1] = f_raw[0, :]
                            elif f_raw.ndim == 1:
                                alpha_ml[idm, ip0:ip1] = f_raw[:]
                            else:
                                raise ValueError(f"Unexpected f_raw shape: {f_raw.shape}")
            ip0 = ip1

        # Apply NLDF potential
        for idm in range(nset):
            wv_full[idm, :, :] += ni.nldfgen.get_potential(vxc_nldf_full[idm])

    else:
        # Non-NLDF evaluation using regular block loop
        ao_deriv = 1
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")

        def non_nldf_block_loop():
            for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
                sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, ao)
                rho = make_rho(i, ao, mask, xctype)

                # Extract hybrid features for this block
                hyb_feat = None
                if ni.hyb_plan is not None:
                    # Slice the exact exchange data for this block
                    eps_block = eps_exx[i, ip0:ip1]  # Shape: (ngrids_block,)
                    hyb_feat = eps_block[None, :]  # Shape: (1, ngrids_block) for single feature

                if hasattr(ni, 'timer'):
                    ni.timer.start("sdmx fwd")
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dms[i],
                        mol,
                        coords,
                        ao=sdmx_ao,
                        cao=sdmx_cao,
                    )
                else:
                    sdmx_feat = None
                if hasattr(ni, 'timer'):
                    ni.timer.stop("sdmx fwd")
                if hasattr(ni, 'timer'):
                    ni.timer.start("xc cider")
                xc_result = ni.eval_xc_cider(
                    xc_code, rho, None, sdmx_feat, hyb_feat=hyb_feat, deriv=1, xctype=xctype
                )
                exc = xc_result[0]
                vxc, vxc_nldf, vxc_sdmx, vxc_hyb = xc_result[1]
                # Extract raw ML output (alpha) if available
                f_raw = xc_result[4] if len(xc_result) > 4 else None
                if hasattr(ni, 'timer'):
                    ni.timer.stop("xc cider")
                assert vxc_nldf is None

                # Build K matrix contribution for local hybrid if needed
                if ni.hyb_plan is not None and vxc_hyb is not None:
                    if hasattr(ni, 'timer'):
                        ni.timer.start("hybrid K build")

                    # Extract derivative from vxc_hyb (this is local contribution to d(E_xc)/d(eps_exx))
                    dalpha_deps[i, ip0:ip1] = vxc_hyb[0]

                    if f_raw is not None:
                        # f_raw contains the raw ML output (alpha values) for this block only
                        if isinstance(f_raw, np.ndarray):
                            if f_raw.ndim == 2:
                                alpha_ml[i, ip0:ip1] = f_raw[0, :]
                            elif f_raw.ndim == 1:
                                alpha_ml[i, ip0:ip1] = f_raw[:]
                            else:
                                raise ValueError(f"Unexpected f_raw shape: {f_raw.shape}")
                        elif isinstance(f_raw, list) and len(f_raw) > 0:
                            raw_output = f_raw[0]
                            if hasattr(raw_output, 'shape'):
                                alpha_ml[i, ip0:ip1] = raw_output.ravel()[:]
                            else:
                                raise ValueError(f"Unexpected raw output format: {type(raw_output)}")
                        else:
                            raise ValueError(f"Unexpected f_raw type: {type(f_raw)}")

                    # Build K matrix contribution
                    if not use_blockwise_hybrid and a_tensor is not None:
                        # Use pre-computed A tensor
                        K_block = ni.hyb_plan.build_k_matrix_block(
                            ao, dms[i], a_tensor[ip0:ip1],
                            dalpha_deps[i, ip0:ip1], weight
                        )
                        K_contrib[i] += K_block
                    # For blockwise mode, K matrix will be built after all dalpha_deps are collected

                    if hasattr(ni, 'timer'):
                        ni.timer.stop("hybrid K build")

                if hasattr(ni, 'timer'):
                    ni.timer.start("sdmx bwd")
                if ni.has_sdmx:
                    ni.sdmxgen.get_vxc_(vmat[i], vxc_sdmx[0] * weight)
                if hasattr(ni, 'timer'):
                    ni.timer.stop("sdmx bwd")
                den = rho[0] * weight
                nelec[i] += den.sum()

                excsum[i] += np.dot(den, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    # Contract weighted vxc to vmat
    buffers = None
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    v1 = np.zeros_like(vmat)

    if ni.has_nldf:
        # Contract from wv_full for NLDF case
        for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv, include_extra_ao=False):
            wv = wv_full[i, :, ip0:ip1]
            vmats, buffers = ni.contract_wv(
                ao,
                wv,
                nbins,
                mask,
                pair_mask,
                ao_loc,
                vmats=(vmat[i], v1[i]),
                buffers=buffers,
            )

            # Build K matrix for hybrid if needed (NLDF case)
            if ni.hyb_plan is not None and a_tensor is not None:
                if dalpha_deps[i, ip0:ip1].any():
                    if hasattr(ni, 'timer'):
                        ni.timer.start("hybrid K build")

                    # Build K matrix contribution using HybridPlan
                    K_block = ni.hyb_plan.build_k_matrix_block(
                        ao, dms[i], a_tensor[ip0:ip1],
                        dalpha_deps[i, ip0:ip1], weight
                    )
                    K_contrib[i] += K_block

                    if hasattr(ni, 'timer'):
                        ni.timer.stop("hybrid K build")
    else:
        # Contract directly for non-NLDF case
        # The non-NLDF block loop yields (i, ao, mask, wv) after computing everything
        for i, ao, mask, wv in non_nldf_block_loop():
            vmats, buffers = ni.contract_wv(
                ao,
                wv,
                nbins,
                mask,
                pair_mask,
                ao_loc,
                vmats=(vmat[i], v1[i]),
                buffers=buffers,
            )

    # Add K contribution and symmetrize
    # Build K matrix blockwise if using blockwise hybrid mode
    if use_blockwise_hybrid and ni.hyb_plan is not None:
        if hasattr(ni, 'timer'):
            ni.timer.start("hybrid K blockwise")

        # Build K matrix blockwise for each density matrix
        for i in range(nset):
            if dalpha_deps[i].any():  # Only if we have non-zero derivatives
                K_contrib[i] = ni.hyb_plan.build_k_matrix_blockwise(
                    mol, grids, dms[i], dalpha_deps[i],
                    max_memory=max_memory, sgx_cache=sgx_cache
                )

        if hasattr(ni, 'timer'):
            ni.timer.stop("hybrid K blockwise")

    if ni.hyb_plan is not None and K_contrib is not None:
        # Finalize K matrix with proper symmetrization and factors
        ni.hyb_plan.finalize_k_matrix(K_contrib, vmat, spin=None)

    vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
    vmat += v1

    if ni.has_sdmx:
        ni.sdmxgen._cached_ao_data = None

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    # if ni.has_sdmx:
    #    ni.sdmxgen.reset_buffers()
    if hasattr(ni, 'timer'):
        ni.timer.stop("nr_rks")
    return nelec, excsum, vmat


def nr_uks(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    """
    Unified nr_uks implementation that supports all features:
    - Semi-local (SL)
    - SDMX
    - Hybrid
    - NLDF
    - All combinations of the above

    Key differences from RKS:
    - Handles spin-polarized density matrices
    - Separate alpha and beta spin channels
    """
    if hasattr(ni, 'timer'):
        ni.timer.start("nr_uks")
    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 2)

    dma, dmb = numint._format_uks_dm(dms)
    if dma.ndim == 2:
        dma = dma[None, :, :]
        dmb = dmb[None, :, :]
        is_2d = True
    else:
        is_2d = False
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

    # Format density matrices for SDMX
    if ni.has_sdmx:
        # Create combined density matrices for SDMX (matching standard nr_uks)
        dm_for_sdmx = [np.stack([dma[i], dmb[i]]) for i in range(nset)]
    else:
        dm_for_sdmx = None

    nelec = np.zeros((2, nset))
    excsum = np.zeros(nset)
    vmat = np.zeros((2, nset, nao, nao))

    # Initialize for NLDF if needed
    rho_full_a = None
    rho_full_b = None
    wv_full = None
    vxc_nldf_full = None
    if ni.has_nldf:
        if not hasattr(grids, "grids_indexer"):
            raise ValueError("Grids object must have indexer for NLDF evaluation")

        nrho = 5 if xctype == "MGGA" else 4
        if not ni.settings.nlof_settings.is_empty:
            nrho += ni.settings.nlof_settings.nrho

        # For UKS, we need spin-up and spin-down densities
        rho_full_a = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")
        rho_full_b = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")
        wv_full = np.zeros((2, nset, nrho, grids.weights.size), dtype=np.float64, order="C")

        num_nldf = ni.settings.nldf_settings.nfeat
        vxc_nldf_full = np.zeros(
            (2, nset, num_nldf, grids.weights.size), dtype=np.float64, order="C"
        )

    # Initialize for hybrid if needed
    eps_exx_a, eps_exx_b, a_tensor_a, a_tensor_b = None, None, None, None
    alpha_ml = None
    dalpha_deps = None
    K_contrib = None
    use_blockwise_hybrid = False
    sgx_cache = {}  # Cache for SGX data

    if ni.hyb_plan is not None:
        if hasattr(ni, 'timer'):
            ni.timer.start("hybrid exx")

        # Check memory requirement for A tensor
        memory_gb = ni.hyb_plan.estimate_memory_gb(grids.coords.shape[0], nao)
        memory_threshold_gb = max_memory / 1000  # Convert MB to GB

        if memory_gb > memory_threshold_gb:
            # Use blockwise mode to avoid OOM
            use_blockwise_hybrid = True
            # Use logger.debug instead of print to reduce verbosity
            lib.logger.debug(mol, f"UKS: Using blockwise hybrid mode: estimated A-tensor memory {memory_gb:.2f} GB > threshold {memory_threshold_gb:.2f} GB")

            # Compute exact exchange energy density for both spins (but no A tensors)
            eps_exx_all_a = np.zeros((nset, grids.coords.shape[0]))
            eps_exx_all_b = np.zeros((nset, grids.coords.shape[0]))

            # Get only exchange energy density for all density matrices
            for i in range(nset):
                # For UKS, match training convention: each spin gets 2*P_spin
                eps_exx_all_a[i] = ni.hyb_plan.compute_eps_exx_only(
                    mol, grids, 2 * dma[i], ni.settings.hyb_settings,
                    sgx_cache=sgx_cache
                )
                eps_exx_all_b[i] = ni.hyb_plan.compute_eps_exx_only(
                    mol, grids, 2 * dmb[i], ni.settings.hyb_settings,
                    sgx_cache=sgx_cache
                )

            a_tensor_a = a_tensor_b = None  # Will compute blockwise later
        else:
            # Use existing full computation mode
            use_blockwise_hybrid = False

            # Compute exact exchange energy density for both spins
            eps_exx_all_a = np.zeros((nset, grids.coords.shape[0]))
            eps_exx_all_b = np.zeros((nset, grids.coords.shape[0]))

            # Get A tensor and exact exchange for first density matrix set
            # For UKS, match training convention: each spin gets 2*P_spin
            # This gives same features as RKS for closed-shell but different for open-shell
            eps_exx_all_a[0], a_tensor_a = ni.hyb_plan.compute_a_tensor_and_exx(
                mol, grids, 2 * dma[0], ni.settings.hyb_settings,
                sgx_cache=sgx_cache, return_a_tensor=True, is_uks=True
            )

            # For beta spin
            eps_exx_all_b[0], a_tensor_b = ni.hyb_plan.compute_a_tensor_and_exx(
                mol, grids, 2 * dmb[0], ni.settings.hyb_settings,
                sgx_cache=sgx_cache, return_a_tensor=True, is_uks=True
            )

            # For additional density matrices
            for i in range(1, nset):
                # Each spin gets features from its own doubled density matrix
                eps_exx_all_a[i], _ = ni.hyb_plan.compute_a_tensor_and_exx(
                    mol, grids, 2 * dma[i], ni.settings.hyb_settings,
                    sgx_cache=sgx_cache, return_a_tensor=False, is_uks=True
                )

                # Beta
                eps_exx_all_b[i], _ = ni.hyb_plan.compute_a_tensor_and_exx(
                    mol, grids, 2 * dmb[i], ni.settings.hyb_settings,
                    sgx_cache=sgx_cache, return_a_tensor=False, is_uks=True
                )

        eps_exx_a = eps_exx_all_a
        eps_exx_b = eps_exx_all_b

        # Initialize arrays for ML outputs
        alpha_ml = np.zeros((2, nset, grids.coords.shape[0]))
        dalpha_deps = np.zeros((2, nset, grids.coords.shape[0]))
        K_contrib = np.zeros((2, nset, nao, nao))

        if hasattr(ni, 'timer'):
            ni.timer.stop("hybrid exx")

    def block_loop(ao_deriv, include_extra_ao=True):
        ip0 = 0
        extra_ao = 0
        if include_extra_ao:
            if ni.has_sdmx:
                extra_ao += ni.sdmxgen.get_extra_ao(mol)
            if ni.has_nldf:
                extra_ao += ni.nldfgen.get_extra_ao()
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            for i in range(nset):
                yield i, ip0, ip1, ao, mask, weight, coords
            ip0 = ip1

    # First pass: compute rho for all blocks (needed for NLDF)
    if ni.has_nldf:
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")
        ao_deriv = 1
        # For NLDF density computation, don't include extra_ao (matching backup implementation)
        for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv, include_extra_ao=False):
            rho_full_a[i, :, ip0:ip1] = make_rhoa(i, ao, mask, xctype)
            rho_full_b[i, :, ip0:ip1] = make_rhob(i, ao, mask, xctype)

        # Compute NLDF features
        nldf_feat = []
        for idm in range(nset):
            # For UKS, compute features separately for each spin
            nldf_feat.append(
                np.stack(
                    [
                        ni.nldfgen.get_features(rho_full_a[idm], spin=0),
                        ni.nldfgen.get_features(rho_full_b[idm], spin=1),
                    ]
                )
            )
    else:
        nldf_feat = None

    # Second pass: compute XC and accumulate
    # For NLDF case, use extra_block_loop; for non-NLDF, use regular block_loop
    if ni.has_nldf:
        # NLDF evaluation using extra_block_loop
        ip0 = 0
        extra_ao = 0
        if ni.has_sdmx:
            extra_ao += ni.sdmxgen.get_extra_ao(mol)
        if ni.has_nldf:
            extra_ao += ni.nldfgen.get_extra_ao()

        for mask, weight, coords in ni.extra_block_loop(
            mol, grids, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, None)

            for idm in range(nset):
                # Get density for this block
                rho_a = rho_full_a[idm, :, ip0:ip1]
                rho_b = rho_full_b[idm, :, ip0:ip1]
                rho = np.array([rho_a, rho_b])

                # Get SDMX features if needed
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dm_for_sdmx[idm], mol, coords, ao=sdmx_ao, cao=sdmx_cao
                    )
                else:
                    sdmx_feat = None

                # Get hybrid features if needed
                if ni.hyb_plan is not None:
                    # For UKS, we have separate exchange densities for each spin
                    hyb_feat_a = eps_exx_a[idm][ip0:ip1]
                    hyb_feat_b = eps_exx_b[idm][ip0:ip1]
                    # Stack them: shape (2, 1, ngrids) - each spin has 1 feature
                    hyb_feat = np.stack([hyb_feat_a[None, :], hyb_feat_b[None, :]])
                else:
                    hyb_feat = None

                # Get NLDF features for this block
                nldf_feat_block = nldf_feat[idm][..., ip0:ip1]

                # Evaluate XC
                result = ni.eval_xc_cider(
                    xc_code,
                    rho,
                    nldf_feat_block,
                    sdmx_feat,
                    hyb_feat=hyb_feat,
                    deriv=1,
                    xctype=xctype,
                )
                exc = result[0]
                vxc, vxc_nldf, vxc_sdmx, vxc_hyb = result[1][:4]
                f_raw = result[4] if len(result) > 4 else None

                # Handle SDMX
                if ni.has_sdmx and vxc_sdmx is not None:
                    ni.sdmxgen.get_vxc_(vmat[:, idm], vxc_sdmx * weight)

                # Handle NLDF potential
                if vxc_nldf is not None:
                    vxc_nldf_full[:, idm, :, ip0:ip1] = vxc_nldf * weight

                # Store weighted vxc for later contraction
                wv_full[0, idm, :, ip0:ip1] = weight * vxc[0]
                wv_full[1, idm, :, ip0:ip1] = weight * vxc[1]

                # Accumulate energy and density
                den_a = rho[0, 0] * weight
                den_b = rho[1, 0] * weight
                nelec[0, idm] += den_a.sum()
                nelec[1, idm] += den_b.sum()
                excsum[idm] += np.dot(den_a + den_b, exc)

                # Handle hybrid ML outputs and K matrix
                if ni.hyb_plan is not None and vxc_hyb is not None:
                    # Store derivative for each spin
                    if vxc_hyb.ndim == 3 and vxc_hyb.shape[1] == 1:
                        dalpha_deps[0, idm, ip0:ip1] = vxc_hyb[0, 0, :]
                        dalpha_deps[1, idm, ip0:ip1] = vxc_hyb[1, 0, :]
                    else:
                        dalpha_deps[0, idm, ip0:ip1] = vxc_hyb[0] if vxc_hyb.ndim > 1 else vxc_hyb
                        dalpha_deps[1, idm, ip0:ip1] = vxc_hyb[1] if vxc_hyb.ndim > 1 else vxc_hyb

                    # Store ML output
                    if f_raw is not None:
                        if f_raw.ndim == 1:
                            # If ML model returns single output, use for both spins
                            alpha_ml[0, idm, ip0:ip1] = f_raw
                            alpha_ml[1, idm, ip0:ip1] = f_raw
                        elif f_raw.ndim == 2 and f_raw.shape[0] == 2:
                            # f_raw has shape (2, ngrids) for UKS
                            alpha_ml[0, idm, ip0:ip1] = f_raw[0]
                            alpha_ml[1, idm, ip0:ip1] = f_raw[1]
                        else:
                            raise ValueError(f"Unexpected f_raw shape: {f_raw.shape}")
            ip0 = ip1

        # Apply NLDF potential (spin must match the get_features calls
        # above so each channel uses its own cached derivatives)
        for idm in range(nset):
            wv_full[0, idm, :, :] += ni.nldfgen.get_potential(
                vxc_nldf_full[0, idm], spin=0
            )
            wv_full[1, idm, :, :] += ni.nldfgen.get_potential(
                vxc_nldf_full[1, idm], spin=1
            )

    else:
        # Non-NLDF evaluation using regular block loop
        ao_deriv = 1
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")

        def non_nldf_block_loop():
            for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
                sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, ao)
                rho_a = make_rhoa(i, ao, mask, xctype)
                rho_b = make_rhob(i, ao, mask, xctype)
                rho = np.array([rho_a, rho_b])

                # Extract hybrid features for this block
                hyb_feat = None
                if ni.hyb_plan is not None:
                    # For UKS, we have separate exchange densities for each spin
                    hyb_feat_a = eps_exx_a[i][ip0:ip1]
                    hyb_feat_b = eps_exx_b[i][ip0:ip1]
                    # Stack them: shape (2, 1, ngrids) - each spin has 1 feature
                    hyb_feat = np.stack([hyb_feat_a[None, :], hyb_feat_b[None, :]])

                if hasattr(ni, 'timer'):
                    ni.timer.start("sdmx fwd")
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dm_for_sdmx[i],
                        mol,
                        coords,
                        ao=sdmx_ao,
                        cao=sdmx_cao,
                    )
                else:
                    sdmx_feat = None
                if hasattr(ni, 'timer'):
                    ni.timer.stop("sdmx fwd")
                if hasattr(ni, 'timer'):
                    ni.timer.start("xc cider")
                xc_result = ni.eval_xc_cider(
                    xc_code, rho, None, sdmx_feat, hyb_feat=hyb_feat, deriv=1, xctype=xctype
                )
                exc = xc_result[0]
                vxc, vxc_nldf, vxc_sdmx, vxc_hyb = xc_result[1][:4]
                # Extract raw ML output (alpha) if available
                f_raw = xc_result[4] if len(xc_result) > 4 else None
                if hasattr(ni, 'timer'):
                    ni.timer.stop("xc cider")
                assert vxc_nldf is None

                # Build K matrix contribution for local hybrid if needed
                if hasattr(ni.settings, 'has_hyb') and ni.settings.has_hyb and vxc_hyb is not None:
                    if hasattr(ni, 'timer'):
                        ni.timer.start("hybrid K build")

                    # Extract derivative from vxc_hyb
                    if vxc_hyb.ndim == 3 and vxc_hyb.shape[1] == 1:
                        dalpha_deps[0, i, ip0:ip1] = vxc_hyb[0, 0, :]
                        dalpha_deps[1, i, ip0:ip1] = vxc_hyb[1, 0, :]
                    else:
                        dalpha_deps[0, i, ip0:ip1] = vxc_hyb[0] if vxc_hyb.ndim > 1 else vxc_hyb
                        dalpha_deps[1, i, ip0:ip1] = vxc_hyb[1] if vxc_hyb.ndim > 1 else vxc_hyb

                    # Extract alpha from ML model
                    if f_raw is not None:
                        if isinstance(f_raw, np.ndarray):
                            if f_raw.ndim == 1:
                                # Single output for both spins
                                alpha_ml[0, i, ip0:ip1] = f_raw
                                alpha_ml[1, i, ip0:ip1] = f_raw
                            elif f_raw.ndim == 2 and f_raw.shape[0] == 2:
                                # Separate outputs for each spin
                                alpha_ml[0, i, ip0:ip1] = f_raw[0]
                                alpha_ml[1, i, ip0:ip1] = f_raw[1]
                            else:
                                raise ValueError(f"Unexpected f_raw shape: {f_raw.shape}")

                    # Build K matrix contributions
                    if not use_blockwise_hybrid and a_tensor_a is not None:
                        # Use pre-computed A tensors
                        # Alpha spin
                        K_block_a = ni.hyb_plan.build_k_matrix_block(
                            ao, 2*dma[i], a_tensor_a[ip0:ip1],
                            dalpha_deps[0, i, ip0:ip1], weight
                        )
                        K_contrib[0, i] += K_block_a

                        # Beta spin
                        K_block_b = ni.hyb_plan.build_k_matrix_block(
                            ao, 2*dmb[i], a_tensor_b[ip0:ip1],
                            dalpha_deps[1, i, ip0:ip1], weight
                        )
                        K_contrib[1, i] += K_block_b
                    # For blockwise mode, K matrix will be built after all dalpha_deps are collected

                    if hasattr(ni, 'timer'):
                        ni.timer.stop("hybrid K build")

                if hasattr(ni, 'timer'):
                    ni.timer.start("sdmx bwd")
                if ni.has_sdmx:
                    ni.sdmxgen.get_vxc_(vmat[:, i], vxc_sdmx * weight)
                if hasattr(ni, 'timer'):
                    ni.timer.stop("sdmx bwd")
                den_a = rho[0, 0] * weight
                den_b = rho[1, 0] * weight
                nelec[0, i] += den_a.sum()
                nelec[1, i] += den_b.sum()
                excsum[i] += np.dot(den_a + den_b, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    # Contract weighted vxc to vmat
    buffers = None
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    v1 = np.zeros_like(vmat)

    if ni.has_nldf:
        # Contract from wv_full for NLDF case
        for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv, include_extra_ao=False):
            wv_a = wv_full[0, i, :, ip0:ip1]
            wv_b = wv_full[1, i, :, ip0:ip1]

            # Contract alpha
            vmats_a, buffers = ni.contract_wv(
                ao,
                wv_a,
                nbins,
                mask,
                pair_mask,
                ao_loc,
                vmats=(vmat[0, i], v1[0, i]),
                buffers=buffers,
            )
            # Contract beta
            vmats_b, buffers = ni.contract_wv(
                ao,
                wv_b,
                nbins,
                mask,
                pair_mask,
                ao_loc,
                vmats=(vmat[1, i], v1[1, i]),
                buffers=buffers,
            )

            # Build K matrix for hybrid if needed (NLDF case)
            if hasattr(ni.settings, 'has_hyb') and ni.settings.has_hyb and a_tensor_a is not None:
                # Check if we have non-zero derivatives for this block
                if dalpha_deps[0, i, ip0:ip1].any() or dalpha_deps[1, i, ip0:ip1].any():
                    if hasattr(ni, 'timer'):
                        ni.timer.start("hybrid K build")

                    # Build K matrix contributions using HybridPlan
                    # Alpha spin
                    if dalpha_deps[0, i, ip0:ip1].any():
                        K_block_a = ni.hyb_plan.build_k_matrix_block(
                            ao, 2*dma[i], a_tensor_a[ip0:ip1],
                            dalpha_deps[0, i, ip0:ip1], weight
                        )
                        K_contrib[0, i] += K_block_a

                    # Beta spin
                    if dalpha_deps[1, i, ip0:ip1].any():
                        K_block_b = ni.hyb_plan.build_k_matrix_block(
                            ao, 2*dmb[i], a_tensor_b[ip0:ip1],
                            dalpha_deps[1, i, ip0:ip1], weight
                        )
                        K_contrib[1, i] += K_block_b

                    if hasattr(ni, 'timer'):
                        ni.timer.stop("hybrid K build")
    else:
        # Contract directly for non-NLDF case
        # The non-NLDF block loop yields (i, ao, mask, wv) after computing everything
        for i, ao, mask, wv in non_nldf_block_loop():
            # Contract alpha
            vmats_a, buffers = ni.contract_wv(
                ao,
                wv[0],
                nbins,
                mask,
                pair_mask,
                ao_loc,
                vmats=(vmat[0, i], v1[0, i]),
                buffers=buffers,
            )
            # Contract beta
            vmats_b, buffers = ni.contract_wv(
                ao,
                wv[1],
                nbins,
                mask,
                pair_mask,
                ao_loc,
                vmats=(vmat[1, i], v1[1, i]),
                buffers=buffers,
            )

    # Build K matrix blockwise if using blockwise hybrid mode
    if use_blockwise_hybrid and ni.hyb_plan is not None:
        if hasattr(ni, 'timer'):
            ni.timer.start("hybrid K blockwise")

        # Build K matrix blockwise for each density matrix and spin
        for i in range(nset):
            # Alpha spin
            if dalpha_deps[0, i].any():  # Only if we have non-zero derivatives
                K_contrib[0, i] = ni.hyb_plan.build_k_matrix_blockwise(
                    mol, grids, 2 * dma[i], dalpha_deps[0, i],
                    max_memory=max_memory, sgx_cache=sgx_cache
                )

            # Beta spin
            if dalpha_deps[1, i].any():  # Only if we have non-zero derivatives
                K_contrib[1, i] = ni.hyb_plan.build_k_matrix_blockwise(
                    mol, grids, 2 * dmb[i], dalpha_deps[1, i],
                    max_memory=max_memory, sgx_cache=sgx_cache
                )

        if hasattr(ni, 'timer'):
            ni.timer.stop("hybrid K blockwise")

    # Add K contribution and symmetrize
    if ni.hyb_plan is not None and K_contrib is not None:
        # Finalize K matrix with proper symmetrization and factors for UKS
        ni.hyb_plan.finalize_k_matrix(K_contrib, vmat, spin='uks')

    vmat = lib.hermi_sum(vmat.reshape(-1, nao, nao), axes=(0, 2, 1)).reshape(
        2, nset, nao, nao
    )
    vmat += v1

    if ni.has_sdmx:
        ni.sdmxgen._cached_ao_data = None

    if isinstance(dma, np.ndarray) and is_2d:
        vmat = vmat[:, 0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = np.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    if hasattr(ni, 'timer'):
        ni.timer.stop("nr_uks")
    return nelec, excsum, vmat


class CiderNumIntMixin:

    sl_plan: SemilocalPlan
    fl_plan: FracLaplPlan
    sdmxgen: EXXSphGenerator
    nldfgen: PyscfNLDFGenerator
    settings: FeatureSettings
    xmix: float
    rhocut: float

    def nlc_coeff(self, xc_code):
        if self._nlc_coeff is not None:
            return self._nlc_coeff
        else:
            return super().nlc_coeff(xc_code)

    @property
    def settings(self):
        return self.mlxc.settings

    @property
    def has_sdmx(self):
        return self.settings.has_sdmx

    @property
    def has_nldf(self):
        return self.settings.has_nldf

    def build(self, mol=None):
        self.mol = mol
        self.timer = Timer()
        self.sl_plan = None
        self.fl_plan = None
        self.sdmxgen = None
        self.nldfgen = None

    def reset(self, mol=None):
        self.mol = mol
        self.sl_plan = None
        self.fl_plan = None
        self.sdmxgen = None
        self.nldfgen = None

    def initialize_feature_generators(self, mol, grids, nspin):
        self.sl_plan = SemilocalPlan(self.settings.sl_settings, nspin)
        self.fl_plan = FracLaplPlan(self.settings.nlof_settings, nspin)

        # Initialize HybridPlan if needed
        if self.settings.has_hyb:
            self.hyb_plan = HybridPlan(self.settings.hyb_settings, nspin)
        else:
            self.hyb_plan = None

        cond = self.sdmxgen is None
        cond = cond or self.mol != mol
        cond = cond or self.sdmxgen.plan.nspin != nspin
        cond = cond and self.settings.has_sdmx
        if cond:
            self.sdmxgen = self.sdmx_init.initialize_sdmx_generator(mol, nspin)
        self.mol = mol

    def eval_xc_cider(
        self,
        xc_code,
        rho,
        nldf_feat,
        sdmx_feat,
        hyb_feat=None,
        deriv=1,
        omega=None,
        xctype=None,
        verbose=None,
    ):
        xc_code = self.slxc
        rho = np.asarray(rho, order="C", dtype=np.double)
        if deriv > 1:
            raise NotImplementedError
        if rho.ndim == 2:
            closed = True
            rho = rho[None, :]
        else:
            closed = False
        nfeat = self.settings.nfeat
        ngrids = rho.shape[-1]
        nspin = rho.shape[0]
        vxc = np.zeros_like(rho)
        xctype = self._xc_type(xc_code)
        if xctype == "LDA":
            nvar = 1
        elif xctype == "GGA":
            nvar = 4
        elif xctype == "MGGA":
            nvar = 5
        else:
            exc = np.zeros(ngrids)
        if xctype in ["LDA", "GGA", "MGGA"]:
            if nspin == 1:
                rhotmp = np.ascontiguousarray(rho[0, :nvar])
                exc, vxc[0, :nvar] = self.eval_xc_eff(
                    xc_code,
                    rhotmp,
                    deriv=deriv,
                    omega=omega,
                    xctype=xctype,
                    verbose=verbose,
                )[:2]
            else:
                exc, vxc[:, :nvar] = self.eval_xc_eff(
                    xc_code,
                    rho[:, :nvar],
                    deriv=deriv,
                    omega=omega,
                    xctype=xctype,
                    verbose=verbose,
                )[:2]

        start = 0
        X0T = np.empty((nspin, nfeat, ngrids))
        has_sl = not self.settings.sl_settings.is_empty
        has_nldf = not self.settings.nldf_settings.is_empty
        has_nlof = not self.settings.nlof_settings.is_empty
        has_sdmx = not self.settings.sdmx_settings.is_empty
        has_hyb = not self.settings.hyb_settings.is_empty
        if has_sl:
            nfeat_tmp = self.settings.sl_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = self.sl_plan.get_feat(rho[:, :5])
            start += nfeat_tmp
        if has_nldf:
            if nldf_feat.ndim == 2:
                nldf_feat = nldf_feat[None, :]
            assert nldf_feat.shape[0] == nspin
            assert nldf_feat.ndim == 3
            nfeat_tmp = self.settings.nldf_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = nldf_feat
            start += nfeat_tmp
        if has_nlof:
            nfeat_tmp = self.settings.nlof_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = self.fl_plan.get_feat(rho)
            start += nfeat_tmp
        if has_sdmx:
            if sdmx_feat.ndim == 2:
                sdmx_feat = sdmx_feat[None, :]
            assert sdmx_feat.shape[0] == nspin
            assert sdmx_feat.ndim == 3
            nfeat_tmp = self.settings.sdmx_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = sdmx_feat
            start += nfeat_tmp
        if has_hyb:
            if hyb_feat is not None:
                if hyb_feat.ndim == 2:
                    hyb_feat = hyb_feat[None, :]
                assert hyb_feat.shape[0] == nspin, f"hyb_feat.shape[0]={hyb_feat.shape[0]} != nspin={nspin}"
                assert hyb_feat.ndim == 3
                nfeat_tmp = self.settings.hyb_settings.nfeat
                X0T[:, start : start + nfeat_tmp] = hyb_feat
                start += nfeat_tmp
        if start != nfeat:
            raise RuntimeError("nfeat mismatch, this should not happen!")

        X0TN = self.settings.normalizers.get_normalized_feature_vector(X0T)
        xmix = self.xmix
        f_raw = None
        if isinstance(self.mlxc, MappedXC):
            # For hybrid functionals, we need to get the raw ML output
            if has_hyb and hyb_feat is not None:
                exc_ml, dexcdX0TN_ml, f_raw_total, df_raw_total = self.mlxc(
                    X0TN, rhocut=self.rhocut, return_raw_ml_output=True
                )
                # Extract the raw output for this block
                if hasattr(f_raw_total, 'shape'):
                    if f_raw_total.ndim == 2 and f_raw_total.shape[0] == 1:
                        f_raw = f_raw_total[0, :]
                    elif f_raw_total.ndim == 1:
                        f_raw = f_raw_total
                    elif f_raw_total.ndim == 2 and f_raw_total.shape[0] == nspin:
                        # For UKS, keep the spin dimension
                        f_raw = f_raw_total
                    else:
                        # For other cases, flatten (maintains RKS compatibility)
                        f_raw = f_raw_total.ravel()
            else:
                exc_ml, dexcdX0TN_ml = self.mlxc(X0TN, rhocut=self.rhocut)
        elif isinstance(self.mlxc, MappedXC2):
            rho_tuple = get_rho_tuple_with_grad_cross(rho, is_mgga=True)
            exc_ml, dexcdX0TN_ml, vrho_tuple = self.mlxc(
                X0TN, rho_tuple, rhocut=self.rhocut
            )
            vxc[:] += xmix * vxc_tuple_to_array(rho, vrho_tuple)
        else:
            raise TypeError("mlxc must be MappedXC or MappedXC2")
        exc_ml *= xmix
        dexcdX0TN_ml *= xmix
        vxc_ml = self.settings.normalizers.get_derivative_wrt_unnormed_features(
            X0T, dexcdX0TN_ml
        )
        exc[:] += exc_ml / (rho[:, 0].sum(axis=0) + 1e-16)

        start = 0
        if has_sl:
            nfeat_tmp = self.settings.sl_settings.nfeat
            self.sl_plan.get_vxc(
                rho[:, :5], vxc_ml[:, start : start + nfeat_tmp], vxc=vxc[:, :5]
            )
            start += nfeat_tmp
        if has_nldf:
            nfeat_tmp = self.settings.nldf_settings.nfeat
            vxc_nldf = vxc_ml[:, start : start + nfeat_tmp]
            start += nfeat_tmp
        else:
            vxc_nldf = None
        if has_nlof:
            nfeat_tmp = self.settings.nlof_settings.nfeat
            self.fl_plan.get_vxc(vxc_ml[:, start : start + nfeat_tmp], vxc=vxc)
            start += nfeat_tmp
        if has_sdmx:
            nfeat_tmp = self.settings.sdmx_settings.nfeat
            vxc_sdmx = vxc_ml[:, start : start + nfeat_tmp]
            start += nfeat_tmp
        else:
            vxc_sdmx = None
        if has_hyb:
            if hyb_feat is not None:
                nfeat_tmp = self.settings.hyb_settings.nfeat
                vxc_hyb = vxc_ml[:, start : start + nfeat_tmp]
                start += nfeat_tmp
            else:
                vxc_hyb = None
        else:
            vxc_hyb = None
        if start != nfeat:
            raise RuntimeError("nfeat mismatch, this should not happen!")

        if closed:
            vxc = vxc[0]
            if vxc_hyb is not None:
                vxc_hyb = vxc_hyb[0]

        return exc, (vxc, vxc_nldf, vxc_sdmx, vxc_hyb), None, None, f_raw


class CiderNumInt(CiderNumIntMixin, numint.NumInt):
    def __init__(
        self, mlxc, slxc, nldf_init, sdmx_init, xmix=1.0, rhocut=None, nlc_coeff=None
    ):
        """

        Args:
            mlxc (MappedXC): Model for XC energy
            slxc (str): semilocal contribution to XC energy
            nldf_init (PySCFNLDFInitializer)
            sdmx_init (PySCFSDMXInitializer)
            xmix (float): Mixing fraction of ML functional
            rhocut (float): Low density cutoff for numerical stability
        """
        self.mlxc = mlxc
        self.slxc = slxc
        self.xmix = xmix
        self.rhocut = DEFAULT_RHOCUT if rhocut is None else rhocut
        self.mol = None
        self.nldf_init = nldf_init
        self.sdmx_init = sdmx_init
        self.sdmxgen = None
        self.nldfgen = None
        self.hyb_plan = None
        self._nlc_coeff = nlc_coeff
        super(CiderNumInt, self).__init__()

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    nr_rks = nr_rks
    nr_uks = nr_uks

    nr_rks_fxc = method_not_implemented
    nr_uks_fxc = method_not_implemented
    nr_rks_fxc_st = method_not_implemented
    cache_xc_kernel = method_not_implemented
    cache_xc_kernel1 = method_not_implemented

    def contract_wv(
        self,
        ao,
        wv,
        nbins,
        mask,
        pair_mask,
        ao_loc,
        buffers=None,
        vmats=None,
    ):
        if vmats is None:
            vmats = (None, None)
        if not wv.flags.c_contiguous:
            wv = np.ascontiguousarray(wv)
        vmat, v1 = vmats
        aow = buffers
        wv[0] *= 0.5
        aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
        vmat = _dot_ao_ao_sparse(
            ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=vmat
        )
        if len(wv) > 4:
            wv[4] *= 0.5
            v1 = _tau_dot_sparse(
                ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1, aow=aow
            )
        return (vmat, v1), aow

    def block_loop(
        self,
        mol,
        grids,
        nao=None,
        deriv=0,
        max_memory=2000,
        non0tab=None,
        blksize=None,
        buf=None,
        extra_ao=0,
    ):
        """Define this macro to loop over grids by blocks.
        Same as pyscf block_loop except extra_ao allows accounting for
        extra ao terms in blksize calculation.
        """
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = mol.nao
        ngrids = grids.coords.shape[0]
        comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(
                max_memory * 1e6 / (((comp + 1) * nao + extra_ao) * 8 * BLKSIZE)
            )
            blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
        assert blksize % BLKSIZE == 0

        if non0tab is None and mol is grids.mol:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = np.empty(
                ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
            )
            non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
        screen_index = non0tab

        # the xxx_sparse() functions require ngrids 8-byte aligned
        allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > numint.SWITCH_SIZE

        if buf is None:
            buf = numint._empty_aligned(comp * blksize * nao)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            mask = screen_index[ip0 // BLKSIZE :]
            ao = self.eval_ao(
                mol, coords, deriv=deriv, non0tab=mask, cutoff=grids.cutoff, out=buf
            )
            if not allow_sparse and not numint._sparse_enough(mask):
                # Unset mask for dense AO tensor. It determines which eval_rho
                # to be called in make_rho
                mask = None
            yield ao, mask, weight, coords


class _FLNumIntMixin:

    feat_plan = None
    settings: FeatureSettings = None

    nr_rks = nr_rks
    nr_uks = nr_uks

    nr_nlc_vxc = numint.nr_nlc_vxc
    nr_sap = nr_sap_vxc = numint.nr_sap_vxc

    make_mask = staticmethod(nlof.make_mask)
    eval_ao = staticmethod(numint.eval_ao)
    eval_kao = staticmethod(nlof.eval_kao)
    eval_rho = staticmethod(nlof.eval_rho)
    eval_rho1 = lib.module_method(nlof.eval_rho1, absences=["cutoff"])
    eval_rho2 = staticmethod(nlof.eval_rho2)
    get_rho = numint.get_rho

    def block_loop(
        ni,
        mol,
        grids,
        nao=None,
        deriv=0,
        max_memory=2000,
        non0tab=None,
        blksize=None,
        buf=None,
        extra_ao=None,
    ):
        """Define this macro to loop over grids by blocks. Expands on
        CiderNumInt.block_loop by also returning the fractional laplacian-
        like operators on the orbitals, as set up by the FracLaplSettings
        """
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = mol.nao
        ngrids = grids.coords.shape[0]
        comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
        flsettings = ni.settings.nlof_settings
        kcomp = flsettings.npow + 3 * flsettings.nd1
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(
                max_memory * 1e6 / (((comp + kcomp + 1) * nao + extra_ao) * 8 * BLKSIZE)
            )
            blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
        assert blksize % BLKSIZE == 0

        if non0tab is None and mol is grids.mol:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = np.empty(
                ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
            )
            non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
        screen_index = non0tab

        # the xxx_sparse() functions require ngrids 8-byte aligned
        allow_sparse = ngrids % ALIGNMENT_UNIT == 0

        if buf is None:
            buf = nlof._empty_aligned((comp + kcomp) * blksize * nao)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            mask = screen_index[ip0 // BLKSIZE :]
            ao_buf = np.ndarray((comp + kcomp, weight.size * nao), buffer=buf)
            ao = ni.eval_ao(
                mol, coords, deriv=deriv, non0tab=mask, cutoff=grids.cutoff, out=ao_buf
            )
            kao = ni.eval_kao(
                flsettings.slist,
                mol,
                coords,
                deriv=0,
                non0tab=None,
                cutoff=grids.cutoff,
                out=ao_buf[comp:],
                n1=flsettings.nd1,
            )
            if not allow_sparse and not nlof._sparse_enough(mask):
                # Unset mask for dense AO tensor. It determines which eval_rho
                # to be called in make_rho
                mask = None
            yield (ao, kao), mask, weight, coords

    def contract_wv(
        self,
        ao,
        wv,
        nbins,
        mask,
        pair_mask,
        ao_loc,
        buffers=None,
        vmats=None,
    ):
        if buffers is None:
            buffers = (None, None)
        if vmats is None:
            vmats = (None, None)
        aow1, aow2, vmat, v1 = nlof._odp_dot_sparse_(
            ao,
            wv,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            self.settings.nlof_settings,
            aow1=buffers[0],
            aow2=buffers[1],
            vmat=vmats[0],
            v1=vmats[1],
        )
        return (vmat, v1), (aow1, aow2)

    _gen_rho_evaluator = nlof.FLNumInt._gen_rho_evaluator


class _NLDFMixin:

    nr_rks = nr_rks
    nr_uks = nr_uks

    def extra_block_loop(
        self,
        mol,
        grids,
        max_memory=2000,
        non0tab=None,
        blksize=None,
        make_mask=False,
        extra_ao=None,
    ):
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ngrids = grids.coords.shape[0]
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory * 1e6 / ((1 + extra_ao) * 8 * BLKSIZE))
            blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
        assert blksize % BLKSIZE == 0

        if make_mask:
            if non0tab is None and mol is grids.mol:
                non0tab = grids.non0tab
            if non0tab is None:
                non0tab = np.empty(
                    ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
                )
                non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
            screen_index = non0tab
        else:
            screen_index = None

        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            if make_mask:
                mask = screen_index[ip0 // BLKSIZE :]
            else:
                mask = None
            yield mask, weight, coords


class NLOFNumInt(_FLNumIntMixin, CiderNumInt):
    pass


class NLDFNumInt(_NLDFMixin, CiderNumInt):

    grids = None

    def initialize_feature_generators(self, mol, grids, nspin):
        cond = self.nldfgen is None
        cond = cond or self.grids != grids
        cond = cond or self.mol != mol
        cond = cond or self.nldfgen.plan.nspin != nspin
        if cond:
            self.nldfgen = self.nldf_init.initialize_nldf_generator(
                mol, grids.grids_indexer, nspin
            )
            self.nldfgen.interpolator.set_coords(grids.coords)
        super().initialize_feature_generators(mol, grids, nspin)
        self.grids = grids


class NLDFNLOFNumInt(_NLDFMixin, _FLNumIntMixin, CiderNumInt):

    grids = None

    def initialize_feature_generators(self, mol, grids, nspin):
        cond = self.nldfgen is None
        cond = cond or self.grids != grids
        cond = cond or self.mol != mol
        cond = cond or self.nldfgen.plan.nspin != nspin
        if cond:
            self.nldfgen = self.nldf_init.initialize_nldf_generator(
                mol, grids.grids_indexer, nspin
            )
        super().initialize_feature_generators(mol, grids, nspin)
        self.grids = grids


class HybridNLDFNumInt(_NLDFMixin, CiderNumInt):
    """NumInt class that supports both NLDF and Hybrid features"""

    grids = None
    nr_rks = nr_rks
    nr_uks = nr_uks

    def initialize_feature_generators(self, mol, grids, nspin):
        """Initialize both NLDF and standard feature generators"""
        # Initialize NLDF generator
        cond = self.nldfgen is None
        cond = cond or self.grids != grids
        cond = cond or self.mol != mol
        cond = cond or self.nldfgen.plan.nspin != nspin
        if cond:
            self.nldfgen = self.nldf_init.initialize_nldf_generator(
                mol, grids.grids_indexer, nspin
            )
            self.nldfgen.interpolator.set_coords(grids.coords)

        # Initialize standard generators (including any hybrid setup)
        CiderNumInt.initialize_feature_generators(self, mol, grids, nspin)
        self.grids = grids
