#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Edits by Kyle Bystrom <kylebystrom@gmail.com>
#

import time

import numpy
from pyscf.sgx.sgx_jk import _gen_batch_nuc, _gen_jk_direct, lib, logger


def get_jk_densities(sgx, dm, hermi=1, direct_scf_tol=1e-13):
    t0 = time.monotonic(), time.time()
    mol = sgx.mol
    grids = sgx.grids
    gthrd = sgx.grids_thrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1, nao, nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(
            mol, "s2", True, True, direct_scf_tol, sgx._opt, pjs=False
        )
    logger.timer_debug1(mol, "sgX initialziation", *t0)

    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory * 1e6 / 8 / nao**2))))
    tnuc = 0, 0
    ej, ek = numpy.zeros((nset, ngrids)), numpy.zeros((nset, ngrids))
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto("GTOval", coords)
        wao = ao * grids.weights[i0:i1, None]

        fg = lib.einsum("gi,xij->xgj", wao, dms)
        fgnw = lib.einsum("gi,xij->xgj", ao, dms)
        mask = numpy.zeros(i1 - i0, dtype=bool)
        for i in range(nset):
            mask |= numpy.any(fg[i] > gthrd, axis=1)
            mask |= numpy.any(fg[i] < -gthrd, axis=1)

        inds = numpy.arange(i0, i1)

        numpy.einsum("xgu,gu->xg", fg, ao)
        rho = numpy.einsum("xgu,gu->xg", fgnw, ao)

        if sgx.debug:
            tnuc = tnuc[0] - time.monotonic(), tnuc[1] - time.time()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + time.monotonic(), tnuc[1] + time.time()
            jg = numpy.einsum("gij,xij->xg", gbn, dms)
            gv = lib.einsum("gvt,xgt->xgv", gbn, fgnw)
            gbn = None
        else:
            tnuc = tnuc[0] - time.monotonic(), tnuc[1] - time.time()
            jg, gv = batch_jk(mol, coords, dms, fgnw.copy(), None)
            tnuc = tnuc[0] + time.monotonic(), tnuc[1] + time.time()

        jg = numpy.sum(jg, axis=0)
        ej[:, inds] = jg * rho / 2.0
        for i in range(nset):
            ek[i, inds] = lib.einsum("gu,gu->g", fgnw[i], gv[i])

        jg = gv = None

    ek *= -0.5
    return ej, ek


def get_jk_densities_and_a_tensor(sgx, dm, hermi=1, direct_scf_tol=1e-13, return_a_tensor=False):
    """
    Get J/K densities with optional A tensor computation.
    
    Parameters
    ----------
    sgx : SGX object
        Semi-grid exchange object from PySCF
    dm : array-like
        Density matrix or matrices
    hermi : int, optional
        Hermiticity flag (default: 1)
    direct_scf_tol : float, optional
        Direct SCF tolerance (default: 1e-13)
    return_a_tensor : bool, optional
        If True, also return the three-center integrals A_tensor (default: False)
        
    Returns
    -------
    ej : array (nset, ngrids)
        Coulomb energy densities
    ek : array (nset, ngrids)
        Exchange energy densities  
    a_tensor : array (ngrids, nao, nao), optional
        Three-center integrals A_{gij} if return_a_tensor=True
    """
    t0 = time.monotonic(), time.time()
    mol = sgx.mol
    grids = sgx.grids
    gthrd = sgx.grids_thrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1, nao, nao)
    nset = dms.shape[0]

    # Always use batch_nuc when A tensor is needed
    if return_a_tensor or sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    if not sgx.debug and not return_a_tensor:
        batch_jk = _gen_jk_direct(
            mol, "s2", True, True, direct_scf_tol, sgx._opt, pjs=False
        )
    logger.timer_debug1(mol, "sgX initialziation", *t0)

    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory * 1e6 / 8 / nao**2))))
    tnuc = 0, 0
    ej, ek = numpy.zeros((nset, ngrids)), numpy.zeros((nset, ngrids))
    
    # Initialize A tensor storage if requested
    if return_a_tensor:
        a_tensor = numpy.zeros((ngrids, nao, nao))
    
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto("GTOval", coords)
        wao = ao * grids.weights[i0:i1, None]

        fg = lib.einsum("gi,xij->xgj", wao, dms)
        fgnw = lib.einsum("gi,xij->xgj", ao, dms)

        inds = numpy.arange(i0, i1)

        rho = numpy.einsum("xgu,gu->xg", fgnw, ao)

        if sgx.debug or return_a_tensor:
            tnuc = tnuc[0] - time.monotonic(), tnuc[1] - time.time()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + time.monotonic(), tnuc[1] + time.time()
            jg = numpy.einsum("gij,xij->xg", gbn, dms)
            gv = lib.einsum("gvt,xgt->xgv", gbn, fgnw)
            if return_a_tensor:
                a_tensor[i0:i1] = gbn
            gbn = None
        else:
            tnuc = tnuc[0] - time.monotonic(), tnuc[1] - time.time()
            jg, gv = batch_jk(mol, coords, dms, fgnw.copy(), None)
            tnuc = tnuc[0] + time.monotonic(), tnuc[1] + time.time()

        jg = numpy.sum(jg, axis=0)
        ej[:, inds] = jg * rho / 2.0
        for i in range(nset):
            ek[i, inds] = lib.einsum("gu,gu->g", fgnw[i], gv[i])

        jg = gv = None

    ek *= -0.5
    
    if return_a_tensor:
        return ej, ek, a_tensor
    else:
        return ej, ek


def build_k_matrix_blockwise(sgx, dm, dalpha_deps, grids_weights=None, max_memory=2000):
    """
    Build K matrix contributions blockwise without storing full A tensor.
    
    This function computes the K matrix contribution for local hybrid functionals
    by processing the grid in blocks, avoiding the memory issue of storing the
    full A tensor (ngrids x nao x nao).
    
    Parameters
    ----------
    sgx : SGX object
        Semi-grid exchange object from PySCF
    dm : array (nao, nao)
        Density matrix (already scaled: dm for RKS, 2*dm_spin for UKS)
    dalpha_deps : array (ngrids,)
        Derivative of alpha (mixing parameter) w.r.t. exchange energy density
    grids_weights : array (ngrids,), optional
        Grid weights. If None, will use sgx.grids.weights
    max_memory : float, optional
        Maximum memory in MB (default: 2000)
        
    Returns
    -------
    K_contrib : array (nao, nao)
        K matrix contribution (NOT symmetrized, factor of +0.5 already
        applied; see HybridPlan.build_k_matrix_block for the convention)
    """
    t0 = time.monotonic(), time.time()
    mol = sgx.mol
    grids = sgx.grids
    nao = mol.nao
    ngrids = grids.coords.shape[0]
    
    if grids_weights is None:
        grids_weights = grids.weights
    
    # Calculate safe block size considering A tensor memory
    # Memory needed per block: block_size * nao * nao * 8 bytes
    max_memory_bytes = max_memory * 1e6 - lib.current_memory()[0] * 1e6
    sblk = getattr(sgx, 'blockdim', 240)
    
    # Calculate block size based on available memory
    # We need memory for: A_tensor_block (blksize x nao x nao) + working arrays
    bytes_per_gridpoint = nao * nao * 8 * 1.5  # 1.5x for safety margin
    max_block_size = int(max_memory_bytes / bytes_per_gridpoint)
    blksize = min(ngrids, max(128, min(sblk, max_block_size)))
    
    # Align to BLKSIZE for efficiency (if BLKSIZE is defined)
    try:
        from pyscf.dft.gen_grid import BLKSIZE
        blksize = (blksize // BLKSIZE) * BLKSIZE
        blksize = max(BLKSIZE, blksize)
    except ImportError:
        pass
    
    logger.debug(mol, f"build_k_matrix_blockwise: ngrids={ngrids}, blksize={blksize}, nao={nao}")
    
    # Initialize K matrix contribution
    K_contrib = numpy.zeros((nao, nao))
    
    # Use batch_nuc for A tensor computation
    batch_nuc = _gen_batch_nuc(mol)
    
    tnuc = 0, 0
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        block_size = i1 - i0
        
        # Evaluate atomic orbitals at grid points
        ao = mol.eval_gto("GTOval", coords)  # Shape: (block_size, nao)
        
        # Compute A tensor for this block only
        tnuc = tnuc[0] - time.monotonic(), tnuc[1] - time.time()
        a_tensor_block = batch_nuc(mol, coords)  # Shape: (block_size, nao, nao)
        tnuc = tnuc[0] + time.monotonic(), tnuc[1] + time.time()
        
        # Build K matrix contribution for this block (all-BLAS; see
        # HybridPlan.build_k_matrix_block for the equations/conventions)
        F = numpy.dot(ao, dm.T)  # (block_size, nao)
        F *= (grids_weights[i0:i1] * dalpha_deps[i0:i1])[:, None]
        gv = numpy.matmul(a_tensor_block, F[:, :, None])[:, :, 0]
        K_contrib += numpy.dot(ao.T, gv)

        # Clear large arrays to free memory
        a_tensor_block = None
        gv = None
    
    # Factor +0.5: matches HybridPlan.build_k_matrix_block -- the raw
    # hybrid feature is feat = -0.5*ek = +0.25 F^T A F, so
    # d(feat_g)/dP_{mu nu} = +0.5 chi_mu (A F)_nu (unsymmetrized).
    K_contrib *= 0.5
    
    logger.timer_debug1(mol, f"build_k_matrix_blockwise ({ngrids} points, {blksize} blocksize)", *t0)
    logger.timer_debug1(mol, "  nuclear integral time", *tnuc)
    
    return K_contrib
