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
Second-order SCF (SOSCF / newton) support for CIDER functionals.

Two response-kernel strategies are available for the orbital-Hessian
vector products that pyscf's newton solver requests via mf.gen_response():

1. Proxy semilocal kernel (default). The CIDER numint implements
   cache_xc_kernel / nr_rks_fxc / nr_uks_fxc / nr_rks_fxc_st by delegating
   to a plain pyscf NumInt with a semilocal stand-in functional (see
   CiderNumIntMixin.get_fxc_proxy_xc). pyscf's stock _gen_rhf_response /
   _gen_uhf_response then work unmodified. Because the SCF gradient (Fock
   matrix) is exact, newton converges to the exact CIDER solution; the
   proxy only affects the step direction/rate. This mirrors how pyscf
   treats VV10/NLC in response functions (excluded with a warning).

2. Exact finite-difference response (opt-in, mf.fd_response = True).
   gen_fd_response() computes the XC part of the Hessian-vector product as
   a central finite difference of the full CIDER XC potential:
       fxc . dm1 ~= [vxc(dm0 + eps*u) - vxc(dm0 - eps*u)] / (2*eps)
   with u the hermitian part of dm1. This captures ALL feature
   nonlocality (NLDF, SDMX, and local-hybrid K contributions, since
   nr_rks/nr_uks fold them into vmat), at the cost of two full CIDER XC
   builds per inner Davidson iteration.

This module also provides _CiderSOSCF, a mixin injected by
_CiderKS.newton() so that the post-SCF vdW energy contract survives the
SOSCF wrapper (whose kernel() would otherwise shadow _CiderKS.kernel()).
NOTE: use mf.newton(), not scf.newton(mf) / newton_ah.newton(mf) --
the function form bypasses this mixin, and the vdW contract delta would
be silently skipped (the SCF energy itself would still be correct).
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import rohf, uhf
from pyscf.scf._response_functions import _gen_rhf_response, _gen_uhf_response

DEFAULT_FD_RESPONSE_DELTA = 1e-4


class _CiderSOSCF:
    """Mixin placed in front of pyscf's _CIAH_SOSCF in the MRO so the CIDER
    vdW post-SCF energy contract survives mf.newton().kernel()."""

    def kernel(self, *args, **kwargs):
        from ciderpress.pyscf.dft import (
            _apply_post_density_vdw_energy,
            _pre_kernel_vdw_setup,
        )

        _pre_kernel_vdw_setup(self)
        super().kernel(*args, **kwargs)  # -> _CIAH_SOSCF.kernel
        return _apply_post_density_vdw_energy(self)

    def undo_soscf(self):
        mf = super().undo_soscf()
        return lib.view(mf, lib.drop_class(mf.__class__, _CiderSOSCF))

    undo_newton = undo_soscf


def gen_fd_response(mf, mo_coeff=None, mo_occ=None, **kwargs):
    """Exact CIDER response function via central finite differences of the
    XC potential. Captures all feature nonlocality (NLDF, SDMX, and
    local-hybrid K contributions), since nr_rks/nr_uks fold them all into
    the potential matrix.

    Cost: two full CIDER XC builds per vind(dm1) call. The FD step is
    controlled by mf.fd_response_delta (default 1e-4; raise to 1e-3 if the
    augmented-Hessian inner iterations oscillate due to rough higher
    derivatives of the ML model).
    """
    if isinstance(mf, (uhf.UHF, rohf.ROHF)):
        return _gen_fd_response_uks(mf, mo_coeff, mo_occ, **kwargs)
    else:
        return _gen_fd_response_rks(mf, mo_coeff, mo_occ, **kwargs)


def _fd_xc_deriv(ni, mol, grids, xc_code, nr_xc, dm0, u, eps, max_memory):
    """Central-difference directional derivative of the XC potential matrix
    along the (hermitian) density-matrix direction u."""
    vp = nr_xc(mol, grids, xc_code, dm0 + eps * u, max_memory=max_memory)[2]
    vm = nr_xc(mol, grids, xc_code, dm0 - eps * u, max_memory=max_memory)[2]
    return (vp - vm) / (2 * eps)


def _gen_fd_response_rks(
    mf, mo_coeff=None, mo_occ=None, singlet=None, hermi=0, max_memory=None
):
    if singlet is not None:
        # Singlet/triplet TDDFT kernels are not formulated as a plain
        # directional derivative of the ground-state potential; fall back
        # to the proxy semilocal kernel for these.
        logger.warn(
            mf,
            "CIDER FD response supports only the ground-state orbital "
            "hessian (singlet=None); falling back to the proxy semilocal "
            "kernel for singlet/triplet response.",
        )
        return _gen_rhf_response(mf, mo_coeff, mo_occ, singlet, hermi, max_memory)
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mol = mf.mol
    ni = mf._numint
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    delta = getattr(mf, "fd_response_delta", DEFAULT_FD_RESPONSE_DELTA)
    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * 0.8 - mem_now)

    def vind(dm1):
        dm1 = np.asarray(dm1)
        if np.iscomplexobj(dm1):
            raise NotImplementedError("complex dm1 in CIDER FD response")
        if hermi == 2:
            # Antisymmetric dm1: the (real-basis) density response vanishes,
            # so the XC and Coulomb responses are zero. Matches
            # _gen_rhf_response.
            return np.zeros_like(dm1)
        single = dm1.ndim == 2
        dms = dm1[None] if single else dm1
        v1 = np.zeros_like(dms)
        for i, d in enumerate(dms):
            # For real AO bases the density (and hence vxc) depends only on
            # the hermitian part of the density matrix, so symmetrizing is
            # exact (and required for CIDER's hermi=1 rho evaluation).
            u = (d + d.T) * 0.5
            s = np.abs(u).max()
            if s < 1e-14:
                continue
            eps = delta / s
            v1[i] = _fd_xc_deriv(
                ni, mol, mf.grids, mf.xc, ni.nr_rks, dm0, u, eps, max_memory
            )
        v1 += mf.get_j(mol, dms, hermi=hermi)
        return v1[0] if single else v1

    return vind


def _gen_fd_response_uks(
    mf, mo_coeff=None, mo_occ=None, with_j=True, hermi=0, max_memory=None
):
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mol = mf.mol
    ni = mf._numint
    dm0 = np.asarray(mf.make_rdm1(mo_coeff, mo_occ))
    delta = getattr(mf, "fd_response_delta", DEFAULT_FD_RESPONSE_DELTA)
    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * 0.8 - mem_now)

    def vind(dm1):
        dm1 = np.asarray(dm1)
        if np.iscomplexobj(dm1):
            raise NotImplementedError("complex dm1 in CIDER FD response")
        if hermi == 2:
            v1 = np.zeros_like(dm1)
        else:
            single = dm1.ndim == 3
            dms = dm1[:, None] if single else dm1
            nset = dms.shape[1]
            v1 = np.zeros_like(dms)
            for i in range(nset):
                # Perturb both spin channels together: the derivative along
                # the joint spin direction is what the UKS orbital hessian
                # requires.
                d = dms[:, i]
                u = (d + d.transpose(0, 2, 1)) * 0.5
                s = np.abs(u).max()
                if s < 1e-14:
                    continue
                eps = delta / s
                v1[:, i] = _fd_xc_deriv(
                    ni, mol, mf.grids, mf.xc, ni.nr_uks, dm0, u, eps, max_memory
                )
            v1 = v1[:, 0] if single else v1
        if with_j:
            # Matches _gen_uhf_response: vj is added regardless of hermi.
            vj = mf.get_j(mol, dm1, hermi=hermi)
            v1 += vj[0] + vj[1]
        return v1

    return vind
