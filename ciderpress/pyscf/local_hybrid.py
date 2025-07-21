import numpy
from pyscf.dft import numint
from pyscf.dft.numint import (
    NBINS,
    _dot_ao_ao_sparse,
    _format_uks_dm,
    _scale_ao_sparse,
    _tau_dot_sparse,
)
from pyscf.dft.rks import RKS, lib, logger
from pyscf.dft.uks import UKS


def apply_exx_(ni, mol, vmat, dms, amat, get_k, nset, hermi, nspin=1):
    if dms.ndim == 2:
        dms = dms[None, :, :]
    apmat = lib.einsum("xij,xjk->xik", amat, dms)
    edms = (0.5 * nspin) * lib.einsum("xij,xjk->xik", dms, apmat)
    ovlp = mol.intor("int1e_ovlp")
    # NOTE we are going to assume the result of get_k is symmetric
    if ni.symmetric:
        vk = get_k(mol, edms, hermi)
        vfac = 0.25
    else:
        vk = get_k(mol, numpy.append(dms, edms, axis=0), hermi)
        vmat[:] -= 0.25 * vk[nset:]
        vk = vk[:nset]
        vfac = 0.125
    vfac *= nspin
    exc = (-0.25 * nspin) * numpy.einsum("xij,xji->x", edms, vk)
    for idm in range(nset):
        vktmp = apmat[idm].dot(vk[idm])
        if ni.correct_eigvals:
            sk = numpy.linalg.solve(ovlp, vk[idm])
            ask = amat[idm].dot(sk)
            lib.hermi_sum(ask, axes=(1, 0), inplace=True)
            sp = ovlp.dot(dms[idm])
            vktmp[:] += -0.5 * sp.dot(ask) + 0.5 / nspin * ask
        vmat[idm] -= vfac * lib.hermi_sum(vktmp, axes=(1, 0))
        vk[idm] = dms[idm].dot(vk[idm]).dot(dms[idm])
    return exc, vk


def _get_xc(ni, xc_code):
    xctype = ni._xc_type(xc_code)
    try:
        x_code, c_code = xc_code.split(",")
    except Exception:
        raise ValueError("Need comma-separated XC for this")
    if xctype == "LDA":
        ao_deriv = 0
        nrho = 1
    elif xctype == "GGA":
        ao_deriv = 1
        nrho = 4
    elif xctype == "MGGA":
        ao_deriv = 1
        nrho = 5
    else:
        raise ValueError
    return xctype, x_code, c_code, ao_deriv, nrho


def _eval_xc(ni, x_code, c_code, rho, xctype):
    rho = numpy.array(rho)
    if isinstance(rho, tuple) or rho.ndim == 3:
        den = rho[0][0] + rho[1][0]
    else:
        den = rho[0]
    ex, vx = ni.eval_xc_eff(x_code + ",", rho, deriv=1, xctype=xctype)[:2]
    ec, vc = ni.eval_xc_eff("," + c_code, rho, deriv=1, xctype=xctype)[:2]
    wa, da = ni.eval_amix_eff(rho, xctype)
    exc = ex * (1 - wa) + ec
    vxc = vx * (1 - wa) + vc
    vxc[:] -= ex * den * da
    return exc, vxc, wa, da


def nr_rks_lh(
    ni,
    mol,
    grids,
    xc_code,
    dms,
    get_k,
    relativity=0,
    hermi=1,
    max_memory=2000,
    verbose=None,
):
    xctype, x_code, c_code, ao_deriv, nrho = _get_xc(ni, xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset, nao, nao))
    amat = numpy.zeros((nset, nao, nao))

    ngrids = grids.weights.size

    wv_full = numpy.empty((nset, nrho, ngrids))
    wda_full = numpy.empty((nset, nrho, ngrids))

    aow = None
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    ip0 = 0
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        ip1 = ip0 + weight.size
        for i in range(nset):
            rho = make_rho(i, ao, mask, xctype)
            exc, vxc, wa, da = _eval_xc(ni, x_code, c_code, rho, xctype)
            if xctype == "LDA":
                den = rho * weight
            else:
                den = rho[0] * weight
            nelec[i] += den.sum()
            excsum[i] += numpy.dot(den, exc)
            wv = weight * vxc
            if ni.symmetric:
                wa = numpy.sqrt(wa)
                da[:] /= wa
            wda_full[i, :, ip0:ip1] = weight * da
            wa[:] = wa * weight
            wv_full[i, :, ip0:ip1] = wv
            _dot_ao_ao_sparse(
                ao[0], ao[0], wa, nbins, mask, pair_mask, ao_loc, hermi, amat[i]
            )
        ip0 = ip1

    exc, vk = apply_exx_(ni, mol, vmat, dms, amat, get_k, nset, hermi)
    excsum[:] += exc
    make_vrho = ni._gen_rho_evaluator(mol, -0.125 * vk, hermi, False, grids)[0]

    v2 = numpy.zeros_like(vmat)
    if xctype == "MGGA":
        v1 = numpy.zeros_like(vmat)
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")
    else:
        v1 = None
    ip0 = 0
    mask_dat = nbins, mask, pair_mask, hermi, ao_loc
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        ip1 = ip0 + weight.size
        for i in range(nset):
            va = make_vrho(i, ao[0], mask, "LDA")
            wda = wda_full[i, :, ip0:ip1].copy()
            wv = va * wda + wv_full[i, :, ip0:ip1]
            ni.contract_wv_(xctype, ao, wv, mask_dat, v1, v2, aow, i)
        ip0 = ip1
    if xctype in ["GGA", "MGGA"]:
        lib.hermi_sum(v2, axes=(0, 2, 1), inplace=True)
    if xctype == "MGGA":
        v2[:] += v1
    vmat[:] += v2

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


def nr_uks_lh(
    ni,
    mol,
    grids,
    xc_code,
    dms,
    get_k,
    relativity=0,
    hermi=1,
    max_memory=2000,
    verbose=None,
):
    xctype, x_code, c_code, ao_deriv, nrho = _get_xc(ni, xc_code)
    dma, dmb = _format_uks_dm(dms)
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
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    nelec = numpy.zeros((2, nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((2, nset, nao, nao))
    amat = numpy.zeros((nset, nao, nao))

    ngrids = grids.weights.size

    wv_full = numpy.empty((2, nset, nrho, ngrids))
    wda_full = numpy.empty((2, nset, nrho, ngrids))

    aow = None
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    ip0 = 0
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        ip1 = ip0 + weight.size
        for i in range(nset):
            rhoa = make_rhoa(i, ao, mask, xctype)
            rhob = make_rhob(i, ao, mask, xctype)
            rho = (rhoa, rhob)
            exc, vxc, wa, da = _eval_xc(ni, x_code, c_code, rho, xctype)
            den_a = rhoa[0] * weight
            den_b = rhob[0] * weight
            nelec[0, i] += den_a.sum()
            nelec[1, i] += den_b.sum()
            excsum[i] += numpy.dot(den_a + den_b, exc)
            wv = weight * vxc
            if ni.symmetric:
                wa = numpy.sqrt(wa)
                da[:] /= wa + 1e-16
            wda_full[:, i, :, ip0:ip1] = weight * da
            wa[:] = wa * weight
            wv_full[:, i, :, ip0:ip1] = wv
            _dot_ao_ao_sparse(
                ao[0], ao[0], wa, nbins, mask, pair_mask, ao_loc, hermi, amat[i]
            )
        ip0 = ip1

    # TODO need to play around with the array shapes and such
    # to handle nset > 1
    assert nset == 1 and dms.ndim == 3 and dms.shape[0] == 2
    amat = numpy.append(amat, amat, axis=0)
    exc, vk = apply_exx_(
        ni, mol, vmat[:, 0], dms, amat, get_k, 2 * nset, hermi, nspin=2
    )
    vmat[:] *= 2
    excsum[0] += exc[0] + exc[1]
    make_vrho = ni._gen_rho_evaluator(mol, -0.5 * (vk[0] + vk[1]), hermi, False, grids)[
        0
    ]

    v2 = numpy.zeros_like(vmat)
    if xctype == "MGGA":
        v1 = numpy.zeros_like(vmat)
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")
    else:
        v1 = (None, None)
    ip0 = 0
    mask_dat = nbins, mask, pair_mask, hermi, ao_loc
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        ip1 = ip0 + weight.size
        for i in range(nset):
            va = make_vrho(i, ao[0], mask, "LDA")
            wda = wda_full[:, i, :, ip0:ip1].copy()
            wv = va * wda + wv_full[:, i, :, ip0:ip1]
            ni.contract_wv_(xctype, ao, wv[0], mask_dat, v1[0], v2[0], aow, i)
            ni.contract_wv_(xctype, ao, wv[1], mask_dat, v1[1], v2[1], aow, i)
        ip0 = ip1
    if xctype in ["GGA", "MGGA"]:
        lib.hermi_sum(v2[0], axes=(0, 2, 1), inplace=True)
        lib.hermi_sum(v2[1], axes=(0, 2, 1), inplace=True)
    if xctype == "MGGA":
        v2[:] += v1
    vmat[:] += v2

    if isinstance(dma, numpy.ndarray) and is_2d:
        vmat = vmat[:, 0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = numpy.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    # exit()
    return nelec, excsum, vmat


def get_veff_rks(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    """Coulomb + XC functional

    .. note::
        This function will modify the input ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference Vxc potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    """
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = isinstance(dm, numpy.ndarray) and dm.ndim == 2

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_rks(
            mol, ks.grids, ks.xc, dm, ks.get_k, max_memory=max_memory
        )
        logger.debug(ks, "nelec by numeric integration = %s", n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(
                mol, ks.nlcgrids, xc, dm, max_memory=max_memory
            )
            exc += enlc
            vxc += vnlc
            logger.debug(ks, "nelec with nlc grids = %s", n)
        t0 = logger.timer(ks, "vxc", *t0)

    if ks._eri is None and ks.direct_scf and getattr(vhf_last, "vj", None) is not None:
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        vj = ks.get_j(mol, ddm, hermi)
        vj += vhf_last.vj
    else:
        vj = ks.get_j(mol, dm, hermi)
    vxc += vj

    if ground_state:
        ecoul = numpy.einsum("ij,ji", dm, vj).real * 0.5
    else:
        ecoul = None

    vk = None
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


def get_veff_uks(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    """Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    """
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()
    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        logger.warn(ks, "Incompatible dm dimension. Treat dm as RHF density matrix.")
        dm = numpy.repeat(dm[None] * 0.5, 2, axis=0)
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = dm.ndim == 3 and dm.shape[0] == 2

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0, 0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(
            mol, ks.grids, ks.xc, dm, ks.get_k, max_memory=max_memory
        )
        logger.debug(ks, "nelec by numeric integration = %s", n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(
                mol, ks.nlcgrids, xc, dm[0] + dm[1], max_memory=max_memory
            )
            exc += enlc
            vxc += vnlc
            logger.debug(ks, "nelec with nlc grids = %s", n)
        t0 = logger.timer(ks, "vxc", *t0)

    incremental_jk = (
        ks._eri is None and ks.direct_scf and getattr(vhf_last, "vj", None) is not None
    )
    if incremental_jk:
        dm_last = numpy.asarray(dm_last)
        dm = numpy.asarray(dm)
        assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
        _dm = dm - dm_last
    else:
        _dm = dm

    vk = None
    vj = ks.get_j(mol, _dm[0] + _dm[1], hermi)
    if incremental_jk:
        vj += vhf_last.vj
    vxc += vj

    if ground_state:
        ecoul = numpy.einsum("ij,ji", dm[0] + dm[1], vj).real * 0.5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


class LHNumInt(numint.NumInt):
    def __init__(self, symmetric=False, correct_eigvals=True):
        """
        Initialize a local hybrid numerical integrator.

        Args:
            symmetric: Whether effective exchange integral is symmetric
                For symmetric=True, EXX = Ptilde(r,r') Ptilde(r',r) / |r-r'|.
                For symmetric=False, EXX = Ptilde(r,r') P(r',r) / |r-r'|
            correct_eigvals: Whether to add a term to make Ptilde "more linear"
                in P. correct_eigvals=True ensures that if the mixing fraction
                is a constant, then the eigenvalues are equal to those of the
                corresponding global hybrid.
                For correct_eigvals=True, Ptilde = P A P (matrix multiplication)
                For correct_eigvals=False,
                Ptilde = P A P + (P A + A P) / 2 - (P P A + A P P) / 2
                Note these formulas assume orthogonal basis, but the overlap
                is taken into account in the code. Note that
                (P A + A P) / 2 - (P P A + A P P) / 2 = 0 for idempotent matrices,
                so in practice Ptilde = P A P regardless, but this term
                affects the functional derivative / eigenvalues.
        """
        super(LHNumInt, self).__init__()
        self.symmetric = symmetric
        self.correct_eigvals = correct_eigvals

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    nr_rks = nr_rks_lh
    nr_uks = nr_uks_lh

    def eval_amix_eff(self, rho, xctype):
        a1 = 0.25
        # a1 = 0
        a2 = -0.13
        # a2 = -0.13
        if isinstance(rho, tuple) or rho.ndim == 3:
            spinpol = True
            rho = rho[0] + rho[1]
        else:
            spinpol = False
        sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
        amix = sigma**0.25 / rho[0] ** 0.5
        amix[rho[0] < 1e-10] = 0
        dwdrho = -0.5 * amix / rho[0]
        dwdsigma = 0.25 * amix / sigma
        damix = a2 / (1 + amix) ** 2
        amix[:] = a1 + a2 * amix / (1 + amix)
        agrad = numpy.zeros_like(rho)
        agrad[0] = damix * dwdrho
        agrad[1:4] = 2 * damix * dwdsigma * rho[1:4]
        agrad[:, rho[0] < 1e-10] = 0
        if spinpol:
            agrad = numpy.stack([agrad, agrad])

        return amix, agrad

    def contract_wv_(self, xctype, ao, wv, mask_dat, v1, v2, aow, i):
        nbins, mask, pair_mask, hermi, ao_loc = mask_dat
        if xctype == "LDA":
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc, hermi, v2[i])
        elif xctype == "GGA":
            wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=v2[i]
            )
            # lib.hermi_sum(v2, axes=(0, 2, 1), inplace=True)
        elif xctype == "MGGA":
            wv[0] *= 0.5  # *.5 for v+v.conj().T
            wv[4] *= 0.5  # *.5 for 1/2 in tau
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=v2[i]
            )
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])
            # lib.hermi_sum(v2, axes=(0, 2, 1), inplace=True)
            # v2[:] += v1
        else:
            raise ValueError

    nr_rks_fxc = method_not_implemented
    nr_uks_fxc = method_not_implemented
    nr_rks_fxc_st = method_not_implemented
    cache_xc_kernel = method_not_implemented
    cache_xc_kernel1 = method_not_implemented


class LHRKS(RKS):
    def __init__(self, mol, xc="LDA,VWN"):
        RKS.__init__(self, mol, xc=xc)
        self._numint = LHNumInt()

    get_veff = get_veff_rks


class LHUKS(UKS):
    def __init__(self, mol, xc="LDA,VWN"):
        UKS.__init__(self, mol, xc=xc)
        self._numint = LHNumInt()

    get_veff = get_veff_uks


if __name__ == "__main__":
    from pyscf import gto

    # mol = gto.M(atom="H 0 0 0; H 0 0 0.7", basis="def2-qzvppd", verbose=4)
    # mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-qzvppd", verbose=4)
    mol = gto.M(atom="F 0 0 0; F 0 0 1.4", basis="def2-qzvppd", verbose=4)

    # ks = LHRKS(mol, xc="GGA_X_PBE,GGA_C_PBE")
    # ks = LHRKS(mol, xc="MGGA_X_R2SCAN,MGGA_C_R2SCAN")
    ks = LHUKS(mol, xc="GGA_X_PBE,GGA_C_PBE")
    # ks = LHUKS(mol, xc="MGGA_X_R2SCAN,MGGA_C_R2SCAN")

    ks.init_guess_breaksym = 0

    ks.grids.level = 3
    ks._numint.correct_eigvals = True
    ks._numint.symmetric = False
    ks.kernel()
    # ks = dft.RKS(mol, xc="0.25*HF+0.75*GGA_X_PBE+GGA_C_PBE")
    # ks.kernel()

    mo_coeff = ks.mo_coeff.copy()
    mo_occ = ks.mo_occ
    nocc = int(numpy.round(ks.mo_occ.sum())) // 2
    etot = ks.e_tot

    def _rotate(coeff, angle, spin=None):
        rot = numpy.identity(coeff.shape[-1])
        rot[nocc - 1, nocc - 1] = numpy.cos(angle)
        rot[nocc, nocc] = numpy.cos(angle)
        rot[nocc - 1, nocc] = -numpy.sin(angle)
        rot[nocc, nocc - 1] = numpy.sin(angle)
        if spin is None:
            coeff = coeff.dot(rot)
        else:
            coeff = coeff.copy()
            coeff[spin] = coeff[spin].dot(rot)
        return coeff

    if ks.mo_coeff.ndim == 2:
        spin = None
    else:
        spin = 1
    mod_coeff = _rotate(ks.mo_coeff.copy(), numpy.pi / 4, spin=spin)
    mod_dm = ks.make_rdm1(mo_coeff=mod_coeff, mo_occ=mo_occ)
    emod = ks.energy_tot(dm=mod_dm)
    print(etot, emod)

    p_coeff = _rotate(mod_coeff, numpy.pi / 1000, spin=spin)
    m_coeff = _rotate(mod_coeff, -numpy.pi / 1000, spin=spin)
    p_dm = ks.make_rdm1(mo_coeff=p_coeff, mo_occ=mo_occ)
    m_dm = ks.make_rdm1(mo_coeff=m_coeff, mo_occ=mo_occ)

    ep = ks.energy_tot(dm=p_dm)
    em = ks.energy_tot(dm=m_dm)

    h1e = ks.get_hcore()
    vhf = ks.get_veff(ks.mol, mod_dm)
    heff = h1e + vhf
    ddm = p_dm - m_dm
    de_ref = ep - em
    de_pred = (ddm * heff).sum()

    import scipy

    ovlp = mol.intor("int1e_ovlp")

    def _test_eigvals(matrix):
        L = scipy.linalg.cholesky(ovlp, lower=True)
        matrix = L.T.dot(matrix).dot(L)
        # print(scipy.linalg.eigvalsh(matrix))

    if spin is not None:
        mod_dm = mod_dm[spin]
        p_dm = p_dm[spin]
        m_dm = m_dm[spin]
    _test_eigvals(mod_dm)
    _test_eigvals(p_dm)
    _test_eigvals(m_dm)

    print(de_pred, de_ref, numpy.sqrt((ddm * ddm).sum()))
    tol = numpy.abs(de_pred - de_ref) / numpy.sqrt((ddm * ddm).sum())
    print(tol)
    assert tol < 1e-8
