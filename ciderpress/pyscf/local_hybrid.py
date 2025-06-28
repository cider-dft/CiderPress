import numpy
from pyscf.dft import numint
from pyscf.dft.numint import NBINS, _dot_ao_ao_sparse, _scale_ao_sparse
from pyscf.dft.rks import RKS, lib, logger


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
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset, nao, nao))
    amat = numpy.zeros((nset, nao, nao))
    a1 = -1.00778
    a2 = 1.10507
    a1 = 0.25
    a2 = -0.13

    try:
        x_code, c_code = xc_code.split(",")
    except Exception:
        raise ValueError("Need comma-separated XC for this")

    ngrids = grids.weights.size
    nrho = 1
    if xctype == "GGA":
        nrho = 4
    else:
        nrho = 5

    wv_full = numpy.empty((nset, nrho, ngrids))
    wda_full = numpy.empty((nset, nrho, ngrids))

    def amix_and_deriv(rho):
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
        return amix, agrad

    def block_loop(ao_deriv):
        ip0 = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            ip1 = ip0 + weight.size
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                # rho_full[i, :, ip0 : ip1] = rho
                ex, vx = ni.eval_xc_eff(x_code + ",", rho, deriv=1, xctype=xctype)[:2]
                ec, vc = ni.eval_xc_eff("," + c_code, rho, deriv=1, xctype=xctype)[:2]
                wa, da = amix_and_deriv(rho)
                exc = ex * (1 - wa) + ec
                vxc = vx * (1 - wa) + vc
                if xctype == "LDA":
                    den = rho * weight
                else:
                    den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)
                wv = weight * (vxc - ex * rho[0] * da)
                wda_full[i, :, ip0:ip1] = weight * da
                wa[:] = wa * weight
                wv_full[i, :, ip0:ip1] = wv
                yield i, ao, mask, wa
            ip0 = ip1

    aow = None
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype == "GGA":
        ao_deriv = 1
    elif xctype == "MGGA":
        ao_deriv = 1
    else:
        raise ValueError
    for i, ao, mask, wa in block_loop(ao_deriv):
        _dot_ao_ao_sparse(
            ao[0], ao[0], wa, nbins, mask, pair_mask, ao_loc, hermi, amat[i]
        )

    """
    if xctype == "LDA":
        for i, ao, mask, wa in block_loop(ao_deriv):
            _dot_ao_ao_sparse(
                ao, ao, wv, nbins, mask, pair_mask, ao_loc, hermi, vmat[i]
            )
    elif xctype == "GGA":
        ao_deriv = 1
        for i, ao, mask, wv, wa in block_loop(ao_deriv):
            wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=vmat[i]
            )
            _dot_ao_ao_sparse(
                ao[0], ao[0], wa, nbins, mask, pair_mask, ao_loc, hermi, amat[i]
            )
        vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
    elif xctype == "MGGA":
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv, wa in block_loop(ao_deriv):
            wv[0] *= 0.5  # *.5 for v+v.conj().T
            wv[4] *= 0.5  # *.5 for 1/2 in tau
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=vmat[i]
            )
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])
            _dot_ao_ao_sparse(
                ao[0], ao[0], wa, nbins, mask, pair_mask, ao_loc, hermi, amat[i]
            )
        vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
        vmat += v1
    elif xctype == "HF":
        pass
    else:
        raise NotImplementedError(f"numint.nr_uks for functional {xc_code}")
    """

    if dms.ndim == 2:
        dms = dms[None, :, :]
    pamat = lib.einsum("xij,xjk->xik", dms, amat)
    edms = 0.5 * lib.einsum("xij,xjk->xik", pamat, dms)
    ovlp = mol.intor("int1e_ovlp")
    ratio = numpy.sum(edms * ovlp) / numpy.sum(dms * ovlp)
    print(ratio)
    print(pamat + pamat.transpose(0, 2, 1))
    # vk = get_k(mol, ratio * dms, hermi)
    vk2 = get_k(mol, edms, hermi)
    vk1 = get_k(mol, dms, hermi)
    # vkp = vk + vk2 * ratio  # numpy.einsum("xij,vjk->xik", vk2, amat)
    # vkp = numpy.einsum("xij,xjk->xik", vk, pamat)
    # vkp = numpy.einsum("xij,xjk->xik", vk, amat)
    # vkp = 0.25 * (vkp + vkp.transpose(0, 2, 1))
    vk2 = 0.5 * (vk2 + vk2.transpose(0, 2, 1))
    vk3 = vk1.copy()
    for idm in range(vk1.shape[0]):
        sk = numpy.linalg.solve(ovlp, vk1[idm])
        ask = amat[idm].dot(sk)
        ask = ask + ask.T
        sp = ovlp.dot(dms[idm])
        apk = amat[idm].dot(dms[idm]).dot(vk1[idm])
        vk3[idm] = dms[idm].dot(vk3[idm]).dot(dms[idm])
        # vk1[idm] = numpy.linalg.solve(ovlp, vk1[idm])
        # vk1[idm] = amat[idm].dot(vk1[idm])
        vk1[idm] = 1.0 * apk - 0.5 * sp.dot(ask) + 0.5 * ask
    vk1 = 0.5 * (vk1 + vk1.transpose(0, 2, 1))
    vk3 = 0.5 * (vk3 + vk3.transpose(0, 2, 1))
    vmat[:] -= 0.25 * (vk2 + vk1)
    excsum[:] -= 0.25 * numpy.einsum("xij,xji->x", dms, vk2)

    make_vrho = ni._gen_rho_evaluator(mol, -0.125 * vk3, hermi, False, grids)[0]

    def block_loop(ao_deriv):
        ip0 = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            ip1 = ip0 + weight.size
            for i in range(nset):
                # rho = make_rho(i, ao, mask, xctype)
                va = make_vrho(i, ao[0], mask, "LDA")
                wda = wda_full[i, :, ip0:ip1].copy()
                wv = va * wda
                yield i, ao, mask, wv + wv_full[i, :, ip0:ip1]
            ip0 = ip1

    v2 = numpy.zeros_like(vmat)
    if xctype == "GGA":
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=v2[i]
            )

        v2 = lib.hermi_sum(v2, axes=(0, 2, 1))
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
                mol, ks.nlcgrids, xc, dm, ks.get_k, max_memory=max_memory
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
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
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

    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        if (
            ks._eri is None
            and ks.direct_scf
            and getattr(vhf_last, "vj", None) is not None
        ):
            dm_last = numpy.asarray(dm_last)
            dm = numpy.asarray(dm)
            assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
            ddm = dm - dm_last
            vj = ks.get_j(mol, ddm[0] + ddm[1], hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm[0] + dm[1], hermi)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if (
            ks._eri is None
            and ks.direct_scf
            and getattr(vhf_last, "vk", None) is not None
        ):
            dm_last = numpy.asarray(dm_last)
            dm = numpy.asarray(dm)
            assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
            ddm = dm - dm_last
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if omega != 0:
                vklr = ks.get_k(mol, ddm, hermi, omega)
                vklr *= alpha - hyb
                vk += vklr
            vj = vj[0] + vj[1] + vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vj = vj[0] + vj[1]
            vk *= hyb
            if omega != 0:
                vklr = ks.get_k(mol, dm, hermi, omega)
                vklr *= alpha - hyb
                vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -= (
                numpy.einsum("ij,ji", dm[0], vk[0]).real
                + numpy.einsum("ij,ji", dm[1], vk[1]).real
            ) * 0.5
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
    nr_uks = numint.nr_uks

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


if __name__ == "__main__":
    from pyscf import gto

    # mol = gto.M(atom="H 0 0 0; H 0 0 0.7", basis="def2-qzvppd", verbose=4)
    # mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-qzvppd", verbose=4)
    mol = gto.M(atom="F 0 0 0; F 0 0 1.4", basis="def2-qzvppd", verbose=4)
    ks = LHRKS(mol, xc="GGA_X_PBE,GGA_C_PBE")
    ks.grids.level = 4
    ks.kernel()
    # ks = dft.RKS(mol, xc="0.25*HF+0.75*GGA_X_PBE+GGA_C_PBE")
    # ks.kernel()

    mo_coeff = ks.mo_coeff.copy()
    mo_occ = ks.mo_occ
    nocc = int(numpy.round(ks.mo_occ.sum())) // 2
    etot = ks.e_tot

    def _rotate(coeff, angle):
        rot = numpy.identity(coeff.shape[0])
        rot[nocc - 1, nocc - 1] = numpy.cos(angle)
        rot[nocc, nocc] = numpy.cos(angle)
        rot[nocc - 1, nocc] = -numpy.sin(angle)
        rot[nocc, nocc - 1] = numpy.sin(angle)
        return coeff.dot(rot)

    mod_coeff = _rotate(ks.mo_coeff.copy(), numpy.pi / 4)
    mod_dm = ks.make_rdm1(mo_coeff=mod_coeff, mo_occ=mo_occ)
    emod = ks.energy_tot(dm=mod_dm)
    print(etot, emod)

    p_coeff = _rotate(mod_coeff, numpy.pi / 1000)
    m_coeff = _rotate(mod_coeff, -numpy.pi / 1000)
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
        print(scipy.linalg.eigvalsh(matrix))

    _test_eigvals(mod_dm)
    _test_eigvals(p_dm)
    _test_eigvals(m_dm)

    print(de_pred, de_ref, numpy.sqrt((ddm * ddm).sum()))
    tol = numpy.abs(de_pred - de_ref) / numpy.sqrt((ddm * ddm).sum())
    print(tol)
    assert tol < 1e-8
