import numpy
from pyscf.dft import numint
from pyscf.dft.numint import NBINS, _dot_ao_ao_sparse, _scale_ao_sparse, _tau_dot_sparse
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
    a2 = 0.13

    try:
        x_code, c_code = xc_code.split(",")
    except Exception:
        raise ValueError("Need comma-separated XC for this")

    def block_loop(ao_deriv):
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                ex, vx = ni.eval_xc_eff(x_code + ",", rho, deriv=1, xctype=xctype)[:2]
                ec, vc = ni.eval_xc_eff("," + c_code, rho, deriv=1, xctype=xctype)[:2]
                sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
                wa = sigma**0.25 / rho[0] ** 0.5
                wa[:] = a1 + a2 * wa / (1 + wa)
                print(numpy.std(sigma**0.25 / rho[0] ** 0.5))
                wa[rho[0] < 1e-10] = 0
                # wa[:] = 0.5
                exc = ex * (1 - wa) + ec
                vxc = vx * (1 - wa) + vc
                if xctype == "LDA":
                    den = rho * weight
                else:
                    den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)
                wv = weight * vxc
                wa[:] = numpy.sqrt(wa) * weight
                yield i, ao, mask, wv, wa

    aow = None
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    if xctype == "LDA":
        ao_deriv = 0
        for i, ao, mask, wv, wa in block_loop(ao_deriv):
            _dot_ao_ao_sparse(
                ao, ao, wv, nbins, mask, pair_mask, ao_loc, hermi, vmat[i]
            )
            _dot_ao_ao_sparse(
                ao, ao, wa, nbins, mask, pair_mask, ao_loc, hermi, amat[i]
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

    if dms.ndim == 2:
        dms = dms[None, :, :]
    pamat = lib.einsum("xij,xjk->xik", dms, amat)
    edms = 0.5 * lib.einsum("xij,xjk->xik", pamat, dms)
    ovlp = mol.intor("int1e_ovlp")
    ratio = numpy.sum(edms * ovlp) / numpy.sum(dms * ovlp)
    print(ratio)
    print(pamat + pamat.transpose(0, 2, 1))
    vk = get_k(mol, edms, hermi)
    vkp = numpy.einsum("xij,xjk->xik", vk, pamat)
    vkp = 0.5 * (vkp + vkp.transpose(0, 2, 1))
    vmat[:] -= 0.5 * vkp
    excsum[:] -= 0.25 * numpy.einsum("xij,xij->x", edms, vk)

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
    def __init__(self):
        """

        Args:
            mlxc (MappedXC): Model for XC energy
            slxc (str): semilocal contribution to XC energy
            nldf_init (PySCFNLDFInitializer)
            sdmx_init (PySCFSDMXInitializer)
            xmix (float): Mixing fraction of ML functional
            rhocut (float): Low density cutoff for numerical stability
        """
        super(LHNumInt, self).__init__()

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
