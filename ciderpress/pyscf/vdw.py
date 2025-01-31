import ctypes

import numpy
from pyscf.dft.numint import NBINS, _dot_ao_ao_sparse, _scale_ao_sparse, xc_deriv

from ciderpress.dft.plans import libcider


def _get_r2vv10_wh(R, M, Ws, W0, N1, N2):
    Wh = N1 * W0 / (M**(4.0 / 3) * R**(4.0 / 9))
    Wh /= (1 + N2 * R**(1.0 / 3))**(2.0 / 3)
    Wh /= 1 + Ws
    return Wh

def _r2vv10a(rho, coords, vvrho, vvweight, vvcoords, nlc_pars, with_local_part=True):
    C1, C2, C3, C4, C5 = nlc_pars
    thresh = 1e-15

    # output
    exc = numpy.zeros(rho[0, :].size)
    # vxc = numpy.zeros([2, rho[0, :].size])

    # outer grid needs threshing
    threshind = rho[0, :] >= thresh
    coords = coords[threshind]
    R = rho[0, :][threshind]
    Gx = rho[1, :][threshind]
    Gy = rho[2, :][threshind]
    Gz = rho[3, :][threshind]
    G = Gx**2.0 + Gy**2.0 + Gz**2.0
    ind = numpy.argmax(G)
    #print("CHECK", (G[ind]**0.5 / R[ind])**4, numpy.max(R))
    #print("CHECK", (G[ind]**0.5 / (2 * (3 * numpy.pi**2)**0.3333333 * R[ind]**1.333333)), numpy.max(R))

    # inner grid needs threshing
    innerthreshind = vvrho[0, :] >= thresh
    vvcoords = vvcoords[innerthreshind]
    vvweight = vvweight[innerthreshind]
    Rp = vvrho[0, :][innerthreshind]
    RpW = Rp * vvweight
    Gxp = vvrho[1, :][innerthreshind]
    Gyp = vvrho[2, :][innerthreshind]
    Gzp = vvrho[3, :][innerthreshind]
    Gp = Gxp**2.0 + Gyp**2.0 + Gzp**2.0

    # constants and parameters
    Pi = numpy.pi
    Pi43 = 4.0 * Pi / 3.0
    # Bvv, Cvv = nlc_pars
    Bvv = 1
    Bvv * 1.5 * Pi * ((9.0 * Pi) ** (-1.0 / 6.0))
    # Beta = ((3.0 / (Bvv * Bvv)) ** (0.75)) / 32.0

    # inner grid
    Wgp = C3 * Gp / (Rp * Rp)
    Ws2p = Rp + Wgp * Wgp / Pi43
    Ws23p = Ws2p**(1.0 / 3)
    Mp = 1 + C1 / Ws23p
    Wsp = C5 * Wgp / (C2 * Rp**(2.0 / 3))
    W0p = (Wgp * Wgp + Pi43 * Rp * Mp / (1 + Wsp)) ** 0.5
    Whp = _get_r2vv10_wh(Rp, Mp, Wsp, W0p, C4, 10*C2)
    PRpW = Whp**1.5 * Mp * RpW
    Qp = W0p * Whp

    # outer grid
    Wg = C3 * G / (R * R)
    Ws2 = R + Wg * Wg / Pi43
    Ws23 = Ws2**(1.0 / 3)
    M = 1 + C1 / Ws23
    Wp2 = Pi43 * R * M
    Ws = C5 * Wg / (C2 * R**(2.0 / 3))
    W0 = (Wg * Wg + Wp2 / (1 + Ws)) ** 0.5
    Wh = _get_r2vv10_wh(R, M, Ws, W0, C4, 10*C2)
    P = Wh**1.5 * M
    Q = W0 * Wh

    vvcoords = numpy.asarray(vvcoords, order="C")
    coords = numpy.asarray(coords, order="C")
    F = numpy.empty_like(R)
    U = numpy.empty_like(R)
    W = numpy.empty_like(R)
    libcider.VXC_r2vv10a(
        F.ctypes.data_as(ctypes.c_void_p),
        U.ctypes.data_as(ctypes.c_void_p),
        W.ctypes.data_as(ctypes.c_void_p),
        vvcoords.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        Qp.ctypes.data_as(ctypes.c_void_p),
        Q.ctypes.data_as(ctypes.c_void_p),
        PRpW.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(vvcoords.shape[0]),
        ctypes.c_int(coords.shape[0]),
    )
    # exc is multiplied by Rho later
    exc[threshind] = 0.5 * F * P
    if with_local_part:
        exc[threshind] += (3 * Wh**1.5 * 9 * Wp2**2) / (512 * R * W0**1.5)
    # vxc[0, threshind] = Beta + F + 1.5 * (U * dKdR + W * dW0dR)
    # vxc[1, threshind] = 1.5 * W * dW0dG
    return exc  # , vxc


def _get_r2vv10c_terms(R, G, params, get_local_part=True):
    C1, C2, C3, C4, C5 = params
    Pi = numpy.pi
    Pi4 = 4.0 * Pi
    Pi43 = Pi4 / 3.0
    Wp2 = Pi4 * R
    R13 = R**(1.0 / 3)
    M = 1 + C1 / R13
    Wg = C3 * G / (R * R)
    x2 = G / R**(8.0 / 3)
    mix = 1.0 / (1.0 + C4 * x2)
    print("MIX", numpy.max(mix), numpy.min(mix))
    Whr = C2 * Wp2**0.5 / (M**(5.0 / 6) * R**(4.0 / 9))
    Whr /= (1 + C5 * R13)**(2.0 / 3)
    Whg = 0.1 * (C4 * R13 / (1 + C4 * R13))**(2.0 / 3)
    P = mix * M * Whr**1.5 + (1 - mix) * Whg**1.5
    Q = mix * Wp2 / 3 * M * Whr * Whr + (1 - mix) * Wg * Wg * Whg * Whg
    # P = mix * M * Whr**1.5 + (1 - mix) * Whg**1.5
    # Q = mix * Wp2 / 3 * M * Whr * Whr + (1 - mix) * Wg * Wg * Whg * Whg
    # Q = mix * (Wp2 * M)**0.5 * Whr + (1 - mix) * Wg * Whg
    Q = Q**0.5
    if get_local_part:
        E = 3.0 / 32 * P * P * Pi * Pi / Q**1.5 * R
    else:
        E = 0
    return P, Q, E


def _r2vv10c(rho, coords, vvrho, vvweight, vvcoords, nlc_pars, with_local_part=True):
    C1, C2, C3, C4, C5 = nlc_pars
    thresh = 1e-15

    # output
    exc = numpy.zeros(rho[0, :].size)
    # vxc = numpy.zeros([2, rho[0, :].size])

    # outer grid needs threshing
    threshind = rho[0, :] >= thresh
    coords = coords[threshind]
    R = rho[0, :][threshind]
    Gx = rho[1, :][threshind]
    Gy = rho[2, :][threshind]
    Gz = rho[3, :][threshind]
    G = Gx**2.0 + Gy**2.0 + Gz**2.0

    # inner grid needs threshing
    innerthreshind = vvrho[0, :] >= thresh
    vvcoords = vvcoords[innerthreshind]
    vvweight = vvweight[innerthreshind]
    Rp = vvrho[0, :][innerthreshind]
    RpW = Rp * vvweight
    Gxp = vvrho[1, :][innerthreshind]
    Gyp = vvrho[2, :][innerthreshind]
    Gzp = vvrho[3, :][innerthreshind]
    Gp = Gxp**2.0 + Gyp**2.0 + Gzp**2.0

    # inner grid
    Pp, Qp, Ep = _get_r2vv10c_terms(Rp, Gp, nlc_pars, False)
    PRpW = Pp * RpW

    # outer grid
    P, Q, E = _get_r2vv10c_terms(R, G, nlc_pars, True)

    vvcoords = numpy.asarray(vvcoords, order="C")
    coords = numpy.asarray(coords, order="C")
    F = numpy.empty_like(R)
    U = numpy.empty_like(R)
    W = numpy.empty_like(R)
    libcider.VXC_r2vv10a(
        F.ctypes.data_as(ctypes.c_void_p),
        U.ctypes.data_as(ctypes.c_void_p),
        W.ctypes.data_as(ctypes.c_void_p),
        vvcoords.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        Qp.ctypes.data_as(ctypes.c_void_p),
        Q.ctypes.data_as(ctypes.c_void_p),
        PRpW.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(vvcoords.shape[0]),
        ctypes.c_int(coords.shape[0]),
    )
    # exc is multiplied by Rho later
    exc[threshind] = 0.5 * F * P
    if with_local_part:
        exc[threshind] += E
    # vxc[0, threshind] = Beta + F + 1.5 * (U * dKdR + W * dW0dR)
    # vxc[1, threshind] = 1.5 * W * dW0dG
    return exc  # , vxc


def nr_nlc_vxc(
    ni, mol, grids, xc_code, dm, relativity=0, hermi=1, max_memory=2000, verbose=None,
    r2c=None
):
    """Calculate NLC functional and potential matrix on given grids

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm : 2D array
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    """
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    assert nset == 1
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    ao_deriv = 1
    vvrho = []
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        vvrho.append(make_rho(0, ao, mask, "GGA"))
    rho = numpy.hstack(vvrho)

    exc = 0
    if r2c is None:
        # nlc_coefs = [([0.029, 0.01, 0.05, 40, 40], 1.0)]  # for PBE
        nlc_coefs = [([0.029, 0.25, 0.1, 25, 25], 1.0)]  # for rPW86+PBE
    else:
        nlc_coefs = [(r2c, 1.0)]
    for nlc_pars, fac in nlc_coefs:
        # e = _r2vv10a(rho, grids.coords, rho, grids.weights, grids.coords, nlc_pars)
        e = _r2vv10c(rho, grids.coords, rho, grids.weights, grids.coords, nlc_pars)
        exc += e * fac
    den = rho[0] * grids.weights
    nelec = den.sum()
    excsum = numpy.dot(den, exc)
    return excsum
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, "GGA", spin=0)

    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    aow = None
    vmat = numpy.zeros((nao, nao))
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:, p0:p1] * weight
        wv[0] *= 0.5
        aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
        _dot_ao_ao_sparse(
            ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=vmat
        )
    vmat = vmat + vmat.T
    return nelec, excsum, vmat
