#!/usr/bin/env python

import numpy as np
import pytest
from gpaw.setup import create_setup
from scipy.interpolate import interp1d

from ciderpress.gpaw.interp_paw import (
    DiffPAWXCCorrection,
    _enforce_fhc_core_tau,
    _get_interpolation_coordinates,
)


class _MappedGrid:
    def __init__(self, mapped):
        self.mapped = np.asarray(mapped)

    def r2g(self, _r_g):
        return self.mapped


def _interpolate_to_correction_grid(xcc, correction, array):
    coordinates = _get_interpolation_coordinates(
        xcc.rgd,
        correction.rgd.r_g,
        xcc.rgd.r_g.size,
    )
    return interp1d(
        np.arange(xcc.rgd.r_g.size),
        array,
        kind="cubic",
    )(coordinates)


def _orbital_products(xcc, correction, phi_jg):
    dphi_jg = np.array([xcc.rgd.derivative(phi_g) for phi_g in phi_jg])
    phi_jg = _interpolate_to_correction_grid(xcc, correction, phi_jg)
    dphi_jg = _interpolate_to_correction_grid(xcc, correction, dphi_jg)

    n_qg = []
    d_qg = []
    for j1 in range(len(phi_jg)):
        for j2 in range(j1, len(phi_jg)):
            n_qg.append(phi_jg[j1] * phi_jg[j2])
            d_qg.append(phi_jg[j1] * dphi_jg[j2] + dphi_jg[j1] * phi_jg[j2])
    return np.asarray(n_qg), np.asarray(d_qg)


def test_interpolation_coordinates_accept_endpoint_roundoff():
    mapped = [np.nextafter(0.0, -1.0), 100.0, 545.0000000000001]
    result = _get_interpolation_coordinates(_MappedGrid(mapped), mapped, 546)
    np.testing.assert_array_equal(result, [0.0, 100.0, 545.0])


@pytest.mark.parametrize("mapped", [[-1e-6, 0.0], [0.0, 545.000001]])
def test_interpolation_coordinates_reject_extrapolation(mapped):
    with pytest.raises(ValueError, match="extends beyond"):
        _get_interpolation_coordinates(_MappedGrid(mapped), mapped, 546)


def test_fhc_core_tau_floor_changes_only_violations():
    n_g = np.ones(2)
    dndr_g = np.array([2.0, 1.0])
    tauw_g = np.sqrt(4 * np.pi) * dndr_g**2 / (8 * n_g + 1e-10)
    tau_g = np.array([0.5 * tauw_g[0], 2.0 * tauw_g[1]])

    result = _enforce_fhc_core_tau(tau_g, n_g, dndr_g)

    np.testing.assert_allclose(result, [tauw_g[0], tau_g[1]])
    np.testing.assert_allclose(tau_g, [0.5 * tauw_g[0], 2.0 * tauw_g[1]])


@pytest.mark.parametrize("symbol", ["H", "Co"])
def test_from_setup_default_path_and_core_fhc_bound(symbol):
    setup = create_setup(symbol, xc="PBE")
    xcc = setup.xc_correction
    correction = DiffPAWXCCorrection.from_setup(setup)

    assert correction.n_qg.shape[-1] == correction.rgd.r_g.size
    assert np.isfinite(correction.n_qg).all()
    assert np.isfinite(correction.d_qg).all()
    np.testing.assert_allclose(
        correction.n_qg,
        _interpolate_to_correction_grid(xcc, correction, xcc.n_qg),
        rtol=2e-14,
        atol=0.0,
    )
    expected_d_qg = np.array([xcc.rgd.derivative(n_g) for n_g in xcc.n_qg])
    np.testing.assert_allclose(
        correction.d_qg,
        _interpolate_to_correction_grid(xcc, correction, expected_d_qg),
        rtol=2e-14,
        atol=0.0,
    )

    tauc_min = np.sqrt(4 * np.pi) * correction.dc_g**2 / (8 * correction.nc_g + 1e-10)
    tauct_min = (
        np.sqrt(4 * np.pi) * correction.dct_g**2 / (8 * correction.nct_g + 1e-10)
    )
    assert np.all(correction.tauc_g >= tauc_min)
    assert np.all(correction.tauct_g >= tauct_min)


@pytest.mark.parametrize("symbol", ["H", "Co"])
def test_from_setup_kinetic_path_uses_orbital_products(symbol):
    setup = create_setup(symbol, xc="PBE")
    xcc = setup.xc_correction
    setup_arrays = {
        name: np.array(getattr(xcc, name), copy=True)
        for name in ("n_qg", "nt_qg", "phi_jg", "phit_jg", "tauc_g", "tauct_g")
    }

    correction = DiffPAWXCCorrection.from_setup(setup, build_kinetic=True)

    expected_n_qg, expected_d_qg = _orbital_products(xcc, correction, xcc.phi_jg)
    expected_nt_qg, expected_dt_qg = _orbital_products(xcc, correction, xcc.phit_jg)
    np.testing.assert_allclose(
        correction.n_qg,
        expected_n_qg,
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        correction.d_qg,
        expected_d_qg,
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        correction.nt_qg,
        expected_nt_qg,
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        correction.dt_qg,
        expected_dt_qg,
        rtol=2e-14,
        atol=0.0,
    )
    assert correction.tau_npg is not None
    assert correction.taut_npg is not None

    for name, original in setup_arrays.items():
        np.testing.assert_array_equal(getattr(xcc, name), original)
