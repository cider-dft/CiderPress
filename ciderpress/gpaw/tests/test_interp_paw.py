#!/usr/bin/env python

import numpy as np
import pytest
from gpaw.setup import create_setup

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
    correction = DiffPAWXCCorrection.from_setup(create_setup(symbol, xc="PBE"))

    assert correction.n_qg.shape[-1] == correction.rgd.r_g.size
    assert np.isfinite(correction.n_qg).all()
    assert np.isfinite(correction.d_qg).all()

    tauc_min = np.sqrt(4 * np.pi) * correction.dc_g**2 / (8 * correction.nc_g + 1e-10)
    tauct_min = (
        np.sqrt(4 * np.pi) * correction.dct_g**2 / (8 * correction.nct_g + 1e-10)
    )
    assert np.all(correction.tauc_g >= tauc_min)
    assert np.all(correction.tauct_g >= tauct_min)
