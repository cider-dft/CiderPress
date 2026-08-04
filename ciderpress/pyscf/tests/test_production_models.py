#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import dft, gto

from ciderpress.dft.model_utils import (
    BUILTIN_MODELS,
    CIDER23X_MODELS,
    CIDER24X_MODELS,
    CIDER26XC_MODELS,
)
from ciderpress.pyscf import dft as cider_dft


@pytest.mark.parametrize("name", BUILTIN_MODELS)
def test_production_model_initializes_pyscf(name):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    calc = cider_dft.make_cider_calc(dft.RKS(mol), name)

    assert calc._numint.mlxc.nfeat == calc._numint.settings.nfeat
    assert calc._cider_vdw_contract_enabled == (name in CIDER26XC_MODELS)
    if name == "CIDER26XCCHEMD4":
        assert calc._cider_vdw_fit_term == "d4"
        assert calc._cider_vdw_fit_info["kind"] == "d4"
    else:
        assert calc._cider_vdw_fit_term is None
        assert calc._cider_vdw_fit_info is None


def test_post_density_vdw_is_applied_exactly_once(monkeypatch):
    calc = SimpleNamespace(
        e_tot=-10.0,
        _cider_vdw_contract_enabled=True,
        _cider_vdw_fit_term="d4",
        _cider_vdw_fit_info={"kind": "d4", "params": {"xc": "PBE"}},
        _cider_vdw_eval_mode="post_density",
        _cider_vdw_applied=False,
    )
    monkeypatch.setattr(
        cider_dft, "_get_present_dispersion_energy_ha", lambda _calc: -0.03
    )
    monkeypatch.setattr(
        cider_dft,
        "_compute_expected_vdw_energy_ha",
        lambda _calc, _info, _mode: -0.10,
    )

    assert cider_dft._apply_post_density_vdw_energy(calc) == pytest.approx(-10.07)
    assert calc.e_tot_base == pytest.approx(-10.0)
    assert calc.e_vdw_present == pytest.approx(-0.03)
    assert calc.e_vdw_expected == pytest.approx(-0.10)
    assert calc.e_vdw_delta == pytest.approx(-0.07)
    assert cider_dft._apply_post_density_vdw_energy(calc) == pytest.approx(-10.07)


@pytest.mark.parametrize("name", CIDER26XC_MODELS)
def test_production_model_scf_high_cost(name):
    """One-cycle end-to-end smoke test; selected explicitly for releases."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    calc = cider_dft.make_cider_calc(dft.RKS(mol), name)
    calc.grids.level = 0
    calc.max_cycle = 1
    energy = calc.kernel()

    assert np.isfinite(energy)
    assert calc._numint.nldfgen.plan._raise_large_expnt_error is False
    assert calc._cider_vdw_applied
    assert energy == pytest.approx(calc.e_tot_base + calc.e_vdw_delta)
    if name == "CIDER26XCCHEMD4":
        assert calc.e_vdw_expected < 0
    else:
        assert calc.e_vdw_expected == 0


@pytest.mark.parametrize(
    "name", (CIDER23X_MODELS[-1], CIDER24X_MODELS[-1])
)
def test_exchange_only_family_initializes_with_explicit_surrogate_mix(name):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    calc = cider_dft.make_cider_calc(
        dft.RKS(mol),
        name,
        xmix=0.25,
        xkernel="GGA_X_PBE",
        ckernel="GGA_C_PBE",
    )

    assert calc._numint.mlxc.nfeat == calc._numint.settings.nfeat
    assert not calc._cider_vdw_contract_enabled
