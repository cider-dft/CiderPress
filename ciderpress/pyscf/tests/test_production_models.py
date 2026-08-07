#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
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


def test_model_d4_gradient_matches_direct_evaluator_and_finite_difference():
    dftd4 = pytest.importorskip("pyscf.dispersion.dftd4")
    mol = gto.M(
        atom="H 0 0 -0.7; F 0 0 0.7",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    params = {"xc": "PBE", "ga": 3.0, "gc": 2.0, "wf": 6.0, "atm": True}
    info = {"kind": "d4", "params": params}
    calc = SimpleNamespace(mol=mol)

    gradient = cider_dft._compute_expected_vdw_gradient_ha_per_bohr(
        calc, info, "post_density"
    )
    direct = dftd4.DFTD4Dispersion(mol, **params).get_dispersion(grad=True)
    assert_allclose(gradient, direct["gradient"], atol=1e-14, rtol=0)

    step = 1e-4
    coords = mol.atom_coords()
    displaced = []
    for sign in (-1, 1):
        new_coords = coords.copy()
        new_coords[1, 2] += sign * step
        new_mol = mol.set_geom_(new_coords, unit="Bohr", inplace=False)
        new_calc = SimpleNamespace(mol=new_mol)
        displaced.append(
            cider_dft._compute_expected_vdw_energy_ha(new_calc, info, "post_density")
        )
    finite_difference = (displaced[1] - displaced[0]) / (2 * step)
    assert gradient[1, 2] == pytest.approx(finite_difference, abs=1e-9)


def test_cider_vdw_gradient_is_added_once_and_honors_atmlst(monkeypatch):
    class ElectronicGradient:
        def __init__(self, base):
            self.base = base
            self.mol = base.mol
            self.atmlst = None

        def grad_nuc(self, mol=None, atmlst=None):
            values = np.arange(9, dtype=float).reshape(3, 3)
            return values if atmlst is None else values[atmlst]

    mol = gto.M(atom="He 0 0 0; He 0 0 1; He 0 0 2", basis="sto-3g", verbose=0)
    base = SimpleNamespace(
        mol=mol,
        _cider_vdw_contract_enabled=True,
        _cider_vdw_fit_term="d4",
        _cider_vdw_fit_info={"kind": "d4", "params": {"xc": "PBE"}},
        _cider_vdw_eval_mode="post_density",
    )
    calls = []

    def expected_gradient(_base, _info, _mode, mol=None, atmlst=None):
        calls.append((mol, atmlst))
        values = np.full((3, 3), 0.25)
        return values if atmlst is None else values[atmlst]

    monkeypatch.setattr(
        cider_dft, "_compute_expected_vdw_gradient_ha_per_bohr", expected_gradient
    )
    gradient = cider_dft._add_cider_vdw_gradient(base, ElectronicGradient(base))
    result = gradient.grad_nuc(atmlst=[2, 0])

    expected = np.arange(9, dtype=float).reshape(3, 3)[[2, 0]] + 0.25
    assert_allclose(result, expected)
    assert calls == [(None, [2, 0])]
    gradient.atmlst = [2, 0]
    assert_allclose(gradient.get_dispersion(), np.zeros((2, 3)))


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("density_fitted", [False, True])
def test_d4_model_decorates_rks_uks_and_density_fitted_gradients(
    unrestricted, density_fitted
):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ks = dft.UKS(mol) if unrestricted else dft.RKS(mol)
    calc = cider_dft.make_cider_calc(ks, "CIDER26XCCHEMD4")
    if density_fitted:
        calc = calc.density_fit()

    gradient = calc.nuc_grad_method()

    assert isinstance(gradient, cider_dft._CiderVDWGradient)
    assert (getattr(gradient.base, "with_df", None) is not None) == density_fitted


def test_external_d4_wrapper_does_not_wrap_cider_gradient_twice():
    dftd4_pyscf = pytest.importorskip("dftd4.pyscf")
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    base = dft.RKS(mol)
    base.xc = "PBE"
    calc = cider_dft.make_cider_calc(dftd4_pyscf.energy(base), "CIDER26XCCHEMD4")

    gradient = calc.nuc_grad_method()
    gradient_mro = gradient.__class__.mro()

    assert gradient_mro.count(cider_dft._CiderVDWGradient) == 1
    assert not any(cls.__module__.startswith("dftd4") for cls in gradient_mro)


def test_production_d4_gradient_high_cost():
    pytest.importorskip("pyscf.dispersion.dftd4")
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    calc = cider_dft.make_cider_calc(dft.RKS(mol), "CIDER26XCCHEMD4")
    calc.grids.level = 0
    calc.max_cycle = 1
    calc.kernel()

    total_gradient = calc.nuc_grad_method().kernel()
    d4_gradient = cider_dft._compute_expected_vdw_gradient_ha_per_bohr(
        calc,
        calc._cider_vdw_fit_info,
        calc._cider_vdw_eval_mode,
    )
    calc._cider_vdw_fit_term = None
    electronic_gradient = calc.nuc_grad_method().kernel()

    assert_allclose(total_gradient - electronic_gradient, d4_gradient, atol=1e-10)


@pytest.mark.parametrize("kind", ["d3", "nlc"])
def test_unsupported_post_density_gradient_fails_clearly(kind):
    mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
    base = SimpleNamespace(
        mol=mol,
        _cider_vdw_contract_enabled=True,
        _cider_vdw_fit_term=kind,
        _cider_vdw_fit_info={"kind": kind, "params": {}},
        _cider_vdw_eval_mode="post_density",
    )
    with pytest.raises(NotImplementedError, match=kind.upper()):
        cider_dft._add_cider_vdw_gradient(base, SimpleNamespace())


def test_full_xc_model_rejects_additional_semilocal_terms():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    with pytest.raises(ValueError, match="complete XC baseline"):
        cider_dft.make_cider_calc(
            dft.RKS(mol),
            "CIDER26XCCHEM",
            xmix=1.0,
            xkernel="GGA_X_PBE",
            ckernel="GGA_C_PBE",
        )


@pytest.mark.parametrize("name", CIDER26XC_MODELS)
def test_external_d4_wrapper_is_reconciled_high_cost(name):
    dftd4_pyscf = pytest.importorskip("dftd4.pyscf")
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

    energies = []
    calculations = []
    for wrapped in (False, True):
        base = dft.RKS(mol)
        base.xc = "PBE"
        if wrapped:
            base = dftd4_pyscf.energy(base)
        calc = cider_dft.make_cider_calc(base, name)
        calc.grids.level = 0
        calc.max_cycle = 1
        energies.append(calc.kernel())
        calculations.append(calc)

    assert energies[1] == pytest.approx(energies[0], abs=1e-12)
    if name == "CIDER26XCCHEMD4":
        assert calculations[1].with_dftd4 is not None
        assert calculations[1].e_vdw_present < 0
        assert calculations[1].e_vdw_expected < 0
    else:
        assert calculations[1].with_dftd4 is None
        assert calculations[1].e_vdw_present == 0
        assert calculations[1].e_vdw_expected == 0


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


@pytest.mark.parametrize("name", (CIDER23X_MODELS[-1], CIDER24X_MODELS[-1]))
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
