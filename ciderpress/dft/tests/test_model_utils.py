#!/usr/bin/env python

import hashlib
from pathlib import Path

import pytest

from ciderpress.data import functionals
from ciderpress.dft.model_utils import (
    BUILTIN_MODELS,
    CIDER24X_MODELS,
    get_builtin_model_name,
    load_cider_model,
    validate_cider_composition,
)
from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2

EXPECTED_HASHES = {
    "CIDER23X_SL_GGA": "2e518727b836cd806c4f27c5566e9401b8c15db77182e34380728efdea24f0ce",
    "CIDER23X_SL_MGGA": "fdd60d58ae0f981dcdf64ec24437454bdc334e65b162e379e3630bb99dc82bba",
    "CIDER23X_NL_GGA": "b4e29d9530eaa7c94a3c61cc5e942a0cf7cfd0dd3c4a265dc35395123ae26c86",
    "CIDER23X_NL_MGGA": "f49060978575ffeb8b18c64df8b9eb917fddf2e0cb8b4cc120e4a11f8a01a537",
    "CIDER23X_NL_MGGA_PBE": "9ce3303986f80aee859c00f8973c7947f02bf3837dfc356c57d7d82fa36660f4",
    "CIDER23X_NL_MGGA_DTR": "93312ddde97a8b8e88a9f875df221a44f65a1a1b7c955a55d9214bd449e363ea",
    "CIDER24Xne": "ab76620ad504ae2934f2f1d599c9c5457c3b882315cbbca710358d641b3e505a",
    "CIDER24Xe": "f2650a9416ca13e0967d932e3b8d43e232fe87bd7bd2b27eb46fe6c82fd3aae1",
    "CIDER26XCCHEM": "fd2e0b5cd7408cd4b0ff09495bb026b109b5066bc2f1267c011ce5f0408bdf1d",
    "CIDER26XCCHEMD4": "ee6824e258625246180efb75fdec7c1e23310a6fe2fcdf2ab836ec90d29bb00c",
    "CIDER26XCSURFSCI": "e141a998359da9a64f3c5d06b4804e06762ab2e53dc6409979ff3eb0eacd793e",
}

OLD_MODEL_NAMES = ("CIDER_mol", "CIDER_D4_mol", "CIDER_comb")


def _model_path(name):
    return Path(functionals.__file__).parent / (name + ".yaml")


def _iter_strings(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, str):
        yield value
        return
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return
    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key, seen)
            yield from _iter_strings(item, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item, seen)
    elif hasattr(value, "__dict__"):
        yield from _iter_strings(vars(value), seen)


@pytest.mark.parametrize("name", BUILTIN_MODELS)
def test_builtin_model_hash_and_aliases(name):
    path = _model_path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_HASHES[name]

    plain = load_cider_model(name)
    suffixed = load_cider_model(name + ".yaml")
    explicit = load_cider_model(path)
    assert isinstance(plain, (MappedXC, MappedXC2))
    for loaded in (suffixed, explicit):
        assert type(loaded) is type(plain)
        assert loaded.nfeat == plain.nfeat
        assert loaded.libxc_baseline == plain.libxc_baseline
        assert len(loaded.kernels) == len(plain.kernels)
        assert loaded.settings.sl_settings.level == plain.settings.sl_settings.level
        assert type(loaded.settings.nldf_settings) is type(plain.settings.nldf_settings)
        assert type(loaded.settings.sdmx_settings) is type(plain.settings.sdmx_settings)


def test_builtin_model_vdw_contracts_and_portability():
    for name in BUILTIN_MODELS:
        model = load_cider_model(name)
        values = list(_iter_strings(model))
        assert not any(value.startswith("/n/") for value in values)
        if name == "CIDER26XCCHEMD4":
            assert model.vdw_fit_term == "d4"
            assert model.vdw_eval_mode == "post_density"
            assert model.vdw_fit_info == {
                "kind": "d4",
                "params": {"xc": "PBE"},
                "term": "d4",
                "units": "Ha",
            }
        else:
            assert getattr(model, "vdw_fit_term", None) is None
            assert getattr(model, "vdw_fit_info", None) is None
            assert getattr(model, "vdw_eval_mode", None) is None


def test_builtin_model_rejects_non_yaml_format():
    with pytest.raises(ValueError, match="Built-in CIDER models use YAML"):
        load_cider_model("CIDER26XCCHEM", "joblib")


def test_builtin_model_name_resolution_respects_explicit_files(tmp_path):
    assert get_builtin_model_name("CIDER24Xne") == "CIDER24Xne"
    assert get_builtin_model_name("CIDER24Xne.yaml") == "CIDER24Xne"

    explicit = tmp_path / "CIDER24Xne.yaml"
    explicit.touch()
    assert get_builtin_model_name(explicit) is None


@pytest.mark.parametrize("name", CIDER24X_MODELS)
def test_cider24_missing_torch_message(name, monkeypatch):
    import ciderpress.dft.model_utils as model_utils

    def missing_torch(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(model_utils.yaml, "load", missing_torch)
    with pytest.raises(ModuleNotFoundError, match=r"ciderpress\[cider24\]"):
        load_cider_model(name)


@pytest.mark.parametrize("name", OLD_MODEL_NAMES)
def test_unreleased_old_model_names_are_not_aliases(name):
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_cider_model(name)


def test_unknown_model_name_is_not_treated_as_builtin():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_cider_model("not_a_cider_model")


def test_full_xc_composition_is_validated():
    full_xc = load_cider_model("CIDER26XCCHEM")
    validate_cider_composition(
        full_xc, xmix=1.0, xkernel=None, ckernel=None, backend="test"
    )
    with pytest.raises(ValueError, match="complete XC baseline"):
        validate_cider_composition(
            full_xc,
            xmix=1.0,
            xkernel="GGA_X_PBE",
            ckernel="GGA_C_PBE",
            backend="test",
        )

    exchange = load_cider_model("CIDER23X_NL_MGGA")
    validate_cider_composition(
        exchange,
        xmix=0.25,
        xkernel="GGA_X_PBE",
        ckernel="GGA_C_PBE",
        backend="test",
    )
