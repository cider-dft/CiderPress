#!/usr/bin/env python

import hashlib
from pathlib import Path

import pytest

from ciderpress.data import functionals
from ciderpress.dft.model_utils import BUILTIN_MODELS, load_cider_model
from ciderpress.dft.xc_evaluator2 import MappedXC2

EXPECTED_HASHES = {
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
    assert isinstance(plain, MappedXC2)
    for loaded in (suffixed, explicit):
        assert type(loaded) is type(plain)
        assert loaded.nfeat == plain.nfeat
        assert loaded.libxc_baseline == plain.libxc_baseline
        assert len(loaded.kernels) == len(plain.kernels)
        assert loaded.settings.sl_settings.level == plain.settings.sl_settings.level
        assert (
            loaded.settings.nldf_settings.feat_spec_list
            == plain.settings.nldf_settings.feat_spec_list
        )


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


@pytest.mark.parametrize("name", OLD_MODEL_NAMES)
def test_unreleased_old_model_names_are_not_aliases(name):
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_cider_model(name)


def test_unknown_model_name_is_not_treated_as_builtin():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_cider_model("not_a_cider_model")
