#!/usr/bin/env python

import pytest

from ciderpress.dft.model_utils import CIDER23X_MODELS, CIDER24X_MODELS
from ciderpress.gpaw.calculator import cider_functional_from_dict, get_cider_functional


@pytest.mark.parametrize("name", ["CIDER26XCCHEM", "CIDER26XCSURFSCI"])
def test_production_model_initializes_gpaw(name):
    assert (
        get_cider_functional(
            name,
            xmix=1.0,
            xkernel=None,
            ckernel=None,
            use_paw=False,
        )
        is not None
    )


@pytest.mark.parametrize("name", CIDER23X_MODELS)
def test_exchange_model_initializes_gpaw(name):
    assert (
        get_cider_functional(
            name,
            xmix=0.25,
            xkernel="GGA_X_PBE",
            ckernel="GGA_C_PBE",
            use_paw=False,
        )
        is not None
    )


@pytest.mark.parametrize("name", CIDER24X_MODELS)
def test_sdmx_model_is_rejected_by_gpaw(name):
    with pytest.raises(NotImplementedError, match="SDMX features in GPAW"):
        get_cider_functional(name, use_paw=False)


def test_d4_production_model_is_rejected_by_gpaw():
    with pytest.raises(NotImplementedError, match="PySCF backend only"):
        get_cider_functional("CIDER26XCCHEMD4", use_paw=False)


def test_production_model_dictionary_round_trip_preserves_spline_grid():
    xc = get_cider_functional(
        "CIDER26XCSURFSCI",
        xmix=1.0,
        xkernel=None,
        ckernel=None,
        qmax=310,
        lambd=1.75,
        use_paw=False,
    )
    data = xc.todict()
    data["mlfunc"] = xc.get_mlfunc_data()
    restored = cider_functional_from_dict(data)

    assert restored.Nalpha == xc.Nalpha
    assert restored.lambd == xc.lambd
    assert restored.encut == xc.encut
