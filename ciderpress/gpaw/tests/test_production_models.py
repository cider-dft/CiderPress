#!/usr/bin/env python

import pytest

from ciderpress.gpaw.calculator import get_cider_functional


@pytest.mark.parametrize("name", ["CIDER_mol", "CIDER_comb"])
def test_production_model_initializes_gpaw(name):
    assert get_cider_functional(name, use_paw=False) is not None


def test_d4_production_model_is_rejected_by_gpaw():
    with pytest.raises(NotImplementedError, match="PySCF backend only"):
        get_cider_functional("CIDER_D4_mol", use_paw=False)
