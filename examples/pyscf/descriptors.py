#!/usr/bin/env python
"""Evaluate the descriptor blocks expected by a packaged CIDER model."""

from pyscf import dft, gto

from ciderpress.dft.model_utils import load_cider_model
from ciderpress.pyscf.analyzers import RHFAnalyzer
from ciderpress.pyscf.descriptors import get_descriptors


def main():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp")
    mf = dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PBE SCF did not converge")

    model = load_cider_model("CIDER26XCCHEM")
    analyzer = RHFAnalyzer.from_calc(mf, grids_level=3)

    semilocal = get_descriptors(analyzer, model.settings.sl_settings)
    nonlocal_density = get_descriptors(analyzer, model.settings.nldf_settings)

    print(f"grid points = {analyzer.grids.weights.size}")
    print(f"semilocal descriptor shape = {semilocal.shape}")
    print(f"NLDF descriptor shape = {nonlocal_density.shape}")


if __name__ == "__main__":
    main()
