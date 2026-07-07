#!/usr/bin/env python
"""Diagnosis campaign for the NLDF UKS open-shell FD inconsistency.

Experiments (see ~/.claude/plans plan "vb2 NLDF-UKS FD"):
  A1: linear-probe (smooth evaluator) FD through nr_rks/nr_uks
      -> discriminates chain-derivative bug (M2) vs domain effect (M1)
  A2: spline-domain violation instrumentation (real vb2 model)
  A3: train/eval NLDF feature parity on the same dm
Run:  python diagnose_nldf_uks.py A1 [A2 A3 ...]
"""

import sys

import numpy as np

sys.path.insert(
    0, "/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF"
)

from pyscf import dft

from ciderpress.dft.model_utils import load_cider_model
from ciderpress.dft.xc_evaluator import FuncEvaluator
from ciderpress.pyscf.dft import make_cider_calc
from ciderpress.pyscf.tests.lh_test_utils import (
    converged_dm,
    fd_energy_potential_check,
    get_hyb_model_path,
    get_mol,
    prep_cider_ni,
)

VB2_NONHYB = get_hyb_model_path("vb2_nonhyb")


class LinearEvaluator(FuncEvaluator):
    """f(X1) = w . X1 (+0.01 const); dres = w. Smooth everywhere -- no
    spline domains, so FD failures with this evaluator indict the
    numint/generator chain, not the ML model."""

    def __init__(self, weights):
        self.weights = np.asarray(weights)

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        if dres is None:
            dres = np.zeros(X1.shape)
        res[:] += X1 @ self.weights + 0.01
        dres[:] += self.weights
        return res, dres


def make_linear_probe_model(seed=0, scale=0.05):
    """vb2 settings/transforms/normalizers with smooth linear evaluators."""
    ml = load_cider_model(VB2_NONHYB, "yaml")
    rng = np.random.RandomState(seed)
    for k in ml.kernels:
        w = scale * rng.uniform(-1, 1, size=k.N1)
        k.fevals = [LinearEvaluator(w)]
    return ml


def _make_calc(mol, mlfunc):
    mf = dft.UKS(mol) if mol.spin != 0 else dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = 1
    mf = make_cider_calc(mf, mlfunc, xkernel=None, ckernel=None, xmix=1.0)
    return mf


def run_a1():
    print("=" * 70)
    print("A1: linear-probe FD through nr_rks/nr_uks (smooth evaluator)")
    print("=" * 70)
    ml = make_linear_probe_model()
    for molname in ("h2o", "h2o+"):
        mol = get_mol(molname)
        dm = converged_dm(mol, grids_level=1)
        mf = _make_calc(mol, ml)
        ni, grids = prep_cider_ni(mf)
        results = fd_energy_potential_check(
            ni, mol, grids, dm, n_dirs=3, deltas=[1e-3, 2.5e-4]
        )
        for i, entry in enumerate(results):
            for dl in sorted(entry, reverse=True):
                lhs, rhs = entry[dl]
                rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
                print(
                    f"  {molname:5s} dir{i} d={dl:g}: dE={lhs:+.8e} "
                    f"Tr(Vu)={rhs:+.8e} rel={rel:.3e}"
                )
    print("interpretation: rel ~<1e-6 both systems => chain consistent (M1);")
    print("open-shell-only rel >>1e-6, delta-independent => chain bug (M2)")


def run_a2():
    print("=" * 70)
    print("A2: spline-domain violations, real vb2 model, closed vs open shell")
    print("=" * 70)
    import ciderpress.dft.xc_evaluator as xce

    orig = xce.get_vec_eval

    def run_case(molname):
        stats = {}

        def hooked(grid, coeffs, X, N):
            for dim in range(N):
                a, b, n = grid[dim]
                key = (N, dim, float(a), float(b))
                s = stats.setdefault(key, [0, 0, np.inf, -np.inf])
                s[0] += int((X[:, dim] < a).sum() + (X[:, dim] > b).sum())
                s[1] += X.shape[0]
                s[2] = min(s[2], float(X[:, dim].min()))
                s[3] = max(s[3], float(X[:, dim].max()))
            return orig(grid, coeffs, X, N)

        xce.get_vec_eval = hooked
        xce.SplineSetEvaluator.__call__.__globals__["get_vec_eval"] = hooked
        try:
            mol = get_mol(molname)
            dm = converged_dm(mol, grids_level=1)
            mf = _make_calc(mol, VB2_NONHYB)
            ni, grids = prep_cider_ni(mf)
            if mol.spin == 0:
                ni.nr_rks(mol, grids, "PBE", dm)
            else:
                ni.nr_uks(mol, grids, "PBE", dm)
        finally:
            xce.get_vec_eval = orig
            xce.SplineSetEvaluator.__call__.__globals__["get_vec_eval"] = orig
        print(f"--- {molname} ---")
        for key in sorted(stats):
            s = stats[key]
            frac = s[0] / max(s[1], 1)
            flag = "  <-- OUT OF DOMAIN" if frac > 1e-6 else ""
            print(
                f"  spline N={key[0]} dim={key[1]} dom=({key[2]:+.3f},"
                f"{key[3]:+.3f}): viol={s[0]}/{s[1]} ({frac:.2%}) "
                f"range=({s[2]:+.4f},{s[3]:+.4f}){flag}"
            )

    for molname in ("h2o", "h2o+", "o2"):
        try:
            run_case(molname)
        except ValueError as e:
            if molname == "o2":
                print(f"--- o2 skipped ({e}) ---")
            else:
                raise


def run_a3():
    print("=" * 70)
    print("A3: train/eval NLDF feature parity on the same H2O+ dm")
    print("=" * 70)
    from pyscf import scf

    from ciderpress.pyscf.analyzers import UHFAnalyzer
    from ciderpress.pyscf.descriptors import get_descriptors

    ml = load_cider_model(VB2_NONHYB, "yaml")
    settings = ml.settings

    mol = get_mol("h2o+")
    mf = dft.UKS(mol)
    mf.xc = "PBE"
    mf.grids.level = 1
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    dm = np.asarray(mf.make_rdm1())

    # ---- training path: analyzer + get_descriptors (2*rdm1[s], nspin=1)
    ana = UHFAnalyzer.from_calc(mf)
    ana.perform_full_analysis()
    desc_train = get_descriptors(ana, settings.nldf_settings)
    print("train-path desc shape:", np.asarray(desc_train).shape)

    # ---- SCF path: the generator exactly as nr_uks drives it
    mfc = _make_calc(get_mol("h2o+"), ml)
    ni, grids = prep_cider_ni(mfc)
    ni.initialize_feature_generators(mol, grids, 2)
    from pyscf.dft.numint import NumInt

    pni = NumInt()
    ao = pni.eval_ao(mol, grids.coords, deriv=2)
    rho_a = pni.eval_rho(mol, ao, dm[0], xctype="MGGA")
    rho_b = pni.eval_rho(mol, ao, dm[1], xctype="MGGA")
    feat_scf = np.stack(
        [
            ni.nldfgen.get_features(np.ascontiguousarray(rho_a[:5]), spin=0),
            ni.nldfgen.get_features(np.ascontiguousarray(rho_b[:5]), spin=1),
        ]
    )
    print("scf-path feat shape:", feat_scf.shape)

    dt = np.asarray(desc_train)
    # align shapes: train (2, nfeat, ngrids) expected
    for s in range(2):
        for i in range(feat_scf.shape[1]):
            a = dt[s, i]
            b = feat_scf[s, i]
            denom = np.maximum(np.abs(a), 1e-10)
            rel = np.abs(a - b) / denom
            # ratio diagnostic on significant points
            sel = np.abs(a) > 1e-6
            ratio = np.median(b[sel] / a[sel]) if sel.any() else np.nan
            print(
                f"  spin{s} feat{i}: max_rel={rel.max():.3e} "
                f"med_rel={np.median(rel):.3e} med_ratio(scf/train)={ratio:.6f}"
            )


if __name__ == "__main__":
    which = sys.argv[1:] or ["A1"]
    if "A1" in which:
        run_a1()
    if "A2" in which:
        run_a2()
    if "A3" in which:
        run_a3()


def _shim_spin_cycling(ni):
    """Runtime-only confirmation shim (NO repo changes): nr_uks calls
    nldfgen.get_potential twice per dm set, alpha then beta, without the
    spin argument (the suspected bug). This wrapper cycles spin=0,1 per
    call, reproducing the hypothesized fix at runtime."""
    gen = ni.nldfgen
    orig = gen.get_potential
    state = {"i": 0}

    def wrapped(vfeat, spin=None, **kwargs):
        s = state["i"] % 2
        state["i"] += 1
        return orig(vfeat, spin=s, **kwargs)

    gen.get_potential = wrapped
    return gen, orig


def run_confirm():
    print("=" * 70)
    print("CONFIRM: linear-probe FD with spin-cycling get_potential shim")
    print("=" * 70)
    ml = make_linear_probe_model()
    for molname in ("h2o", "h2o+"):
        mol = get_mol(molname)
        dm = converged_dm(mol, grids_level=1)
        mf = _make_calc(mol, ml)
        ni, grids = prep_cider_ni(mf)
        # initialize generators so the shim can attach; nr_uks re-inits
        # but keeps the same nldfgen instance for same mol/grids/nspin
        nspin = 1 if mol.spin == 0 else 2
        ni.initialize_feature_generators(mol, grids, nspin)
        if nspin == 2:
            _shim_spin_cycling(ni)
        results = fd_energy_potential_check(
            ni, mol, grids, dm, n_dirs=3, deltas=[1e-3, 2.5e-4]
        )
        for i, entry in enumerate(results):
            for dl in sorted(entry, reverse=True):
                lhs, rhs = entry[dl]
                rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
                print(
                    f"  {molname:5s} dir{i} d={dl:g}: rel={rel:.3e}"
                )


def run_impact():
    print("=" * 70)
    print("IMPACT: converged UKS energies, buggy vs class-patched potential")
    print("=" * 70)
    from pyscf import gto

    def make_mol(name):
        if name == "o2":
            return gto.M(
                atom="O 0 0 0; O 0 0 1.208", basis="def2-svp", spin=2,
                verbose=0, output="/dev/null",
            )
        return get_mol(name)

    import contextlib

    @contextlib.contextmanager
    def class_level_spin_fix(ni_cls_probe_gen):
        """Patch get_potential at the CLASS level so generators recreated
        during mf.kernel() inherit the fix. Spin inferred from per-
        instance call order (nr_uks calls alpha then beta per dm set)."""
        cls = type(ni_cls_probe_gen)
        orig = cls.get_potential

        def wrapped(self, vfeat, spin=None, **kwargs):
            n = getattr(self, "_shim_ncalls", 0)
            self._shim_ncalls = n + 1
            return orig(self, vfeat, spin=n % 2, **kwargs)

        cls.get_potential = wrapped
        try:
            yield
        finally:
            cls.get_potential = orig

    # probe generator instance just to grab the class
    mf0 = _make_calc(make_mol("h2o+"), VB2_NONHYB)
    mf0.grids.build()
    mf0._numint.build(mf0.mol)
    mf0._numint.initialize_feature_generators(mf0.mol, mf0.grids, 2)
    gen_probe = mf0._numint.nldfgen

    for molname in ("h2o+", "o2", "h2+"):
        energies = {}
        dms = {}
        for label in ("buggy", "fixed"):
            mf = _make_calc(make_mol(molname), VB2_NONHYB)
            mf.conv_tol = 1e-10
            if label == "fixed":
                with class_level_spin_fix(gen_probe):
                    e = mf.kernel()
            else:
                e = mf.kernel()
            energies[label] = e
            dms[label] = np.asarray(mf.make_rdm1())
            print(
                f"  {molname:5s} {label:6s}: E={e:.10f} "
                f"converged={mf.converged}"
            )
        dE = energies["fixed"] - energies["buggy"]
        ddm = np.max(np.abs(dms["fixed"] - dms["buggy"]))
        print(f"  {molname:5s} DELTA:  dE={dE:+.3e} Ha  max|d(dm)|={ddm:.3e}")


_EXTRA = {"CONFIRM": run_confirm, "IMPACT": run_impact}
if __name__ == "__main__" and any(k in sys.argv for k in _EXTRA):
    for k, fn in _EXTRA.items():
        if k in sys.argv:
            fn()
