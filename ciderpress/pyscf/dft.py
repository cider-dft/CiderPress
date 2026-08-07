#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

import numpy as np

from pyscf import lib
from pyscf.dft.gen_grid import Grids

from ciderpress.dft.model_utils import (
    get_slxc_settings,
    load_cider_model,
    validate_cider_composition,
)
from ciderpress.pyscf.gen_cider_grid import CiderGrids
from ciderpress.pyscf.nldf_convolutions import PySCFNLDFInitializer
from ciderpress.pyscf.numint import (
    CiderNumInt,
    HybridNLDFNumInt,
    NLDFNLOFNumInt,
    NLDFNumInt,
    NLOFNumInt,
)
from ciderpress.pyscf.sdmx import PySCFSDMXInitializer


def _sanitize_vdw_value(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, dict):
        return {
            str(_sanitize_vdw_value(k)): _sanitize_vdw_value(val)
            for k, val in v.items()
        }
    if isinstance(v, (list, tuple)):
        return type(v)(_sanitize_vdw_value(x) for x in v)
    return v


def _make_d4_dispersion(mol, params):
    """Construct a D4 evaluator from serialized model parameters."""
    try:
        from pyscf.dispersion import dftd4
    except Exception as exc:
        raise RuntimeError(
            "D4 evaluation requires the optional pyscf-dispersion dependency. "
            "Install it with `pip install 'ciderpress[d4]'`."
        ) from exc

    params = _sanitize_vdw_value(params or {})
    allowed = {"xc", "ga", "gc", "wf", "atm"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"Unsupported D4 model parameters: {unknown!r}")
    xc = params.get("xc")
    if not xc:
        raise ValueError("The serialized D4 model parameters must specify 'xc'.")
    return dftd4.DFTD4Dispersion(
        mol,
        xc=xc,
        ga=params.get("ga"),
        gc=params.get("gc"),
        wf=params.get("wf"),
        atm=bool(params.get("atm", False)),
    )


def _validate_post_density_mode(vdw_eval_mode):
    if vdw_eval_mode not in (None, "post_density"):
        raise ValueError(f"Unsupported vdW evaluation mode: {vdw_eval_mode!r}")


def _get_present_dispersion_energy_ha(mf):
    """
    Detect and compute an already-enabled dispersion correction on a PySCF
    mean-field object (for example, a ``dftd4.pyscf.energy`` wrapper).

    Returns zero if no wrapper is present and raises if a detected wrapper
    cannot be evaluated; silently treating that case as zero would double-count
    or omit a dispersion contribution during reconciliation.
    """
    if getattr(mf, "_cider_vdw_applied", False) and hasattr(mf, "e_vdw_expected"):
        return float(getattr(mf, "e_vdw_expected"))
    wd4 = getattr(mf, "with_dftd4", None)
    if wd4 is not None:
        try:
            return float(wd4.kernel()[0])
        except Exception as exc:
            raise RuntimeError(
                "An attached D4 wrapper was detected, but its dispersion "
                "energy could not be evaluated."
            ) from exc
    wd3 = getattr(mf, "with_dftd3", None)
    if wd3 is not None:
        try:
            return float(wd3.kernel()[0])
        except Exception as exc:
            raise RuntimeError(
                "An attached D3 wrapper was detected, but its dispersion "
                "energy could not be evaluated."
            ) from exc
    return 0.0


def _compute_expected_vdw_energy_ha(mf, vdw_fit_info, vdw_eval_mode):
    """
    Compute the vdW term specified by vdw_fit_info.

    Contract: post-density evaluation (energy-only; no vxc feedback).
    """
    if vdw_fit_info is None:
        return 0.0
    vdw_fit_info = _sanitize_vdw_value(vdw_fit_info)
    kind = (vdw_fit_info.get("kind") or "").lower()
    params = vdw_fit_info.get("params") or {}
    mol = mf.mol
    _validate_post_density_mode(vdw_eval_mode)

    if kind == "d3":
        try:
            from pyscf.dispersion import dftd3
        except Exception as e:
            raise RuntimeError(
                "vdW term kind='d3' requested but pyscf-dispersion is unavailable."
            ) from e
        xc = params.get("xc")
        version = params.get("version")
        atm = bool(params.get("atm", False))
        disp = dftd3.DFTD3Dispersion(mol, xc=xc, version=version, atm=atm)
        return float(disp.get_dispersion()["energy"])

    if kind == "d4":
        disp = _make_d4_dispersion(mol, params)
        return float(disp.get_dispersion()["energy"])

    if kind == "nlc":
        from pyscf.dft import numint

        xc_code = params.get("xc_code") or "GGA_XC_VV10"
        grids_level = params.get("grids_level", None)
        grids = Grids(mol)
        if grids_level is None:
            grids.level = getattr(getattr(mf, "grids", None), "level", None) or 3
        else:
            grids.level = int(grids_level)
        grids.build(with_non0tab=True)
        dm = mf.make_rdm1()
        if hasattr(dm, "ndim") and dm.ndim == 3:
            dm = dm[0] + dm[1]
        ni = numint.NumInt()
        _, excsum, _ = numint.nr_nlc_vxc(ni, mol, grids, xc_code, dm)
        return float(excsum)

    raise ValueError(f"Unsupported vdW kind in vdw_fit_info: {vdw_fit_info!r}")


def _compute_expected_vdw_gradient_ha_per_bohr(
    mf, vdw_fit_info, vdw_eval_mode, mol=None, atmlst=None
):
    """Evaluate the analytical gradient prescribed by the model contract."""
    if mol is None:
        mol = mf.mol
    if vdw_fit_info is None:
        gradient = np.zeros((mol.natm, 3))
    else:
        vdw_fit_info = _sanitize_vdw_value(vdw_fit_info)
        kind = (vdw_fit_info.get("kind") or "").lower()
        params = vdw_fit_info.get("params") or {}
        _validate_post_density_mode(vdw_eval_mode)
        if kind == "d4":
            result = _make_d4_dispersion(mol, params).get_dispersion(grad=True)
            gradient = np.asarray(result["gradient"], dtype=float)
        elif kind in {"d3", "nlc"}:
            raise NotImplementedError(
                f"Analytical gradients for post-density {kind.upper()} model "
                "contracts are not implemented."
            )
        else:
            raise ValueError(
                f"Unsupported vdW kind in vdw_fit_info: {vdw_fit_info!r}"
            )
    if atmlst is not None:
        gradient = gradient[np.asarray(atmlst, dtype=int)]
    return gradient


def _apply_post_density_vdw_energy(mf):
    """
    Adjust mf.e_tot to be consistent with the trained-model vdW contract.

    If the incoming mean-field object already includes a dispersion correction
    (e.g. via DFTD4 wrapper), replace that contribution with the expected one.
    """
    if not getattr(mf, "_cider_vdw_contract_enabled", False):
        return mf.e_tot
    # Guard against multiple applications when kernel() is wrapped multiple times
    # (e.g. density_fit objects can have both _CiderDF.kernel and _CiderKS.kernel
    # in the MRO). The outermost kernel() should reset this flag before running SCF.
    if getattr(mf, "_cider_vdw_applied", False):
        return mf.e_tot

    vdw_term = getattr(mf, "_cider_vdw_fit_term", None)
    vdw_info = getattr(mf, "_cider_vdw_fit_info", None)
    vdw_eval_mode = getattr(mf, "_cider_vdw_eval_mode", None)

    present = _get_present_dispersion_energy_ha(mf)
    expected = (
        0.0
        if vdw_term is None
        else _compute_expected_vdw_energy_ha(mf, vdw_info, vdw_eval_mode)
    )

    mf.e_tot_base = float(mf.e_tot)
    mf.e_vdw_present = float(present)
    mf.e_vdw_expected = float(expected)
    mf.e_vdw_delta = float(expected - present)
    mf.e_tot = mf.e_tot_base + mf.e_vdw_delta
    mf._cider_vdw_applied = True
    return mf.e_tot


def _strip_present_dispersion_wrappers(mf):
    """
    If the starting mean-field object has a dispersion wrapper (D3/D4) enabled
    via external PySCF wrappers (e.g. dftd4.pyscf.energy), disable it before
    running SCF. This avoids confusing tags/prints and ensures the base SCF
    energy is not dispersion-shifted when the model contract expects no
    dispersion (or an NLC term).

    Note: D3/D4 are energy-only corrections in these wrappers; they do not affect
    the Fock matrix. Disabling them primarily affects reported energies/logging.
    """

    def _pick_base_cls(method_name: str):
        for cls in mf.__class__.mro():
            mod = getattr(cls, "__module__", "") or ""
            if mod.startswith(("dftd4", "dftd3")):
                continue
            if hasattr(cls, method_name):
                return cls
        return None

    def _rebind(method_name: str):
        base_cls = _pick_base_cls(method_name)
        if base_cls is None:
            return
        meth = getattr(base_cls, method_name, None)
        if meth is None:
            return
        setattr(mf, method_name, meth.__get__(mf, base_cls))

    had = False
    if getattr(mf, "with_dftd4", None) is not None:
        had = True
        mf.with_dftd4 = None
    if getattr(mf, "with_dftd3", None) is not None:
        had = True
        mf.with_dftd3 = None
    if had:
        # Restore base behaviors that wrappers commonly override.
        _rebind("energy_nuc")
        _rebind("energy_elec")
        _rebind("dump_flags")
        if (
            getattr(mf, "with_dftd4", None) is not None
            or getattr(mf, "with_dftd3", None) is not None
        ):
            raise RuntimeError("Failed to disable an attached dispersion wrapper.")


class _CiderVDWGradient:
    """Add the model's post-density correction to a CIDER gradient."""

    def grad_nuc(self, mol=None, atmlst=None):
        nuc_gradient = super().grad_nuc(mol=mol, atmlst=atmlst)
        base = self.base
        vdw_gradient = _compute_expected_vdw_gradient_ha_per_bohr(
            base,
            getattr(base, "_cider_vdw_fit_info", None),
            getattr(base, "_cider_vdw_eval_mode", None),
            mol=mol,
            atmlst=atmlst,
        )
        return nuc_gradient + vdw_gradient

    def get_dispersion(self, *args, **kwargs):
        # grad_nuc includes the model-prescribed correction. Returning zero
        # prevents PySCF's mf.disp hook from adding it a second time.
        atmlst = getattr(self, "atmlst", None)
        natm = self.mol.natm if atmlst is None else len(atmlst)
        return np.zeros((natm, 3))


def _add_cider_vdw_gradient(mf, mf_grad):
    """Validate the model contract and decorate a supported gradient."""
    if not getattr(mf, "_cider_vdw_contract_enabled", False):
        return mf_grad
    vdw_term = getattr(mf, "_cider_vdw_fit_term", None)
    if vdw_term is None:
        return mf_grad

    vdw_info = _sanitize_vdw_value(getattr(mf, "_cider_vdw_fit_info", None))
    if not isinstance(vdw_info, dict):
        raise ValueError("A post-density vdW term requires serialized fit metadata.")
    kind = (vdw_info.get("kind") or "").lower()
    if kind != str(vdw_term).lower():
        raise ValueError(
            "The serialized vdW term and fit metadata disagree: "
            f"{vdw_term!r} != {kind!r}."
        )
    _validate_post_density_mode(getattr(mf, "_cider_vdw_eval_mode", None))
    if kind != "d4":
        raise NotImplementedError(
            f"Analytical gradients for post-density {kind.upper()} model "
            "contracts are not implemented."
        )
    return lib.set_class(mf_grad, (_CiderVDWGradient, mf_grad.__class__))


def make_cider_calc(
    ks,
    mlfunc,
    xmix=1.0,
    xc=None,
    xkernel=None,
    ckernel=None,
    mlfunc_format=None,
    nlc_coeff=None,
    nldf_init=None,
    sdmx_init=None,
    rhocut=None,
):
    """
    Decorate the PySCF DFT object ks with a CIDER functional mlfunc.
    If xc, xkernel, ckernel, and xmix are not specified,
    the equivalent of HF with CIDER in place of EXX is performed.
    The XC energy is::

        E_xc = xmix * E_x^CIDER + (1-xmix) * xkernel + ckernel + xc

    Mapped full-XC models already contain their complete baseline and require
    ``xmix=1.0``, ``xc=None``, ``xkernel=None``, and ``ckernel=None`` (the
    defaults). Other compositions are rejected to prevent double counting.

    NOTE: Only GGA-level XC functionals can be used with GGA-level
    (orbital-independent) CIDER functionals currently.

    Args:
        ks (pyscf.dft.KohnShamDFT): DFT object
        mlfunc (MappedXC, MappedXC2, str): CIDER exchange functional or file name
        xmix (float): Fraction of CIDER exchange used.
        xc (str or None): If specified, this semi-local XC code is evaluated
             and added to the total XC energy.
        xkernel (str or None): Semi-local X code in libxc. Scaled by (1-xmix).
        ckernel (str or None): Semi-local C code in libxc.
        mlfunc_format (str or None): 'joblib' or 'yaml', specifies the format
            of mlfunc if it is a string corresponding to a file name.
            If unspecified, infer from file extension and raise error
            if file type cannot be determined.
        nlc_coeff (tuple or None):
            VV10 coefficients. If None, VV10 term is not evaluated.
        nldf_init (PySCFNLDFInitializer)
        sdmx_init (PySCFSDMXInitializer)
        rhocut (float)

    Returns:
        A decorated Kohn-Sham object for performing a CIDER calculation.
    """
    mlfunc = load_cider_model(mlfunc, mlfunc_format)
    validate_cider_composition(
        mlfunc,
        xmix=xmix,
        xkernel=xkernel,
        ckernel=ckernel,
        xc=xc,
        backend="PySCF",
    )
    ks._xc = get_slxc_settings(xc, xkernel, ckernel, xmix)
    # Assign the PySCF-facing functional to be a simple SL
    # functional to avoid hybrid DFT being called.
    # NOTE this might need to be changed to some nicer
    # approach later.
    if mlfunc.settings.sl_settings.level == "MGGA":
        ks.xc = "R2SCAN"
    else:
        ks.xc = "PBE"
    new_ks = _CiderKS(
        ks,
        mlfunc,
        xmix=xmix,
        nldf_init=nldf_init,
        sdmx_init=sdmx_init,
        rhocut=rhocut,
        nlc_coeff=nlc_coeff,
    )
    new_ks = lib.set_class(new_ks, (_CiderKS, ks.__class__))

    # vdW contract propagation (joblib -> mapped YAML -> SCF):
    # If the mapped model carries vdW metadata, enforce it by replacing any
    # already-enabled dispersion term on the starting mean-field object with the
    # expected one (post-density evaluation).
    if hasattr(mlfunc, "vdw_fit_term") or hasattr(mlfunc, "vdw_fit_info"):
        new_ks._cider_vdw_contract_enabled = True
        new_ks._cider_vdw_fit_term = getattr(mlfunc, "vdw_fit_term", None)
        new_ks._cider_vdw_fit_info = getattr(mlfunc, "vdw_fit_info", None)
        new_ks._cider_vdw_eval_mode = getattr(mlfunc, "vdw_eval_mode", None)
    else:
        new_ks._cider_vdw_contract_enabled = False
        new_ks._cider_vdw_fit_term = None
        new_ks._cider_vdw_fit_info = None
        new_ks._cider_vdw_eval_mode = None

    return new_ks


class _CiderKS:

    grids = None

    def __init__(
        self,
        mf,
        mlxc,
        xmix=1.0,
        nldf_init=None,
        sdmx_init=None,
        rhocut=None,
        nlc_coeff=None,
    ):
        self.__dict__.update(mf.__dict__)
        # self.mlxc = None
        # self.xmix = None
        # self.nldf_init = None
        # self.sdmx_init = None
        # self.rhocut = rhocut
        self.set_mlxc(
            mlxc,
            xmix=xmix,
            nldf_init=nldf_init,
            sdmx_init=sdmx_init,
            rhocut=rhocut,
            nlc_coeff=nlc_coeff,
        )

    def set_mlxc(
        self,
        mlxc,
        xmix=1.0,
        nldf_init=None,
        sdmx_init=None,
        rhocut=None,
        nlc_coeff=None,
    ):
        # self.mlxc = mlxc
        # self.xmix = xmix
        # self.nldf_init = nldf_init
        # self.sdmx_init = sdmx_init
        if nldf_init is None and mlxc.settings.has_nldf:
            nldf_init = PySCFNLDFInitializer(mlxc.settings.nldf_settings)
        if sdmx_init is None and mlxc.settings.has_sdmx:
            sdmx_init = PySCFSDMXInitializer(mlxc.settings.sdmx_settings, lowmem=False)
        old_grids = self.grids
        changed = False
        if mlxc.settings.has_nldf:
            if not isinstance(old_grids, CiderGrids):
                changed = True
                self.grids = CiderGrids(self.mol)
        else:
            if old_grids is None or isinstance(old_grids, CiderGrids):
                changed = True
                self.grids = Grids(self.mol)
        if changed:
            for key in (
                "atom_grid",
                "atomic_radii",
                "radii_adjust",
                "radi_method",
                "becke_scheme",
                "prune",
                "level",
            ):
                self.grids.__setattr__(key, old_grids.__getattribute__(key))
        settings = mlxc.settings
        has_nldf = not settings.nldf_settings.is_empty
        has_nlof = not settings.nlof_settings.is_empty
        has_hyb = settings.has_hyb

        # Choose appropriate NumInt class based on features
        if has_hyb and has_nldf:
            cls = HybridNLDFNumInt
        elif has_hyb:
            cls = CiderNumInt
        elif has_nldf and has_nlof:
            cls = NLDFNLOFNumInt
        elif has_nldf:
            cls = NLDFNumInt
        elif has_nlof:
            cls = NLOFNumInt
        else:
            cls = CiderNumInt
        self._numint = cls(
            mlxc,
            self._xc,
            nldf_init,
            sdmx_init,
            xmix=xmix,
            rhocut=rhocut,
            nlc_coeff=nlc_coeff,
        )

    def build(self, mol=None, **kwargs):
        self._numint.build(mol=mol)
        return super().build(mol, **kwargs)

    def reset(self, mol=None):
        self._numint.reset(mol=mol)
        return super().reset(mol)

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        new_self = super().density_fit(
            auxbasis=auxbasis, with_df=with_df, only_dfj=only_dfj
        )
        # Preserve vdW contract across density_fit() wrapper.
        for k in (
            "_cider_vdw_contract_enabled",
            "_cider_vdw_fit_term",
            "_cider_vdw_fit_info",
            "_cider_vdw_eval_mode",
        ):
            if hasattr(self, k):
                setattr(new_self, k, getattr(self, k))
        lib.set_class(new_self, (_CiderDF, new_self.__class__))
        return new_self

    def kernel(self, *args, **kwargs):
        # Reset per-kernel-call guard to ensure vdW is applied once at the end.
        self._cider_vdw_applied = False
        # If the trained model contract does not want a D3/D4 dispersion term,
        # disable any dispersion wrapper inherited from a restarted baseline job.
        if getattr(self, "_cider_vdw_contract_enabled", False):
            vdw_term = getattr(self, "_cider_vdw_fit_term", None)
            vdw_info = getattr(self, "_cider_vdw_fit_info", None)
            kind = None
            if isinstance(vdw_info, dict):
                kind = (vdw_info.get("kind") or "").lower()
            wants_disp = kind in {"d3", "d4"}
            if (vdw_term is None) or (not wants_disp):
                _strip_present_dispersion_wrappers(self)
        super().kernel(*args, **kwargs)
        return _apply_post_density_vdw_energy(self)

    def nuc_grad_method(self):
        from pyscf import dft

        has_df = hasattr(self, "with_df") and self.with_df is not None
        if isinstance(self, dft.rks.RKS):
            from ciderpress.pyscf import rks_grad

            if has_df:
                mf_grad = rks_grad.DFGradients(self)
            else:
                mf_grad = rks_grad.Gradients(self)
        elif isinstance(self, dft.uks.UKS):
            from ciderpress.pyscf import uks_grad

            if has_df:
                mf_grad = uks_grad.DFGradients(self)
            else:
                mf_grad = uks_grad.Gradients(self)
        else:
            return None
        return _add_cider_vdw_gradient(self, mf_grad)

    Gradients = nuc_grad_method

    Hessian = method_not_implemented
    NMR = method_not_implemented
    NSR = method_not_implemented
    Polarizability = method_not_implemented
    RotationalGTensor = method_not_implemented
    MP2 = method_not_implemented
    CISD = method_not_implemented
    CCSD = method_not_implemented
    CASCI = method_not_implemented
    CASSCF = method_not_implemented


class _CiderDF:
    nuc_grad_method = _CiderKS.nuc_grad_method

    def kernel(self, *args, **kwargs):
        self._cider_vdw_applied = False
        if getattr(self, "_cider_vdw_contract_enabled", False):
            vdw_term = getattr(self, "_cider_vdw_fit_term", None)
            vdw_info = getattr(self, "_cider_vdw_fit_info", None)
            kind = None
            if isinstance(vdw_info, dict):
                kind = (vdw_info.get("kind") or "").lower()
            wants_disp = kind in {"d3", "d4"}
            if (vdw_term is None) or (not wants_disp):
                _strip_present_dispersion_wrappers(self)
        super().kernel(*args, **kwargs)
        return _apply_post_density_vdw_energy(self)
