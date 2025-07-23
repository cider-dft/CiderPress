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

import os

import numpy as np
from pyscf.dft.libxc import eval_xc
from pyscf.lib import chkfile, prange
from scipy.linalg import cho_solve, cholesky

from ciderpress.dft.plans import get_rho_tuple_with_grad_cross, vxc_tuple_to_array
from ciderpress.dft.settings import CFC, FeatureSettings
from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2


def strk_to_tuplek(d, ref=False):
    if not isinstance(d, dict):
        raise NotImplementedError
    nd = {}
    for occ, vocc in d.items():
        for num, vnum in vocc.items():
            if ref:
                nd[occ, int(num)] = vnum
                continue
            if isinstance(vnum, (tuple, list)):
                s = vnum[0]
                v = vnum[1]
            else:
                s = 0
                v = vnum
            nd[occ, int(num)] = (s, v)
    return nd


class MOLGP:
    """
    Gaussian process model for the exchange-correlation functional
    or its components.
    """

    def __init__(
        self,
        kernels,
        settings,
        libxc_baseline=None,
        default_noise=0.030,
    ):
        """
        Args:
            kernels (list[DFTKernel]): List of kernels that sum to the XC energy.
            settings (DescParams or FeatureSettings): Settings for the
                features. Specifies what the feature vector is.
            libxc_baseline (str or None): Additional baseline functional to
                evaluate using the libxc library.
            default_noise (float): Default noise hyperparameter, used if noise
                is not provided for a particular data point
        """
        self.libxc_baseline = libxc_baseline
        if not isinstance(settings, FeatureSettings):
            raise ValueError("settings must be FeatureSettings")
        self.settings = settings
        if self.libxc_baseline is not None:
            raise NotImplementedError
        if not isinstance(kernels, list):
            kernels = [kernels]
        if len(kernels) < 1:
            raise ValueError("Need at least 1 covariance kernel")
        self.kernels = kernels
        self.default_noise = default_noise

        self.numerical_epsilon = 1e-9
        self.args = None

        self.reset_reactions()
        self.Xctrl_exch = None
        self.Xctrl_corr = None
        self.y = None
        self.K_ = None
        self.Kcov_ = None
        self.alpha_mol_ = None
        self.y_mol_ = None
        self.ks_baseline_dict = {}
        self.exx_ref_dict = {}
        self.dexx_ref_dict = {}
        # Reference exact exchange values with baseline subtracted
        self.exch_ref_dict = {}
        # Vector of exchange covariances with control points
        self.exch_cov_dict = {}
        # Vector of derivative of exchange covariances with control points,
        # taken with respect to orbital occupation
        self.exch_dcov_dict = {}
        # Baseline for XC energy
        self.corr_bas_dict = {}
        # Vector of correlation covariances with control points
        self.corr_cov_dict = {}
        # Vector of derivative of correlation covariances with control points
        self.corr_dcov_dict = {}

    def map(self, mapping_plans):
        """
        Map the MOLGP model to an Evaluator object that can efficiently
        evaluate the XC energy and which can also be serialized.

        Returns:
            list(callable): Functions that map each
                kernel to a MappedDFTKernel object
        """
        mapped_kernels = [
            kernel.map(mapping_plan)
            for kernel, mapping_plan in zip(self.kernels, mapping_plans)
        ]
        return MappedXC(
            mapped_kernels,
            self.settings,
            libxc_baseline=self.libxc_baseline,
        )

    @property
    def num_kernels(self):
        return len(self.kernels)

    @property
    def xkernels(self):
        return [k for k in self.kernels if k.component == "x"]

    @property
    def ckernels(self):
        return [k for k in self.kernels if k.component != "x"]

    def fit(self, x=None, sigma_min=0.25):

        # --- DEBUG PRINTING START for fit method ---
        print(f"\nDEBUG fit: Entering fit method.")
        print(f"DEBUG fit: Length of self.rxn_ref_list: {len(self.rxn_ref_list)}")
        print(f"DEBUG fit: Length of self.rxn_noise_list: {len(self.rxn_noise_list)}")
        print(f"DEBUG fit: Number of kernels in self.kernels: {len(self.kernels)}")
        if len(self.kernels) > 0:
            for i_k_debug, k_debug in enumerate(self.kernels):
                 print(f"DEBUG fit:   Kernel #{i_k_debug}: type={type(k_debug).__name__}, component='{k_debug.component}', "
                       f"id={id(k_debug)}, len(rxn_cov_list)={len(k_debug.rxn_cov_list if hasattr(k_debug, 'rxn_cov_list') else -1)}")
        # --- DEBUG PRINTING END ---

        Kimn_list = []
        Knmimn = 0
        for kernel_idx_fit, kernel in enumerate(self.kernels): # Original L148, added kernel_idx_fit
            # --- DEBUG PRINTING START for kernel in fit ---
            print(f"DEBUG fit: Processing kernel #{kernel_idx_fit} in fit loop: type={type(kernel).__name__}, component='{kernel.component}', id={id(kernel)}")
            print(f"DEBUG fit:   Length of this kernel.rxn_cov_list before np.stack: {len(kernel.rxn_cov_list)}")
            if len(kernel.rxn_cov_list) > 0:
                print(f"DEBUG fit:     First element type in kernel.rxn_cov_list: {type(kernel.rxn_cov_list[0])}")
                if hasattr(kernel.rxn_cov_list[0], 'shape'):
                    print(f"DEBUG fit:     First element shape: {kernel.rxn_cov_list[0].shape}")
            else:
                print(f"DEBUG fit:     kernel.rxn_cov_list IS EMPTY, np.stack will fail here!")
            # --- DEBUG PRINTING END ---
            Kmm = kernel.get_kctrl()
            Kmn = np.stack(kernel.rxn_cov_list).T
            M = Kmm.shape[0]
            mini_noise = self.numerical_epsilon * np.identity(M)
            L_ = cholesky(Kmm + mini_noise, lower=True)
            Kimn = cho_solve((L_, True), Kmn)
            Knmimn += Kmn.T.dot(Kimn)
            Kimn_list.append(Kimn)
        noise_nn = np.array(self.rxn_noise_list)
        noise_nn = noise_nn**2  # get noise covariance from noise std deviation
        if x is not None:
            Knmimn[:] *= x[0] ** 2
            Kimn_list = [x[0] ** 2 * k for k in Kimn_list]
            noise_nn[:] *= sigma_min + x[1] ** 2
        y = np.array(self.rxn_ref_list)
        if len(noise_nn) == 0:
            raise ValueError("Need to set reactions using add_reactions")
        K = Knmimn + np.diag(noise_nn)
        K += self.numerical_epsilon * np.identity(noise_nn.size)
        self.Kcov_ = Knmimn
        self.K_ = K
        L_ = cholesky(K, lower=True)
        alpha_new = cho_solve((L_, True), y)
        self.alpha_mol_ = alpha_new
        self.y_mol_ = y
        for ik, kernel in enumerate(self.kernels):
            kernel.alpha = np.dot(Kimn_list[ik], alpha_new)

    def compute_likelihood(self, x=None, sigma_min=0.25):
        assert self.alpha_mol_ is not None
        if x is None:
            x = np.array([1.0, 1.0])
        y = self.y_mol_
        noise = (sigma_min + x[1] ** 2) * (self.K_ - self.Kcov_)
        Kfull = x[0] ** 2 * self.Kcov_ + noise
        Lfull = cholesky(Kfull, lower=True)
        vec = np.linalg.solve(Lfull, y)
        likelihood = -0.5 * vec.dot(vec)
        # likelihood = -0.5 * np.dot(y, np.linalg.solve(Kfull, y))
        likelihood -= 0.5 * np.linalg.slogdet(Kfull)[1]
        n = y.size
        likelihood -= n / 2 * np.log(2 * np.pi)
        return likelihood

    def optimize_cov_and_noise_(self, refit=True, sigma_min=0.5):
        assert self.alpha_mol_ is not None
        from scipy.optimize import minimize

        def fun(x, *args):
            return -1 * self.compute_likelihood(x, sigma_min=sigma_min)

        res = minimize(fun, np.array([1.0, 1.0]), tol=1e-3)
        print(res.success, res.fun, res.x)
        if refit:
            self.fit(x=res.x, sigma_min=sigma_min)

    def set_control_points(self, X0T_list, reduce=True):
        """
        Transform raw descriptors X and use them as
        control points.

        Args:
            X: Numpy array of raw features
        """
        for kernel in self.kernels:
            kernel.set_control_points(X0T_list, reduce=reduce)

    def _get_normalized_features(self, desc):
        return self.settings.normalizers.get_normalized_feature_vector(desc)

    def _get_normalized_feature_derivs(self, desc, ddesc):
        return self.settings.normalizers.get_derivative_of_normed_features(desc, ddesc)

    @staticmethod
    def load_data(ddir, mol_id, get_orb_deriv):
        print("Loading data for ", mol_id)
        print("ddir is ", ddir)
        print("get_orb_deriv is ", get_orb_deriv)
        fname = os.path.join(ddir["REF"], mol_id + ".hdf5")
        print("fname for REF: ", fname)
        all_data = chkfile.load(fname, "train_data")

        # --- DEBUG PRINTING START for MOLGP.load_data (REF file) ---
        print(f"DEBUG load_data: REF file '{fname}' loaded. Content overview:")
        for key_in_all_data, val_in_all_data in all_data.items():
            val_repr = "N/A"
            val_shape_repr = "N/A"
            if isinstance(val_in_all_data, np.ndarray):
                val_shape_repr = str(val_in_all_data.shape)
                # Avoid printing huge arrays, show a small slice or just stats
                if val_in_all_data.size > 20:
                    val_repr = f"<ndarray, mean:{np.mean(val_in_all_data):.2e}, std:{np.std(val_in_all_data):.2e}>"
                else:
                    val_repr = str(val_in_all_data)
            elif isinstance(val_in_all_data, dict):
                val_repr = f"<dict with keys: {list(val_in_all_data.keys())}>"
            elif isinstance(val_in_all_data, (list, tuple)):
                 if len(val_in_all_data) > 5:
                     val_repr = f"<{type(val_in_all_data).__name__} of len {len(val_in_all_data)}, first 5: {val_in_all_data[:5]}>"
                 else:
                     val_repr = str(val_in_all_data)
            else:
                val_repr = str(val_in_all_data)

            print(f"DEBUG load_data:   all_data['{key_in_all_data}']: type={type(val_in_all_data)}, shape={val_shape_repr}, value_summary='{val_repr}'")
        # --- DEBUG PRINTING END ---

        all_data["desc"] = []
        all_data["ddesc"] = []
        for feat_type in ["SL", "NLDF", "NLOF", "SDMX", "HYB"]:
            if ddir[feat_type] is None:
                print(f"Skipping {feat_type} - directory is None")
                if feat_type == "SL":
                    raise ValueError("Need semilocal features")
                else:
                    continue
            else:
                fname = os.path.join(ddir[feat_type], mol_id + ".hdf5")
                print("fname for ", feat_type, ": ", fname)
                data = chkfile.load(fname, "train_data")

                # --- DEBUG PRINTING START for MOLGP.load_data (feat_type file) ---
                print(f"DEBUG load_data: Feature file '{fname}' for feat_type '{feat_type}' loaded. Content overview:")
                for key_in_data, val_in_data in data.items(): # Changed variable names to avoid clash
                    val_repr = "N/A"
                    val_shape_repr = "N/A"
                    if isinstance(val_in_data, np.ndarray):
                        val_shape_repr = str(val_in_data.shape)
                        if val_in_data.size > 20:
                             val_repr = f"<ndarray, mean:{np.mean(val_in_data):.2e}, std:{np.std(val_in_data):.2e}>"
                        else:
                            val_repr = str(val_in_data)
                    elif isinstance(val_in_data, dict):
                        val_repr = f"<dict with keys: {list(val_in_data.keys())}>"
                    elif isinstance(val_in_data, (list, tuple)):
                        if len(val_in_data) > 5:
                            val_repr = f"<{type(val_in_data).__name__} of len {len(val_in_data)}, first 5: {val_in_data[:5]}>"
                        else:
                            val_repr = str(val_in_data)
                    else:
                        val_repr = str(val_in_data)
                    print(f"DEBUG load_data:   data['{key_in_data}']: type={type(val_in_data)}, shape={val_shape_repr}, value_summary='{val_repr}'")
                # --- DEBUG PRINTING END ---

                print(f"Keys available in data: {list(data.keys())}")
                all_data["desc"].append(data["desc"])
                print(f"get_orb_deriv value: {get_orb_deriv}")
                # if get_orb_deriv or (get_orb_deriv is None and "ddesc" in data):
                if get_orb_deriv:
                    print(f"Inside a If statement - Attempting to append ddesc for {feat_type}")
                    all_data["ddesc"].append(data["ddesc"])
                    get_orb_deriv = True
                else:
                    get_orb_deriv = False
        all_data["desc"] = np.concatenate(all_data["desc"], axis=-2)
        assert all_data["desc"].ndim == 3
        if get_orb_deriv:
            ddesc = all_data["ddesc"]
            all_data["ddesc"] = {}
            ddesc0 = ddesc[0]
            for k1, v1 in ddesc0.items():
                all_data["ddesc"][k1] = {}
                for k2, v2 in ddesc0[k1].items():
                    all_data["ddesc"][k1][k2] = []
                    if all_data["desc"].shape[0] == 2:
                        assert len(v2) == 2
                        spin = ddesc0[k1][k2][0]
                        all_data["ddesc"][k1][k2] = []
                        for sub_ddesc in ddesc:
                            assert spin == sub_ddesc[k1][k2][0]
                            all_data["ddesc"][k1][k2].append(sub_ddesc[k1][k2][1])
                        all_data["ddesc"][k1][k2] = (
                            spin,
                            np.concatenate(all_data["ddesc"][k1][k2], axis=0),
                        )
                        assert all_data["ddesc"][k1][k2][1].ndim == 2
                    else:
                        for sub_ddesc in ddesc:
                            all_data["ddesc"][k1][k2].append(sub_ddesc[k1][k2])
                        all_data["ddesc"][k1][k2] = np.concatenate(
                            all_data["ddesc"][k1][k2], axis=0
                        )
                        assert all_data["ddesc"][k1][k2].ndim == 2
        else:
            all_data.pop("ddesc")
        return all_data

    def _compute_mol_covs(
        self, ddir, mol_ids, kernel, get_orb_deriv=None, save_refs=False
    ):
        # mode should be x or c
        blksize = 10000
        if self.args is not None:
            debug_model = self.args.debug_model
            debug_spline = self.args.debug_spline
        else:
            debug_model = None
            debug_spline = None

        deriv = None
        for i, mol_id in enumerate(mol_ids):
            print("MOL ID", mol_id)
            data = self.load_data(ddir, mol_id, get_orb_deriv)
            if deriv is None:
                if get_orb_deriv is None:
                    if "ddesc" in data:
                        deriv = True
                    else:
                        deriv = False
                else:
                    deriv = get_orb_deriv
            if deriv:
                assert "ddesc" in data

            weights = data["wt"]
            # nspin = data['nspin']
            desc = data["desc"]
            vwrtt_tot = 0
            baseline = 0
            if debug_spline is not None:
                etst = 0
            if deriv:
                ddesc = strk_to_tuplek(data["ddesc"])
                self.dexx_ref_dict[mol_id] = strk_to_tuplek(data["dval"], ref=True)

                # --- DEBUG PRINTING START for _compute_mol_covs (dexx_ref_dict) ---
                print(f"DEBUG _compute_mol_covs ({mol_id}): Populating self.dexx_ref_dict[{mol_id}]")
                val_to_print = self.dexx_ref_dict[mol_id]
                if isinstance(val_to_print, dict):
                    for orb_key, orb_val in val_to_print.items():
                        print(f"DEBUG _compute_mol_covs ({mol_id}):   dexx_ref_dict[{mol_id}][{orb_key}]: {orb_val} (type: {type(orb_val)}, shape: {getattr(orb_val, 'shape', 'N/A')})")
                else:
                     print(f"DEBUG _compute_mol_covs ({mol_id}):   dexx_ref_dict[{mol_id}] (overall): {val_to_print} (type: {type(val_to_print)}, shape: {getattr(val_to_print, 'shape', 'N/A')})")
                # --- DEBUG PRINTING END ---

                dvwrtt_tot = {k: 0 for k in ddesc.keys()}
                dbaseline = {k: 0 for k in ddesc.keys()}
            for i0, i1 in prange(0, weights.size, blksize):
                X0T = self._get_normalized_features(desc[..., i0:i1])
                wt = weights[i0:i1]
                m, dm = kernel.multiplicative_baseline(X0T)
                a, da = kernel.additive_baseline(X0T)
                if debug_spline is not None:
                    tmp, dtmp = debug_spline(X0T, rhocut=1e-7)
                    etst += np.dot(tmp, wt)
                # k is (Nctrl, nspin, Nsamp)
                # dk is (Nctrl, nspin, N0, Nsamp)
                if deriv:
                    k, dkdX0T = kernel.get_k_and_deriv(X0T)
                else:
                    k = kernel.get_k(X0T)
                    dkdX0T = None
                # TODO setting nan to zero could cover up more serious issues,
                # but it is the easiest way to take care of small density
                # training data without editing functions used at eval-time.
                if kernel.mode == "SEP":
                    cond = X0T[:, 0] < 1e-6
                    m[cond] = 0.0
                    a[cond] = 0.0
                    k[:, cond] = 0.0
                    for s in range(X0T.shape[0]):
                        dm[s][:, cond[s]] = 0.0
                        da[s][:, cond[s]] = 0.0
                        if deriv:
                            dkdX0T[:, s][:, :, cond[s]] = 0.0
                else:
                    cond = X0T[:, 0].sum(0) < 1e-6
                    m[..., cond] = 0.0
                    dm[..., cond] = 0.0
                    a[..., cond] = 0.0
                    da[..., cond] = 0.0
                    k[..., cond] = 0.0
                    if deriv:
                        print(dkdX0T.shape)
                        dkdX0T[..., cond] = 0.0
                if kernel.mode == "SEP":
                    km = (k * m).sum(1)
                    if deriv:
                        dkm = dkdX0T * m[:, None, :]
                        dkm += k[:, :, None, :] * dm
                else:
                    km = k * m
                    if deriv:
                        dkm = dkdX0T * m + dm * k[:, None, None, :]
                vwrtt_tot += (km * wt).sum(axis=1)
                baseline += (a * wt).sum()
                if deriv:
                    for orb, (s, ddesc_tmp) in ddesc.items():
                        # ddesc_tmp has shape (N0, Nsamp)
                        # avoid overwriting slice
                        ddesc_tmp = wt * self._get_normalized_feature_derivs(
                            desc[s, :, i0:i1], ddesc_tmp[:, i0:i1]
                        )
                        dvwrtt_tot[orb] += np.einsum("cdn,dn->c", dkm[:, s], ddesc_tmp)
                        dbaseline[orb] += (da[s] * ddesc_tmp).sum()
            kernel.cov_dict[mol_id] = vwrtt_tot
            kernel.base_dict[mol_id] = baseline
            if debug_model is not None:
                diff = debug_model.kernels[0].alpha.dot(vwrtt_tot)
                print("REF VWRTT", diff, kernel.base_dict[mol_id])
            if debug_spline is not None:
                print("DEBUG EN", etst)
            print("BASELINE", baseline)
            if save_refs:
                self.exx_ref_dict[mol_id] = (data["val"] * weights).sum()
                print("exx_ref", self.exx_ref_dict[mol_id])
                self.ks_baseline_dict[mol_id] = data["e_tot_orig"] - data["exc_orig"]
            if deriv:
                kernel.dcov_dict[mol_id] = dvwrtt_tot
                kernel.dbase_dict[mol_id] = dbaseline
                print("DBASELINE", dbaseline)
                if debug_model is not None:
                    diff = {
                        k: debug_model.kernels[0].alpha.dot(dvwrtt)
                        for k, dvwrtt in dvwrtt_tot.items()
                    }
                    print("REF DVWRTT", diff, kernel.dbase_dict[mol_id])

    def add_ueg_reactions(
        self, rholist, zeta, xc_code, mode, omega=None, rel_noise=1e-8
    ):
        # TODO test for version 1 and 2, X only, XC, short-range X
        if not isinstance(rholist, np.ndarray):
            rholist = np.array([rholist])
        if zeta is not None:
            zeta = np.minimum(zeta, 1 - 1e-6)
            zeta = np.maximum(zeta, -1 + 1e-6)
            rholist = np.stack([0.5 * rholist * (1 + zeta), 0.5 * rholist * (1 - zeta)])
            nspin = 2
        else:
            rholist = rholist[None, :]
            nspin = 1
        refval = eval_xc(xc_code, rholist, spin=nspin - 1, deriv=0, omega=omega)[0]
        rxn_list = []
        is_mgga = self.settings.sl_settings.level == "MGGA"
        rxn_id_list = []
        for ikernel, kernel in enumerate(self.kernels):
            for i in range(rholist.shape[1]):
                rho = rholist[:, i : i + 1]
                # We want to get energy per particle from energy density
                wt = 1.0 / rho.sum()
                X0T = np.stack(
                    [
                        self.settings.ueg_vector(rho=r * nspin, with_normalizers=True)
                        for r in rho[:, 0]
                    ]
                )[..., None]
                print(X0T.shape)
                baseline = 0
                vwrtt_tot = 0
                if isinstance(self, MOLGP2):
                    tau = CFC * rho * (nspin * rho) ** (2.0 / 3)
                    sigma = np.zeros((2 * nspin - 1, 1))
                    if is_mgga:
                        rho_tuple = (rho, sigma, tau)
                    else:
                        rho_tuple = (rho, sigma)
                    m = kernel.multiplicative_baseline(rho_tuple)[0]
                    a = kernel.additive_baseline(rho_tuple)[0]
                else:
                    m = kernel.multiplicative_baseline(X0T)[0]
                    a = kernel.additive_baseline(X0T)[0]
                k = kernel.get_k(X0T)
                if kernel.mode == "SEP":
                    km = (k * m).sum(1)
                else:
                    km = k * m
                vwrtt_tot += (km * wt).sum(axis=1)
                baseline += (a * wt).sum()
                mol_id = f"_UEG_{i:010d}"
                kernel.cov_dict[mol_id] = vwrtt_tot
                kernel.base_dict[mol_id] = baseline
                if ikernel == 0:
                    rxn_id_list.append(mol_id)
                    rxn = {
                        "unit": 1.0,
                        "structs": [mol_id],
                        "counts": [1],
                        "energy": refval[i] if mode > 0 else None,
                        "noise": rel_noise * np.abs(refval[i])+0.3,
                    }
                    rxn_list.append((mode, rxn))
                    if mode == 0:
                        self.exx_ref_dict[mol_id] = refval[i]
                    else:
                        self.ks_baseline_dict[mol_id] = 0
        self.add_reactions(rxn_list)
        return rxn_id_list

    def store_mol_covs(self, ddir, mol_ids, get_orb_deriv=None, get_correlation=True):
        """
        Reads features (and optionally their derivatives with respect to orbital
        occupations) and computes covariances with the control point sets of all
        the kernels of the models (or only the exchange components if
        get_correlation=False). Also stores the reference energy data
        in the dataset

        Args:
            ddir (str or dict[str]): If using DescParams (old version), should
                be single directory with the feature vectors. If using
                FeatureSettings (new version), should be dictionary of
                directories, with 'REF', 'SL', 'NLDF', 'NLOF', 'SDMX', and
                'HYB' being the keys. All are optional except 'SL' and 'REF'.
                'HYB' stands for reference data. It should contain
                reference energies and raw, spin-polarized
                density/gradient/kinetic energy data.
            mol_ids (list[str]): system ids to search for in ddir
            get_orb_deriv (bool): Whether to read the orbital occupation
                derivatives. If None, reads them iff they are available.
                If True, reads them and throws an error if they are not
                available. If False, does not read them.
            get_correlation (bool): Whether to compute covariance for the
                correlation kernels (True) or just the exchange kernels
                (False).
        """
        for i, kernel in enumerate(self.kernels):
            print(ddir, get_orb_deriv, get_correlation)
            save_refs = i == 0
            if isinstance(get_orb_deriv, (list, tuple)):
                deriv = get_orb_deriv[i]
            else:
                deriv = get_orb_deriv
            if get_correlation or kernel.component == "x":
                self._compute_mol_covs(
                    ddir, mol_ids, kernel, get_orb_deriv=deriv, save_refs=save_refs
                )

    def reset_reactions(self):
        self.rxn_ref_list = []
        self.rxn_noise_list = []
        for kernel in self.kernels:
            kernel.rxn_cov_list = []

    def add_reactions(self, rxn_list):
        """
        Store the reactions in rxn_list in the model. These will be used
        as training points when fit() is called. Note: For every mol_id
        present in the reactions, the corresponding system must have been
        added to the covariance dictionary via the store_mol_covs
        function.

        Args:
            rxn_list: List of tuples. In each tuple, element 0
                is an integer mode (mode: 0 (X), 1 (C), or 2 (XC)),
                and element 1 is a reaction dict. Each dict should have

                * structs: struct codes list
                  (If an element in structs is a tuple, the first
                  tuple element is a structure code, and the second
                  is an orbital code.)
                * counts: count codes list
                * energy: Energy of the reaction
                * unit: Eh per (rxn energy unit)
                * noise (optional): noise in Eh for the reaction
        """

        # --- DEBUG PRINTING START: Initial info for add_reactions ---
        print(f"\nDEBUG add_reactions: Received rxn_list with {len(rxn_list)} reaction entries.")
        # --- DEBUG PRINTING END ---
        
        for reaction_idx, (mode, rxn) in enumerate(rxn_list): # Added reaction_idx for easier tracking
            # --- DEBUG PRINTING START: Reaction Entry Info ---
            rxn_structs_info = str(rxn.get('structs', ['<no_structs>'])).replace(' ', '') # Compact
            if len(rxn_structs_info) > 120: rxn_structs_info = rxn_structs_info[:120] + "..."
            
            print(f"\n--- DEBUG add_reactions: PROCESSING RXN ENTRY #{reaction_idx} ---")
            print(f"DEBUG add_reactions:   Mode: {mode}")
            print(f"DEBUG add_reactions:   Reaction content (structs summary): {rxn_structs_info}")
            # --- DEBUG PRINTING END ---

            if mode == 1:
                raise NotImplementedError
            elif mode > 2:
                raise ValueError("Unsupported mode")
            
            rxn_ref = 0
            # --- DEBUG PRINTING START for rxn_ref ---
            print(f"DEBUG add_reactions:     rxn_ref_step_0 (initial): {rxn_ref} (type: {type(rxn_ref)})")
            # --- DEBUG PRINTING END ---

            if mode == 0:
                print(f"DEBUG add_reactions:   --- Mode 0: Initial rxn_ref calculation (from dexx_ref_dict/exx_ref_dict) ---")
                for sysid_idx, (sysid, count) in enumerate(zip(rxn["structs"], rxn["counts"])):
                    term_val, term_src = (None, "")
                    if isinstance(sysid, tuple):
                        term_val = self.dexx_ref_dict[sysid[0]][sysid[1]]
                        term_src = f"self.dexx_ref_dict[{sysid[0]}][{sysid[1]}]"
                    else:
                        term_src = f"self.exx_ref_dict[{sysid}]"
                        term_val = self.exx_ref_dict[sysid]
                    
                    print(f"DEBUG add_reactions:       mode0_sysid{sysid_idx}: Adding from {term_src} for sysid '{str(sysid)[:50]}...'")
                    print(f"DEBUG add_reactions:         Term value: {term_val} (type: {type(term_val)}, shape: {getattr(term_val, 'shape', 'N/A')}), Count: {count}")
                    rxn_ref += count * term_val
                    print(f"DEBUG add_reactions:         rxn_ref after this term: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")
            
            print(f"DEBUG add_reactions:   --- Applying XKernel Baselines to rxn_ref AND Calculating rxn_cov for XKernels ---")
            for xkernel_idx, kernel in enumerate(self.xkernels):
                kernel_name_info = f"xkernel[{xkernel_idx}]({type(kernel).__name__}, id={id(kernel)}, comp='{kernel.component}')"
                print(f"DEBUG add_reactions:     Processing {kernel_name_info}:")
                
                rxn_cov = 0 # Initialize rxn_cov for this kernel and this reaction
                # --- DEBUG PRINTING START for kernel.rxn_cov_list (xkernel) ---
                print(f"DEBUG add_reactions:       {kernel_name_info}: BEFORE append to its rxn_cov_list: len={len(kernel.rxn_cov_list)}")
                # --- DEBUG PRINTING END ---

                for sysid_idx, (sysid, count) in enumerate(zip(rxn["structs"], rxn["counts"])):
                    term_val_dbase, term_src_dbase = (None, "")
                    term_val_cov, term_src_cov = (None, "")

                    if isinstance(sysid, tuple):
                        # For rxn_cov
                        term_val_cov = kernel.dcov_dict[sysid[0]][sysid[1]]
                        term_src_cov = f"{kernel_name_info}.dcov_dict[{sysid[0]}][{sysid[1]}]"
                        rxn_cov += count * term_val_cov
                        
                        # For rxn_ref baseline subtraction
                        term_val_dbase = kernel.dbase_dict[sysid[0]][sysid[1]]
                        term_src_dbase = f"{kernel_name_info}.dbase_dict[{sysid[0]}][{sysid[1]}]"
                        print(f"DEBUG add_reactions:         sysid{sysid_idx} (tuple) '{str(sysid)[:50]}...':")
                        print(f"DEBUG add_reactions:           Subtracting for rxn_ref from {term_src_dbase}: Term value: {term_val_dbase} (type: {type(term_val_dbase)}, shape: {getattr(term_val_dbase, 'shape', 'N/A')}), Count: {count}")
                        rxn_ref -= count * term_val_dbase
                        print(f"DEBUG add_reactions:           rxn_ref after this dbase term: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")
                        print(f"DEBUG add_reactions:           Adding to rxn_cov from {term_src_cov}: Term value: {term_val_cov} (type: {type(term_val_cov)}, shape: {getattr(term_val_cov, 'shape', 'N/A')}), Count: {count}")

                    else: # sysid is a string
                        # For rxn_cov
                        term_val_cov = kernel.cov_dict[sysid]
                        term_src_cov = f"{kernel_name_info}.cov_dict[{sysid}]"
                        rxn_cov += count * term_val_cov

                        # For rxn_ref baseline subtraction
                        term_val_dbase = kernel.base_dict[sysid]
                        term_src_dbase = f"{kernel_name_info}.base_dict[{sysid}]"
                        print(f"DEBUG add_reactions:         sysid{sysid_idx} (string) '{str(sysid)[:50]}...':")
                        print(f"DEBUG add_reactions:           Subtracting for rxn_ref from {term_src_dbase}: Term value: {term_val_dbase} (type: {type(term_val_dbase)}, shape: {getattr(term_val_dbase, 'shape', 'N/A')}), Count: {count}")
                        rxn_ref -= count * term_val_dbase
                        print(f"DEBUG add_reactions:           rxn_ref after this base term: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")
                        print(f"DEBUG add_reactions:           Adding to rxn_cov from {term_src_cov}: Term value: {term_val_cov} (type: {type(term_val_cov)}, shape: {getattr(term_val_cov, 'shape', 'N/A')}), Count: {count}")
                    
                    print(f"DEBUG add_reactions:         Current rxn_cov for {kernel_name_info} after sysid{sysid_idx}: {rxn_cov} (type: {type(rxn_cov)}, shape: {getattr(rxn_cov, 'shape', 'N/A')})")

                # --- DEBUG PRINTING START for kernel.rxn_cov_list (xkernel) ---
                print(f"DEBUG add_reactions:       {kernel_name_info}: FINAL rxn_cov to append: {rxn_cov} (type: {type(rxn_cov)}, shape: {getattr(rxn_cov, 'shape', 'N/A')})")
                # --- DEBUG PRINTING END ---
                kernel.rxn_cov_list.append(rxn_cov)
                # --- DEBUG PRINTING START for kernel.rxn_cov_list (xkernel) ---
                print(f"DEBUG add_reactions:       {kernel_name_info}: AFTER append to its rxn_cov_list: len={len(kernel.rxn_cov_list)}")
                # --- DEBUG PRINTING END ---
            
            if mode == 2:
                print(f"DEBUG add_reactions:   --- Mode 2: Additional rxn_ref calculations & CKernel Processing ---")
                _original_rxn_unit_for_debug = rxn.get("unit")
                if rxn.get("unit") is None:
                    rxn["unit"] = 0.00159360109742136
                
                term_yaml_energy_val = rxn.get("energy", 0.0)
                term_yaml_unit_val = rxn["unit"]
                term_from_yaml_energy = term_yaml_energy_val * term_yaml_unit_val
                print(f"DEBUG add_reactions:     mode2_energy_term: Adding from rxn YAML: energy={term_yaml_energy_val}, unit={term_yaml_unit_val} (original unit: {_original_rxn_unit_for_debug}), term_value={term_from_yaml_energy}")
                rxn_ref += term_from_yaml_energy
                print(f"DEBUG add_reactions:     rxn_ref after YAML energy: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")

                for sysid_idx, (sysid, count) in enumerate(zip(rxn["structs"], rxn["counts"])):
                    term_val = self.ks_baseline_dict[sysid]
                    term_src = f"self.ks_baseline_dict[{sysid}]"
                    print(f"DEBUG add_reactions:       mode2_ks_baseline_sysid{sysid_idx}: Subtracting from {term_src} for sysid '{str(sysid)[:50]}...'")
                    print(f"DEBUG add_reactions:         Term value: {term_val} (type: {type(term_val)}, shape: {getattr(term_val, 'shape', 'N/A')}), Count: {count}")
                    rxn_ref -= count * term_val
                    print(f"DEBUG add_reactions:         rxn_ref after this term: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")

                print(f"DEBUG add_reactions:   --- Applying CKernel Baselines to rxn_ref AND Calculating rxn_cov for CKernels (mode 2) ---")
                for ckernel_idx, kernel in enumerate(self.ckernels):
                    kernel_name_info = f"ckernel[{ckernel_idx}]({type(kernel).__name__}, id={id(kernel)}, comp='{kernel.component}')"
                    print(f"DEBUG add_reactions:     Processing {kernel_name_info}:")
                    
                    rxn_cov = 0 # Initialize rxn_cov for this ckernel and this reaction
                    # --- DEBUG PRINTING START for kernel.rxn_cov_list (ckernel mode 2) ---
                    print(f"DEBUG add_reactions:       {kernel_name_info}: BEFORE append to its rxn_cov_list: len={len(kernel.rxn_cov_list)}")
                    # --- DEBUG PRINTING END ---

                    for sysid_idx, (sysid, count) in enumerate(zip(rxn["structs"], rxn["counts"])):
                        term_val_dbase, term_src_dbase = (None, "")
                        term_val_cov, term_src_cov = (None, "")
                        if isinstance(sysid, tuple):
                            term_val_cov = kernel.dcov_dict[sysid[0]][sysid[1]]
                            term_src_cov = f"{kernel_name_info}.dcov_dict[{sysid[0]}][{sysid[1]}]"
                            rxn_cov += count * term_val_cov
                            
                            term_val_dbase = kernel.dbase_dict[sysid[0]][sysid[1]]
                            term_src_dbase = f"{kernel_name_info}.dbase_dict[{sysid[0]}][{sysid[1]}]"
                            print(f"DEBUG add_reactions:         sysid{sysid_idx} (tuple) '{str(sysid)[:50]}...':")
                            print(f"DEBUG add_reactions:           Subtracting for rxn_ref from {term_src_dbase}: Term value: {term_val_dbase} (type: {type(term_val_dbase)}, shape: {getattr(term_val_dbase, 'shape', 'N/A')}), Count: {count}")
                            rxn_ref -= count * term_val_dbase
                            print(f"DEBUG add_reactions:           rxn_ref after this dbase term: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")
                            print(f"DEBUG add_reactions:           Adding to rxn_cov from {term_src_cov}: Term value: {term_val_cov} (type: {type(term_val_cov)}, shape: {getattr(term_val_cov, 'shape', 'N/A')}), Count: {count}")
                        else: # sysid is string
                            term_val_cov = kernel.cov_dict[sysid]
                            term_src_cov = f"{kernel_name_info}.cov_dict[{sysid}]"
                            rxn_cov += count * term_val_cov

                            term_val_dbase = kernel.base_dict[sysid]
                            term_src_dbase = f"{kernel_name_info}.base_dict[{sysid}]"
                            print(f"DEBUG add_reactions:         sysid{sysid_idx} (string) '{str(sysid)[:50]}...':")
                            print(f"DEBUG add_reactions:           Subtracting for rxn_ref from {term_src_dbase}: Term value: {term_val_dbase} (type: {type(term_val_dbase)}, shape: {getattr(term_val_dbase, 'shape', 'N/A')}), Count: {count}")
                            rxn_ref -= count * term_val_dbase
                            print(f"DEBUG add_reactions:           rxn_ref after this base term: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")
                            print(f"DEBUG add_reactions:           Adding to rxn_cov from {term_src_cov}: Term value: {term_val_cov} (type: {type(term_val_cov)}, shape: {getattr(term_val_cov, 'shape', 'N/A')}), Count: {count}")
                        print(f"DEBUG add_reactions:         Current rxn_cov for {kernel_name_info} after sysid{sysid_idx}: {rxn_cov} (type: {type(rxn_cov)}, shape: {getattr(rxn_cov, 'shape', 'N/A')})")
                    
                    # --- DEBUG PRINTING START for kernel.rxn_cov_list (ckernel mode 2) ---
                    print(f"DEBUG add_reactions:       {kernel_name_info}: FINAL rxn_cov to append: {rxn_cov} (type: {type(rxn_cov)}, shape: {getattr(rxn_cov, 'shape', 'N/A')})")
                    # --- DEBUG PRINTING END ---
                    kernel.rxn_cov_list.append(rxn_cov)
                    # --- DEBUG PRINTING START for kernel.rxn_cov_list (ckernel mode 2) ---
                    print(f"DEBUG add_reactions:       {kernel_name_info}: AFTER append to its rxn_cov_list: len={len(kernel.rxn_cov_list)}")
                    # --- DEBUG PRINTING END ---
            else: # mode != 2 (this includes mode == 0 for ckernels)
                print(f"DEBUG add_reactions:   --- Processing CKernels (mode != 2): Appending zeros to rxn_cov_list ---")
                for ckernel_idx_else, kernel_else in enumerate(self.ckernels):
                    kernel_name_info = f"ckernel_else[{ckernel_idx_else}]({type(kernel_else).__name__}, id={id(kernel_else)}, comp='{kernel_else.component}')"
                    # --- DEBUG PRINTING START for kernel.rxn_cov_list (ckernel, mode != 2) ---
                    print(f"DEBUG add_reactions:     RXN_ENTRY #{reaction_idx}, {kernel_name_info}:")
                    print(f"DEBUG add_reactions:       BEFORE append (zeros) to rxn_cov_list: len={len(kernel_else.rxn_cov_list)}")
                    zeros_to_add = np.zeros(kernel_else.Nctrl)
                    print(f"DEBUG add_reactions:       Value of zeros_to_add: shape={zeros_to_add.shape}")
                    # --- DEBUG PRINTING END ---
                    kernel_else.rxn_cov_list.append(zeros_to_add)
                    # --- DEBUG PRINTING START for kernel.rxn_cov_list (ckernel, mode != 2) ---
                    print(f"DEBUG add_reactions:       AFTER append (zeros) to rxn_cov_list: len={len(kernel_else.rxn_cov_list)}")
                    # --- DEBUG PRINTING END ---
            
            # --- DEBUG PRINTING START: Final rxn_ref before appending to self.rxn_ref_list ---
            print(f"DEBUG add_reactions:   FINAL rxn_ref for reaction #{reaction_idx} before append to self.rxn_ref_list: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})")
            # --- DEBUG PRINTING END ---
            self.rxn_ref_list.append(rxn_ref)

            # --- Noise Calculation and Debug Printing (Copied from your "latest version" in prompt 16, assumed to be correct by you) ---
            raw_noise_val = rxn.get("noise")
            raw_noise_factor = rxn.get("noise_factor")
            raw_noise_rel_factor = rxn.get("noise_rel_factor")
            raw_weight = rxn.get("weight")

            if raw_noise_val is not None: # Renamed 'noise' to 'current_noise_calc' to avoid conflict if 'noise' key exists in rxn
                current_noise_calc = raw_noise_val
            elif raw_noise_factor is not None:
                current_noise_calc = raw_noise_factor * self.default_noise
            else:
                current_noise_calc = self.default_noise
            
            noise_after_factor_or_direct = current_noise_calc

            if raw_noise_rel_factor is not None:
                abs_rxn_ref_val = np.abs(rxn_ref) if isinstance(rxn_ref, (np.ndarray, int, float)) else 0 
                term_rel_factor = raw_noise_rel_factor * abs_rxn_ref_val
                current_noise_calc += term_rel_factor
            
            noise_after_rel_factor = current_noise_calc

            if raw_weight is not None:
                if isinstance(raw_weight, (int, float, np.number)) and raw_weight > 0:
                    if isinstance(current_noise_calc, (int, float, np.number, np.ndarray)):
                        current_noise_calc /= np.sqrt(raw_weight)
                    else:
                        print(f"!!- WARNING add_reactions: current_noise_calc is unexpected type {type(current_noise_calc)} before weight division for reaction #{reaction_idx} -!!")
                else:
                    print(f"!!- WARNING add_reactions: raw_weight is {raw_weight} (type {type(raw_weight)}), skipping division for noise calc for reaction #{reaction_idx} -!!")
            
            final_calculated_noise = current_noise_calc 

            print(f"DEBUG add_reactions:   --- Noise Calculation Summary for reaction #{reaction_idx} (Mode={mode}) ---")
            print(f"DEBUG add_reactions:     Raw input 'noise': {raw_noise_val} (type: {type(raw_noise_val)})")
            print(f"DEBUG add_reactions:     Raw input 'noise_factor': {raw_noise_factor} (type: {type(raw_noise_factor)})")
            print(f"DEBUG add_reactions:     self.default_noise: {self.default_noise} (type: {type(self.default_noise)})")
            print(f"DEBUG add_reactions:     Noise after factor/direct init: {noise_after_factor_or_direct} (type: {type(noise_after_factor_or_direct)}, shape: {getattr(noise_after_factor_or_direct, 'shape', 'N/A')})")
            print(f"DEBUG add_reactions:     Value of rxn_ref used for rel_factor: {rxn_ref} (type: {type(rxn_ref)}, shape: {getattr(rxn_ref, 'shape', 'N/A')})") # This is the final rxn_ref for this reaction
            print(f"DEBUG add_reactions:     Raw input 'noise_rel_factor': {raw_noise_rel_factor} (type: {type(raw_noise_rel_factor)})")
            print(f"DEBUG add_reactions:     Noise after applying rel_factor: {noise_after_rel_factor} (type: {type(noise_after_rel_factor)}, shape: {getattr(noise_after_rel_factor, 'shape', 'N/A')})")
            print(f"DEBUG add_reactions:     Raw input 'weight': {raw_weight} (type: {type(raw_weight)})")
            print(f"DEBUG add_reactions:     FINAL calculated noise to be appended: {final_calculated_noise} (type: {type(final_calculated_noise)}, shape: {getattr(final_calculated_noise, 'shape', 'N/A')})")
            if not isinstance(final_calculated_noise, (int, float, np.number)): 
                print(f"!!- WARNING add_reactions: Non-scalar noise detected for reaction #{reaction_idx}! Summary for identification: {rxn_structs_info} -!!")
            
            self.rxn_noise_list.append(final_calculated_noise)
            print(f"--- END DEBUG add_reactions: FINISHED RXN ENTRY #{reaction_idx} ---")

class MOLGP2(MOLGP):
    """
    Gaussian process model for the exchange-correlation functional
    or its components.
    """

    def __init__(
        self,
        kernels,
        settings,
        libxc_baseline=None,
        default_noise=0.030,
    ):
        """
        Same as MOLGP, except kernels should be of type DFTKernel2.
        This requires a new class because DFTKernel2 has a different approach
        to evaluating baseline functional contributions, so a few
        functions in MOLGP need to be modified.

        Args:
            kernels (list[DFTKernel2]): List of kernels that sum to the XC energy.
            settings (DescParams or FeatureSettings): Settings for the
                features. Specifies what the feature vector is.
            libxc_baseline (str or None): Additional baseline functional to
                evaluate using the libxc library.
            default_noise (float): Default noise hyperparameter, used if noise
                is not provided for a particular data point
        """
        super(MOLGP2, self).__init__(
            kernels,
            settings,
            libxc_baseline=libxc_baseline,
            default_noise=default_noise,
        )

    def _compute_mol_covs(
        self, ddir, mol_ids, kernel, get_orb_deriv=None, save_refs=False
    ):
        # need to have different _compute_mol_covs function to handle baselines differently.
        blksize = 10000

        deriv = None
        for mol_id in mol_ids:
            print("MOL ID", mol_id)
            data = self.load_data(ddir, mol_id, get_orb_deriv)
            if deriv is None:
                if get_orb_deriv is None:
                    if "ddesc" in data:
                        deriv = True
                    else:
                        deriv = False
                else:
                    deriv = get_orb_deriv
            if deriv:
                assert "ddesc" in data
                assert "drho_data" in data

            weights = data["wt"]
            nspin = data["nspin"]
            desc = data["desc"]
            rho_data = data["rho_data"] / nspin
            vwrtt_tot = 0
            baseline = 0
            is_mgga = self.settings.sl_settings.level == "MGGA"
            if deriv:
                ddesc = strk_to_tuplek(data["ddesc"])
                drho_data = strk_to_tuplek(data["drho_data"])
                for orb, (s, drho) in drho_data.items():
                    drho[:] /= nspin
                self.dexx_ref_dict[mol_id] = strk_to_tuplek(data["dval"], ref=True)
                dvwrtt_tot = {k: 0 for k in ddesc.keys()}
                dbaseline = {k: 0 for k in ddesc.keys()}
            for i0, i1 in prange(0, weights.size, blksize):
                X0T = self._get_normalized_features(desc[..., i0:i1])
                rho_tuple = get_rho_tuple_with_grad_cross(
                    rho_data[..., i0:i1], is_mgga=is_mgga
                )
                wt = weights[i0:i1]
                m_res = kernel.multiplicative_baseline(rho_tuple)
                a_res = kernel.additive_baseline(rho_tuple)
                dm = vxc_tuple_to_array(rho_data[..., i0:i1], m_res[1:])
                m = m_res[0]
                if a_res is None:
                    da = np.zeros_like(dm)
                    a = np.zeros_like(m)
                else:
                    da = vxc_tuple_to_array(rho_data[..., i0:i1], a_res[1:])
                    a = a_res[0]

                # k is (Nctrl, nspin, Nsamp)
                # dk is (Nctrl, nspin, N0, Nsamp)
                if deriv:
                    k, dkdX0T = kernel.get_k_and_deriv(X0T)
                else:
                    k = kernel.get_k(X0T)
                    dkdX0T = None
                # TODO setting nan to zero could cover up more serious issues,
                # but it is the easiest way to take care of small density
                # training data without editing functions used at eval-time.
                cond = rho_tuple[0] < 1e-6
                if kernel.mode == "SEP":
                    m[cond] = 0.0
                    a[cond] = 0.0
                    k[:, cond] = 0.0
                    if deriv:
                        for s in range(X0T.shape[0]):
                            dm[s][:, cond[s]] = 0.0
                            da[s][:, cond[s]] = 0.0
                            dkdX0T[:, s][:, :, cond[s]] = 0.0
                else:
                    if cond.shape[0] == 1:
                        scond = cond[0]
                    else:
                        scond = np.logical_and(cond[0], cond[1])
                    m[..., scond] = 0.0
                    a[..., scond] = 0.0
                    k[..., scond] = 0.0
                    if deriv:
                        for s in range(X0T.shape[0]):
                            dkdX0T[:, s, cond[s], :] = 0.0
                            dm[s][:, cond[s]] = 0.0
                            da[s][:, cond[s]] = 0.0
                if kernel.mode == "SEP":
                    km = (k * m).sum(1)
                    if deriv:
                        dkm1 = dkdX0T * m[:, None, :]
                        dkm2 = k[:, :, None, :] * dm
                else:
                    km = k * m
                    if deriv:
                        dkm1 = dkdX0T * m
                        dkm2 = k[:, None, None, :] * dm
                vwrtt_tot += (km * wt).sum(axis=1)
                baseline += (a * wt).sum()
                if deriv:
                    for orb, (s, ddesc_tmp) in ddesc.items():
                        # ddesc_tmp has shape (N0, Nsamp)
                        # avoid overwriting slice
                        ddesc_tmp = wt * self._get_normalized_feature_derivs(
                            desc[s, :, i0:i1], ddesc_tmp[:, i0:i1]
                        )
                        drho_tmp = wt * drho_data[orb][1][:, i0:i1]
                        dvwrtt_tot[orb] += np.einsum("cdn,dn->c", dkm1[:, s], ddesc_tmp)
                        dvwrtt_tot[orb] += np.einsum("cdn,dn->c", dkm2[:, s], drho_tmp)

                        dbaseline[orb] += np.sum(da[s] * drho_tmp)
                        
                        # dbaseline[orb] += np.dot(da * drho_tmp, wt)
                        # The below is what in misc_fixes_and_misc branch ab4295a (Modifies models.train.MOLGP2._compute_mol_covs and add some print statements. This version works for training complete band-gap scider model (trained with molecules and solids data), see execution_train_2025-05-16-05-38-54)
                        # dbaseline[orb] += np.sum(da[s] * drho_tmp * wt)
            kernel.cov_dict[mol_id] = vwrtt_tot
            kernel.base_dict[mol_id] = baseline

            # --- DEBUG PRINTING START for _compute_mol_covs (kernel.base_dict) ---
            val_to_print = kernel.base_dict[mol_id]
            # baseline is usually a scalar sum
            print(f"DEBUG _compute_mol_covs ({mol_id}, kernel {type(kernel).__name__}): Populating kernel.base_dict[{mol_id}]: {val_to_print} (type: {type(val_to_print)}, shape: {getattr(val_to_print, 'shape', 'N/A')})")
            # --- DEBUG PRINTING END ---

            if save_refs:
                self.exx_ref_dict[mol_id] = (data["val"] * weights).sum()

                # --- DEBUG PRINTING START for _compute_mol_covs (exx_ref_dict) ---
                val_to_print = self.exx_ref_dict[mol_id]
                print(f"DEBUG _compute_mol_covs ({mol_id}): Populating self.exx_ref_dict[{mol_id}]: {val_to_print} (type: {type(val_to_print)}, shape: {getattr(val_to_print, 'shape', 'N/A')})")
                # --- DEBUG PRINTING END ---

                self.ks_baseline_dict[mol_id] = data["e_tot_orig"] - data["exc_orig"]

                # --- DEBUG PRINTING START for _compute_mol_covs (ks_baseline_dict) ---
                val_to_print = self.ks_baseline_dict[mol_id]
                print(f"DEBUG _compute_mol_covs ({mol_id}): Populating self.ks_baseline_dict[{mol_id}]: {val_to_print} (type: {type(val_to_print)}, shape: {getattr(val_to_print, 'shape', 'N/A')})")
                # --- DEBUG PRINTING END ---

            if deriv:
                kernel.dcov_dict[mol_id] = dvwrtt_tot

                # --- DEBUG PRINTING START for _compute_mol_covs (kernel.dbase_dict population) ---
                print(f"DEBUG _compute_mol_covs ({mol_id}, kernel {type(kernel).__name__}): Will assign the following 'dbaseline' to kernel.dbase_dict[{mol_id}]")
                if isinstance(dbaseline, dict):
                    for orb_key_debug, orb_val_debug in dbaseline.items(): # orb_key_debug is an orb_tuple like ('U',0)
                        print(f"DEBUG _compute_mol_covs ({mol_id}, kernel {type(kernel).__name__}):   dbaseline entry for orb_key [{orb_key_debug}]: type={type(orb_val_debug)}, shape={getattr(orb_val_debug, 'shape', 'N/A')}, value_summary={str(orb_val_debug)[:100]+'...' if isinstance(orb_val_debug, np.ndarray) and orb_val_debug.size > 10 else orb_val_debug}")
                else:
                    print(f"DEBUG _compute_mol_covs ({mol_id}, kernel {type(kernel).__name__}):   dbaseline itself is not a dict: {dbaseline} (type: {type(dbaseline)})")
                # --- DEBUG PRINTING END ---
                
                kernel.dbase_dict[mol_id] = dbaseline

                # --- DEBUG PRINTING START for _compute_mol_covs (kernel.dbase_dict) ---
                print(f"DEBUG _compute_mol_covs ({mol_id}, kernel {type(kernel).__name__}): Populating kernel.dbase_dict[{mol_id}]")
                db_val_to_print = kernel.dbase_dict[mol_id] # This is the 'dbaseline' dictionary
                if isinstance(db_val_to_print, dict):
                    for orb_key, orb_val in db_val_to_print.items():
                        print(f"DEBUG _compute_mol_covs ({mol_id}, kernel {type(kernel).__name__}):   dbase_dict[{mol_id}][{orb_key}]: {orb_val} (type: {type(orb_val)}, shape: {getattr(orb_val, 'shape', 'N/A')})")
                else: # Should be a dict, but just in case
                    print(f"DEBUG _compute_mol_covs ({mol_id}, kernel {type(kernel).__name__}):   dbase_dict[{mol_id}] (overall): {db_val_to_print} (type: {type(db_val_to_print)}, shape: {getattr(db_val_to_print, 'shape', 'N/A')})")
                # --- DEBUG PRINTING END ---
                

    def map(self, mapping_plans):
        """
        Map the MOLGP model to an Evaluator object that can efficiently
        evaluate the XC energy and which can also be serialized.

        Returns:
            list(callable): Functions that map each
                kernel to a MappedDFTKernel object
        """
        mapped_kernels = [
            kernel.map(mapping_plan)
            for kernel, mapping_plan in zip(self.kernels, mapping_plans)
        ]
        return MappedXC2(
            mapped_kernels,
            self.settings,
            libxc_baseline=self.libxc_baseline,
        )
