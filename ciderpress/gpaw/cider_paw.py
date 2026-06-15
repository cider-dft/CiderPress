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

import ctypes

import numpy as np
from gpaw.core.atom_arrays import AtomArrays, AtomArraysLayout, AtomDistribution
from gpaw.utilities import pack_density
from gpaw.xc.gga import GGA
from gpaw.xc.mgga import MGGA

from ciderpress.gpaw.atom_utils import FastAtomPASDWSlice, FastPASDWCiderKernel
from ciderpress.gpaw.cider_fft import (
    CIDERPW_GRAD_MODE_FORCE,
    CIDERPW_GRAD_MODE_NONE,
    CIDERPW_GRAD_MODE_STRESS,
    CiderGGA,
    CiderMGGA,
)
from ciderpress.gpaw.nldf_interface import pwutil as libpwutil


class CiderPASDW_MPRoutines:
    has_paw = True

    def get_D_asp(self):
        if hasattr(self.atomdist, "to_work"):
            D_asp = self.atomdist.to_work(self.dens.D_asp)
            D_asp.partition
            self.atom_partition
            # assert p0.comm.size == p1.comm.size
            # if not (p0.rank_a == p1.rank_a).all():
            #    print(p0.rank_a, p1.rank_a)
            #    assert False
            # print(len(D_asp), D_asp[list(D_asp.keys())[0]].shape, self.world.rank)
            return D_asp
        else:
            # print("ATOMS", self.world.rank, self.dens.D_asii)
            # atom_array_layout = AtomArraysLayout(
            #    [(setup.ni * (setup.ni + 1) // 2) for setup in builder.setups],
            #    atomdist=builder.atomdist, dtype=dtype)
            D_asp = {}
            rank_a, comm = self.get_atom_rank_and_comm()
            D_asii = self.dens.D_asii.gather(broadcast=True)
            for a, D_sii in D_asii.items():
                if rank_a[a] == comm.rank:
                    D_asp[a] = np.array([pack_density(D_ii.real) for D_ii in D_sii])
            return D_asp

    def add_forces(self, F_av):
        if self.new_gpaw:
            nt_sR = self.dens.nt_sR
            nt_sr = self._interpolate(nt_sR)
            taut_sR = self.dens.taut_sR
            assert taut_sR is not None
            taut_sr = self._interpolate(taut_sR)
            # print(nrt_sR.data.shape, taut_sR.data.shape, nt_sr.data.shape, taut_sr.data.shape)
            nt_sr.xp
            nt_sr = nt_sr.to_xp(np)
            taut_sr = taut_sr.to_xp(np)
            self.calculate_force_contribs(F_av, nt_sr.data, taut_sg=taut_sr.data)
        else:
            nt_sg = self.dens.nt_xg
            redist = self._hamiltonian.xc_redistributor
            if redist is not None:
                nt_sg = redist.distribute(nt_sg)
            self.calculate_force_contribs(F_av, nt_sg)

    def _collect_paw_corrections(self):
        """
        For use in xc_tools, only after calculate_paw_correction
        is called. Collects potential using from_work
        """
        dH_asp_tmp = self.dH_asp_tmp
        E_a_tmp = self.E_a_tmp
        # dH_asp_new = self.setups.empty_atomic_matrix(
        #    self.nspin, self.atom_partition
        # )
        # E_a_new = self.atom_partition.arraydict([(1,)] * len(self.setups), float)
        dH_asp_new = self._empty_atomic_matrix()
        E_a_new = self._empty_atomic_matrix([(1,)] * len(self.setups))
        for a, E in E_a_tmp.items():
            E_a_new[a][:] = E
            dH_asp_new[a][:] = dH_asp_tmp[a]
        dist = self.atomdist
        # TODO
        self.E_a_tmp = dist.from_work(E_a_new)
        self.dH_asp_tmp = dist.from_work(dH_asp_new)

    def initialize_paw_kernel(self, cider_kernel_inp):
        self.paw_kernel = FastPASDWCiderKernel(
            cider_kernel_inp,
            self._plan,
            self.aux_gd,
            self.cut_xcgrid,
        )
        # TODO seems redundant
        self.paw_kernel.initialize(self.setups)
        self.paw_kernel.initialize_more_things(self.setups)
        self.atom_slices_s = None

    @property
    def new_gpaw(self):
        # TODO better way of detecting new_gpaw
        return hasattr(self.atomdist, "rank_a")

    def get_atom_rank_and_comm(self):
        if self.new_gpaw:
            # print(self.world.rank, self.atomdist.comm.rank,
            #      self.atomdist.comm.size,
            #      (self.atomdist.rank_a == self.atomdist.comm.rank).sum())
            natm = len(self.setups)
            natm_per_proc = (natm + self.world.size - 1) // self.world.size
            rank_a = np.arange(natm) // natm_per_proc
            return rank_a, self.world
        else:
            return self.atom_partition.rank_a, self.atom_partition.comm

    def _empty_atomic_matrix(self, shapes=None, dtype=float):
        # TODO this won't work for new_gpaw
        if shapes is None:
            shapes = [
                (self.nspin, setup.ni * (setup.ni + 1) // 2) for setup in self.setups
            ]
        if self.new_gpaw:
            layout = AtomArraysLayout(
                shapes,
                self.atomdist,
                dtype=dtype,
            )
            return layout.empty(self.nspin)
        else:
            return self.atom_partition.arraydict(shapes, dtype)

    def _setup_atom_slices(self):
        self.timer.start("ATOM SLICE SETUP")
        self.atom_slices_s = {}
        self.atom_slices_a = {}
        self.global_slices_s = {}
        self.global_slices_a = {}
        self.D_asiq = None
        self.vc_asiq = None
        rank_a, comm = self.get_atom_rank_and_comm()
        for a in range(len(self.setups)):
            args = [
                self.aux_gd,
                self.spos_ac[a],
                self.setups[a].ps_setup,
                self.fft_obj,
            ]
            kwargs = dict(
                rmax=self.setups[a].ps_setup.rcut_func,
                sphere=True,
                store_funcs=self.pasdw_store_funcs,
            )
            self.atom_slices_s[a] = FastAtomPASDWSlice.from_gd_and_setup(
                *args, **kwargs
            )
            if rank_a[a] == comm.rank and self.pasdw_ovlp_fit:
                self.global_slices_s[a] = FastAtomPASDWSlice.from_gd_and_setup(
                    *args, is_global=True, ovlp_fit=True, **kwargs
                )
            args[2] = self.setups[a].pa_setup
            kwargs["rmax"] = self.setups[a].pa_setup.rcut_func
            self.atom_slices_a[a] = FastAtomPASDWSlice.from_gd_and_setup(
                *args, **kwargs
            )
            if rank_a[a] == comm.rank and self.pasdw_ovlp_fit:
                self.global_slices_a[a] = FastAtomPASDWSlice.from_gd_and_setup(
                    *args, is_global=True, ovlp_fit=True, **kwargs
                )
        atoms = [[] for r in range(comm.size)]
        for a in range(len(self.setups)):
            atoms[rank_a[a]].append(a)
        self._atom_lists_r = atoms
        self._my_atom_list = atoms[comm.rank]
        self.timer.stop()

    def write_augfeat_to_rbuf(self, spin, pot=False):
        self.timer.start("PAW TO GRID")
        if self.atom_slices_s is None:
            self._setup_atom_slices()

        self.timer.start("setup")
        if pot:
            atom_slices = self.atom_slices_a
            global_slices = self.global_slices_a
            X_asiq = self.vD_asiq
        else:
            atom_slices = self.atom_slices_s
            global_slices = self.global_slices_s
            X_asiq = self.c_asiq

        shapes = [
            (atom_slices[a].num_funcs, self._plan.nalpha)
            for a in range(len(self.setups))
        ]
        sizes = np.array([s[0] * s[1] for s in shapes])
        rank_a, comm = self.get_atom_rank_and_comm()
        if self.fft_obj.has_mpi:
            alocs = {}
            rlocs = []
            rsizes = []
            tot = 0
            for r in range(comm.size):
                rlocs.append(tot)
                for a in self._atom_lists_r[r]:
                    alocs[a] = tot
                    tot += sizes[a]
                rsizes.append(tot - rlocs[-1])
            sendbuf = np.empty(rsizes[comm.rank], dtype=np.float64)
            rtot = 0
            for a in self._my_atom_list:
                c_iq = X_asiq[a][spin]
                if self.pasdw_ovlp_fit:
                    sendbuf[rtot : rtot + sizes[a]] = np.ascontiguousarray(
                        # c_abi[a].dot(global_slices[a].sinv_pf.T)
                        global_slices[a].sinv_pf.dot(c_iq)
                    ).ravel()
                else:
                    sendbuf[rtot : rtot + sizes[a]] = np.ascontiguousarray(c_iq).ravel()
                rtot += sizes[a]
            np.asarray([rtot] * comm.size)
            np.asarray([0] * comm.size)
            recvbuf = np.empty(tot, dtype=np.float64)
            self.timer.stop()
            self.timer.start("comm")
            rsizes = np.asarray(rsizes, order="C", dtype=np.int32)
            rlocs = np.asarray(rlocs, order="C", dtype=np.int32)
            libpwutil.ciderpw_all_gatherv_from_gpaw(
                ctypes.py_object(comm.get_c_object()),
                sendbuf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(rtot),
                recvbuf.ctypes.data_as(ctypes.c_void_p),
                rsizes.ctypes.data_as(ctypes.c_void_p),
                rlocs.ctypes.data_as(ctypes.c_void_p),
            )
        else:
            max_size = np.max([s[0] * s[1] for s in shapes])
            buf = np.empty(max_size, dtype=np.float64)
        self.timer.stop()

        for a in range(len(self.setups)):
            atom_slice = atom_slices[a]
            if self.fft_obj.has_mpi:
                coefs_iq = np.ndarray(shapes[a], buffer=recvbuf[alocs[a] :])
            else:
                coefs_iq = np.ndarray(shapes[a], buffer=buf)
                coefs_iq = np.ndarray(shapes[a], buffer=buf)
                if comm.rank == rank_a[a]:
                    coefs_iq[:] = X_asiq[a][spin]
                comm.broadcast(coefs_iq, rank_a[a])
            if atom_slice is not None and atom_slices[a].rad_g.size > 0:
                self.timer.start("funcs")
                funcs_ig = atom_slice.get_funcs()
                funcs_gq = np.dot(funcs_ig.T, coefs_iq)
                self.timer.stop()
                self.timer.start("paw2grid")
                self.fft_obj.add_paw2grid(funcs_gq, atom_slice.indset)
                self.timer.stop()
        self.timer.stop()

    def _build_asiq_dict_(self, X_asiq, shapes, xshape=None):
        if xshape is None:
            xshape = ()
        for a in self._my_atom_list:
            have_a = a in X_asiq
            ashape = xshape + shapes[a]
            if not have_a or X_asiq[a].shape != ashape:
                X_asiq[a] = np.zeros(ashape)

    def interpolate_rbuf_onto_atoms(self, spin, pot=False):
        # edits X_asiq
        self.timer.start("GRID TO PAW")
        if self.atom_slices_s is None:
            self._setup_atom_slices()

        if pot:
            atom_slices = self.atom_slices_s
            global_slices = self.global_slices_s
            if self.vc_asiq is None:
                self.vc_asiq = {}
            X_asiq = self.vc_asiq
        else:
            atom_slices = self.atom_slices_a
            global_slices = self.global_slices_a
            if self.D_asiq is None:
                self.D_asiq = {}
            X_asiq = self.D_asiq

        shapes = [
            (atom_slices[a].num_funcs, self._plan.nalpha)
            for a in range(len(self.setups))
        ]
        if spin == 0:
            self._build_asiq_dict_(X_asiq, shapes, xshape=(self.nspin,))

        rank_a, comm = self.get_atom_rank_and_comm()
        coefs_a_iq = {}
        for a in range(len(self.setups)):
            atom_slice: FastAtomPASDWSlice = atom_slices[a]
            if atom_slice.rad_g.size > 0:
                funcs_ig = atom_slice.get_funcs()
                funcs_gq = np.empty((funcs_ig.shape[1], self._plan.nalpha))
                self.fft_obj.set_grid2paw(funcs_gq, atom_slice.indset)
                coefs_iq = np.dot(funcs_ig, funcs_gq)
                coefs_iq[:] *= atom_slice.dv
            else:
                coefs_iq = np.zeros(shapes[a])
            coefs_a_iq[a] = coefs_iq
        for a in range(len(self.setups)):
            comm.sum(coefs_a_iq[a], rank_a[a])
        for a in self._my_atom_list:
            coefs_iq = coefs_a_iq[a]
            assert coefs_iq is not None
            if self.pasdw_ovlp_fit:
                X_asiq[a][spin] = global_slices[a].sinv_pf.T.dot(coefs_iq)
            else:
                X_asiq[a][spin] = coefs_iq
        self.timer.stop()
        return X_asiq

    def interpolate_rbuf_onto_atoms_grad(self, s, pot=False):
        self.timer.start("GRID TO PAW GRAD")
        if self.atom_slices_s is None:
            self._setup_atom_slices()

        if pot:
            atom_slices = self.atom_slices_s
            global_slices = self.global_slices_s
            X_asiq = self.vc_asiq
        else:
            atom_slices = self.atom_slices_a
            global_slices = self.global_slices_a
            X_asiq = self.D_asiq

        shapes = [
            (atom_slices[a].num_funcs, self._plan.nalpha)
            for a in range(len(self.setups))
        ]
        if s == 0:
            self._save_v_avsiq = {}
            self._build_asiq_dict_(
                self._save_v_avsiq,
                shapes,
                xshape=(
                    3,
                    self.nspin,
                ),
            )

        rank_a, comm = self.get_atom_rank_and_comm()
        coefs_a_viq = {}
        for a in range(len(self.setups)):
            atom_slice: FastAtomPASDWSlice = atom_slices[a]
            if atom_slice.rad_g.size > 0:
                funcs_vig = atom_slice.get_grads()
                funcs_gq = np.empty((funcs_vig.shape[2], self._plan.nalpha))
                self.fft_obj.set_grid2paw(
                    funcs_gq,
                    atom_slice.indset,
                )
                coefs_viq = np.dot(funcs_vig, funcs_gq)
                coefs_viq[:] *= atom_slice.dv
            else:
                coefs_viq = np.zeros((3,) + shapes[a])
            coefs_a_viq[a] = coefs_viq
        for a in range(len(self.setups)):
            comm.sum(coefs_a_viq[a], rank_a[a])
        for a in self._my_atom_list:
            coefs_viq = coefs_a_viq[a]
            assert coefs_viq is not None
            for v in range(3):
                if self.pasdw_ovlp_fit:
                    self._save_v_avsiq[a][v, s] = global_slices[a].sinv_pf.T.dot(
                        coefs_viq[v]
                    )
                else:
                    self._save_v_avsiq[a][v, s] = coefs_viq[v]
            if self.pasdw_ovlp_fit:
                sl = global_slices[a]
                X_vii = sl.get_ovlp_deriv(sl.get_funcs(), sl.get_grads(), stress=False)
                # TODO not sure if this line below is quite right but it seems to work
                # could also just be X_asiq[a][s]
                # but in principle we need to remove the sinv_pf from X_asiq before multiplying
                # in the derivative matrix
                coefs_iq = np.linalg.solve(sl.sinv_pf.T, X_asiq[a][s])
                self._save_v_avsiq[a][:, s] += np.einsum("vij,iq->vjq", X_vii, coefs_iq)
        self.timer.stop()

    def interpolate_rbuf_onto_atoms_stress(self, s, pot=False):
        self.timer.start("GRID TO PAW GRAD")
        if self.atom_slices_s is None:
            self._setup_atom_slices()

        if pot:
            atom_slices = self.atom_slices_s
            global_slices = self.global_slices_s
            X_asiq = self.vc_asiq
        else:
            atom_slices = self.atom_slices_a
            global_slices = self.global_slices_a
            X_asiq = self.D_asiq

        shapes = [
            (atom_slices[a].num_funcs, self._plan.nalpha)
            for a in range(len(self.setups))
        ]
        if s == 0:
            self._save_v_avsiq = {}
            self._build_asiq_dict_(
                self._save_v_avsiq,
                shapes,
                xshape=(
                    3,
                    3,
                    self.nspin,
                ),
            )

        rank_a, comm = self.get_atom_rank_and_comm()
        coefs_a_vviq = {}
        for a in range(len(self.setups)):
            atom_slice: FastAtomPASDWSlice = atom_slices[a]
            if atom_slice.rad_g.size > 0:
                funcs_vvig = atom_slice.get_stress_funcs()
                funcs_gq = np.empty((funcs_vvig.shape[-1], self._plan.nalpha))
                self.fft_obj.set_grid2paw(
                    funcs_gq,
                    atom_slice.indset,
                )
                coefs_vviq = np.dot(funcs_vvig, funcs_gq)
                coefs_vviq[:] *= atom_slice.dv
            else:
                coefs_vviq = np.zeros((3, 3) + shapes[a])
            coefs_a_vviq[a] = coefs_vviq
        for a in range(len(self.setups)):
            comm.sum(coefs_a_vviq[a], rank_a[a])
        for a in self._my_atom_list:
            coefs_vviq = coefs_a_vviq[a]
            assert coefs_vviq is not None
            for u in range(3):
                for v in range(3):
                    if self.pasdw_ovlp_fit:
                        self._save_v_avsiq[a][u, v, s] = global_slices[a].sinv_pf.T.dot(
                            coefs_vviq[u, v]
                        )
                    else:
                        self._save_v_avsiq[a][u, v, s] = coefs_vviq[u, v]
            if self.pasdw_ovlp_fit:
                sl = global_slices[a]
                X_vvii = sl.get_ovlp_deriv(sl.get_funcs(), sl.get_grads(), stress=True)
                coefs_iq = np.linalg.solve(sl.sinv_pf.T, X_asiq[a][s])
                self._save_v_avsiq[a][:, :, s] += np.einsum(
                    "uvij,iq->uvjq", X_vvii, coefs_iq
                )
        self.timer.stop()

    def set_fft_work(self, s, pot=False):
        self.write_augfeat_to_rbuf(s, pot=pot)

    def get_fft_work(self, s, pot=False, grad_mode=CIDERPW_GRAD_MODE_NONE):
        self.interpolate_rbuf_onto_atoms(s, pot=pot)
        if grad_mode == CIDERPW_GRAD_MODE_FORCE:
            self.interpolate_rbuf_onto_atoms_grad(s, pot=pot)
        elif grad_mode == CIDERPW_GRAD_MODE_STRESS:
            self.interpolate_rbuf_onto_atoms_stress(s, pot=pot)

    def _set_paw_terms(self):
        self.timer.start("PAW TERMS")
        D_asp = self.get_D_asp()
        # TODO add sanity check for new_gpaw
        if (
            not self.new_gpaw
            and not (D_asp.partition.rank_a == self.atom_partition.rank_a).all()
        ):
            raise ValueError("rank_a mismatch")
        self.c_asiq = self.paw_kernel.calculate_paw_feat_corrections(self.setups, D_asp)
        self.D_asp = D_asp
        self.timer.stop()

    def _calculate_paw_energy_and_potential(self, grad_mode=CIDERPW_GRAD_MODE_NONE):
        self.timer.start("PAW ENERGY")
        self.vD_asiq, deltaE, deltaV = self.paw_kernel.calculate_paw_feat_corrections(
            self.setups, self.D_asp, D_asiq=self.D_asiq
        )
        self.E_a_tmp = deltaE
        self.deltaV = deltaV
        self.timer.stop()
        if grad_mode == CIDERPW_GRAD_MODE_STRESS:
            tot = 0
            for a in self._my_atom_list:
                tot += np.einsum("siq,siq->", self.vD_asiq[a], self.D_asiq[a])
            return tot

    def _get_dH_asp(self):
        self.timer.start("PAW POTENTIAL")
        dH_asp = self.paw_kernel.calculate_paw_feat_corrections(
            self.setups, self.D_asp, vc_asiq=self.vc_asiq
        )
        if self.new_gpaw:
            rank_a, comm = self.get_atom_rank_and_comm()
            dist = AtomDistribution(rank_a, comm)
            shapes = [
                (self.nspin, setup.ni * (setup.ni + 1) // 2) for setup in self.setups
            ]
            layout = AtomArraysLayout(shapes, dist, dtype=float)
            self.dH_asp_tmp = AtomArrays(layout)
            for a in self.D_asp.keys():
                self.dH_asp_tmp[a] = dH_asp[a] + self.deltaV[a]
            self.dH_asp_tmp = self.dH_asp_tmp.gather(broadcast=True)
            layout = AtomArraysLayout([1] * len(shapes), dist, dtype=float)
            E_a_tmp = AtomArrays(layout)
            for a in self.D_asp.keys():
                E_a_tmp[a] = self.E_a_tmp[a]
            self.E_a_tmp = E_a_tmp.gather(broadcast=True)
            self.E_a_tmp = {a: self.E_a_tmp[a].item() for a in range(len(shapes))}
        else:
            for a in self.D_asp.keys():
                dH_asp[a] += self.deltaV[a]
            self.dH_asp_tmp = dH_asp
        self.timer.stop()

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        if a is None:
            raise ValueError("Atom index a not provided.")
        if dEdD_sp is not None:
            dEdD_sp += self.dH_asp_tmp[a]
        return self.E_a_tmp[a]

    @property
    def atom_partition(self):
        return self.atomdist.work_partition

    def initialize_more_things(self, setups=None, atomdist=None):
        if setups is None:
            self.setups = self._hamiltonian.setups
        else:
            self.setups = setups
        if atomdist is None:
            self.atomdist = self._hamiltonian.atomdist
        else:
            self.atomdist = atomdist

        encut_atom = self.encut
        Nalpha_atom = self.Nalpha
        while encut_atom < self.encut_atom_min:
            encut_atom *= self.lambd
            Nalpha_atom += 1

        self.initialize_paw_kernel(self.cider_kernel)

    @property
    def is_initialized(self):
        if (
            (self._plan is None)
            or (self._plan.nspin != self.nspin)
            or (self.setups is None)
            or (self.atom_slices_s is None)
        ):
            return False
        else:
            return True


class CiderGGAPASDW(CiderPASDW_MPRoutines, CiderGGA):
    def __init__(self, cider_kernel, **kwargs):
        CiderGGA.__init__(
            self,
            cider_kernel,
            **kwargs,
        )
        defaults_list = {
            "encut_atom_min": 8000.0,  # float
            "cut_xcgrid": False,  # bool
            "pasdw_ovlp_fit": True,
            "pasdw_store_funcs": False,
        }
        settings = {}
        for k, v in defaults_list.items():
            settings[k] = kwargs.get(k)
            if settings[k] is None:
                settings[k] = defaults_list[k]
        self.__dict__.update(settings)
        self.k2_k = None
        self.setups = None
        self.atomdist = None

    def todict(self):
        d = super(CiderGGAPASDW, self).todict()
        d["_cider_type"] = "CiderGGAPASDW"
        d["xc_params"]["pasdw_store_funcs"] = self.pasdw_store_funcs
        d["xc_params"]["pasdw_ovlp_fit"] = self.pasdw_ovlp_fit
        return d

    def add_forces(self, F_av):
        GGA.add_forces(self, F_av)
        CiderPASDW_MPRoutines.add_forces(self, F_av)

    def stress_tensor_contribution(self, n_sg):
        return CiderGGA.stress_tensor_contribution(self, n_sg)

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac
        self.atom_slices_s = None

    def initialize(self, density, hamiltonian, wfs):
        self.dens = density
        self._hamiltonian = hamiltonian
        self.timer = wfs.timer
        self.world = wfs.world
        self._plan = None
        self.setups = None
        self._check_parallelization(wfs)

    def calculate_force_contribs(self, F_av, n_sg, taut_sg=None):
        e_g = np.zeros(n_sg.shape[1:])
        Ftmp_av = np.zeros_like(F_av)
        if taut_sg is None:
            taut_sg = self._get_taut(n_sg)[0]
        _e, rho_sxg, dedrho_sxg = self._get_cider_inputs(n_sg, taut_sg)
        e_g[:] = 0.0
        fs = self.calc_cider(_e, rho_sxg, dedrho_sxg, grad_mode=CIDERPW_GRAD_MODE_FORCE)
        for a, f in fs.items():
            Ftmp_av[a] = f
        self.world.sum(Ftmp_av, 0)
        if self.world.rank == 0:
            F_av[:] += Ftmp_av


class CiderMGGAPASDW(CiderPASDW_MPRoutines, CiderMGGA):
    def __init__(self, cider_kernel, **kwargs):
        CiderMGGA.__init__(
            self,
            cider_kernel,
            **kwargs,
        )
        defaults_list = {
            "encut_atom_min": 8000.0,  # float
            "cut_xcgrid": False,  # bool
            "pasdw_ovlp_fit": True,
            "pasdw_store_funcs": False,
        }
        settings = {}
        for k, v in defaults_list.items():
            settings[k] = kwargs.get(k)
            if settings[k] is None:
                settings[k] = defaults_list[k]
        self.__dict__.update(settings)
        self.k2_k = None
        self.setups = None
        self.atomdist = None

    def todict(self):
        d = super(CiderMGGAPASDW, self).todict()
        d["_cider_type"] = "CiderMGGAPASDW"
        d["xc_params"]["pasdw_store_funcs"] = self.pasdw_store_funcs
        d["xc_params"]["pasdw_ovlp_fit"] = self.pasdw_ovlp_fit
        return d

    def add_forces(self, F_av):
        MGGA.add_forces(self, F_av)
        CiderPASDW_MPRoutines.add_forces(self, F_av)

    def set_positions(self, spos_ac):
        if not self.new_gpaw:
            MGGA.set_positions(self, spos_ac)
        self.spos_ac = spos_ac
        self.atom_slices_s = None

    def initialize(self, density, hamiltonian, wfs):
        MGGA.initialize(self, density, hamiltonian, wfs)
        self.dens = density
        self._hamiltonian = hamiltonian
        self.timer = wfs.timer
        self.world = wfs.world
        self.setups = None
        self.atom_slices_s = None
        self._plan = None
        self._check_parallelization(wfs)

    def _get_taut(self, n_sg):
        self.timer.start("KED SETUP")
        if self.type == "MGGA":
            if self.fixed_ke:
                taut_sG = self._fixed_taut_sG
                if not self._taut_gradv_init:
                    self._taut_gradv_init = True
                    # ensure initialization for calculation potential
                    self.wfs.calculate_kinetic_energy_density()
            else:
                taut_sG = self.wfs.calculate_kinetic_energy_density()

            if taut_sG is None:
                taut_sG = self.wfs.gd.zeros(len(n_sg))

            taut_sg = np.empty_like(n_sg)

            for taut_G, taut_g in zip(taut_sG, taut_sg):
                taut_G += 1.0 / self.wfs.nspins * self.tauct_G
                self.distribute_and_interpolate(taut_G, taut_g)

            dedtaut_sg = np.zeros_like(n_sg)
        else:
            taut_sg = None
            dedtaut_sg = None
            taut_sG = None
        self.timer.stop()
        return taut_sg, dedtaut_sg, taut_sG

    def _core_kin_term(self):
        return self.tauct_G * (1.0 / self.wfs.nspins)

    def calculate_force_contribs(self, F_av, n_sg, taut_sg=None):
        e_g = np.zeros(n_sg.shape[1:])
        Ftmp_av = np.zeros_like(F_av)
        if taut_sg is None:
            taut_sg = self._get_taut(n_sg)[0]
        _e, rho_sxg, dedrho_sxg = self._get_cider_inputs(n_sg, taut_sg)
        e_g[:] = 0.0
        fs = self.calc_cider(_e, rho_sxg, dedrho_sxg, grad_mode=CIDERPW_GRAD_MODE_FORCE)
        for a, f in fs.items():
            Ftmp_av[a] = f
        self.world.sum(Ftmp_av, 0)
        if self.world.rank == 0:
            F_av[:] += Ftmp_av

    def stress_tensor_contribution_new(self, nt_sg, vt_sg, e_g, taut_sg, dedtaut_sg):
        stress1_vv = np.zeros((3, 3))
        _e, rho_sxg, dedrho_sxg = self._get_cider_inputs(nt_sg, taut_sg)
        np.zeros((len(nt_sg), rho_sxg.shape[1]) + e_g.shape)
        fs = self.calc_cider(
            _e, rho_sxg, dedrho_sxg, grad_mode=CIDERPW_GRAD_MODE_STRESS
        )
        stress1_vv[:] += 0.5 * (fs + fs.T)
        P = 0
        for s in range(self.nspin):
            for x in range(1, 4):
                P -= np.sum(dedrho_sxg[s, x] * rho_sxg[s, x]) * self.gd.dv
        for v in range(3):
            stress1_vv[v, v] += P
        for s in range(self.nspin):
            for v1 in range(3):
                for v2 in range(3):
                    stress1_vv[v1, v2] -= (
                        np.sum(rho_sxg[s, v1 + 1] * dedrho_sxg[s, v2 + 1]) * self.gd.dv
                    )
        self.world.sum(stress1_vv, 0)
        self.world.broadcast(stress1_vv, 0)
        # if self.world.rank != 0:
        #    stress1_vv[:] = 0.0

        e_g[:] = 0.0
        vt_sg[:] = 0.0
        dedtaut_sg[:] = 0.0
        self._add_from_cider_grid(e_g[None, :], _e[None, :])
        for s in range(len(nt_sg)):
            self._add_from_cider_grid(vt_sg[s, None], dedrho_sxg[s, 0:1])
            self._add_from_cider_grid(dedtaut_sg[s, None], dedrho_sxg[s, 4:5])
        return stress1_vv

    def stress_tensor_contribution(self, n_sg, taut_sg=None):
        e_g = np.zeros(n_sg.shape[1:])
        stress1_vv = np.zeros((3, 3))
        if taut_sg is None:
            taut_sg, dedtaut_sg, taut_sG = self._get_taut(n_sg)
        else:
            pass
        _e, rho_sxg, dedrho_sxg = self._get_cider_inputs(n_sg, taut_sg)
        tmp_dedrho_sxg = np.zeros((len(n_sg), rho_sxg.shape[1]) + e_g.shape)
        e_g[:] = 0.0
        fs = self.calc_cider(
            _e, rho_sxg, dedrho_sxg, grad_mode=CIDERPW_GRAD_MODE_STRESS
        )
        stress1_vv[:] += 0.5 * (fs + fs.T)
        self.world.sum(stress1_vv, 0)

        self._add_from_cider_grid(e_g[None, :], _e[None, :])
        for s in range(len(n_sg)):
            self._add_from_cider_grid(tmp_dedrho_sxg[s], dedrho_sxg[s])
        dedrho_sxg = tmp_dedrho_sxg
        tmp_rho_sxg = np.zeros_like(tmp_dedrho_sxg)
        for s in range(len(n_sg)):
            self._add_from_cider_grid(tmp_rho_sxg[s], rho_sxg[s])
        rho_sxg = tmp_rho_sxg

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        nspin = n_sg.shape[0]
        for s in range(nspin):
            for x in range(5):
                P -= integrate(dedrho_sxg[s, x], rho_sxg[s, x])

        dedtaut_sg[:] = tmp_dedrho_sxg[:, 4]
        tau_svvG = self.wfs.calculate_kinetic_energy_density_crossterms()

        stress_vv = P * np.eye(3)
        tau_cross_g = self.gd.empty()
        # TODO eventually we should update this to use more concise GPAW routines
        for s in range(nspin):
            if tau_svvG.shape[1] == 3:
                for v1 in range(3):
                    for v2 in range(3):
                        stress_vv[v1, v2] -= integrate(
                            rho_sxg[s, v1 + 1], dedrho_sxg[s, v2 + 1]
                        )
                        self.distribute_and_interpolate(
                            tau_svvG[s, v1, v2], tau_cross_g
                        )
                        stress_vv[v1, v2] -= integrate(tau_cross_g, dedtaut_sg[s])
            else:
                assert tau_svvG.shape[1] == 6
                tau_swG = tau_svvG
                w = 0
                for v1 in range(3):
                    for v2 in range(v1, 3):
                        stress_vv[v1, v2] -= integrate(
                            rho_sxg[s, v1 + 1], dedrho_sxg[s, v2 + 1]
                        )
                        if v1 != v2:
                            stress_vv[v2, v1] -= integrate(
                                rho_sxg[s, v2 + 1], dedrho_sxg[s, v1 + 1]
                            )
                        self.distribute_and_interpolate(tau_swG[s, w], tau_cross_g)
                        stress_contrib = integrate(tau_cross_g, dedtaut_sg[s])
                        stress_vv[v1, v2] -= stress_contrib
                        if v1 != v2:
                            stress_vv[v2, v1] -= stress_contrib
                        w += 1

        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        for s in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])

        stress_vv = 0.5 * (stress_vv + stress_vv.T)

        self.gd.comm.sum(stress_vv)

        stress_vv += stress1_vv

        self.gd.comm.broadcast(stress_vv, 0)
        return stress_vv
