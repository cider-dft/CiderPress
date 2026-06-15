import numpy as np
from ase import Atoms
from gpaw.dft import (
    IO,
    PARAMETER_NAMES,
    XC,
    Eigensolver,
    ExtensionInput,
    KptsType,
    Logger,
    Mixer,
    Mode,
    MonkhorstPack,
    MPIComm,
    Occupations,
    Parameters,
    Path,
    PoissonSolver,
    Sequence,
    Symmetry,
)
from gpaw.mpi import world
from gpaw.new import trace
from gpaw.new.ase_interface import ASECalculator, DFTCalculation
from gpaw.new.calculation import (
    check_atoms_too_close,
    check_atoms_too_close_to_boundary,
    write_atoms,
)
from gpaw.new.xc import (
    Functional,
    GGAFunctional,
    HybridXC,
    MGGAFunctional,
    OldXCFunctional,
    PWFDWaveFunctions,
    UGArray,
    UGDesc,
    zips,
)

from ciderpress.gpaw.calculator import get_cider_functional
from ciderpress.gpaw.cider_fft import add_gradient_correction


def read_gpw(
    filename,
    *,
    log=None,
    comm=None,
    parallel=None,
    dtype=None,
    force_complex_dtype=False,
    object_hooks=None,
):
    """
    Read gpw file

    Returns
    -------
    atoms, calculation, params, builder
    """
    from gpaw.new.gpw import mpi, read_atoms, read_dft_state, ulm, units

    parallel = parallel or {}

    if not isinstance(log, Logger):
        log = Logger(log, comm or mpi.world)

    comm = log.comm

    log(f"Reading from {filename}")

    reader = ulm.Reader(filename)
    singlep = reader.get("precision", "double") == "single"

    atoms = read_atoms(reader.atoms)
    kwargs = reader.parameters.asdict()
    kwargs["parallel"] = parallel

    if "dtype" in kwargs:
        kwargs["dtype"] = np.dtype(kwargs["dtype"])

    if dtype is not None:
        kwargs["dtype"] = dtype

    shape = reader.wave_functions.eigenvalues.shape
    if len(shape) == 3:
        kwargs["nbands"] = shape[-1]
    else:
        kwargs["nbands"] = shape[-1] // 2

    for old_keyword in ["fixdensity", "txt"]:
        kwargs.pop(old_keyword, None)

    if object_hooks:
        for key, hook in object_hooks.items():
            if key in kwargs:
                kwargs[key] = hook(kwargs[key])

    params = CiderParameters(**kwargs)
    if force_complex_dtype:
        params.mode.force_complex_dtype = True
    builder, params, state = read_dft_state(
        reader,
        atoms=atoms,
        params=params,
        comm=comm,
        singlep=singlep,
        log=log,
        **kwargs,
    )
    ibzwfs, density, potential, energies = state

    dft = CiderDFTCalculation(
        atoms,
        ibzwfs,
        density,
        potential,
        builder.setups,
        builder.create_scf_loop(),
        pot_calc=builder.create_potential_calculator(),
        params=params,
        energies=energies,
        log=log,
    )

    results = {
        key: value / units[key] for key, value in reader.results.asdict().items()
    }

    if results:
        log(f'Read {", ".join(sorted(results))}')

    if reader.version < 4 and "magmoms" in results:
        magmom_a = results["magmoms"]
        magmom_av = np.pad(magmom_a[:, np.newaxis], [(0, 0), (2, 0)])
        results["non_collinear_magmoms"] = magmom_av

    dft.results = results

    if builder.mode in ["pw", "fd"]:  # fd = finite-difference
        data = ibzwfs._wfs_u[0].psit_nX.data
        if not hasattr(data, "fd"):  # fd = file-descriptor
            reader.close()
    else:
        reader.close()

    return atoms, dft, params, builder


def create_functional(
    xc: OldXCFunctional | str | dict, grid: UGDesc, xp=np, comms=None
) -> Functional:
    assert comms is not None
    exx_fraction = 0.0
    exx_omega = 0.0
    exx_yukawa = False
    if isinstance(xc, (str, dict)):
        xc = XC(xc)

    if xc.type == "HYB":
        assert isinstance(xc, HybridXC)
        exx_fraction = xc.exx_fraction
        exx_omega = xc.omega
        exx_yukawa = xc.yukawa
        xc = xc.xc

    if xc.type == "GGA":
        functional = CiderGGAFunctional(xc, grid)
        functional._world = comms["w"]
        functional._kptband_comm = comms["D"]
    elif xc.type == "MGGA":
        functional = CiderMGGAFunctional(xc, grid)
        functional._world = comms["w"]
        functional._kptband_comm = comms["D"]
    else:
        raise ValueError(f"{xc.type} not supported")

    functional.exx_fraction = exx_fraction
    functional.exx_omega = exx_omega
    functional.exx_yukawa = exx_yukawa

    return functional


def _initialize_cider_xc(self, nspin):
    if self.xc.is_initialized:
        return
    from ase.utils.timing import Timer

    self.xc.dens = self._density
    self.xc._interpolate = self._interpolate
    self.xc._hamiltonian = None
    self.xc.timer = Timer()
    self.xc.world = self._world
    self.xc.kptband_comm = self._kptband_comm
    self.xc.setups = None
    self.xc.atom_slices_s = None
    self.xc._plan = None
    self.xc.nspin = nspin
    self.xc._check_parallelization()
    self.xc._setup_plan()
    self.xc.fft_obj.initialize_backend()
    self.xc.initialize_more_things(setups=self._setups, atomdist=self._atomdist)
    self.xc._setup_atom_slices()
    assert self.xc.is_initialized


def _move_xc(self, relpos_ac, atomdist, density):
    self.xc.spos_ac = relpos_ac
    self._atomdist = atomdist
    self._density = density
    self.xc._plan = None
    self.xc.atom_slices_s = None


class CiderGGAFunctional(GGAFunctional):
    def __init__(self, xc, grid, xp=np):
        if xp != np:
            raise NotImplementedError("Cider on GPU")
        super().__init__(xc, grid, xp)

    move = _move_xc

    def calculate(
        self, nt_sr: UGArray, taut_sr: UGArray | None = None
    ) -> tuple[float, UGArray, UGArray | None]:
        # gradn_svr, sigma_xr = gradient_and_sigma(self.grad_v, nt_sr)
        vxct_sr = nt_sr.new(zeroed=True)
        e_r = self.grid.empty(xp=self.xp)
        args = [nt_sr, vxct_sr, e_r]
        args = [a.data for a in args]
        _initialize_cider_xc(self, args[0].shape[0])
        dedgrad_svg = self.xc.calculate_impl_new(*args)
        add_gradient_correction(
            [grad.apply for grad in self.grad_v], dedgrad_svg, vxct_sr.data
        )
        exc = e_r.integrate()
        return exc, vxct_sr, None


def _taut(ibzwfs, grid):
    dpsit1_vR = grid.new(comm=None, dtype=ibzwfs.dtype).empty(3)
    taut1_swR = grid.new(comm=None).zeros((ibzwfs.nspins, 6))
    assert isinstance(taut1_swR, UGArray)  # Argggghhh!
    domain_comm = grid.comm

    for wfs in ibzwfs:
        assert isinstance(wfs, PWFDWaveFunctions)
        psit_nG = wfs.psit_nX
        pw = psit_nG.desc

        pw1 = pw.new(comm=None)
        psit1_G = pw1.empty()
        iGpsit1_G = pw1.empty()
        Gplusk1_Gv = pw1.reciprocal_vectors()

        occ_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n

        (N,) = psit_nG.mydims
        for n1 in range(0, N, domain_comm.size):
            n2 = min(n1 + domain_comm.size, N)
            psit_nG[n1:n2].gather_all(psit1_G)
            n = n1 + domain_comm.rank
            if n >= N:
                continue
            f = occ_n[n]
            if f == 0.0:
                continue
            for Gplusk1_G, dpsit1_R in zips(Gplusk1_Gv.T, dpsit1_vR):
                iGpsit1_G.data[:] = psit1_G.data
                iGpsit1_G.data *= 1j * Gplusk1_G
                iGpsit1_G.ifft(out=dpsit1_R)
            w = 0
            for v1 in range(3):
                for v2 in range(v1, 3):
                    taut1_swR[wfs.spin, w].data += (
                        f * (dpsit1_vR[v1].data.conj() * dpsit1_vR[v2].data).real
                    )
                    w += 1

    ibzwfs.kpt_comm.sum(taut1_swR.data, 0)
    ibzwfs.band_comm.sum(taut1_swR.data, 0)
    domain_comm.sum(taut1_swR.data, 0)
    symmetries = ibzwfs.ibz.symmetries
    taut1_swR.symmetrize(symmetries.rotation_scc, symmetries.translation_sc)
    taut_swR = grid.empty((ibzwfs.nspins, 6))
    taut_swR.scatter_from(taut1_swR)
    return taut_swR


def stress_contribution(self, ibzwfs, density, interpolate):
    # args, kwargs = self._args(ibzwfs, density, interpolate)
    assert self.xp is np
    # e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr, taut_sr, dedtaut_sr = args

    nt_sR = density.nt_sR
    e_r = self.grid.empty(xp=self.xp)
    nt_sr = interpolate(nt_sR)
    vt_sr = nt_sr.new(zeroed=True)
    nspins = nt_sr.dims[0]
    gradn_svr = nt_sr.desc.empty((nspins, 3), xp=self.xp)
    sigma_xr = nt_sr.desc.empty(nspins * 2 - 1, xp=self.xp)
    dedsigma_xr = sigma_xr.new()
    kwargs = {"gradn_svr": gradn_svr}
    sigma_xr.data[:] = 0
    dedsigma_xr.data[:] = 0
    gradn_svr.data[:] = 0
    taut_swR = _taut(ibzwfs, density.nt_sR.desc)
    taut_sR = density.taut_sR
    assert taut_sR is not None
    taut_sr = interpolate(taut_sR)
    dedtaut_sr = taut_sr.new()
    args = [e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr, taut_sr, dedtaut_sr]
    kwargs["taut_swR"] = taut_swR
    kwargs["interpolate"] = interpolate

    args2 = [nt_sr, vt_sr, e_r, taut_sr, dedtaut_sr]
    args2 = [a.data for a in args2]
    stress1_vv = self.xc.stress_tensor_contribution_new(*args2)
    if ibzwfs.kpt_band_comm.rank == 0:
        stress_vv = self._stress(*args, **kwargs)
        if ibzwfs.domain_comm.rank == 0:
            stress_vv += stress1_vv
    else:
        stress_vv = self.xp.zeros((3, 3))
    return stress_vv


class CiderMGGAFunctional(MGGAFunctional):
    def __init__(self, xc, grid, xp=np):
        if xp != np:
            raise NotImplementedError("Cider on GPU")
        super().__init__(xc, grid, xp)

    move = _move_xc

    stress_contribution = stress_contribution

    def calculate(
        self, nt_sr: UGArray, taut_sr: UGArray | None = None
    ) -> tuple[float, UGArray, UGArray | None]:
        xp = nt_sr.xp
        nt_sr = nt_sr.to_xp(np)
        e_r = self.grid.empty()
        if taut_sr is None:
            taut_sr = nt_sr.new(zeroed=True)
        else:
            taut_sr = taut_sr.to_xp(np)
        dedtaut_sr = taut_sr.new()
        vxct_sr = taut_sr.new()
        vxct_sr.data[:] = 0.0
        args = [nt_sr, vxct_sr, e_r, taut_sr, dedtaut_sr]
        args = [array.data for array in args]
        _initialize_cider_xc(self, args[0].shape[0])
        dedgrad_svg = self.xc.calculate_impl_new(*args)
        add_gradient_correction(
            [grad.apply for grad in self.grad_v], dedgrad_svg, vxct_sr.data
        )
        vxct_sr = vxct_sr.to_xp(xp)
        dedtaut_sr = dedtaut_sr.to_xp(xp)
        return e_r.integrate(), vxct_sr, dedtaut_sr


class CiderXC(XC):
    def __init__(self, **kwargs):
        name = kwargs.get("name", "CIDER")
        # TODO remove and raise error if model is missing
        # model = kwargs.get("model", "/mnt/home/kbystrom/Production/CiderPress/functionals/SCIDER_NST.yaml")
        model = kwargs["model"]
        kwargs = {k: v for k, v in kwargs.items() if k not in ["name", "model"]}
        assert name == "CIDER"
        self.name = "CIDER"
        self.model = model
        self.kwargs = kwargs

    def todict(self):
        return {"name": self.name, "model": self.model, **self.kwargs}

    def functional(self, *, collinear: bool, atoms: Atoms | None = None):
        assert collinear
        return get_cider_functional(self.model, **self.kwargs)


class CiderDFTCalculation(DFTCalculation):
    @classmethod
    def from_parameters(
        cls, atoms: Atoms, params: Parameters, comm=None, log=None
    ) -> DFTCalculation:
        """Create DFTCalculation object from parameters and atoms."""
        check_atoms_too_close(atoms)
        check_atoms_too_close_to_boundary(atoms)

        if not isinstance(log, Logger):
            log = Logger(log, comm or world)

        builder = params.dft_component_builder(atoms, log=log)

        basis_set = builder.create_basis_set()

        # The SCF-loop has a Hamiltonian that has an fft-plan that is
        # cached for later use, so best to create the SCF-loop first
        # FIX this!
        scf_loop = builder.create_scf_loop()

        pot_calc = builder.create_potential_calculator()

        density = builder.density_from_superposition(basis_set)
        if len(atoms) == 0:
            density.nt_sR.data[:] = 1.0
        density.normalize(pot_calc.charge)

        builder.xc._interpolate = pot_calc.interpolate
        builder.xc.move(pot_calc.relpos_ac, builder.atomdist, density)

        potential, energies, _ = pot_calc.calculate_without_orbitals(
            density, kpt_band_comm=builder.communicators["D"]
        )
        ibzwfs = builder.create_ibz_wave_functions(basis_set, potential)

        if ibzwfs._wfs_u[0].has_eigs:
            nelectrons = density.nvalence - density.charge + pot_calc.charge
            ibzwfs.calculate_occs(scf_loop.occ_calc, nelectrons)

        if False:
            density.update(ibzwfs, ked=pot_calc.xc.type == "MGGA")

        write_atoms(atoms, builder.initial_magmom_av, builder.grid, log)
        log(ibzwfs)
        log(density)
        log(potential)
        log(builder.setups)
        log(scf_loop)
        log(pot_calc)

        return cls(
            atoms,
            ibzwfs,
            density,
            potential,
            builder.setups,
            scf_loop,
            pot_calc,
            log,
            params=params,
            energies=energies,
        )

    def new(self, atoms, params, log=None):
        """Create new DFTCalculation object."""

        if params.mode.name != "pw":
            raise ReuseWaveFunctionsError

        ibzwfs = self.ibzwfs
        # if ibzwfs.domain_comm.size != 1:
        #     raise ReuseWaveFunctionsError

        check_atoms_too_close(atoms)
        check_atoms_too_close_to_boundary(atoms)

        builder = params.dft_component_builder(atoms, log=log)

        kpt_kc = builder.ibz.kpt_kc
        old_kpt_kc = ibzwfs.ibz.kpt_kc
        if len(kpt_kc) != len(old_kpt_kc):
            raise ReuseWaveFunctionsError
        if abs(kpt_kc - old_kpt_kc).max() > 1e-9:
            raise ReuseWaveFunctionsError

        log("Interpolating wave functions to new cell")

        scf_loop = builder.create_scf_loop()
        pot_calc = builder.create_potential_calculator()

        density = self.density.new(
            builder.grid,
            builder.interpolation_desc,
            builder.relpos_ac,
            builder.atomdist,
        )
        density.normalize(pot_calc.charge)

        # Make sure all have exactly the same density.
        # Not quite sure it is needed???
        # At the moment we skip it on GPU's because it doesn't
        # work!
        if density.nt_sR.xp is np:
            ibzwfs.kpt_band_comm.broadcast(density.nt_sR.data, 0)

        builder.xc._interpolate = pot_calc.interpolate
        builder.xc.move(pot_calc.relpos_ac, builder.atomdist, density)

        if False:
            density.update(ibzwfs, ked=pot_calc.xc.type == "MGGA")

        potential, energies, _ = pot_calc.calculate(density)

        ibzwfs.make_sure_wfs_are_read_from_gpw_file()
        old_ibzwfs = ibzwfs

        def create_wfs(spin, q, k, kpt_c, weight):
            wfs = old_ibzwfs._get_wfs(k, spin)
            return wfs.morph(builder.wf_desc, builder.relpos_ac, builder.atomdist)

        ibzwfs = ibzwfs.create(
            ibz=builder.ibz,
            ncomponents=old_ibzwfs.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=old_ibzwfs.kpt_comm,
            kpt_band_comm=old_ibzwfs.kpt_band_comm,
            comm=self.comm,
        )

        write_atoms(atoms, builder.initial_magmom_av, builder.grid, log)
        log(ibzwfs)
        log(density)
        log(potential)
        log(builder.setups)
        log(scf_loop)
        log(pot_calc)

        return CiderDFTCalculation(
            atoms,
            ibzwfs,
            density,
            potential,
            builder.setups,
            scf_loop,
            pot_calc,
            log,
            params=params,
            energies=energies,
        )

    def move_atoms(self, atoms):
        if self.pot_calc.xc.xc.atomdist is None:
            self.pot_calc.xc.move(
                self.relpos_ac,
                self.density.D_asii.layout.atomdist,
                self.density,
            )
            self.pot_calc.xc.xc.atomdist = self.pot_calc.xc._atomdist
            self.pot_calc.xc._interpolate = self.pot_calc.interpolate
        super().move_atoms(atoms)
        self.pot_calc.xc.move(
            self.relpos_ac, self.density.D_asii.layout.atomdist, self.density
        )
        self.pot_calc.xc._interpolate = self.pot_calc.interpolate
        return self


class CiderASECalculator(ASECalculator):
    def _fix_xc(self):
        xc = self._dft.pot_calc.xc
        if xc.type == "GGA":
            xc = CiderGGAFunctional(xc.xc, xc.grid, xc.xp)
        elif xc.type == "MGGA":
            xc = CiderMGGAFunctional(xc.xc, xc.grid, xc.xp)
        else:
            raise NotImplementedError

    @trace
    def create_new_calculation(self, atoms: Atoms) -> None:
        with self.timer("Init"):
            self._dft = CiderDFTCalculation.from_parameters(
                atoms, self.params, self.comm, self.log
            )
            # self._fix_xc()
        self._atoms = atoms.copy()

    def create_new_calculation_from_old(self, atoms: Atoms) -> None:
        with self.timer("Morph"):
            self._dft = self.dft.new(atoms, self.params, self.log)
            # self._fix_xc()
        self._atoms = atoms.copy()

    @trace
    def create_new_calculation(self, atoms: Atoms) -> None:
        with self.timer("Init"):
            self._dft = CiderDFTCalculation.from_parameters(
                atoms, self.params, self.comm, self.log
            )
        self._atoms = atoms.copy()

    def new(self, **kwargs):
        kwargs = {**self.params.todict(), **kwargs}
        return CiderGPAW(**kwargs)


"""
class CiderDFTComponentsBuilder(DFTComponentsBuilder):
    def __init__(self,
                 atoms: Atoms,
                 params: Parameters,
                 *,
                 log=None,
                 comm=None):
        super().__init__(atoms, params, log=log, comm=comm)
        xcfunc = params.xc.functional(collinear=(self.ncomponents < 4),
                                      atoms=self.atoms)
        self.xc = create_functional(xcfunc, self.fine_grid, self.xp)
"""


class CiderParameters(Parameters):
    def dft_component_builder(self, atoms, *, comm=None, log=None):
        if not isinstance(self.xc, CiderXC):
            self.xc = CiderXC(**(self.xc.kwargs))
        builder = self.mode.dft_components_builder(atoms, self, comm=comm, log=log)
        xcfunc = self.xc.functional(
            collinear=(builder.ncomponents < 4), atoms=builder.atoms
        )
        # assert self.xc.name == "CIDER"
        # kwargs = {k:v for k,v in self.xc.kwargs.items() if k != "model"}
        # xcfunc = get_cider_functional(self.xc.kwargs["model"], **kwargs)
        builder.xc = create_functional(
            xcfunc, builder.fine_grid, builder.xp, comms=builder.communicators
        )
        builder.xc._setups = builder.setups
        return builder


def CiderGPAW(
    filename: str | Path | IO[str] | None = None,
    *,
    basis: str | dict[str | int | None, str] | None = None,
    charge: float | None = None,
    convergence: dict | None = None,
    eigensolver: str | dict | Eigensolver | None = None,
    experimental: dict | None = None,
    extensions: Sequence[ExtensionInput] | None = None,
    gpts: Sequence[int] | None = None,
    h: float | None = None,
    hund: bool | None = None,
    interpolation: int | None = None,
    kpts: KptsType | MonkhorstPack | None = None,
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
    maxiter: int | None = None,
    mixer: dict | Mixer | None = None,
    mode: str | dict | Mode | None = None,
    nbands: int | str | None = None,
    occupations: dict | Occupations | None = None,
    parallel: dict | None = None,
    poissonsolver: dict | PoissonSolver | None = None,
    random: bool | None = None,
    setups: str | dict | None = None,
    soc: bool | None = None,
    spinpol: bool | None = None,
    symmetry: str | dict | Symmetry | None = None,
    xc: str | dict | XC | None = None,
    txt: str | Path | IO[str] | None = "?",
    communicator: MPIComm | Sequence[int] | None = None,
    object_hooks=None,
) -> CiderASECalculator:
    """Create ASE-compatible GPAW calculator.

    See :class:`gpaw.dft.Parameters` for the complete list of parameters.

    Parameters
    ==========
    filename:
        Name of gpw-file to restart from.
    txt:
        Text log-file.  Use ``None`` for no logging and ``'-'``
        for using standard out.
    communicator:
        MPI-communicator.  Default is to use ``gpaw.mpi.world``.
    object_hooks:
        Dictionart of hook-functions to create custom parameter-objects.
    """
    # from gpaw.new.gpw import read_gpw

    if txt == "?":
        txt = "-" if filename is None else None

    log = Logger(txt, communicator)

    if mode is None:
        del mode

    kwargs = {key: value for key, value in locals().items() if key in PARAMETER_NAMES}

    if filename is not None:
        args = CiderParameters(mode="pw", **kwargs)._non_defaults
        if set(args) > {"mode", "parallel"}:
            raise ValueError(
                "Illegal argument(s) when reading from a file: " f'{", ".join(args)}'
            )
        atoms, dft, params, _ = read_gpw(
            filename, log=log, parallel=parallel, object_hooks=object_hooks
        )
        return CiderASECalculator(params, log=log, dft=dft, atoms=atoms)

    params = CiderParameters(**kwargs)
    return CiderASECalculator(params, log=log)
