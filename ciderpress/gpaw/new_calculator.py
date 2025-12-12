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
from gpaw.new import trace
from gpaw.new.ase_interface import ASECalculator, DFTCalculation
from gpaw.new.xc import GGAFunctional, MGGAFunctional, UGArray, gradient_and_sigma

from ciderpress.gpaw.calculator import get_cider_functional
from ciderpress.gpaw.cider_fft import add_gradient_correction


class CiderGGAFunctional(GGAFunctional):
    def __init__(self, xc, grid, xp=np):
        if xp != np:
            raise NotImplementedError("Cider on GPU")
        super().__init__(xc, grid, xp)

    def calculate(
        self, nt_sr: UGArray, taut_sr: UGArray | None = None
    ) -> tuple[float, UGArray, UGArray | None]:
        gradn_svr, sigma_xr = gradient_and_sigma(self.grad_v, nt_sr)

        vxct_sr = nt_sr.new(zeroed=True)
        e_r = self.grid.empty(xp=self.xp)
        args = [nt_sr, vxct_sr, e_r]
        args = [a.data for a in args]
        dedgrad_svg = self.xc.calculate_impl_new(*args)
        add_gradient_correction(
            [grad.apply for grad in self.grad_v], dedgrad_svg, vxct_sr.data
        )
        exc = e_r.integrate()
        return exc, vxct_sr, None


class CiderMGGAFunctional(MGGAFunctional):
    def __init__(self, xc, grid, xp=np):
        if xp != np:
            raise NotImplementedError("Cider on GPU")
        super().__init__(xc, grid, xp)

    def calculate(
        self, nt_sr: UGArray, taut_sr: UGArray | None = None
    ) -> tuple[float, UGArray, UGArray | None]:
        xp = nt_sr.xp
        nt_sr = nt_sr.to_xp(np)

        gradn_svr, sigma_xr = gradient_and_sigma(self.grad_v, nt_sr)
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
        dedgrad_svg = self.xc.calculate_impl_new(*args)
        add_gradient_correction(
            [grad.apply for grad in self.grad_v], dedgrad_svg, vxct_sr.data
        )
        vxct_sr = vxct_sr.to_xp(xp)
        dedtaut_sr = dedtaut_sr.to_xp(xp)

        return e_r.integrate(), vxct_sr, dedtaut_sr


class CiderXC(XC):
    def __init__(self, **kwargs):
        self.name = "CIDER"
        self.path = kwargs.pop("path")
        self.kwargs = kwargs

    def functional(self, *, collinear: bool, atoms: Atoms | None = None):
        assert collinear
        return get_cider_functional(self.path, **self.kwargs)


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
            self._dft = DFTCalculation.from_parameters(
                atoms, self.params, self.comm, self.log
            )
            self._fix_xc()
        self._atoms = atoms.copy()

    def create_new_calculation_from_old(self, atoms: Atoms) -> None:
        with self.timer("Morph"):
            self._dft = self.dft.new(atoms, self.params, self.log)
            self._fix_xc()
        self._atoms = atoms.copy()


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
    from gpaw.new.gpw import read_gpw

    if txt == "?":
        txt = "-" if filename is None else None

    log = Logger(txt, communicator)

    if mode is None:
        del mode

    kwargs = {key: value for key, value in locals().items() if key in PARAMETER_NAMES}

    if filename is not None:
        args = Parameters(mode="pw", **kwargs)._non_defaults
        if set(args) > {"mode", "parallel"}:
            raise ValueError(
                "Illegal argument(s) when reading from a file: " f'{", ".join(args)}'
            )
        atoms, dft, params, _ = read_gpw(
            filename, log=log, parallel=parallel, object_hooks=object_hooks
        )
        return CiderASECalculator(params, log=log, dft=dft, atoms=atoms)

    params = Parameters(**kwargs)
    return CiderASECalculator(params, log=log)
