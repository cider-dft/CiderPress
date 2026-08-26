<div align="left">
  <img src="docs/logos/cider_logo_and_name.svg" height="80px" alt="CiderPress"/>
</div>

CiderPress: Machine-Learned Exchange-Correlation Functionals
------------------------------------------------------------
![Build Status](https://github.com/cider-dft/CiderPress/actions/workflows/basic_tests.yml/badge.svg)

CiderPress provides tools for training and evaluating CIDER functionals in
density functional theory calculations. Interfaces to PySCF and classic GPAW
are included to enable full DFT calculations with CIDER functionals.

## Installation

CiderPress contains compiled C/C++ numerical kernels. A working C/C++ build
toolchain, CMake, BLAS/LAPACK, and an FFT implementation are required. PySCF
is installed as a core Python dependency; GPAW is a separate optional host
code.

Install a published release with:

```bash
pip install ciderpress
```

From a source checkout, use:

```bash
git clone https://github.com/cider-dft/CiderPress.git
cd CiderPress
pip install .
```

Install the optional runtime required by the selected model:

```bash
pip install 'ciderpress[cider24]'  # PyTorch-backed CIDER24X models
pip install 'ciderpress[d4]'       # CIDER26XCCHEMD4 post-density energy (pyscf-dispersion + dftd4)
```

See the [installation guide](https://cider-dft.github.io/CiderPress/installation/installation.html)
for compiler, MPI, FFT, MKL, GPAW, and source-build details.

## Documentation

- [Documentation index](https://cider-dft.github.io/CiderPress/)
- [Choosing a packaged model](https://cider-dft.github.io/CiderPress/usage/production_models.html)
- [PySCF guide](https://cider-dft.github.io/CiderPress/usage/pyscf.html)
- [GPAW guide](https://cider-dft.github.io/CiderPress/usage/gpaw.html)
- [SCF convergence recommendations](https://cider-dft.github.io/CiderPress/usage/convergence.html)
- [CIDER framework and model lineage](https://cider-dft.github.io/CiderPress/theory/framework.html)

## Packaged Functional Families

CiderPress 0.5.0 includes mapped models from three CIDER generations:

- CIDER23X: six semilocal/NLDF exchange models, for PySCF and classic GPAW.
- CIDER24X: `CIDER24Xne` and `CIDER24Xe`, SDMX exchange models for PySCF.
- CIDER26XC: `CIDER26XCCHEM`, `CIDER26XCCHEMD4`, and
  `CIDER26XCSURFSCI`, full XC models for molecular chemistry and combined
  molecular/solid/surface-science applications. `CIDER26XCSURFSCI` is supported
  in both PySCF and GPAW. `CIDER26XCCHEM` is supported in both codes as well,
  but only recommended for molecular calculations in PySCF. `CIDER26XCCHEMD4`
  is only supported in PySCF currently, due to the need for the D4 dispersion
  term.

All of these functionals can be selected and loaded by their short name
anywhere an `mlfunc` path is accepted. The
[model guide](https://cider-dft.github.io/CiderPress/usage/production_models.html)
lists every name, feature representation, supported backend, functional
composition, and release checksum.

CIDER23X and CIDER24X store exchange and normally use the explicit
PBE0/CIDER surrogate-hybrid composition. CIDER26XC stores full XC and uses the
semilocal baseline contained in its model file.

## Quick starts

For a molecular calculation:

```python
from pyscf import dft
from ciderpress.pyscf.dft import make_cider_calc

mf = make_cider_calc(dft.RKS(mol), "CIDER26XCCHEM")
energy = mf.kernel()
```

For an exchange-only model:

```python
mf = make_cider_calc(
    dft.RKS(mol),
    "CIDER23X_NL_MGGA_DTR",
    xmix=0.25,
    xkernel="GGA_X_PBE",
    ckernel="GGA_C_PBE",
)
```

For a classic GPAW calculation, load the periodic model as a full-XC
functional:

```python
from ciderpress.gpaw.calculator import get_cider_functional

xc = get_cider_functional(
    "CIDER26XCSURFSCI",
    xmix=1.0,
    xkernel=None,
    ckernel=None,
    pasdw_store_funcs=False,
)
```

D4 evaluation is supported by PySCF and is added once after the density SCF.
The current CIDER molecular gradient contains the electronic contribution.
The GPAW interface reports an error when a D4 model is selected. CiderPress
0.5.0 supports classic GPAW with PAW setups; `gpaw.new` is outside this
release interface.

Complete examples and fallback ladders to fix SCF convergence problems
are linked in the documentation list above.

## Support

Use the [GitHub issue tracker](https://github.com/cider-dft/CiderPress/issues)
for bug reports, documentation problems, and technical questions. Scientific
inquiries may also be sent to Kyle Bystrom at kylebystrom@gmail.com.

## Citing

For work using CiderPress or CIDER functionals, cite the following article:
```
@article{PhysRevB.110.075130,
  title = {Nonlocal machine-learned exchange functional for molecules and solids},
  author = {Bystrom, Kyle and Kozinsky, Boris},
  journal = {Phys. Rev. B},
  volume = {110},
  issue = {7},
  pages = {075130},
  numpages = {30},
  year = {2024},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.110.075130},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.110.075130}
}
```
This article introduces the CIDER23X functionals and much of the numerical
framework in CiderPress.  Work using CIDER24X should also cite:
```
@article{doi:10.1021/acs.jctc.4c00999,
  author = {Bystrom, Kyle and Falletta, Stefano and Kozinsky, Boris},
  title = {Training Machine-Learned Density Functionals on Band Gaps},
  journal = {Journal of Chemical Theory and Computation},
  volume = {20},
  number = {17},
  pages = {7516-7532},
  year = {2024},
  doi = {10.1021/acs.jctc.4c00999},
  note ={PMID: 39178337},
  URL = {https://doi.org/10.1021/acs.jctc.4c00999}
}
```

The CIDER26XC models accompany the forthcoming manuscript *Machine-Learned
Exchange-Correlation Functionals in the CIDER Framework and Application to
Chemistry and Surface Science*. See the
[citation guide](https://cider-dft.github.io/CiderPress/reference/citing.html)
for the current citation record.
