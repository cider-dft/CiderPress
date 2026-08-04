<div align="left">
  <img src="https://github.com/mir-group/CiderPress/blob/main/docs/logos/cider_logo_and_name.png" height="80px"/>
</div>

CiderPress: Machine Learned Exchange-Correlation Functionals
------------------------------------------------------------
![Build Status](https://github.com/mir-group/CiderPress/actions/workflows/basic_tests.yml/badge.svg)

CiderPress provides tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations. Interfaces to the GPAW and PySCF codes are included.

Please see the [CiderPress website](https://mir-group.github.io/CiderPress/) for installation instructions and documentation.

## Packaged Functional Families

CiderPress 0.5.0 includes the published mapped models from three generations:

- CIDER23X: six semilocal/NLDF exchange models, for PySCF and classic GPAW.
- CIDER24X: `CIDER24Xne` and `CIDER24Xe`, SDMX exchange models for PySCF.
- CIDER26XC: `CIDER26XCCHEM`, `CIDER26XCCHEMD4`, and
  `CIDER26XCSURFSCI`, full-XC models for molecular chemistry and combined
  molecular/solid/surface-science applications.

All are selected by short name anywhere an `mlfunc` path is accepted. The
[model guide](https://mir-group.github.io/CiderPress/usage/production_models.html)
lists every name, feature representation, supported backend, functional
composition, and release checksum.

CIDER23X and CIDER24X store exchange and normally use the explicit
PBE0/CIDER surrogate-hybrid composition. CIDER26XC stores full XC and must not
be combined with an additional semilocal XC baseline.

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

For a classic GPAW calculation, load the production model as a full XC
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

Install optional runtimes only when needed:

```bash
pip install 'ciderpress[cider24]'  # PyTorch-backed CIDER24X models
pip install 'ciderpress[d4]'       # CIDER26XCCHEMD4 post-density energy
```

D4 evaluation is supported by PySCF and is added once after the density SCF;
its derivative is not part of the CIDER molecular gradient. GPAW rejects the
D4 model rather than silently omitting the fitted correction. CiderPress
0.5.0 supports classic GPAW with PAW setups, not `gpaw.new`.

The documentation contains a [framework overview](https://mir-group.github.io/CiderPress/theory/framework.html),
complete [PySCF](https://mir-group.github.io/CiderPress/usage/pyscf.html) and
[GPAW](https://mir-group.github.io/CiderPress/usage/gpaw.html) workflows, and
calculation-class-specific [SCF fallback settings](https://mir-group.github.io/CiderPress/usage/convergence.html).

## Questions and Comments

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the [Github page](https://github.com/mir-group/CiderPress/).

## Citing

If you find CiderPress or CIDER functionals useful in your research, please cite the following article
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
The above article introduces the CIDER23X functionals and much of the algorithms in CiderPress. If you use the CIDER24X functionals, please also cite
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
  URL = {https://doi.org/10.1021/acs.jctc.4c00999},
  eprint = {https://doi.org/10.1021/acs.jctc.4c00999}
}
```
