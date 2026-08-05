"""GPAW 25.7 serial configuration for the OpenBLAS CI builds."""

import os
import sys

mpi = False
fftw = True

dependency_root = os.environ["CIDER_DEPS"]
library_dirs += [os.path.join(dependency_root, "lib")]
include_dirs += [os.path.join(dependency_root, "include")]

if sys.platform == "darwin":
    brew_prefix = os.environ["HOMEBREW_PREFIX"]
    libraries += ["openblas", "fftw3"]
    library_dirs += [
        os.path.join(brew_prefix, "lib"),
        os.path.join(brew_prefix, "opt", "openblas", "lib"),
    ]
    include_dirs += [
        os.path.join(brew_prefix, "include"),
        os.path.join(brew_prefix, "opt", "openblas", "include"),
    ]
else:
    libraries += ["blas", "lapack", "fftw3"]
