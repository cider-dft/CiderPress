import warnings

warnings.warn(
    "ciderpress.pyscf.pbc is an experimental reproduction interface. "
    "It supports semilocal and SDMX features only and can require substantial "
    "memory; use the classic GPAW interface for the documented periodic NLDF "
    "production route.",
    stacklevel=2,
)
