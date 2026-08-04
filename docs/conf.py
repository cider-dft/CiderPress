# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append("./tools/extensions")

from ciderpress import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CiderPress'
author = 'Kyle Bystrom'
year = datetime.now().year
copyright = f"{year}, Kyle Bystrom"
# The short X.Y version.
v,sv = __version__.split('.')[:2]
version = "%s.%s"%(v,sv)
# The full version, including alpha/beta/rc tags.
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'ciderdocext',
    'breathe',
]
bibtex_bibfiles = ['refs/refs.bib', 'refs/cider_refs.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'

breathe_projects = {"CiderPress": "xml"}
breathe_default_project = "CiderPress"
breathe_default_members = ("members", "undoc-members")
breathe_domain_by_extension = {
    "h" : "c",
}

napoleon_google_docstring = True
# napoleon_numpy_docstring = True
napoleon_use_param = False

# Doxygen exposes these implementation and platform types in C signatures,
# but they are defined by the C standard library, MPI, or the FFT backend
# rather than by CiderPress itself.
nitpick_ignore = [
    ("c:identifier", "size_t"),
    ("c:identifier", "int64_t"),
    ("c:identifier", "uint8_t"),
    ("c:identifier", "MPI_Comm"),
    ("c:identifier", "mpi_fft3d_plan_t"),
    # Shorthand types and field names in inherited NumPy/PySCF/scikit-learn
    # docstrings.  They are rendered as types by autodoc but are not objects
    # owned by this documentation set.
    ("py:class", "np.ndarray"),
    ("py:class", "ndarray"),
    ("py:class", "norms"),
    ("py:class", "Returns"),
    ("py:class", "Nctrl"),
    ("py:class", "N1"),
    ("py:class", "shape"),
    ("py:class", "n_samples_X"),
    ("py:class", "n_dims"),
    ("py:class", "2"),
    ("py:class", "callable"),
    ("py:class", "vxc"),
    ("py:class", "nspin x nao x nao"),
    ("py:class", "KohnShamDFT"),
    ("py:class", "MappedXC"),
    ("py:class", "MappedXC2"),
    ("py:class", "SemilocalSettings"),
    ("py:func", "get_descriptors"),
    ("py:func", "make_cider_calc"),
    ("py:func", "density_fit"),
    ("py:mod", "pbc"),
    ("py:obj", "mode"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = "logos/cider_logo_and_name.png"
html_theme_options = {'logo_only': True}
