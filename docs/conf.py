# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import sys
from pkg_resources import get_distribution

sys.path.insert(0, str(Path("./ext").resolve()))

project = "spey"
copyright = "2023, Jack Y. Araz"
author = "Jack Y. Araz"
release = get_distribution("spey").version
version = ".".join(release.split(".")[:3])
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "sphinx.ext.napoleon",
    "sphinx_click.ext",
    "nbsphinx",
    "sphinx_issues",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "xref",
    "myst_parser",
]

# external links
xref_links = {
    "1007.1727": ("[arXiv:1007.1727]", "https://doi.org/10.48550/arXiv.1007.1727"),
    "pyhf": ("pyhf", "https://pyhf.readthedocs.io/"),
    "HEPData": ("HEPData", "https://www.hepdata.net"),
    "1809.05548": ("[arXiv:1809.05548]", "https://doi.org/10.48550/arXiv.1809.05548"),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

bibtex_default_style = "unsrt"

bibtex_bibfiles = ["bib/references.bib"]

# exclude_patterns = ["releases/changelog-dev.md", "figs"]

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/", None),
#     "matplotlib": ("https://matplotlib.org/stable/", None),
#     "iminuit": ("https://iminuit.readthedocs.io/en/stable/", None),
# }

# GitHub repo
issues_github_path = "SpeysideHEP/spey"

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns = ["introduction.rst"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst"]

# The master toctree document.
master_doc = "index"
man_pages = [(master_doc, "spey", "spey Documentation", [author], 1)]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

mathjax3_config = {
    "tex2jax": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]},
    "tex": {
        "macros": {
            "bm": ["\\boldsymbol{#1}", 1],  # \usepackage{bm}, see mathjax/MathJax#1219
            "HiFa": r"\texttt{HistFactory}",
            "Root": r"\texttt{ROOT}",
            "RooStats": r"\texttt{RooStats}",
            "RooFit": r"\texttt{RooFit}",
            "pyhf": r"\texttt{pyhf}",
            "CLs": r"\mathrm{CL}_{s}",
            "freeset": r"\bm{\eta}",
            "constrset": r"\bm{\chi}",
            "singleconstr": r"\chi",
            "channelcounts": r"\bm{n}",
            "auxdata": r"\bm{a}",
            "poiset": r"\bm{\psi}",
            "nuisset": r"\bm{\theta}",
            "fullset": r"\bm{\phi}",
            "singlefull": r"\phi",
            "TeV": r"\textrm{TeV}",
        }
    },
}

# Output file base name for HTML help builder.
htmlhelp_basename = "speydoc"

# sphinx-copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_here_doc_delimiter = "EOF"

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "spey.tex",
        "spey Documentation",
        "Jack Y. Araz",
        "manual",
    )
]
