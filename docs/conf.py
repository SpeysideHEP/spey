# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import re
import sys
import warnings
from pathlib import Path

from pkg_resources import get_distribution

sys.path.insert(0, str(Path("./ext").resolve()))

project = "spey"
copyright = "2025, Jack Y. Araz"
author = "Jack Y. Araz"
release = get_distribution("spey").version
version = ".".join(release.split(".")[:3])
language = "en"

# html_title = f'Spey <span id="release">v{release}</span>'

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
    # "myst_parser",
    # "sphinx_rtd_size",
    "myst_nb",
]
nb_execution_mode = "off"

suppress_warnings = ["autosectionlabel.*"]
# set the width of the page
# sphinx_rtd_size_width = "100%"

# external links
xref_links = {
    "1007.1727": ("[arXiv:1007.1727]", "https://doi.org/10.48550/arXiv.1007.1727"),
    "pyhf": ("pyhf", "https://pyhf.readthedocs.io/"),
    "HEPData": ("HEPData", "https://www.hepdata.net"),
    "1809.05548": ("[arXiv:1809.05548]", "https://doi.org/10.48550/arXiv.1809.05548"),
    "1202.3415": ("[arXiv:1202.3415]", "https://doi.org/10.48550/arXiv.1202.3415"),
    "physics/0406120": (
        "[arXiv:physics/0406120]",
        "https://10.48550/arXiv.physics/0406120",
    ),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

bibtex_default_style = "unsrt"

bibtex_bibfiles = ["./bib/references.bib", "./bib/cited.bib"]

# GitHub repo
issues_github_path = "https://github.com/SpeysideHEP/spey/issues"

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = True

templates_path = ["_templates"]
exclude_patterns = ["introduction.rst", "requirements.txt"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst"]

# The master toctree document.
master_doc = "index"
man_pages = [(master_doc, "spey", "spey Documentation", [author], 1)]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_provider": "github",
    "repository_url": "https://github.com/SpeysideHEP/spey",
    "use_repository_button": True,  # add a 'link to repository' button
    "use_issues_button": False,  # add an 'Open an Issue' button
    "use_download_button": True,  # add a 'Download' button
    "home_page_in_toc": False,
    "repository_branch": "main",
    # "announcement": "Some announcement",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_fullscreen_button": True,
    # "launch_buttons": {
    #     "colab_url": "https://colab.research.google.com/",
    # },
    "show_navbar_depth": 1,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SpeysideHEP/spey",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "arXiv",
            "url": "https://arxiv.org/abs/2307.06996",
            "icon": "https://img.shields.io/static/v1?style=plastic&label=arXiv&message=2307.06996&color=brightgreen",
            "type": "url",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/spey/",
            "icon": "https://img.shields.io/pypi/v/spey?style=plastic",
            "type": "url",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_logo = "img/spey-logo.png"
html_favicon = "img/spey-logo.png"
logo_only = True

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

# adapted from https://astrodata.nyc/posts/2021-04-23-zenodo-sphinx/
zenodo_path = Path("ZENODO.rst")
if not zenodo_path.exists():
    import textwrap

    try:
        import requests

        headers = {"accept": "application/x-bibtex"}
        response = requests.get(
            "https://zenodo.org/api/records/10156353", headers=headers, timeout=5
        )
        response.encoding = "utf-8"

        txt = response.text
        linker = re.search("@software{(.+?),\n", txt).group(1)
        txt = txt.replace(linker, "spey_zenodo")
        zenodo_record = ".. code-block:: bibtex\n\n" + textwrap.indent(txt, " " * 4)
    except Exception as e:
        warnings.warn("Failed to retrieve Zenodo record for Spey: " f"{str(e)}")
        zenodo_record = (
            "`Retrieve the Zenodo record here. <https://zenodo.org/record/10156353>`_"
        )

    with open(zenodo_path, "w", encoding="utf-8") as f:
        f.write(zenodo_record)
