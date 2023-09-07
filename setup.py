from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as f:
    long_description = f.read()

with open("src/spey/_version.py", mode="r", encoding="UTF-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "numpy>=1.21.6",
    "scipy==1.10.0",
    "autograd==1.5",
    "semantic_version~=2.10",
    "tqdm>=4.64.0",
]

docs = [
    "sphinx==6.2.1",
    "sphinxcontrib-bibtex~=2.1",
    "sphinx-click",
    "sphinx_rtd_theme",
    "nbsphinx!=0.8.8",
    "sphinx-issues",
    "sphinx-copybutton>=0.3.2",
    "sphinx-togglebutton>=0.3.0",
    "myst-parser",
    "sphinx-rtd-size",
]

backend_plugins = [
    "default_pdf.uncorrelated_background = spey.backends.default_pdf:UncorrelatedBackground",
    "default_pdf.correlated_background = spey.backends.default_pdf:CorrelatedBackground",
    "default_pdf.third_moment_expansion = spey.backends.default_pdf:ThirdMomentExpansion",
    "default_pdf.effective_sigma = spey.backends.default_pdf:EffectiveSigma",
]

setup(
    name="spey",
    version=version,
    description=("Smooth inference for reinterpretation studies"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpeysideHEP/spey",
    project_urls={
        "Bug Tracker": "https://github.com/SpeysideHEP/spey/issues",
    },
    download_url=f"https://github.com/SpeysideHEP/spey/archive/refs/tags/v{version}.tar.gz",
    author="Jack Y. Araz",
    author_email=("jack.araz@durham.ac.uk"),
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={"spey.backend.plugins": backend_plugins},
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    extras_require={
        "dev": ["pytest>=7.1.2", "pytest-cov>=3.0.0", "twine>=3.7.1", "wheel>=0.37.1"],
        "doc": docs,
    },
)
