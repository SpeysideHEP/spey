from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("src/spey/_version.py", encoding="UTF-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "numpy>=1.21.6, <3.0.0",
    "scipy>=1.10.0",
    "autograd>=1.7.0",
    "semantic_version~=2.10",
    "tqdm>=4.64.0",
    "requests>=2.31.0",
    "setuptools",
]

backend_plugins = [
    "default.uncorrelated_background = spey.backends.default_pdf:UncorrelatedBackground",
    "default.correlated_background = spey.backends.default_pdf:CorrelatedBackground",
    "default.third_moment_expansion = spey.backends.default_pdf:ThirdMomentExpansion",
    "default.effective_sigma = spey.backends.default_pdf:EffectiveSigma",
    "default.poisson = spey.backends.default_pdf.simple_pdf:Poisson",
    "default.normal = spey.backends.default_pdf.simple_pdf:Gaussian",
    "default.multivariate_normal = spey.backends.default_pdf.simple_pdf:MultivariateNormal",
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
        "Documentation": "https://spey.readthedocs.io",
        "Repository": "https://github.com/SpeysideHEP/spey",
        "Homepage": "https://github.com/SpeysideHEP/spey",
        "Download": f"https://github.com/SpeysideHEP/spey/archive/refs/tags/v{version}.tar.gz",
    },
    docs_url="https://spey.readthedocs.io",
    download_url=f"https://github.com/SpeysideHEP/spey/archive/refs/tags/v{version}.tar.gz",
    author="Jack Y. Araz",
    author_email=("j.araz@ucl.ac.uk"),
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={"spey.backend.plugins": backend_plugins},
    install_requires=requirements,
    python_requires=">3.8, <3.14",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    extras_require={
        "dev": ["pytest>=7.1.2", "pytest-cov>=3.0.0", "twine>=3.7.1", "wheel>=0.37.1"],
        "iminuit": ["iminuit>=2.22.0"],
    },
)
