from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read()
requirements = [x for x in requirements.split("\n") if x != ""]

with open("src/ma5_expert/_version.py", "r") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setup(
    name="madstats",
    version=version,
    description=("A universal statistics package for LHC reinterpretation"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MadAnalysis/madstats",
    project_urls={
        "Bug Tracker": "https://github.com/MadAnalysis/madstats/issues",
    },
    #download_url=f"https://github.com/MadAnalysis/madstats/archive/refs/tags/v{version}.tar.gz",
    author="Jack Y. Araz",
    author_email=("jack.araz@durham.ac.uk"),
    license="MIT",
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
