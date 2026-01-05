# Release notes v0.2

Specific upgrades for the latest release can be found [here](https://github.com/SpeysideHEP/spey/releases/latest).

## New features since the last release

* Ability to compute two-sided tests added.
  ([#45](https://github.com/SpeysideHEP/spey/pull/45))

* Add plug-in registry that does not require creating an entire package.
  ([#60](https://github.com/SpeysideHEP/spey/pull/60))

* Iminuit is available to be used for optimisation as alternative to scipy.
  ([#59](https://github.com/SpeysideHEP/spey/pull/59))

## Improvements

* Autograd has been upgraded to v1.7 which supports numpy 2.0
  ([#50](https://github.com/SpeysideHEP/spey/pull/50))

* Enabled user control to allow negative signal yields in $\chi^2$-test.
  ([#50](https://github.com/SpeysideHEP/spey/pull/50))

* Implement python version compatibility check for new releases.
  ([#52](https://github.com/SpeysideHEP/spey/pull/52))

* Added a helper function to merge correlated bins.
  ([#53](https://github.com/SpeysideHEP/spey/pull/53))

* Some errors are reduced to warnings in chi2 test to reduce the verbosity.
  ([#54](https://github.com/SpeysideHEP/spey/pull/54))

* Enable compatibility for Python 3.13, and the deprecation of `pkg_resources` has been addressed.
  ([#56](https://github.com/SpeysideHEP/spey/pull/56))

* Multivariate Normal distribution now supports callable covariance matrix function.
  ([#59](https://github.com/SpeysideHEP/spey/pull/59))

## Bug Fixes

* Control mechanism added in case of infinite determinant in covariance matrix.
  ([#47](https://github.com/SpeysideHEP/spey/pull/47))

* Error during signal uncertainty insertion due to valid domain has been fixed.
  ([#51](https://github.com/SpeysideHEP/spey/pull/51))

* A python version depended bug fixed in `about` function. This bug only effected python v3.9 and below.
  ([#61](https://github.com/SpeysideHEP/spey/pull/61))

## Contributors

This release contains contributions from (in alphabetical order):

* [Jack Araz](https://github.com/jackaraz)
* [Jon Butterworth](https://github.com/jonbutterworth)
* [Joe Egan](https://github.com/joes-git)
