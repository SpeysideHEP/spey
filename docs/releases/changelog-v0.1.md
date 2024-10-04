# Release notes v0.1

Specific upgrades for the latest release can be found [here](https://github.com/SpeysideHEP/spey/releases/latest).

## New features since the last release

* The exclusion limit calculator has been extended to include p-value computation from chi-square.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

* $\chi^2$ function has been extended to compute background + signal vs background only model.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

* A Poisson-based likelihood constructor without uncertainties has been added
  (Request by Veronica Sanz for EFT studies).
  ([#22](https://github.com/SpeysideHEP/spey/pull/22))

* Normal and multivariate normal distributions have been added (for integration to Contur, requested by Jon Butterworth and Joe Egan).
  ([#40](https://github.com/SpeysideHEP/spey/pull/40))

## Improvements

* Backend inspection has been added for the models that act like intermediate functions.
  ([#15](https://github.com/SpeysideHEP/spey/pull/15))

* Backend inspection has been converted to inheritance property via ``ConverterBase``.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

* During POI upper limit computation, `sigma_mu` will be computed from Hessian, if available
  before approximating through $q_{\mu,A}$.
  [#25](https://github.com/SpeysideHEP/spey/pull/25)

* Update clarification on text-based keyword arguments.
  ([#30](https://github.com/SpeysideHEP/spey/pull/30))

* Adding logging across the software and implementing tools to silence them.
  ([#32](https://github.com/SpeysideHEP/spey/pull/32))

* Spey will automatically look for updates during initiation.
  ([#32](https://github.com/SpeysideHEP/spey/pull/32))

* Utilities to retrieve BibTeX information for third-party plug-ins.
  ([#32](https://github.com/SpeysideHEP/spey/pull/32))

* Add math utilities for users to extract gradient and hessian of negative log-likelihood
  ([#31](https://github.com/SpeysideHEP/spey/pull/31))

* Improve gradient execution for `default_pdf`.
  ([#31](https://github.com/SpeysideHEP/spey/pull/31))

* Add more tests for code coverage.
  ([#33](https://github.com/SpeysideHEP/spey/pull/33))

## Bug Fixes

* In accordance with the latest updates, `UnCorrStatisticsCombiner` has been updated with
  a chi-square computer. See issue [#19](https://github.com/SpeysideHEP/spey/issues/19).
  ([#20](https://github.com/SpeysideHEP/spey/pull/20))

* Execution error fix during likelihood computation for models with single nuisance parameter.
  ([#22](https://github.com/SpeysideHEP/spey/pull/22))

* The numeric problem rising from `==` which has been updated to `np.isclose`
  see issue [#23](https://github.com/SpeysideHEP/spey/issues/23).
  ([#25](https://github.com/SpeysideHEP/spey/pull/25))

* Typofix during computation of $\sigma_\mu$.
  ([#29](https://github.com/SpeysideHEP/spey/pull/29))

* Bug fix in signal uncertainty synthesizer
  ([#34](https://github.com/SpeysideHEP/spey/pull/34))

* Signal uncertainties were causing a narrower $\chi^2$ distribution due to the weight of the constraint term.
  ([#38](https://github.com/SpeysideHEP/spey/pull/38))

* Scipy and Autograd version limitations have been relieved. But there is a clash with numpy v2.0
  so numpy version has been limited to less than 2.0.
  ([#40](https://github.com/SpeysideHEP/spey/pull/40))

* Compatibility issues have been fixed in Python 3.12
  ([#39](https://github.com/SpeysideHEP/spey/pull/39))

* p-value computation in $\chi^2$ and toy calculators have been fixed.
  ([#42](https://github.com/SpeysideHEP/spey/pull/42))

## Contributors

This release contains contributions from (in alphabetical order):

* [Jack Araz](https://github.com/jackaraz)
