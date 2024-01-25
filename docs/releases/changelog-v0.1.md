# Release notes v0.1

## New features since last release

* Exclusion limit calculator has been extended to include p-value computation
  from chi-square.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

* $\chi^2$ function has been extended to compute background + signal vs background only model.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

* Poisson based likelihood constructor without uncertainties has been added
  (Request by Veronica Sanz for EFT studies).
  ([#22](https://github.com/SpeysideHEP/spey/pull/22))

## Improvements

* Backend inspection has been added for the models that act like intermediate functions.
  ([#15](https://github.com/SpeysideHEP/spey/pull/15))

* Backend inspection has been converted to inheritance property via ``ConverterBase``.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

* During POI upper limit computation, `sigma_mu` will be computed from Hessian, if available
  before approximating through $q_{\mu,A}$.
  [#25](https://github.com/SpeysideHEP/spey/pull/25)

## Bug Fixes

* In accordance to the latest updates ```UnCorrStatisticsCombiner``` has been updated with
  chi-square computer. See issue [#19](https://github.com/SpeysideHEP/spey/issues/19).
  ([#20](https://github.com/SpeysideHEP/spey/pull/20))

* Execution error fix during likelihood computation for models with single nuisance parameter.
  ([#22](https://github.com/SpeysideHEP/spey/pull/22))

* Numeric problem rising from `==` which has been updated to `np.isclose`
  see issue [#23](https://github.com/SpeysideHEP/spey/issues/23).
  ([#25](https://github.com/SpeysideHEP/spey/pull/25))

* Typofix during computation of $\sigma_\mu$.
  ([#29](https://github.com/SpeysideHEP/spey/pull/29))

## Contributors

This release contains contributions from (in alphabetical order):

* [Jack Araz](https://github.com/jackaraz)
