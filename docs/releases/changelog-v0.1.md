# Release notes v0.1.2

## New features since last release

* Exclusion limit calculator has been extended to include p-value computation
  from chi-square.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

* $\chi^2$ function has been extended to compute background + signal vs background only model.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

## Improvements

* Backend inspection has been added for the models that act like intermediate functions.
  ([#15](https://github.com/SpeysideHEP/spey/pull/15))

* Backend inspection has been converted to inheritance property via ``ConverterBase``.
  ([#17](https://github.com/SpeysideHEP/spey/pull/17))

## Bug Fixes

* In accordance to the latest updates ```UnCorrStatisticsCombiner``` has been updated with
  chi-square computer.
  ([#20](https://github.com/SpeysideHEP/spey/pull/20))

## Contributors

This release contains contributions from (in alphabetical order):

* [Jack Araz](https://github.com/jackaraz)
