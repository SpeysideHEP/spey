# Release notes v0.2

Specific upgrades for the latest release can be found [here](https://github.com/SpeysideHEP/spey/releases/latest).

## New features since the last release

* Multi-parameter-of-interest support: `poi_test` now accepts a
  `Dict[Union[int, str], float]` in addition to a plain `float`.
  When a dictionary is supplied, every entry fixes the corresponding
  parameter (identified by index or by name as recorded in
  `ModelConfig.parameter_names`) during the fit, while the remaining
  parameters are optimised freely.  A plain `float` retains exactly the
  previous single-POI behaviour.  The new type alias `PoiTest` is
  exported from `spey.interface.statistical_model` for type-annotation
  convenience.  Affected interfaces: `StatisticalModel.likelihood`,
  `StatisticalModel.asimov_likelihood`,
  `StatisticalModel.fixed_poi_sampler`,
  `StatisticalModel.sigma_mu_from_hessian`,
  `HypothesisTestingBase.chi2`, `HypothesisTestingBase.sigma_mu`,
  `HypothesisTestingBase.exclusion_confidence_level`, and the
  corresponding methods in `UnCorrStatisticsCombiner`.

* `ModelConfig` gains two new helpers: `resolve_poi_indices` (converts a
  `poi_test` value to a `{param_index: value}` mapping) and
  `fixed_poi_bounds_multi` (generalises `fixed_poi_bounds` to multiple
  simultaneously fixed parameters).

* `optimizer.fit` extended to accept `fixed_poi_value` as a
  `Dict[int, float]`, fixing an arbitrary set of parameters in a single
  optimisation call.

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

* `MultivariateNormal` backend now accepts `signal_yields` as a callable with signature
  `(extra_pars: np.ndarray) -> np.ndarray` in addition to a plain array.  A companion
  keyword argument `n_signal_parameters` (default `0`) declares how many free parameters
  beyond :math:`\mu` the callable requires; they are appended to the optimiser parameter
  vector as `signal_par_0`, `signal_par_1`, … and are included in `ModelConfig` with
  `(None, None)` bounds.  The plain-array path is unchanged.

* `maximize_likelihood` and `maximize_asimov_likelihood` gain a new optional keyword
  argument `poi_indices` (`Optional[List[Union[int, str]]]`, default `None`).  When
  `None` the methods return the usual `(muhat, nll)` tuple.  When a list of parameter
  indices or names is provided they return `(Dict[Union[int, str], float], nll)` with
  the fitted values for each requested parameter.  Supported on `StatisticalModel` and
  `UnCorrStatisticsCombiner`.

* All default-PDF backends (`DefaultPDFBase`, `UncorrelatedBackground`,
  `CorrelatedBackground`, `ThirdMomentExpansion`, `EffectiveSigma`) now accept a
  `signal_parameter_bounds` keyword argument
  (`Optional[List[Tuple[Optional[float], Optional[float]]]]`, default `None`).  When
  provided, each entry supplies the `(lower, upper)` optimiser bound for the
  corresponding extra signal parameter (indexed by position in the parameter vector
  immediately after :math:`\mu`); a `None` element leaves that side unbounded.  When
  the argument is `None` (the previous behaviour), every extra signal parameter
  continues to receive `(None, None)` bounds.  The list must have exactly
  `n_signal_parameters` entries.  These bounds are stored in `ModelConfig.suggested_bounds`
  and propagated to the optimiser automatically.

* For `pdf_type="multivariategauss"` with a non-callable cov, the old code was:
  `self._pdf = lambda pars: MultivariateNormal(mean=loc(pars), cov=cov)` which re-instantiated
  MultivariateNormal (and re-ran `np.linalg.inv` + `np.linalg.slogdet`, both O(n³)) on every
  likelihood call. The fix creates a single MultivariateNormal once at construction time and
  only updates `_mv.mean = loc(pars)` on each call — the cached `_inv_cov` /`_logdet_cov` closures
  are reused unchanged. When cov is callable, the original per-call re-instantiation is preserved.

* Minor inefficiencies fixed `Normal.log_prob: value[self.domain]` was sliced twice; now sliced once into x.
  `MultivariateNormal.log_prob: value[self.domain]` was sliced four times; now sliced once into x.
  Removed unused from copy import deepcopy import.

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
