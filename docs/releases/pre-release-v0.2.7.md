# Pre-release notes: v0.2.7-b2

> **Status:** beta 2 (`v0.2.7-b2`)
> **Branch merged:** `SpeysideHEP/multiparam-fit` via [PR #63](https://github.com/SpeysideHEP/spey/pull/63)

---

## Highlights

This release is centred on two substantial extensions: support for fits with multiple
parameters of interest (multi-POI), and a richer treatment of signal uncertainties
through callable signal-yield functions. A new confidence-contour mapping algorithm,
an improved `chi2_test` profiler, a thread-safe result cache, and a migration from
`setup.py` to `pyproject.toml` round out the changes.

---

## Multi-parameter-of-interest fit

### `poi_test` now accepts a dictionary

Every method that previously required a single `float` for `poi_test` now also accepts
`Dict[Union[int, str], float]`. Each entry in the dictionary fixes the corresponding
parameter (identified by its integer index or by the name stored in
`ModelConfig.parameter_names`) at the given value during the fit; the remaining
parameters are optimised freely. A plain `float` retains exactly the previous
single-POI behaviour.

The new type alias `PoiTest` is exported from `spey.interface.statistical_model` for
type-annotation convenience. Affected interfaces include
`StatisticalModel.likelihood`, `StatisticalModel.asimov_likelihood`,
`StatisticalModel.fixed_poi_sampler`, `StatisticalModel.sigma_mu_from_hessian`,
`HypothesisTestingBase.chi2`, `HypothesisTestingBase.sigma_mu`,
`HypothesisTestingBase.exclusion_confidence_level`, and the corresponding methods on
`UnCorrStatisticsCombiner`.

### `ModelConfig` helpers

Two new methods have been added to `ModelConfig`:

- `resolve_poi_indices` converts any `PoiTest` value into a normalised
  `{param_index: value}` mapping.
- `fixed_poi_bounds_multi` generalises the existing `fixed_poi_bounds` to
  an arbitrary set of simultaneously fixed parameters.

### `optimizer.fit` extended

The `fixed_poi_value` keyword argument of `optimizer.fit` now accepts
`Dict[int, float]`, allowing any combination of parameters to be held fixed
in a single optimisation call.

### Retrieving multi-POI fit results

`maximize_likelihood` and `maximize_asimov_likelihood` gain an optional
`poi_indices` keyword argument (`Optional[List[Union[int, str]]]`, default `None`).
When a list of indices or names is supplied, both methods return
`(Dict[Union[int, str], float], nll)` with the fitted value for each requested
parameter, rather than the usual `(muhat, nll)` scalar form.

```python
muhat_dict, nll = model.maximize_likelihood(poi_indices=[0, 1])
```

Both `StatisticalModel` and `UnCorrStatisticsCombiner` support this interface.

---

## Confidence-contour mapping: `spey.multiparameter.find_contour`

A new sub-package, `spey.multiparameter`, provides `find_contour`, which maps the
$(1-\alpha)$ chi-squared confidence region boundary for a
`StatisticalModel` backed by `"default.multivariate_normal"` in its full
parameter space.

The algorithm operates in four stages:

1. **Pre-whitening.** The Hessian of the negative log-likelihood at the MLE is
   Cholesky-factored to produce a whitened coordinate system in which the
   contour is approximately a sphere of radius $\sqrt{\Delta_\alpha}$. This
   removes the strong anisotropy that is typical of correlated parameter spaces
   and gives approximately uniform coverage when directions are sampled at random.

2. **Radial search.** $N$ random rays are fired from the MLE in whitened space.
   Each ray's crossing of the NLL threshold $T = \mathrm{NLL}(\hat\theta) + \Delta_\alpha/2$
   is located by Brent's method.

3. **Gap detection.** A large pool of candidate directions is scored by distance to
   the radial set; the least-covered directions seed the subsequent HMC chains.

4. **Constrained RATTLE HMC.** Each seed direction initialises a Hamiltonian Monte
   Carlo chain that walks along the constraint manifold
   $\mathrm{NLL}(\theta) = T$, projecting both position and momentum at every step
   to fill in gaps left by the radial pass.

The function returns a `ContourResult` dataclass containing the MLE, the NLL minimum,
the threshold, the chi-squared quantile, the array of contour points, a Boolean mask
distinguishing radial from HMC-derived points, optional parameter names, the
confidence level, and the degrees of freedom.

A multi-start scan (`n_multistart`) over the parameter space is performed before the
root search to reduce the risk of the NLL minimum anchor being trapped in a local
minimum.

---

## Signal uncertainties and callable signal yields

### Callable `signal_yields` in default PDF backends

All default-PDF backends (`UncorrelatedBackground`, `CorrelatedBackground`,
`ThirdMomentExpansion`, `EffectiveSigma`, and `MultivariateNormal`) now accept
`signal_yields` as a callable with signature
`(extra_pars: np.ndarray) -> np.ndarray` in addition to a plain array. The
companion keyword argument `n_signal_parameters` (default `0`) declares how many
free parameters beyond $\mu$ the callable requires; they are appended to the
optimiser parameter vector as `signal_par_0`, `signal_par_1`, and so on, and
are included in `ModelConfig` with `(None, None)` bounds. The plain-array path is
unchanged.

### `signal_parameter_bounds`

All default-PDF backends gain a `signal_parameter_bounds` keyword argument
(`Optional[List[Tuple[Optional[float], Optional[float]]]]`, default `None`).
When provided, each entry supplies the `(lower, upper)` optimiser bound for the
corresponding extra signal parameter; a `None` element leaves that side unbounded.
The list must have exactly `n_signal_parameters` entries. These bounds are stored
in `ModelConfig.suggested_bounds` and propagated to the optimiser automatically.

### `signal_uncertainty_synthesizer` updated

The `signal_uncertainty_synthesizer` helper now accepts `n_signal_parameters` so
that the domain index offsets for nuisance parameters are shifted correctly when
the backend uses a callable `signal_yields`. This keeps the parameter-vector
layout consistent across all default-PDF backends.

### Improved handling of zero yields and absolute uncertainty regions

Bins with zero background yields are now handled without raising an exception.
Absolute uncertainty regions are validated on construction, with informative
warnings rather than silent failures when `delta_up` or `delta_dn` would resolve
to a physically dubious value.

---

## `chi2_test` improvements

The `chi2_test` method has been updated to handle non-convex profile likelihoods
with disjoint confidence regions correctly. The profile is now enumerated by a
coarse grid scan followed by bracketed root refinement using `toms748`, so every
crossing of the threshold is found and returned in ascending order. Two new
keyword arguments control the precision-vs-cost trade-off:

- `n_scan` (default `3`): number of uniformly-spaced grid points for the coarse
  scan. Values below 3 are clamped to 3.
- `n_multistart` (default `2`): number of evenly-spaced evaluations used by the
  internal scan that re-anchors the NLL minimum. Values below 2 are clamped to 2.

Additionally, `chi2_test` can now profile any nuisance parameter by setting the
`parameter` keyword to the index or name of the parameter of interest, whilst
fixing the primary POI to a specified `poi_value`. Setting `poi_value=None` allows
the primary POI to float freely during the scan.

---

## Backend NLL dispatch

`StatisticalModel.likelihood`, `StatisticalModel.asimov_likelihood`, and
`StatisticalModel.maximize_likelihood` now attempt to call native
`negative_loglikelihood`, `asimov_negative_loglikelihood`, and
`minimize_negative_loglikelihood` methods on the backend before falling back to the
general numerical-optimisation path. Backends that implement these stubs can bypass
the generic fitting loop entirely and supply their own analytically or numerically
superior NLL evaluations. Backends that do not implement a stub raise
`NotImplementedError`, which is caught silently at debug level.

---

## Thread-safe result cache: `spey.system.cache`

A new `cache_results` decorator is provided in `spey.system.cache`. Unlike
`functools.lru_cache`, it handles non-hashable arguments (NumPy arrays, lists,
dicts) by constructing a deterministic cache key from the SHA-256 digest of each
array buffer and a recursive tokenisation of other containers. A threading lock
with per-key events serialises concurrent reads and writes, ensuring that each
unique input is computed at most once even under contention. When applied to
instance methods, `_PerInstanceCacheDescriptor` provides per-instance storage so
that two distinct model objects never share a cache entry.

`generate_asimov_data` on `StatisticalModel` is now decorated with
`cache_results(maxsize=128, copy_on_return=True, per_instance=True)`.

---

## Performance improvements

- For `pdf_type="multivariategauss"` with a constant covariance matrix, a single
  `MultivariateNormal` instance is now created at construction time and only its
  `mean` attribute is updated on each likelihood call. The previous implementation
  re-instantiated the distribution on every call, incurring two $O(n^3)$ operations
  (`np.linalg.inv` and `np.linalg.slogdet`) per evaluation.
- Redundant array slices in `Normal.log_prob` and `MultivariateNormal.log_prob`
  have been reduced from four to one per call.

---

## Build system

`setup.py` has been removed. The project now uses `pyproject.toml` exclusively,
including `sdist` generation in the CI build workflow.

---

## Bug fixes

- `fixed_poi_value` is now correctly resolved through `mle` when passed as a
  keyword argument.
- `chi2_test` no longer miscategorises the sign of the `right` limit when
  `allow_negative_signal` is inferred automatically.
- Config caching that could return stale `ModelConfig` objects after parameter
  changes has been removed.
