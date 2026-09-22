# Release notes (development version)

## New features since the last release

* Added `CorrelatedStatisticsCombiner` (`default.correlated_combiner`), a plug-in that
  combines any number of `StatisticalModel` instances which **share** parameters —
  parameters of interest, EFT/Wilson coefficients, or ordinary nuisance parameters.
  Shared parameters are declared by name or by index and are profiled jointly, so the
  combination is correlated rather than a simple independent product. Unlike
  `UnCorrStatisticsCombiner`, the class is a `BackendBase` plug-in, hence the result is
  an ordinary `StatisticalModel` that works with the full hypothesis-testing toolchain
  and with `spey.multiparameter.find_contour`.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

  ```python
  import spey

  combiner = spey.get_backend("default.correlated_combiner")
  combined = combiner(
      statistical_models=[model_a, model_b, model_c],
      shared_parameters=[
          "mu",
          {"name": "c1", "members": {"SR_A": "signal_par_0", "SR_B": 1}},
      ],
      analysis="combination",
  )
  combined.exclusion_confidence_level()
  ```

  The combined log-likelihood, gradient and Hessian are assembled from the constituent
  backends through the gather map `theta_m = p[I_m]`, whose Jacobian is a selection
  matrix; gradients are therefore accumulated with a single `numpy.bincount` and
  Hessians by block scatter-add, without ever forming a Jacobian product. Constituent
  backends may compute their derivatives with any framework (`autograd`, `jax`,
  `tensorflow`, `pytorch`, or closed form): the combiner consumes only the derivatives
  a backend advertises and never traces through it. The combined log-pdf is exposed as
  an `autograd` primitive with hand-wired vector-Jacobian products, so external
  differentiation — e.g. by `find_contour` — keeps working for any backend. If a
  constituent model cannot provide a gradient or Hessian, the combination reports the
  corresponding method as unavailable and spey falls back to its numerical optimiser.

## Improvements

* Added a tutorial on correlated combinations,
  `docs/tutorials/correlated_combination.ipynb`.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

* `spey.optimizer.core.fit` now falls back to `scipy` when `iminuit` is installed but
  cannot be imported (e.g. built against a different `numpy`), instead of letting the
  `ImportError` propagate out of the fit.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

* `compute_teststatistics` no longer tests the Asimov test statistic for *exact*
  equality with zero. Eq. (66) of arXiv:1007.1727 divides by `2 * sqrt(q_mu,A)`, and
  `sqrt_qmuA == 0` never fired for a merely tiny value, so the division went ahead with a
  near-zero divisor and returned a p-value dominated by numerical noise — in practice a
  spurious, near-certain exclusion. The comparison is now against
  `ASIMOV_TESTSTAT_TOLERANCE` (`1e-3`, overridable per call via
  `asimov_teststat_tolerance`), and `AsimovTestStatZero` is raised before the division.
  Every caller already interprets that exception as "no exclusion".

  The threshold is far from any physical value: at a genuine 95% CL upper limit
  `sqrt(q_mu,A)` is 0.8-2.1 for the built-in backends, while the degenerate regime sits
  at 1e-5 to 1e-4. Sweeping all seven built-in backends, the only quantity that changes
  is `CLs` at `mu = 1e-6`, which moves from ~3e-6 to exactly 0 — both meaning "not
  excluded". All upper limits, expected bands and significances are unchanged.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

* `find_poi_upper_limit` now validates the root it finds. A bracketing solver converges
  on any sign change, including one that is not an upper limit, so the returned value is
  rejected (returning `inf` with a warning) when the bracket is oriented so that
  increasing `mu` *leaves* the exclusion, or when the exclusion does not persist above
  the root — the signature of a non-monotonic `CL_s` curve, which a bimodal likelihood
  can produce. Pass `validate_limit=False` to recover the previous behaviour.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

## Bug fixes

* `HypothesisTestingBase.exclusion_confidence_level` and `sigma_mu` crashed with
  `TypeError: '>' not supported between instances of 'float' and 'dict'` whenever
  `poi_test` was a `dict` — the value meant to identify and fix a multi-POI point (e.g.
  EFT coefficients shared with other analyses) was passed straight into the `qmu`/
  `qmu_tilde`/`q0` scalar comparisons instead of being resolved first. A new
  `HypothesisTestingBase._split_poi_test` helper now splits a dict-valued `poi_test`
  into the scalar value of the primary POI (used for the test-statistic comparisons)
  and any other parameters to keep fixed, and threads the latter through the
  unconstrained fits and every constrained likelihood evaluation — including the `toy`
  and `chi_square` calculators — so the extra parameters stay fixed throughout the test,
  not just at the tested point.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

* `StatisticalModel.generate_asimov_data` silently discarded any `fixed_poi_value`
  keyword argument (with a warning), so a multi-POI `poi_test` lost its extra fixed
  parameters specifically during Asimov data generation, even though the same fit
  respected them everywhere else. `fixed_poi_value` may now be a `dict`: the primary POI
  still keeps its canonical Asimov value (`1.0` for `test_statistic="q0"`, `0.0`
  otherwise) unless the dict explicitly overrides it, while any other entries are
  applied to the fit as before. A plain `float` (which can only refer to the primary
  POI) is still ignored with a warning.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

* `spey.get_backend` could not resolve a backend registered with
  `spey.register_backend`. The registry stores a *class* for a locally registered
  backend and an `importlib` `EntryPoint` for a discovered plug-in, but the lookup
  tested `isinstance(backend, BackendBase)`, which is `False` for a class, and then
  called `backend.load()` on it. Every locally registered backend was therefore
  unreachable through `get_backend`.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

* `StatisticalModel.is_asymptotic_calculator_available` and
  `is_toy_calculator_available` always returned `True`. They compared a *bound method*
  (`self.backend.expected_data`) against a plain function
  (`BackendBase.expected_data`), which can never be equal, so every backend advertised
  every calculator regardless of what it implemented.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

* `StatisticalModel.sigma_mu_from_hessian` returned a bare `nan` accompanied by a raw
  `invalid value encountered in sqrt` runtime warning whenever the observed information
  matrix was not positive definite at the requested point. It now logs an explanatory
  warning and returns `nan` explicitly.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

## Contributors

This release contains contributions from (in alphabetical order):

[Jack Y. Araz](https://github.com/jackaraz)
