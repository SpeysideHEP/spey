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

* `find_poi_upper_limit` now validates the root it finds. A bracketing solver converges
  on any sign change, including one that is not an upper limit, so the returned value is
  now rejected (returning `inf` with a warning) when the exclusion does not persist above
  it, when the bracket is oriented so that increasing `mu` *leaves* the exclusion, or
  when the Asimov test statistic at the root is numerically zero. The last case is the
  common one: eq. (66) of arXiv:1007.1727 divides by `sqrt(q_mu,A)`, so a model whose
  Asimov dataset barely constrains `mu` — for instance a signal strength nearly
  degenerate with another free parameter — produced a `CL_s` curve driven by numerical
  noise, and a finite but meaningless limit. Pass `validate_limit=False` to recover the
  previous behaviour, and `asimov_teststat_tolerance` to tune the threshold.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

## Bug fixes

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
