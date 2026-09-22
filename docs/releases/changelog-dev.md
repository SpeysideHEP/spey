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
  ([#67](https://github.com/SpeysideHEP/spey/pull/67))

* `find_poi_upper_limit` now validates the root it finds. A bracketing solver converges
  on any sign change, including one that is not an upper limit, so the returned value is
  rejected (returning `inf` with a warning) when the bracket is oriented so that
  increasing `mu` *leaves* the exclusion, or when the exclusion does not persist above
  the root — the signature of a non-monotonic `CL_s` curve, which a bimodal likelihood
  can produce. Pass `validate_limit=False` to recover the previous behaviour.
  ([#TBD](https://github.com/SpeysideHEP/spey/pull/))

## Bug fixes

* The Asimov dataset built by the `default_pdf` backends kept the auxiliary measurements
  at the centre of the constraint instead of moving them with the nuisance parameters,
  so it did not satisfy the defining property of an Asimov dataset — that the maximum
  likelihood estimators of the parameters equal the values it was generated at, eq. (25)
  of arXiv:1007.1727.

  `ConstraintModel.log_prob` took no data argument, and `DefaultPDFBase.get_logpdf_func`
  sliced the data vector as `data[: len(self.data)]`, so the auxiliary half of the vector
  that `expected_data` had just produced was discarded on the way back in. The main
  counts of the Asimov set were built at the profiled `theta_hat_hat(mu=0)` while the
  constraint kept pulling the fit back towards `theta = 0`. Refitting the Asimov data of
  the two-bin `default.uncorrelated_background` example returned `theta = (0.007, -0.220)`
  where it was generated at `(0.050, -0.738)`.

  `Normal.log_prob`, `MultivariateNormal.log_prob` and `ConstraintModel.log_prob` now
  accept the auxiliary measurements; `ConstraintModel.expected_data(pars)` returns the
  Asimov auxiliary data for a parameter point; `ConstraintModel.sample` draws
  pseudo-experiments from `p(a | theta)` around the hypothesis rather than around the
  nominal centre, which also fixes the auxiliary half of the toy calculator; and
  `DefaultPDFBase._split_data` splits a data vector into its main and auxiliary parts.
  A vector carrying only the bin counts is still accepted and falls back to the nominal
  auxiliary data, so existing calls such as `model.likelihood(data=[...])` keep working;
  a vector whose auxiliary part has the wrong length now raises `InvalidInput` instead of
  being silently truncated.

  This changes results. `sigma_mu` is unaffected (it is computed from the observed
  information matrix), as are all quantities in `apriori` mode, where the conditional
  MLE is `theta = 0` and the old convention happened to be correct. Post-fit quantities
  move, by an amount that grows with how hard the data pull the nuisance parameters —
  for the two-bin test models with 24-33% background uncertainties:

  | model | quantity | before | after |
  |---|---|---|---|
  | `default.uncorrelated_background` | `1 - CLs` | 0.97018 | 0.94662 |
  | `default.uncorrelated_background` | `poi_upper_limit` | 0.85633 | 1.01810 |
  | `default.correlated_background` | `1 - CLs` | 0.96351 | 0.93529 |
  | `default.correlated_background` | `poi_upper_limit` | 0.90710 | 1.07561 |
  | `default.third_moment_expansion` | `1 - CLs` | 0.96143 | 0.93181 |
  | `default.effective_sigma` | `1 - CLs` | 0.85670 | 0.77355 |

  The new values were verified against `pyhf`, which implements the same convention: for
  an equivalent `histosys` workspace the two log-likelihoods agree to `1e-14` at
  arbitrary parameter points, the Asimov datasets are identical entry by entry, and
  `1 - CLs` agrees to `7e-8` (0.94661532 against 0.94661539). Note that this comparison
  must not be made against `pyhf.uncorrelated_background`, which is a `shapesys` model
  with an asymmetric Poisson constraint on a multiplicative nuisance parameter and is a
  genuinely different likelihood — see
  `spey-pyhf/docs/tutorials/uncorrelated_background_comparison.ipynb`.
  ([#67](https://github.com/SpeysideHEP/spey/pull/67))

* `covariance_to_correlation` did not enforce symmetry, so an asymmetric covariance
  matrix silently produced an asymmetric "correlation" matrix and, through it, an
  ill-defined multivariate normal: the quadratic form of a Gaussian only sees the
  symmetric part of a matrix, but its inverse and its determinant do not, so the
  resulting likelihood was neither the one implied by `Sigma` nor the one implied by
  `Sigma^T`.

  A new helper, `spey.helper_functions.symmetrise_matrix`, replaces such an input with
  `(Sigma + Sigma.T) / 2` — the closest symmetric matrix in the Frobenius norm, and
  therefore the least-assumption reading of an input whose two off-diagonal entries
  disagree — and warns that it did so. It is applied to `covariance_to_correlation`,
  `correlation_to_covariance`, the `covariance_matrix` of `DefaultPDFBase` (hence
  `default.correlated_background` and `default.third_moment_expansion`) and of
  `default.multivariate_normal`, and the `correlation_matrix` of
  `default.effective_sigma`. A non-square matrix now raises `InvalidInput`.

  On top of symmetry, the matrix is now also required to be **positive definite** by a
  second helper, `spey.helper_functions.ensure_positive_definite`, which is what the
  model ingestion points call. Positive *semi*-definiteness is not enough: a singular
  matrix has `det(Sigma) = 0` and no inverse, so the multivariate normal has no density,
  and a negative eigenvalue leaves the exponent unbounded above, so the "likelihood" can
  be made arbitrarily large. The test is whether a Cholesky decomposition exists — the
  condition under which the inverse and determinant the likelihood needs can be computed
  — and eigenvalues are only computed when it fails, to distinguish a singular matrix
  from an indefinite one and to name the offending eigenvalue. An ill-conditioned but
  positive definite matrix is still accepted; the condition number is not policed.

  `default.third_moment_expansion` validates the correlation matrix it *derives* from
  the third moments, not just the covariance matrix it is given: the derived matrix is
  assembled element by element from a discriminant, so a valid input does not guarantee
  a valid constraint. A `callable` covariance matrix is exempt from both checks, since
  validating it would break the `autograd` trace.

  Several docstring examples and most of the test suite used the asymmetric matrix
  `[[144, 13], [25, 256]]`, which is how the pattern spread; they now use its symmetric
  part `[[144, 19], [19, 256]]`. Results obtained with an asymmetric matrix change to
  the values its symmetric part gives — for `default.correlated_background` with the
  matrix above, `1 - CLs` moves from 0.93529 to 0.93542 and `poi_upper_limit` from
  1.07561 to 1.07500, both now agreeing with `pyhf` to `7e-8`.
  ([#67](https://github.com/SpeysideHEP/spey/pull/67))

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
