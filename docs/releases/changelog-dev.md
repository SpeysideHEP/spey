# Release notes (development version)

## Bug fixes

* `HypothesisTestingBase.chi2_test` silently ignored `init_pars` (and `par_bounds`):
  anything passed through `**kwargs` was dropped, so every fit started from the
  backend's `suggested_init` and, for a likelihood with more than one minimum, the
  limits were computed around whichever well that start fell into. Both keywords are now
  documented and forwarded to the global fit that anchors the threshold, to every profile
  fit evaluated along the scan and to the `sigma_mu` estimate. The bounds of the profiled
  parameter additionally limit the range searched for threshold crossings. They are also
  no longer leaked into the legacy root-finder fallback.
  ([#69](https://github.com/SpeysideHEP/spey/issues/69))

* The optional multi-start search of `HypothesisTestingBase.chi2_test` (`n_multistart`)
  that re-anchors the global minimum used for the χ² threshold now runs a bounded local
  minimisation in each of its `n_multistart - 1` cells instead of merely sampling the
  profile on a grid. A grid point on the slope of a narrow basin never beats the incumbent
  minimum, so the old search could not find a second well unless it was wider than the grid
  spacing, however many samples were requested. The search is now disabled by default
  (`n_multistart=1`, previously `2`, which only evaluated the two ends of the window), so
  the default path performs no additional likelihood evaluations; set e.g.
  `n_multistart=9` for likelihoods with several minima when no good `init_pars` is known.
