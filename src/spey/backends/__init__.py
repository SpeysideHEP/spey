"""
Backends shipped with :mod:`spey`.

The :mod:`spey.backends` subpackage hosts the built-in statistical-model
backends and the differentiable distribution helpers they are built on:

* :mod:`spey.backends.distributions` — autograd-compatible
  :class:`~spey.backends.distributions.Poisson`,
  :class:`~spey.backends.distributions.Normal`,
  :class:`~spey.backends.distributions.MultivariateNormal` distributions plus
  the :class:`~spey.backends.distributions.MainModel` /
  :class:`~spey.backends.distributions.ConstraintModel` wrappers.
* :mod:`spey.backends.default_pdf` — the four registered default backends
  (``default.uncorrelated_background``, ``default.correlated_background``,
  ``default.third_moment_expansion``, ``default.effective_sigma``) plus the
  three simple variants (``default.poisson``, ``default.normal``,
  ``default.multivariate_normal``) and the signal-uncertainty synthesizer
  used by the morphing-aware variants.

Third-party backends register themselves via the ``spey.backend.plugins``
entry point; see :func:`spey.register_backend` and the plugin tutorial.
"""
