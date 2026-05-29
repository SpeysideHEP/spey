"""
System-level utilities used across :mod:`spey`.

The :mod:`spey.system` subpackage groups infrastructure that is not part of
the statistical-model API itself but is needed to operate it:

* :mod:`spey.system.logger` — coloured logger setup and a
  :func:`~spey.system.logger.capture_logs` context manager that deduplicates
  noisy messages emitted from hot loops.
* :mod:`spey.system.exceptions` — package-specific exception classes (e.g.
  :class:`~spey.system.exceptions.PluginError`,
  :class:`~spey.system.exceptions.NegativeExpectedYields`).
* :mod:`spey.system.cache` — :func:`~spey.system.cache.cache_results`, a
  thread-safe, mutable-argument-aware decorator with per-instance support.
* :mod:`spey.system.webutils` — small HTTP helpers for retrieving BibTeX
  entries and checking for new ``spey`` releases on PyPI.
"""
