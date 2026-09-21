import logging
import os

import pytest

import spey


def test_version_matches_module():
    # spey.version() should return the package __version__ string
    v = spey.version()
    assert isinstance(v, str)


def test_available_backends_and_get_backend_bibtex():
    # prepare fake metadata and monkeypatch get_backend_metadata + get_bibtex
    out = spey.get_backend_bibtex("default.effective_sigma")
    assert "inspire" in out and "doi.org" in out and "zenodo" in out
    assert any("Barlow, Roger" in s for s in out["inspire"]) or len(out["inspire"]) > 0
    assert (
        any("10.1142/9781860948985_0013" in s or s for s in out["doi.org"])
        or len(out["doi.org"]) > 0
    )


def test_set_log_level_changes_logger_level():
    spey.set_log_level(3)
    # spey.log is the package logger
    assert spey.log.getEffectiveLevel() == logging.DEBUG


def test_set_optimiser_minuit_and_unknown(monkeypatch, caplog):
    # simulate iminuit available
    monkeypatch.setattr(
        spey, "find_spec", lambda name: True if name == "iminuit" else None
    )
    # setenv (not delenv) so that monkeypatch records the key and restores it on
    # teardown; set_optimiser writes to os.environ directly, so without this the
    # "minuit" setting leaks into every test that runs afterwards.
    monkeypatch.setenv("SPEY_OPTIMISER", "scipy")
    spey.set_optimiser("minuit")
    assert os.environ.get("SPEY_OPTIMISER") == "minuit"

    # unknown optimiser logs an error and leaves env unchanged
    monkeypatch.setenv("SPEY_OPTIMISER", "scipy")
    caplog.clear()
    spey.set_optimiser("nope")
    assert os.environ.get("SPEY_OPTIMISER") == "scipy"


def test_register_backend_is_retrievable_via_get_backend():
    """A locally registered backend class must survive the `get_backend` lookup.

    `get_backend` distinguishes a registered *class* from an `importlib` EntryPoint,
    which it has to resolve with `.load()`. Getting that check wrong made every
    `register_backend` model unreachable.
    """
    from spey.base.model_config import ModelConfig

    class LocallyRegistered(spey.BackendBase):
        name = "test.locally_registered"
        version = "1.0.0"
        author = "test"
        spey_requires = spey.__version__

        def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
            return ModelConfig(0, -1.0, [1.0], [(-1.0, poi_upper_bound)])

        def get_logpdf_func(self, expected=spey.ExpectationType.observed, data=None):
            return lambda pars: -0.5 * (pars[0] - 0.5) ** 2

    try:
        spey.register_backend(LocallyRegistered)
        assert "test.locally_registered" in spey.AvailableBackends()

        wrapper = spey.get_backend("test.locally_registered")
        model = wrapper(analysis="locally_registered")
        assert model.backend_type == "test.locally_registered"
        assert model.likelihood(1.0) == pytest.approx(0.125)
    finally:
        spey._backend_entries.pop("test.locally_registered", None)


def test_get_backend_still_resolves_entry_points():
    """The entry-point path (`EntryPoint.load()`) must keep working."""
    wrapper = spey.get_backend("default.poisson")
    model = wrapper(
        signal_yields=[3.0], background_yields=[10.0], data=[11], analysis="ep"
    )
    assert model.backend_type == "default.poisson"
