import logging
import os

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
    monkeypatch.delenv("SPEY_OPTIMISER", raising=False)
    spey.set_optimiser("minuit")
    assert os.environ.get("SPEY_OPTIMISER") == "minuit"

    # unknown optimiser logs an error and leaves env unchanged
    monkeypatch.setenv("SPEY_OPTIMISER", "scipy")
    caplog.clear()
    spey.set_optimiser("nope")
    assert os.environ.get("SPEY_OPTIMISER") == "scipy"
