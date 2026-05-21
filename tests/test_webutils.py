import types

import pytest

from spey.system import webutils


def make_response(status=200, text="", encoding="utf-8", json_obj=None):
    resp = types.SimpleNamespace()
    resp.status_code = status
    resp.text = text
    resp.encoding = encoding

    def _json():
        return json_obj

    resp.json = _json if json_obj is not None else lambda: {}
    return resp


def test_get_bibtex_success(monkeypatch):
    fake_text = "@article{key, title={Test}}"

    def fake_get(url, headers=None, timeout=None):
        return make_response(status=200, text=fake_text)

    monkeypatch.setattr(webutils.requests, "get", fake_get)
    out = webutils.get_bibtex("doi", "10.0/xyz")
    assert out == fake_text


def test_get_bibtex_non_2xx_returns_empty(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return make_response(status=404, text="Not found")

    monkeypatch.setattr(webutils.requests, "get", fake_get)
    out = webutils.get_bibtex("zenodo", "12345")
    assert out == ""


def test_get_bibtex_raises_connection_error(monkeypatch):
    # make requests.get raise a requests.ConnectionError -> should raise webutils.ConnectionError
    def fake_get(url, headers=None, timeout=None):
        raise webutils.requests.ConnectionError("no net")

    monkeypatch.setattr(webutils.requests, "get", fake_get)
    with pytest.raises(webutils.ConnectionError):
        webutils.get_bibtex("doi", "10.0/xyz")


def test_check_updates_handles_exceptions(monkeypatch):
    # Make requests.get raise a generic exception; function should not raise
    def fake_get(url, timeout=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(webutils.requests, "get", fake_get)
    # Should not raise
    webutils.check_updates()


def test_pypi_cache_round_trip(monkeypatch, tmp_path):
    """
    _save_pypi_info writes a JSON file under _SPEY_DIR and _load_cached_pypi_info
    reads it back when the file is younger than 24 hours.
    """
    monkeypatch.setattr(webutils, "_SPEY_DIR", tmp_path)
    payload = {"info": {"version": "9.9.9", "requires_python": ">=3.8"}}

    webutils._save_pypi_info(payload)
    files = list(tmp_path.glob("pypi_latest_*.json"))
    assert len(files) == 1, f"Expected one cache file, got {len(files)}"

    cached = webutils._load_cached_pypi_info()
    assert cached == payload, "Cache round-trip lost or mutated payload."


def test_pypi_cache_miss_when_dir_absent(monkeypatch, tmp_path):
    """_load_cached_pypi_info returns None when the cache directory is missing."""
    missing = tmp_path / "absent"
    monkeypatch.setattr(webutils, "_SPEY_DIR", missing)
    assert webutils._load_cached_pypi_info() is None


def test_get_pypi_info_uses_cache_without_network(monkeypatch, tmp_path):
    """
    _get_pypi_info returns the cached payload and does NOT touch the network
    when a fresh cache file exists.
    """
    monkeypatch.setattr(webutils, "_SPEY_DIR", tmp_path)
    payload = {"info": {"version": "1.2.3"}}
    webutils._save_pypi_info(payload)

    def fail_get(url, timeout=None, stream=None):
        raise AssertionError("network should not be hit when cache is fresh")

    monkeypatch.setattr(webutils.requests, "get", fail_get)
    assert webutils._get_pypi_info() == payload
