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
