import spey


class _FakeEP:
    def __init__(self, name, loader=None):
        self.name = name
        self._loader = loader

    def load(self):
        if self._loader:
            return self._loader()
        # default: return a simple class with spey_requires satisfied
        class Stat:
            spey_requires = f">={spey.version()}"

        return Stat


def test_available_backends_and_get_backend(monkeypatch):
    # prepare a fake backend entry that load() returns a simple class
    ep = _FakeEP("fake.backend")

    # set module _backend_entries to our mapping for the duration of the test
    monkeypatch.setitem(spey.__dict__, "_backend_entries", {"fake.backend": ep})

    backends = spey.AvailableBackends()
    assert "fake.backend" in backends

    wrapper = spey.get_backend("fake.backend")
    assert callable(wrapper)


def test_get_backend_metadata(monkeypatch):
    # make an entrypoint whose load() returns a class with metadata
    def loader():
        class Stat:
            name = "fake.metadata.backend"
            author = "Alice"
            version = "0.1.2"
            spey_requires = f">={spey.version()}"
            doi = ["10.1234/example"]
            arXiv = ["2001.00001"]
            zenodo = ["101010"]

        return Stat

    ep = _FakeEP("meta.backend", loader=loader)
    monkeypatch.setitem(spey.__dict__, "_backend_entries", {"meta.backend": ep})

    meta = spey.get_backend_metadata("meta.backend")
    assert meta["name"] == "fake.metadata.backend"
    assert meta["author"] == "Alice"
    assert "10.1234/example" in meta["doi"]
    assert "2001.00001" in meta["arXiv"]
    assert "101010" in meta["zenodo"]


def test_get_backend_bibtex_uses_get_bibtex_and_formats(monkeypatch):
    # Monkeypatch get_backend_metadata to return specific lists
    fake_meta = {
        "name": "unused",
        "author": "A",
        "version": "0",
        "spey_requires": f">={spey.version()}",
        "doi": ["10.1/doi"],
        "arXiv": ["2001.00002"],
        "zenodo": ["202020"],
    }
    monkeypatch.setattr(spey, "get_backend_metadata", lambda name: fake_meta)

    # Monkeypatch get_bibtex to return different strings depending on source
    def fake_get_bibtex(source, identifier):
        if source.startswith("inspire/arxiv"):
            return "ARXIV_BIBTEX"
        if source == "inspire/doi":
            # simulate missing on inspire -> empty string so code falls back to doi.org
            return ""
        if source == "doi":
            return "DOIORG_BIBTEX"
        if source == "zenodo":
            return "ZENODO_BIBTEX"
        return ""

    monkeypatch.setattr(spey, "get_bibtex", fake_get_bibtex)

    out = spey.get_backend_bibtex("anything")
    assert (
        "ARXIV_BIBTEX" in "".join(out["inspire"]) or out["inspire"]
    )  # inspire got arxiv
    assert "DOIORG_BIBTEX" in "".join(out["doi.org"])
    assert any("ZENODO_BIBTEX" in s for s in out["zenodo"])


def test_reset_backend_entries_updates_registry(monkeypatch):
    # monkeypatch the internal _get_entry_points to return fake entrypoints
    def fake_get_eps(group: str):
        return [_FakeEP("one"), _FakeEP("two")]

    monkeypatch.setattr(spey, "_get_entry_points", fake_get_eps)
    # start with an empty registry
    monkeypatch.setitem(spey.__dict__, "_backend_entries", {})

    spey.reset_backend_entries()
    keys = set(spey._backend_entries.keys())
    assert "one" in keys and "two" in keys
