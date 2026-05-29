import importlib
from importlib.metadata import version


def test_about_outputs(capsys):
    about_module = importlib.import_module("spey.about")

    about_module.about()
    out = capsys.readouterr().out

    # Header block
    assert f"spey v{version('spey')}" in out
    assert "Smooth inference" in out

    # System section
    assert "System:" in out
    assert "Platform:" in out
    assert "Python:" in out

    # Core dependency versions appear exactly once each
    for pkg in ("numpy", "scipy", "autograd", "tqdm", "joblib", "semantic_version"):
        assert out.count(version(pkg)) == 1, f"{pkg} version should appear exactly once"

    # Backends section header
    assert "Installed backends:" in out

    # Built-in default backends all present
    assert "- default.correlated_background" in out
    assert "- default.third_moment_expansion" in out
    assert "- default.uncorrelated_background" in out

    # Backends section header appears exactly once (no duplicate dist blocks)
    assert out.count("Installed backends:") == 1
    assert out.count("Core dependencies:") == 1
