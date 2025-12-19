import importlib
import numpy as np


def test_about_outputs(capsys):
    about_module = importlib.import_module("spey.about")

    about_module.about()
    out = capsys.readouterr().out

    assert "Version:" in out
    assert "Platform info:" in out
    assert "Python version:" in out
    assert f"Numpy version:            {np.__version__}" in out
    assert "- default.correlated_background" in out
    assert "- default.third_moment_expansion" in out
