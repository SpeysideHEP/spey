import types

import numpy as np
import pytest

from spey.combiner import uncorrelated_statistics_combiner as combiner_mod
from spey.combiner.uncorrelated_statistics_combiner import UnCorrStatisticsCombiner
from spey.system.exceptions import AnalysisQueryError, NegativeExpectedYields


class _FakeBackend:
    def __init__(self, minimum_poi=0.0):
        self._minimum_poi = minimum_poi

    def config(self):
        return types.SimpleNamespace(minimum_poi=self._minimum_poi)


class _FakeStatModel:
    def __init__(
        self,
        analysis,
        likelihood_value=0.0,
        asimov_return=None,
        muhat=0.5,
        sigma_mu_value=1.0,
        minimum_poi=0.0,
        raises_negative=False,
        s95exp=1.0,
    ):
        self.analysis = analysis
        self.backend_type = "fake.backend"
        self.backend = _FakeBackend(minimum_poi)
        self._likelihood_value = likelihood_value
        self._asimov_return = asimov_return if asimov_return is not None else []
        self._muhat = muhat
        self._sigma_mu = sigma_mu_value
        self._raises_negative = raises_negative
        self.s95exp = s95exp

        # capabilities
        self.is_alive = True
        self.is_asymptotic_calculator_available = True
        self.is_chi_square_calculator_available = True

    def likelihood(self, poi_test=1.0, expected=None, data=None, **kwargs):
        if self._raises_negative:
            raise NegativeExpectedYields("neg yields")
        return float(self._likelihood_value)

    def generate_asimov_data(self, expected=None, test_statistic="qtilde", **kwargs):
        return list(self._asimov_return)

    def maximize_likelihood(self, expected=None, **kwargs):
        return float(self._muhat), 0.0

    def sigma_mu(self, poi_test=0.0, expected=None, **kwargs):
        return float(self._sigma_mu)


def test_append_len_analyses_items_and_getitem(monkeypatch):
    # make the isinstance check permissive by treating StatisticalModel symbol as object
    monkeypatch.setitem(combiner_mod.__dict__, "StatisticalModel", object)

    comb = UnCorrStatisticsCombiner()
    m1 = _FakeStatModel("A", likelihood_value=1.0)
    m2 = _FakeStatModel("B", likelihood_value=2.0)

    comb.append(m1)
    comb.append(m2)

    assert len(comb) == 2
    assert set(comb.analyses) == {"A", "B"}

    # items yields tuples (analysis, model)
    items = dict(comb.items())
    assert "A" in items and items["A"] is m1

    # get by index and by name
    assert comb[0] is m1
    assert comb["B"] is m2

    # slicing returns tuple of models
    assert isinstance(comb[:], tuple) and comb[0:2][1] is m2


def test_remove_and_not_found(monkeypatch):
    monkeypatch.setitem(combiner_mod.__dict__, "StatisticalModel", object)

    comb = UnCorrStatisticsCombiner()
    m1 = _FakeStatModel("A")
    comb.append(m1)

    comb.remove("A")
    assert len(comb) == 0

    with pytest.raises(AnalysisQueryError):
        comb.remove("nonexistent")


def test_likelihood_combines_and_handles_negative_expected(monkeypatch):
    monkeypatch.setitem(combiner_mod.__dict__, "StatisticalModel", object)

    comb = UnCorrStatisticsCombiner()
    good = _FakeStatModel("G", likelihood_value=1.5)
    bad = _FakeStatModel("B", raises_negative=True)

    comb.append(good)
    comb.append(bad)

    # when a model raises NegativeExpectedYields, a RuntimeWarning is expected and result should be nan
    with pytest.warns(RuntimeWarning):
        res = comb.likelihood(poi_test=1.0, expected=None, return_nll=True)
    assert np.isnan(res)

    # if all good, sum of likelihoods returned
    comb2 = UnCorrStatisticsCombiner()
    comb2.append(_FakeStatModel("X", likelihood_value=0.3))
    comb2.append(_FakeStatModel("Y", likelihood_value=0.7))
    res2 = comb2.likelihood(poi_test=1.0, expected=None, return_nll=True)
    assert pytest.approx(1.0, rel=1e-6) == res2


def test_generate_asimov_data_combines(monkeypatch):
    monkeypatch.setitem(combiner_mod.__dict__, "StatisticalModel", object)

    comb = UnCorrStatisticsCombiner()
    comb.append(_FakeStatModel("one", asimov_return=[5.0, 6.0]))
    comb.append(_FakeStatModel("two", asimov_return=[1.0]))

    data = comb.generate_asimov_data(expected=None, test_statistic="qtilde")
    assert "one" in data and "two" in data
    assert data["one"] == [5.0, 6.0]
    assert data["two"] == [1.0]


def test_maximize_likelihood_uses_fit_and_returns_expected(monkeypatch):
    monkeypatch.setitem(combiner_mod.__dict__, "StatisticalModel", object)

    # Prepare combiner with two models such that mu_init computation path is used
    m1 = _FakeStatModel("m1", muhat=0.2, sigma_mu_value=0.1, minimum_poi=0.0)
    m2 = _FakeStatModel("m2", muhat=0.4, sigma_mu_value=0.2, minimum_poi=0.0)
    comb = UnCorrStatisticsCombiner(m1, m2)

    # Monkeypatch fit to return a known twice_nll and best-fit params
    def fake_fit(func, model_configuration=None, **kwargs):
        # ensure func is callable and return deterministic values
        return 8.0, [0.77]

    monkeypatch.setattr(combiner_mod, "fit", fake_fit)

    muhat, nll = comb.maximize_likelihood(return_nll=True)
    # fake_fit returned twice_nll=8.0 -> nll = 8.0/2 = 4.0
    assert pytest.approx(0.77, rel=1e-12) == muhat
    assert pytest.approx(4.0, rel=1e-12) == nll
