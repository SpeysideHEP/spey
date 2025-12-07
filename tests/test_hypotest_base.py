import pytest

from spey.base.hypotest_base import HypothesisTestingBase
from spey.system.exceptions import CalculatorNotAvailable


class _SimpleHypo(HypothesisTestingBase):
    """Minimal concrete implementation for testing HypothesisTestingBase helpers."""

    def __init__(
        self, *, alive=True, asymp_available=True, toy_available=True, chi_available=True
    ):
        super().__init__(ntoys=10)
        self._alive = alive
        self._asym = asymp_available
        self._toy = toy_available
        self._chi = chi_available
        # configurable responses
        self._likelihood_map = {}
        self._mle = (0.0, 0.0)
        self._asimov_mle = (0.0, 0.0)

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_asymptotic_calculator_available(self) -> bool:
        return self._asym

    @property
    def is_toy_calculator_available(self) -> bool:
        return self._toy

    @property
    def is_chi_square_calculator_available(self) -> bool:
        return self._chi

    def likelihood(
        self, poi_test=1.0, expected=None, return_nll=True, data=None, **kwargs
    ):
        # return a configurable value if present otherwise a simple quadratic dependence
        return float(
            self._likelihood_map.get(float(poi_test), 0.5 * float(poi_test) + 1.0)
        )

    def maximize_likelihood(
        self,
        return_nll=True,
        expected=None,
        allow_negative_signal=True,
        data=None,
        **kwargs,
    ):
        return self._mle

    def asimov_likelihood(
        self,
        poi_test=1.0,
        expected=None,
        return_nll=True,
        test_statistics="qtilde",
        **kwargs,
    ):
        # mimic different behaviour for asimov
        return float(0.25 * float(poi_test) + 0.5)

    def maximize_asimov_likelihood(
        self, return_nll=True, expected=None, test_statistics="qtilde", **kwargs
    ):
        return self._asimov_mle


def test_prepare_for_hypotest_wrappers_call_likelihoods_and_return_values():
    hypo = _SimpleHypo()
    # set maximize returns
    hypo._mle = (1.23, 2.0)  # muhat, nll
    hypo._asimov_mle = (0.5, 0.75)

    (muhat, nll), logpdf, (muhatA, nllA), logpdf_asimov = hypo._prepare_for_hypotest(
        expected=None, test_statistics="qtilde"
    )

    # check returned maxima propagate
    assert muhat == pytest.approx(1.23)
    assert nll == pytest.approx(2.0)
    assert muhatA == pytest.approx(0.5)
    assert nllA == pytest.approx(0.75)

    # logpdf should return -likelihood(poi)
    val = logpdf(2.5)
    assert val == pytest.approx(-hypo.likelihood(poi_test=2.5))

    # asimov wrapper should return negative of asimov_likelihood
    av = logpdf_asimov(1.5)
    assert av == pytest.approx(-hypo.asimov_likelihood(poi_test=1.5))


def test_chi2_uses_likelihood_and_maximize_likelihood_to_compute_value():
    hypo = _SimpleHypo()
    # make likelihood return specific values for poi_test 2.0 and maximize_likelihood denominator
    hypo._likelihood_map[2.0] = 3.0
    hypo._mle = (0.1, 1.0)  # denominator = 1.0
    # chi2 = 2*(likelihood(2.0) - denominator) = 2*(3 - 1) = 4
    chi2_val = hypo.chi2(poi_test=2.0)
    assert chi2_val == pytest.approx(4.0)


def test_poi_upper_limit_returns_inf_when_not_alive():
    hypo = _SimpleHypo(alive=False)
    res = hypo.poi_upper_limit(expected=None, confidence_level=0.95)
    # nominal should be inf for not alive case
    assert res == float("inf")


def test_exclusion_confidence_level_raises_when_calculator_not_available():
    hypo = _SimpleHypo(asymp_available=False)
    with pytest.raises(CalculatorNotAvailable):
        hypo.exclusion_confidence_level(
            poi_test=1.0, expected=None, calculator="asymptotic"
        )
