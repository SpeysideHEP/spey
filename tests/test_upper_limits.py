r"""
Tests for :mod:`spey.hypothesis_testing.upper_limits`.

The focus is the validity guard around :func:`find_poi_upper_limit`.  A
:math:`(1-\alpha)` CL upper limit is the boundary of the excluded set
:math:`\{\mu : CL_s(\mu) < \alpha\}`, which is only an upper limit when that set is a
ray.  A bracketing root finder, however, converges on *any* sign change — including
one produced by a :math:`CL_s` curve that is not monotonically decreasing, or by
numerical noise when the asymptotic approximation has broken down.

The models here are synthetic and analytic: a Gaussian profile likelihood
:math:`\ln\mathcal{L}(\mu) = -\tfrac{1}{2}((\mu-\hat\mu)/\sigma)^2` with a second,
independent width for the Asimov dataset.  Making the Asimov width huge reproduces the
pathological regime in which :math:`\sqrt{q_{\mu,A}} \to 0`, so eq. (66) of
:xref:`1007.1727` divides by (almost) zero.
"""

import logging

import numpy as np
import pytest
from scipy.stats import norm

import spey
from spey.hypothesis_testing.upper_limits import (
    _exclusion_persists,
    find_poi_upper_limit,
)
from spey.utils import ExpectationType


def gaussian_model(sigma: float, sigma_asimov: float, muhat: float = 0.0):
    r"""
    Build the four ingredients :func:`find_poi_upper_limit` needs.

    Args:
        sigma (``float``): Width of the observed profile likelihood.
        sigma_asimov (``float``): Width of the Asimov profile likelihood.  A large
          value means the Asimov dataset barely constrains :math:`\mu`.
        muhat (``float``, default ``0.0``): Best-fit parameter of interest.

    Returns:
        ``Tuple[Tuple[float, float], Callable, Tuple[float, float], Callable]``:
        ``(maximum_likelihood, logpdf, maximum_asimov_likelihood, asimov_logpdf)``.
    """

    def logpdf(mu):
        """Observed log-likelihood."""
        return -0.5 * ((float(mu) - muhat) / sigma) ** 2

    def asimov_logpdf(mu):
        """Asimov log-likelihood, peaked at zero."""
        return -0.5 * (float(mu) / sigma_asimov) ** 2

    return (muhat, -logpdf(muhat)), logpdf, (0.0, 0.0), asimov_logpdf


def upper_limit(model, **kwargs):
    """Run :func:`find_poi_upper_limit` on a :func:`gaussian_model` tuple."""
    maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf = model
    return find_poi_upper_limit(
        maximum_likelihood=maximum_likelihood,
        logpdf=logpdf,
        maximum_asimov_likelihood=maximum_asimov,
        asimov_logpdf=asimov_logpdf,
        expected=ExpectationType.observed,
        allow_negative_signal=False,
        **kwargs,
    )


def capture_spey_warnings(caplog):
    """Context manager enabling propagation of the non-propagating Spey logger."""

    class _Ctx:
        def __enter__(self):
            self.logger = logging.getLogger("Spey")
            self.previous = self.logger.propagate
            self.logger.propagate = True
            self.level = caplog.at_level(logging.WARNING, logger="Spey")
            self.level.__enter__()
            return caplog

        def __exit__(self, *exc):
            self.level.__exit__(*exc)
            self.logger.propagate = self.previous
            return False

    return _Ctx()


class TestHealthyLimit:
    """A well-behaved CLs curve must be untouched by the guard."""

    def test_matches_the_analytic_gaussian_limit(self):
        r"""
        For a unit Gaussian centred at zero the 95 % CLs upper limit is the 97.5 %
        normal quantile, :math:`\mu_{\rm UL} = \Phi^{-1}(0.975) \simeq 1.96`.
        """
        limit = upper_limit(gaussian_model(sigma=1.0, sigma_asimov=1.0))
        assert limit == pytest.approx(norm.ppf(0.975), rel=1e-6)

    def test_guard_does_not_change_the_result(self):
        model = gaussian_model(sigma=1.0, sigma_asimov=1.0)
        assert upper_limit(model, validate_limit=True) == pytest.approx(
            upper_limit(model, validate_limit=False), rel=1e-12
        )

    @pytest.mark.parametrize("sigma,sigma_asimov", [(1.0, 1.5), (2.0, 2.0), (0.5, 0.8)])
    def test_limit_is_finite_and_positive(self, sigma, sigma_asimov):
        limit = upper_limit(gaussian_model(sigma=sigma, sigma_asimov=sigma_asimov))
        assert np.isfinite(limit) and limit > 0.0

    def test_expected_pvalue_ranges_still_return_lists(self):
        model = gaussian_model(sigma=1.0, sigma_asimov=1.0)
        maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf = model
        for expected_pvalue, size in (("1sigma", 3), ("2sigma", 5)):
            limits = find_poi_upper_limit(
                maximum_likelihood=maximum_likelihood,
                logpdf=logpdf,
                maximum_asimov_likelihood=maximum_asimov,
                asimov_logpdf=asimov_logpdf,
                expected=ExpectationType.apriori,
                allow_negative_signal=False,
                expected_pvalue=expected_pvalue,
            )
            assert len(limits) == size
            assert all(np.isfinite(value) for value in limits)


class TestDegenerateAsimov:
    r"""
    When :math:`\sqrt{q_{\mu,A}} \to 0` the asymptotic :math:`CL_s` is meaningless.

    The root finder still sees a sign change and used to return it as an upper limit.
    """

    MODEL = dict(sigma=1.0, sigma_asimov=1e8)

    def test_returns_inf(self):
        assert upper_limit(gaussian_model(**self.MODEL)) == np.inf

    def test_is_reproducible(self):
        """The unguarded result rides on numerical noise; the guard must not."""
        model = gaussian_model(**self.MODEL)
        assert [upper_limit(model) for _ in range(3)] == [np.inf] * 3

    def test_warns_about_the_asimov_test_statistic(self, caplog):
        with capture_spey_warnings(caplog):
            upper_limit(gaussian_model(**self.MODEL))
        assert "Asimov test statistic" in caplog.text
        assert "numerically zero" in caplog.text

    def test_unguarded_result_is_spurious(self):
        """Without the guard a nonsensically small 'limit' comes back."""
        spurious = upper_limit(gaussian_model(**self.MODEL), validate_limit=False)
        assert np.isfinite(spurious)
        assert spurious < 1e-3  # orders of magnitude below any sensible limit

    def test_tolerance_is_configurable(self):
        """Lowering the tolerance below the actual value disables the rejection."""
        model = gaussian_model(**self.MODEL)
        assert np.isfinite(upper_limit(model, asimov_teststat_tolerance=0.0))


class TestExclusionPersists:
    """Unit tests for the monotonicity probe."""

    @staticmethod
    def _monotone(mu):
        """Exclusion holds for every mu above 1.0."""
        return mu - 1.0

    @staticmethod
    def _loses_exclusion(mu):
        """Excluded just above 1.0, but not excluded again further out."""
        return -1.0 if mu > 2.0 else mu - 1.0

    def test_true_for_a_monotone_curve(self):
        assert _exclusion_persists(self._monotone, 1.0, hig_bound=1e5)

    def test_false_when_exclusion_is_lost(self):
        assert not _exclusion_persists(self._loses_exclusion, 1.0, hig_bound=1e5)

    def test_probes_stop_at_the_upper_bound(self):
        """No probe is above hig_bound, so nothing can invalidate the root."""
        assert _exclusion_persists(self._loses_exclusion, 1.0, hig_bound=1.2)

    @pytest.mark.parametrize("root", [np.nan, np.inf, 0.0, -1.0])
    def test_non_positive_or_non_finite_roots_pass_through(self, root):
        assert _exclusion_persists(self._loses_exclusion, root, hig_bound=1e5)

    def test_non_finite_values_are_skipped(self):
        """A probe that cannot be evaluated must not reject the root."""
        assert _exclusion_persists(lambda mu: np.nan, 1.0, hig_bound=1e5)


class TestRealBackendsAreUnaffected:
    """The guard must stay out of the way for ordinary models."""

    def test_poisson(self):
        model = spey.get_backend("default.poisson")(
            signal_yields=[12.0, 15.0],
            background_yields=[50.0, 48.0],
            data=[36, 33],
            analysis="poisson",
        )
        assert model.poi_upper_limit() == pytest.approx(0.314087, rel=1e-4)

    def test_uncorrelated_background(self):
        model = spey.get_backend("default.uncorrelated_background")(
            signal_yields=[12.0, 15.0],
            background_yields=[50.0, 48.0],
            data=[36, 33],
            absolute_uncertainties=[12.0, 13.0],
            analysis="uncorrelated",
        )
        assert model.poi_upper_limit() == pytest.approx(0.748390, rel=1e-4)
        expected = model.poi_upper_limit(
            expected=spey.ExpectationType.apriori, expected_pvalue="1sigma"
        )
        assert len(expected) == 3 and all(np.isfinite(value) for value in expected)
