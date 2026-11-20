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


def _two_well_model():
    """Single-parameter EFT-style fit whose NLL has two wells of different depth.

    The global minimum sits at ``c ~ -0.29`` and a local one at ``c ~ +0.31``;
    the default start ``[1, 1]`` lands in the local one (issue #69).
    """
    import numpy as np
    import spey

    interference = np.array([1.0, 0.0, -1.0])
    quadratic = np.array([2.0, 4.0, 3.0])
    model = spey.get_backend("default.multivariate_normal")(
        signal_yields=lambda pars: pars[0] * interference + pars[0] ** 2 * quadratic,
        background_yields=np.zeros(3),
        data=np.array([0.2, 0.5, 0.4]),
        covariance_matrix=np.eye(3),
        n_signal_parameters=1,
    )
    model.backend.config().change_parameter_names({"signal_par_0": "c"})
    return model, interference, quadratic, np.array([0.2, 0.5, 0.4])


def test_chi2_test_init_pars_selects_basin():
    """``init_pars`` must reach the optimiser inside ``chi2_test`` (issue #69)."""
    import numpy as np
    from scipy.stats import chi2

    _, interference, quadratic, data = _two_well_model()

    # Brute-force 95% interval around the *global* minimum.
    grid = np.linspace(-3.0, 3.0, 600001)
    nll = 0.5 * np.sum(
        (data[None] - (grid[:, None] * interference + grid[:, None] ** 2 * quadratic))
        ** 2,
        axis=1,
    )
    inside = grid[nll <= nll.min() + 0.5 * chi2.isf(0.05, df=1)]
    expected = [inside.min(), inside.max()]

    kwargs = dict(parameter="c", poi_value=1.0, allow_negative_signal=True)
    default = _two_well_model()[0].chi2_test(**kwargs)
    steered = _two_well_model()[0].chi2_test(init_pars=[1.0, 0.0], **kwargs)
    steered_other = _two_well_model()[0].chi2_test(init_pars=[1.0, -0.3], **kwargs)

    assert np.allclose(steered, expected, atol=1e-3), "init_pars was not honoured"
    assert np.allclose(steered_other, expected, atol=1e-3)
    assert not np.allclose(default, steered, atol=1e-3), "init_pars had no effect"


def test_chi2_test_init_pars_does_not_mutate_input():
    """The caller's ``init_pars`` list must be left untouched."""
    init = [1.0, 0.0]
    _two_well_model()[0].chi2_test(
        parameter="c", poi_value=1.0, allow_negative_signal=True, init_pars=init
    )
    assert init == [1.0, 0.0]


def test_chi2_test_par_bounds_limit_profiled_parameter():
    """``par_bounds`` of the profiled parameter caps the returned crossings."""
    model = _two_well_model()[0]
    limits = model.chi2_test(
        parameter="c",
        poi_value=1.0,
        init_pars=[1.0, 1.0],
        par_bounds=[(None, None), (0.0, 2.0)],
    )
    assert all(0.0 - 1e-9 <= lim <= 2.0 + 1e-9 for lim in limits)


def test_chi2_test_poi_profile_accepts_init_pars():
    """The POI-profiling branch forwards ``init_pars`` / ``par_bounds`` as well."""
    import numpy as np
    import spey

    model = spey.get_backend("default.normal")(
        signal_yields=[12.0],
        background_yields=[50.0],
        data=[36.0],
        absolute_uncertainties=[20.0],
    )
    reference = model.chi2_test()
    with_kwargs = model.chi2_test(init_pars=[0.5], par_bounds=[(-10.0, 10.0)])
    assert np.allclose(reference, with_kwargs, rtol=1e-3)


def test_chi2_test_n_multistart_finds_global_basin_without_init_pars():
    """Opt-in multi-start recovers the global basin from the default start."""
    import numpy as np
    from scipy.stats import chi2

    _, interference, quadratic, data = _two_well_model()
    grid = np.linspace(-3.0, 3.0, 600001)
    nll = 0.5 * np.sum(
        (data[None] - (grid[:, None] * interference + grid[:, None] ** 2 * quadratic))
        ** 2,
        axis=1,
    )
    inside = grid[nll <= nll.min() + 0.5 * chi2.isf(0.05, df=1)]

    kwargs = dict(parameter="c", poi_value=1.0, allow_negative_signal=True)
    default = _two_well_model()[0].chi2_test(**kwargs)
    searched = _two_well_model()[0].chi2_test(n_multistart=9, **kwargs)

    assert np.allclose(searched, [inside.min(), inside.max()], atol=1e-3)
    assert not np.allclose(default, searched, atol=1e-3), "default must stay single-start"


def test_chi2_test_n_multistart_unimodal_unchanged():
    """The multi-start search must not alter a well-behaved (convex) profile."""
    import numpy as np
    import spey

    model = spey.get_backend("default.normal")(
        signal_yields=[12.0],
        background_yields=[50.0],
        data=[36.0],
        absolute_uncertainties=[20.0],
    )
    assert np.allclose(model.chi2_test(), model.chi2_test(n_multistart=7), rtol=1e-6)


def test_refine_global_min_1d_resolves_narrow_basin():
    """A basin narrower than the cell width is found; a worse search is a no-op."""
    import numpy as np

    from spey.base.utils import _refine_global_min_1d

    f = lambda x: 0.01 * x**2 - 3.0 * np.exp(-(((x - 7.0) / 0.5) ** 2))
    # incumbent: the shallow minimum at x = 0; the deep basin (width 0.5) is 10x
    # narrower than the cell width (5)
    mll, ref = _refine_global_min_1d(f, f(0.0), 0.0, -10.0, 10.0, n_samples=1)
    assert (mll, ref) == (f(0.0), 0.0), "n_samples < 2 must be a no-op"
    mll, ref = _refine_global_min_1d(f, f(0.0), 0.0, -10.0, 10.0, n_samples=5)
    assert mll < f(0.0) - 1.0 and abs(ref - 7.0) < 0.1
    assert mll <= f(ref) + 1e-12
    mll2, ref2 = _refine_global_min_1d(lambda x: x**2, 0.0, 0.0, -10.0, 10.0, 9)
    assert (mll2, ref2) == (0.0, 0.0)
