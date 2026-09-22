r"""
Unit tests for :class:`~spey.combiner.CorrelatedStatisticsCombiner`.

The core of this module is an **independent closed-form implementation** of a
three-model combination with shared parameters.  Three ``default.normal`` models
share a signal strength ``mu`` and two EFT-like Wilson coefficients; a third
coefficient belongs to a single model:

    global parameters   p = [mu, c1, c2, c3]

    SR_A  2 bins  local [mu, c1, c2]  -> global [0, 1, 2]   s = c1*a1 + c2*a2 + c1*c2*a12
    SR_B  3 bins  local [mu, c1, c3]  -> global [0, 1, 3]   s = c1^2*b1 + c3*b3
    SR_C  2 bins  local [mu, c2]      -> global [0, 2]      s = c2*d1 + c2^2*d2

Every model has ``lambda = mu * s(c) + b`` and a Gaussian likelihood with fixed
widths, so with residual ``r = lambda - n``:

    logL   = sum  -0.5 (r/sig)^2 - log sig - 0.5 log 2pi
    dlogL  = -sum  r/sig^2 * dlambda
    d2logL = -sum [ dlambda dlambda / sig^2 + r/sig^2 * d2lambda ]

The analytic gradient and Hessian below are hand-derived from those expressions and
are additionally cross-checked against finite differences, so the comparison against
the plug-in is a genuine analytic-vs-numeric test rather than a tautology.

The remaining test classes cover parameter mapping, the accepted forms of the
``shared_parameters`` declaration, shared *nuisance* parameters, constraint lifting,
backends whose derivatives come from a foreign autodiff framework (jax / tensorflow /
pytorch style return values), graceful degradation when a constituent model is not
differentiable, data round-tripping, and integration with ``find_contour``.
"""

import numpy as np
import pytest

import spey
from spey.base.backend_base import BackendBase
from spey.base.model_config import ModelConfig
from spey.combiner import CorrelatedStatisticsCombiner
from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.system.exceptions import AnalysisQueryError, InvalidInput
from spey.utils import ExpectationType

# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------
BKG_A = np.array([50.0, 48.0])
SIG_A = np.array([7.0, 6.0])
OBS_A = np.array([63.0, 55.0])
A1 = np.array([12.0, 15.0])
A2 = np.array([6.0, 16.0])
A12 = np.array([2.0, -3.0])

BKG_B = np.array([30.0, 22.0, 17.0])
SIG_B = np.array([5.0, 4.5, 4.0])
OBS_B = np.array([35.0, 20.0, 21.0])
B1 = np.array([4.0, 3.0, 2.0])
B3 = np.array([1.0, 5.0, 7.0])

BKG_C = np.array([80.0, 65.0])
SIG_C = np.array([9.0, 8.0])
OBS_C = np.array([92.0, 60.0])
D1 = np.array([10.0, 4.0])
D2 = np.array([1.5, 2.5])

NPAR = 4
IDX_A = np.array([0, 1, 2])
IDX_B = np.array([0, 1, 3])
IDX_C = np.array([0, 2])
_LOG2PI = np.log(2.0 * np.pi)

TEST_POINTS = [
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([0.7, -0.4, 0.9, 1.3]),
    np.array([1.4, 0.2, -0.6, 0.1]),
    np.array([-0.25, 1.184, -0.987, 1.053]),
]


def signal_a(c1, c2):
    """Signal template of SR_A with a c1-c2 interference term."""
    return c1 * A1 + c2 * A2 + c1 * c2 * A12


def signal_b(c1, c3):
    """Signal template of SR_B, quadratic in c1."""
    return c1**2 * B1 + c3 * B3


def signal_c(c2):
    """Signal template of SR_C, quadratic in c2."""
    return c2 * D1 + c2**2 * D2


def _derivs_a(mu, c1, c2):
    """``(lambda, dlambda, d2lambda)`` of SR_A in global parameter indices."""
    sig = signal_a(c1, c2)
    grad = np.zeros((NPAR, len(BKG_A)))
    grad[0] = sig
    grad[1] = mu * (A1 + c2 * A12)
    grad[2] = mu * (A2 + c1 * A12)
    hess = np.zeros((NPAR, NPAR, len(BKG_A)))
    hess[0, 1] = hess[1, 0] = A1 + c2 * A12
    hess[0, 2] = hess[2, 0] = A2 + c1 * A12
    hess[1, 2] = hess[2, 1] = mu * A12
    return mu * sig + BKG_A, grad, hess


def _derivs_b(mu, c1, c3):
    """``(lambda, dlambda, d2lambda)`` of SR_B in global parameter indices."""
    sig = signal_b(c1, c3)
    grad = np.zeros((NPAR, len(BKG_B)))
    grad[0] = sig
    grad[1] = mu * 2.0 * c1 * B1
    grad[3] = mu * B3
    hess = np.zeros((NPAR, NPAR, len(BKG_B)))
    hess[0, 1] = hess[1, 0] = 2.0 * c1 * B1
    hess[0, 3] = hess[3, 0] = B3
    hess[1, 1] = mu * 2.0 * B1
    return mu * sig + BKG_B, grad, hess


def _derivs_c(mu, c2):
    """``(lambda, dlambda, d2lambda)`` of SR_C in global parameter indices."""
    sig = signal_c(c2)
    grad = np.zeros((NPAR, len(BKG_C)))
    grad[0] = sig
    grad[2] = mu * (D1 + 2.0 * c2 * D2)
    hess = np.zeros((NPAR, NPAR, len(BKG_C)))
    hess[0, 2] = hess[2, 0] = D1 + 2.0 * c2 * D2
    hess[2, 2] = mu * 2.0 * D2
    return mu * sig + BKG_C, grad, hess


def _gaussian_block(lam, dlam, d2lam, obs, sigma):
    """Value, gradient and Hessian of one Gaussian block."""
    resid = lam - obs
    inv_var = 1.0 / sigma**2
    value = np.sum(-0.5 * resid**2 * inv_var - np.log(sigma) - 0.5 * _LOG2PI)
    grad = -np.einsum("i,ai->a", resid * inv_var, dlam)
    hess = -np.einsum("i,ai,bi->ab", inv_var, dlam, dlam) - np.einsum(
        "i,abi->ab", resid * inv_var, d2lam
    )
    return value, grad, hess


def analytic(pars, data=None):
    r"""
    Closed-form ``(logL, gradient, Hessian)`` of the combination.

    Args:
        pars (``np.ndarray``): Global parameter vector ``[mu, c1, c2, c3]``.
        data (``np.ndarray``, default ``None``): Concatenated observations replacing
          ``(OBS_A, OBS_B, OBS_C)``; used for the Asimov / apriori checks.

    Returns:
        ``Tuple[float, np.ndarray, np.ndarray]``
    """
    mu, c1, c2, c3 = pars
    if data is None:
        obs_a, obs_b, obs_c = OBS_A, OBS_B, OBS_C
    else:
        data = np.asarray(data, dtype=float)
        obs_a, obs_b, obs_c = data[:2], data[2:5], data[5:]

    total_value, total_grad, total_hess = 0.0, np.zeros(NPAR), np.zeros((NPAR, NPAR))
    for lam, dlam, d2lam, obs, sigma in (
        (*_derivs_a(mu, c1, c2), obs_a, SIG_A),
        (*_derivs_b(mu, c1, c3), obs_b, SIG_B),
        (*_derivs_c(mu, c2), obs_c, SIG_C),
    ):
        value, grad, hess = _gaussian_block(lam, dlam, d2lam, obs, sigma)
        total_value += value
        total_grad += grad
        total_hess += hess
    return total_value, total_grad, total_hess


def analytic_expected_data(pars):
    """Concatenated per-model expected yields, in model order."""
    mu, c1, c2, c3 = pars
    return np.concatenate(
        [
            mu * signal_a(c1, c2) + BKG_A,
            mu * signal_b(c1, c3) + BKG_B,
            mu * signal_c(c2) + BKG_C,
        ]
    )


def _finite_difference_gradient(func, pars, eps=1e-5):
    """Central-difference gradient of a scalar function."""
    pars = np.asarray(pars, dtype=float)
    out = np.zeros_like(pars)
    for i in range(pars.size):
        up, down = pars.copy(), pars.copy()
        up[i] += eps
        down[i] -= eps
        out[i] = (func(up) - func(down)) / (2 * eps)
    return out


def _finite_difference_hessian(func, pars, eps=1e-4):
    """Central-difference Hessian of a scalar function."""
    pars = np.asarray(pars, dtype=float)
    n = pars.size
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pp, pm, mp, mm = (pars.copy() for _ in range(4))
            pp[i] += eps
            pp[j] += eps
            pm[i] += eps
            pm[j] -= eps
            mp[i] -= eps
            mp[j] += eps
            mm[i] -= eps
            mm[j] -= eps
            out[i, j] = (func(pp) - func(pm) - func(mp) + func(mm)) / (4 * eps**2)
    return out


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_models():
    """Return the three constituent ``default.normal`` statistical models."""
    normal = spey.get_backend("default.normal")
    model_a = normal(
        signal_yields=lambda pars: signal_a(pars[0], pars[1]),
        background_yields=BKG_A,
        data=OBS_A,
        absolute_uncertainties=SIG_A,
        n_signal_parameters=2,
        analysis="SR_A",
    )
    model_b = normal(
        signal_yields=lambda pars: signal_b(pars[0], pars[1]),
        background_yields=BKG_B,
        data=OBS_B,
        absolute_uncertainties=SIG_B,
        n_signal_parameters=2,
        analysis="SR_B",
    )
    model_c = normal(
        signal_yields=lambda pars: signal_c(pars[0]),
        background_yields=BKG_C,
        data=OBS_C,
        absolute_uncertainties=SIG_C,
        n_signal_parameters=1,
        analysis="SR_C",
    )
    return model_a, model_b, model_c


SHARED = [
    "mu",
    {"name": "c1", "members": {"SR_A": "signal_par_0", "SR_B": 1}},
    {"name": "c2", "members": {"SR_A": 2, "SR_C": "signal_par_0"}},
]


def build_combiner(**kwargs):
    """Return the combined :class:`CorrelatedStatisticsCombiner` backend."""
    return CorrelatedStatisticsCombiner(
        statistical_models=list(build_models()),
        shared_parameters=kwargs.pop("shared_parameters", SHARED),
        **kwargs,
    )


def build_combined_model(**kwargs):
    """Return the combination wrapped in a :class:`~spey.StatisticalModel`."""
    return statistical_model_wrapper(CorrelatedStatisticsCombiner)(
        statistical_models=list(build_models()),
        shared_parameters=kwargs.pop("shared_parameters", SHARED),
        analysis="combined",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Analytic vs numeric
# ---------------------------------------------------------------------------
class TestAnalyticAgreement:
    """Compare every derivative the plug-in produces against the closed form."""

    def test_analytic_reference_is_self_consistent(self):
        """The hand-derived gradient/Hessian agree with finite differences."""
        for pars in TEST_POINTS:
            _, grad, hess = analytic(pars)
            fd_grad = _finite_difference_gradient(lambda p: analytic(p)[0], pars)
            fd_hess = _finite_difference_hessian(lambda p: analytic(p)[0], pars)
            assert np.allclose(grad, fd_grad, rtol=1e-5, atol=1e-6)
            assert np.allclose(hess, fd_hess, rtol=1e-4, atol=1e-4)

    def test_logpdf(self):
        backend = build_combiner()
        logpdf = backend.get_logpdf_func()
        for pars in TEST_POINTS:
            assert logpdf(pars) == pytest.approx(analytic(pars)[0], rel=1e-12)

    def test_objective_and_gradient(self):
        backend = build_combiner()
        objective = backend.get_objective_function(do_grad=True)
        for pars in TEST_POINTS:
            value, grad = objective(pars)
            expected_value, expected_grad, _ = analytic(pars)
            assert value == pytest.approx(-expected_value, rel=1e-12)
            # the backend returns the gradient of -logL
            assert np.allclose(-grad, expected_grad, rtol=1e-10, atol=1e-10)

    def test_objective_without_gradient(self):
        backend = build_combiner()
        objective = backend.get_objective_function(do_grad=False)
        for pars in TEST_POINTS:
            assert objective(pars) == pytest.approx(-analytic(pars)[0], rel=1e-12)

    def test_hessian(self):
        backend = build_combiner()
        hessian = backend.get_hessian_logpdf_func()
        for pars in TEST_POINTS:
            computed = hessian(pars)
            assert computed.shape == (NPAR, NPAR)
            assert np.allclose(computed, analytic(pars)[2], rtol=1e-9, atol=1e-9)

    def test_hessian_off_diagonal_of_shared_parameters(self):
        """A shared slot accumulates cross terms from every model that owns it."""
        backend = build_combiner()
        hessian = backend.get_hessian_logpdf_func()
        pars = TEST_POINTS[1]
        computed = hessian(pars)
        # d2/dmu dc3 only exists in SR_B; d2/dc1 dc3 likewise.
        assert computed[0, 3] != 0.0
        # c3 (SR_B only) and c2 (SR_A, SR_C) never appear in the same model, so the
        # joint second derivative has to vanish exactly.
        assert computed[2, 3] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(computed, computed.T, atol=1e-10)

    def test_expected_data(self):
        backend = build_combiner()
        for pars in TEST_POINTS:
            assert np.allclose(backend.expected_data(pars), analytic_expected_data(pars))

    def test_apriori_expectation(self):
        """``apriori`` replaces every model's data with its background."""
        backend = build_combiner()
        logpdf = backend.get_logpdf_func(expected=ExpectationType.apriori)
        background = np.concatenate([BKG_A, BKG_B, BKG_C])
        for pars in TEST_POINTS:
            expected = analytic(pars, data=background)[0]
            assert logpdf(pars) == pytest.approx(expected, rel=1e-12)

    def test_maximum_likelihood_matches_analytic_optimum(self):
        """``spey``'s fit reproduces an independent minimisation of the closed form."""
        from scipy.optimize import minimize  # pylint: disable=import-outside-toplevel

        model = build_combined_model()
        poi_dict, nll = model.maximize_likelihood(
            poi_indices=list(range(NPAR)), return_nll=True
        )
        fit_pars = np.array([poi_dict[idx] for idx in range(NPAR)])

        reference = minimize(
            lambda pars: -analytic(pars)[0],
            x0=np.ones(NPAR),
            jac=lambda pars: -analytic(pars)[1],
            method="BFGS",
            tol=1e-14,
        )
        # the analytic NLL at the fitted point is exactly what spey reports ...
        assert -analytic(fit_pars)[0] == pytest.approx(nll, rel=1e-10)
        # ... and that point is the same optimum an independent minimiser finds.
        # The basin is shallow along c3, so the parameters are compared loosely while
        # the NLL — the quantity that actually matters — is compared tightly.
        assert abs(nll - reference.fun) < 1e-4
        assert np.allclose(fit_pars, reference.x, atol=5e-2)


class TestAutogradCompatibility:
    """The combined log-pdf must survive external :mod:`autograd` differentiation."""

    def test_grad_and_hessian_through_concatenate(self):
        """Exactly the pattern :func:`find_contour` uses to fix ``mu``."""
        autograd = pytest.importorskip("autograd")
        anp = pytest.importorskip("autograd.numpy")

        backend = build_combiner()
        logpdf = backend.get_logpdf_func()

        def nll(signal_pars):
            return -logpdf(anp.concatenate([anp.array([1.0]), signal_pars]))

        point = np.array([-0.4, 0.9, 1.3])
        full = np.concatenate([[1.0], point])
        _, expected_grad, expected_hess = analytic(full)

        assert np.allclose(autograd.grad(nll)(point), -expected_grad[1:], atol=1e-10)
        assert np.allclose(
            autograd.hessian(nll)(point), -expected_hess[1:, 1:], atol=1e-8
        )


# ---------------------------------------------------------------------------
# Parameter bookkeeping
# ---------------------------------------------------------------------------
class TestParameterMapping:
    def test_index_maps(self):
        backend = build_combiner()
        maps = backend.index_map
        assert np.array_equal(maps["SR_A"], IDX_A)
        assert np.array_equal(maps["SR_B"], IDX_B)
        assert np.array_equal(maps["SR_C"], IDX_C)

    def test_names_and_size(self):
        backend = build_combiner()
        assert backend.npar == NPAR
        assert backend.parameter_names == ["mu", "c1", "c2", "signal_par_1"]
        assert backend.shared_parameter_names == ["mu", "c1", "c2"]
        assert backend.config().poi_index == 0

    def test_nothing_shared_gives_block_diagonal_vector(self):
        backend = CorrelatedStatisticsCombiner(list(build_models()))
        assert backend.npar == 3 + 3 + 2
        assert backend.shared_parameter_names == []
        # colliding private names are qualified with the analysis identifier
        assert "SR_A::mu" in backend.parameter_names
        assert "SR_B::signal_par_0" in backend.parameter_names

    def test_bounds_are_intersected(self):
        normal = spey.get_backend("default.normal")
        first = normal(
            signal_yields=lambda p: p[0] * A1,
            background_yields=BKG_A,
            data=OBS_A,
            absolute_uncertainties=SIG_A,
            n_signal_parameters=1,
            signal_parameter_bounds=[(-5.0, 5.0)],
            analysis="first",
        )
        second = normal(
            signal_yields=lambda p: p[0] * A2,
            background_yields=BKG_A,
            data=OBS_A,
            absolute_uncertainties=SIG_A,
            n_signal_parameters=1,
            signal_parameter_bounds=[(-1.0, 8.0)],
            analysis="second",
        )
        backend = CorrelatedStatisticsCombiner(
            [first, second],
            shared_parameters=["mu", {"name": "c", "members": {"first": 1, "second": 1}}],
        )
        assert backend.config().suggested_bounds[1] == (-1.0, 5.0)

    def test_disjoint_bounds_raise(self):
        normal = spey.get_backend("default.normal")
        first = normal(
            signal_yields=lambda p: p[0] * A1,
            background_yields=BKG_A,
            data=OBS_A,
            absolute_uncertainties=SIG_A,
            n_signal_parameters=1,
            signal_parameter_bounds=[(2.0, 5.0)],
            analysis="first",
        )
        second = normal(
            signal_yields=lambda p: p[0] * A2,
            background_yields=BKG_A,
            data=OBS_A,
            absolute_uncertainties=SIG_A,
            n_signal_parameters=1,
            signal_parameter_bounds=[(-5.0, 1.0)],
            analysis="second",
        )
        with pytest.raises(InvalidInput, match="do not overlap"):
            CorrelatedStatisticsCombiner(
                [first, second],
                shared_parameters=[{"name": "c", "members": {"first": 1, "second": 1}}],
            )

    def test_poi_selection_by_name_and_index(self):
        assert build_combiner(poi_name="c1").config().poi_index == 1
        assert build_combiner(poi_index=2).config().poi_index == 2

    def test_config_rescales_poi_bounds(self):
        backend = build_combiner()
        config = backend.config(allow_negative_signal=False, poi_upper_bound=3.0)
        assert config.suggested_bounds[0] == (0.0, 3.0)
        # the unrelated parameters keep their bounds
        assert config.suggested_bounds[1:] == backend.config().suggested_bounds[1:]

    def test_introspection_helpers(self):
        backend = build_combiner()
        assert len(backend) == 3
        assert backend.analyses == ["SR_A", "SR_B", "SR_C"]
        assert backend["SR_B"].analysis == "SR_B"
        assert backend[0].analysis == "SR_A"
        assert [model.analysis for model in backend] == backend.analyses
        assert backend.is_alive
        assert "SR_A" in repr(backend)


class TestSharedParameterDeclarations:
    """All accepted spellings of ``shared_parameters`` and their failure modes."""

    def test_equivalent_spellings(self):
        reference = build_combiner().index_map
        spellings = [
            ["mu", {"c1": None}] if False else None,  # placeholder, replaced below
        ]
        spellings = [
            # bare mapping, no explicit name
            [
                "mu",
                {"SR_A": "signal_par_0", "SR_B": 1},
                {"SR_A": 2, "SR_C": "signal_par_0"},
            ],
            # sequence of (model, parameter) pairs
            [
                "mu",
                [("SR_A", 1), ("SR_B", 1)],
                [(0, 2), (2, 1)],
            ],
            # positional model keys with named parameters
            [
                "mu",
                {"name": "c1", "members": {0: "signal_par_0", 1: "signal_par_0"}},
                {"name": "c2", "members": {0: "signal_par_1", 2: "signal_par_0"}},
            ],
        ]
        for spelling in spellings:
            maps = CorrelatedStatisticsCombiner(
                list(build_models()), shared_parameters=spelling
            ).index_map
            for analysis, expected in reference.items():
                assert np.array_equal(maps[analysis], expected), spelling

    def test_single_declaration_is_accepted(self):
        backend = CorrelatedStatisticsCombiner(list(build_models()), "mu")
        assert backend.shared_parameter_names == ["mu"]

    def test_unknown_parameter_name(self):
        with pytest.raises(InvalidInput, match="No model declares a parameter"):
            CorrelatedStatisticsCombiner(list(build_models()), ["not_a_parameter"])

    def test_unknown_model_key(self):
        with pytest.raises(AnalysisQueryError, match="not among the analyses"):
            CorrelatedStatisticsCombiner(
                list(build_models()), [{"name": "x", "members": {"SR_Z": 1}}]
            )

    def test_out_of_range_parameter_index(self):
        with pytest.raises(InvalidInput, match="out of range"):
            CorrelatedStatisticsCombiner(
                list(build_models()), [{"name": "x", "members": {"SR_C": 7}}]
            )

    def test_parameter_in_two_groups(self):
        with pytest.raises(InvalidInput, match="more than one shared group"):
            CorrelatedStatisticsCombiner(
                list(build_models()),
                [
                    {"name": "x", "members": {"SR_A": 1, "SR_B": 1}},
                    {"name": "y", "members": {"SR_A": 1, "SR_C": 1}},
                ],
            )

    def test_duplicate_group_name(self):
        with pytest.raises(InvalidInput, match="declared twice"):
            CorrelatedStatisticsCombiner(
                list(build_models()),
                [
                    {"name": "x", "members": {"SR_A": 1, "SR_B": 1}},
                    {"name": "x", "members": {"SR_A": 2, "SR_C": 1}},
                ],
            )

    def test_name_without_members(self):
        with pytest.raises(InvalidInput, match="must also"):
            CorrelatedStatisticsCombiner(list(build_models()), [{"name": "x"}])

    def test_unsupported_declaration_type(self):
        with pytest.raises(InvalidInput, match="Can not interpret"):
            CorrelatedStatisticsCombiner(list(build_models()), [3.14])

    def test_single_member_group_warns(self, caplog):
        import logging  # pylint: disable=import-outside-toplevel

        # the Spey logger does not propagate by default, see spey.system.logger
        spey_logger = logging.getLogger("Spey")
        previous = spey_logger.propagate
        spey_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="Spey"):
                CorrelatedStatisticsCombiner(
                    list(build_models()), [{"name": "lonely", "members": {"SR_A": 1}}]
                )
        finally:
            spey_logger.propagate = previous
        assert "single member" in caplog.text

    def test_both_poi_arguments(self):
        with pytest.raises(InvalidInput, match="either"):
            build_combiner(poi_name="mu", poi_index=0)

    def test_unknown_poi_name(self):
        with pytest.raises(InvalidInput, match="not among the combined parameters"):
            build_combiner(poi_name="nope")

    def test_out_of_range_poi_index(self):
        with pytest.raises(InvalidInput, match="out of range"):
            build_combiner(poi_index=99)


class TestConstructorValidation:
    def test_empty_input(self):
        with pytest.raises(InvalidInput, match="At least one"):
            CorrelatedStatisticsCombiner([])

    def test_wrong_type(self):
        with pytest.raises(TypeError, match="Can not combine"):
            CorrelatedStatisticsCombiner(["not a model"])

    def test_duplicate_analysis_names(self):
        model_a, _, _ = build_models()
        with pytest.raises(AnalysisQueryError, match="unique"):
            CorrelatedStatisticsCombiner([model_a, model_a])

    def test_single_model_is_allowed(self):
        model_a, _, _ = build_models()
        backend = CorrelatedStatisticsCombiner(model_a)
        assert backend.npar == 3
        pars = np.array([1.0, 0.5, -0.5])
        assert backend.get_logpdf_func()(pars) == pytest.approx(
            float(model_a.backend.get_logpdf_func()(pars))
        )


# ---------------------------------------------------------------------------
# Shared nuisance parameters and constraints
# ---------------------------------------------------------------------------
class TestSharedNuisanceParameters:
    """A shared parameter need not be a POI."""

    @staticmethod
    def _models():
        uncorrelated = spey.get_backend("default.uncorrelated_background")
        first = uncorrelated(
            signal_yields=[12.0, 15.0],
            background_yields=[50.0, 48.0],
            data=[51, 48],
            absolute_uncertainties=[10.0, 8.0],
            analysis="A",
        )
        second = uncorrelated(
            signal_yields=[8.0],
            background_yields=[30.0],
            data=[33],
            absolute_uncertainties=[6.0],
            analysis="B",
        )
        return first, second

    def test_poi_shareable_by_name_without_declared_names(self):
        """Backends without ``parameter_names`` still expose their POI as ``mu``."""
        first, second = self._models()
        assert first.backend.config().parameter_names is None
        backend = CorrelatedStatisticsCombiner([first, second], ["mu"])
        assert backend.parameter_names[0] == "mu"
        assert backend.shared_parameter_names == ["mu"]

    def test_shared_nuisance_reduces_parameter_count(self):
        first, second = self._models()
        backend = CorrelatedStatisticsCombiner(
            [first, second],
            ["mu", {"name": "theta_shared", "members": {"A": 1, "B": 1}}],
        )
        # mu + theta_shared + A's second nuisance
        assert backend.npar == 3
        assert backend.parameter_names == ["mu", "theta_shared", "A_par_2"]
        assert np.array_equal(backend.index_map["A"], [0, 1, 2])
        assert np.array_equal(backend.index_map["B"], [0, 1])

    def test_logpdf_equals_manual_sum(self):
        first, second = self._models()
        backend = CorrelatedStatisticsCombiner(
            [first, second],
            ["mu", {"name": "theta_shared", "members": {"A": 1, "B": 1}}],
        )
        logpdf = backend.get_logpdf_func()
        for pars in (
            np.array([1.0, 0.0, 0.0]),
            np.array([1.1, 0.3, -0.2]),
            np.array([0.4, -0.7, 0.8]),
        ):
            manual = float(
                first.backend.get_logpdf_func()(pars[backend.index_map["A"]])
            ) + float(second.backend.get_logpdf_func()(pars[backend.index_map["B"]]))
            assert logpdf(pars) == pytest.approx(manual, rel=1e-12)

    def test_gradient_of_shared_nuisance_sums_both_models(self):
        first, second = self._models()
        backend = CorrelatedStatisticsCombiner(
            [first, second],
            ["mu", {"name": "theta_shared", "members": {"A": 1, "B": 1}}],
        )
        pars = np.array([1.1, 0.3, -0.2])
        _, grad = backend.get_objective_function(do_grad=True)(pars)

        grad_a = first.backend.get_objective_function(do_grad=True)(
            pars[backend.index_map["A"]]
        )[1]
        grad_b = second.backend.get_objective_function(do_grad=True)(
            pars[backend.index_map["B"]]
        )[1]
        # slot 1 receives a contribution from both models, slot 2 only from A
        assert grad[0] == pytest.approx(grad_a[0] + grad_b[0], rel=1e-10)
        assert grad[1] == pytest.approx(grad_a[1] + grad_b[1], rel=1e-10)
        assert grad[2] == pytest.approx(grad_a[2], rel=1e-10)

    def test_joint_profiling_differs_from_independent_profiling(self):
        """Sharing a nuisance is not the same as treating the models as independent."""
        first, second = self._models()
        correlated = statistical_model_wrapper(CorrelatedStatisticsCombiner)(
            statistical_models=[first, second],
            shared_parameters=["mu", {"name": "theta", "members": {"A": 1, "B": 1}}],
            analysis="correlated",
        )
        uncorrelated = spey.UnCorrStatisticsCombiner(first, second)
        assert correlated.likelihood(1.0) != pytest.approx(
            uncorrelated.likelihood(1.0), rel=1e-6
        )


class TestConstraintLifting:
    @staticmethod
    def _models():
        poisson = spey.get_backend("default.poisson")
        first = poisson(
            signal_yields=[5.0, 3.0],
            background_yields=[50.0, 30.0],
            data=[55, 31],
            absolute_uncertainties=[5.0, 3.0],
            analysis="P1",
        )
        second = poisson(
            signal_yields=[4.0],
            background_yields=[20.0],
            data=[22],
            absolute_uncertainties=[2.0],
            analysis="P2",
        )
        return first, second

    def test_constraints_are_remapped(self):
        first, second = self._models()
        backend = CorrelatedStatisticsCombiner([first, second], ["mu"])
        assert len(backend.constraints) == 2

        pars = np.array([1.0, 0.1, -0.1, 0.2])
        local_first = first.backend.constraints[0]
        local_second = second.backend.constraints[0]

        assert np.allclose(
            backend.constraints[0].fun(pars),
            local_first.fun(pars[backend.index_map["P1"]]),
        )
        assert np.allclose(
            backend.constraints[1].fun(pars),
            local_second.fun(pars[backend.index_map["P2"]]),
        )

    def test_lifted_jacobian_scatters_columns(self):
        first, second = self._models()
        backend = CorrelatedStatisticsCombiner([first, second], ["mu"])
        pars = np.array([1.0, 0.1, -0.1, 0.2])

        for constraint, analysis, local in zip(
            backend.constraints,
            ("P1", "P2"),
            (first.backend.constraints[0], second.backend.constraints[0]),
        ):
            index_map = backend.index_map[analysis]
            global_jac = np.atleast_2d(constraint.jac(pars))
            local_jac = np.atleast_2d(local.jac(pars[index_map]))
            assert global_jac.shape == (local_jac.shape[0], backend.npar)
            assert np.allclose(global_jac[:, index_map], local_jac)
            # every column outside the model's slots has to be zero
            outside = [i for i in range(backend.npar) if i not in set(index_map)]
            assert np.allclose(global_jac[:, outside], 0.0)

    def test_constraints_reach_the_optimiser(self):
        first, second = self._models()
        model = statistical_model_wrapper(CorrelatedStatisticsCombiner)(
            statistical_models=[first, second],
            shared_parameters=["mu"],
            analysis="PP",
        )
        assert len(model.prepare_for_fit()["constraints"]) == 2
        _, nll = model.maximize_likelihood(return_nll=True)
        assert np.isfinite(nll)

    def test_unsupported_constraint_type(self):
        model_a, model_b, _ = build_models()
        model_a.backend.constraints = ["not a scipy constraint"]
        with pytest.raises(InvalidInput, match="can not be mapped"):
            CorrelatedStatisticsCombiner([model_a, model_b], ["mu"])


# ---------------------------------------------------------------------------
# Foreign autodiff frameworks
# ---------------------------------------------------------------------------
class _TorchLikeTensor:
    """
    Stand-in for a ``pytorch`` tensor that still carries a gradient tape.

    Conversion requires ``detach()`` then ``cpu()`` then ``numpy()``; a direct
    :func:`numpy.asarray` raises, exactly as it does for a real tensor with
    ``requires_grad=True``.
    """

    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32)

    def __array__(self, dtype=None):
        raise RuntimeError("Can't call numpy() on Tensor that requires grad")

    def __float__(self):
        return float(self._value.reshape(()))

    def detach(self):
        """Return a tape-free view, still on the (pretend) device."""
        return _TorchLikeDetached(self._value)


class _TorchLikeDetached:
    """Detached ``pytorch`` tensor: still needs ``cpu()`` before conversion."""

    def __init__(self, value):
        self._value = value

    def __array__(self, dtype=None):
        raise RuntimeError("Tensor lives on a device")

    def cpu(self):
        """Move the tensor to host memory."""
        return _TorchLikeHost(self._value)


class _TorchLikeHost:
    """Host-resident ``pytorch`` tensor exposing ``numpy()``."""

    def __init__(self, value):
        self._value = value

    def numpy(self):
        """Zero-copy conversion to :obj:`numpy.ndarray`."""
        return self._value


class _TensorFlowLikeTensor:
    """Stand-in for an eager ``tensorflow`` tensor: only ``numpy()``, no ``float()``."""

    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32)

    def numpy(self):
        """Convert to :obj:`numpy.ndarray`."""
        return self._value


class _JaxLikeArray:
    """Stand-in for a ``jax`` array: convertible only through ``__array__``."""

    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32)

    def __array__(self, dtype=None):
        return self._value if dtype is None else self._value.astype(dtype)


class _ForeignFrameworkBackend(BackendBase):
    r"""
    Gaussian backend whose derivatives are returned as foreign-framework objects.

    The likelihood is the same :math:`\mathcal{N}(n \mid \mu s(c) + b, \sigma)` used by
    the analytic reference, but every value, gradient and Hessian is wrapped in a
    ``wrapper`` type that mimics ``jax`` / ``tensorflow`` / ``pytorch`` return values.
    """

    name = "test.foreign_framework"
    version = "0.0.1"
    author = "test"
    spey_requires = spey.__version__

    def __init__(self, derivs, background, sigma, data, npar_local, wrapper):
        self._derivs = derivs
        self._background = background
        self._sigma = sigma
        self._data = np.asarray(data, dtype=float)
        self._npar_local = npar_local
        self._wrapper = wrapper

    def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
        """Local configuration: ``mu`` followed by the model's own coefficients."""
        return ModelConfig(
            poi_index=0,
            minimum_poi=-np.inf,
            suggested_init=[1.0] * self._npar_local,
            suggested_bounds=[(-10.0, poi_upper_bound)]
            + [(None, None)] * (self._npar_local - 1),
            parameter_names=["mu"] + [f"coeff_{i}" for i in range(self._npar_local - 1)],
        )

    def _blocks(self, pars, data):
        """Value, gradient and Hessian in *local* parameter indices."""
        lam, dlam, d2lam = self._derivs(*pars)
        value, grad, hess = _gaussian_block(lam, dlam, d2lam, data, self._sigma)
        return value, grad, hess

    def get_logpdf_func(self, expected=ExpectationType.observed, data=None):
        """Log-pdf wrapped in the foreign-framework scalar type."""
        current = self._background if expected == ExpectationType.apriori else self._data
        current = current if data is None else np.asarray(data, dtype=float)
        return lambda pars: self._wrapper(self._blocks(pars, current)[0])

    def get_objective_function(
        self, expected=ExpectationType.observed, data=None, do_grad=True
    ):
        """Objective (and gradient) wrapped in the foreign-framework types."""
        current = self._background if expected == ExpectationType.apriori else self._data
        current = current if data is None else np.asarray(data, dtype=float)
        if not do_grad:
            return lambda pars: self._wrapper(-self._blocks(pars, current)[0])

        def objective(pars):
            value, grad, _ = self._blocks(pars, current)
            return self._wrapper(-value), self._wrapper(-grad)

        return objective

    def get_hessian_logpdf_func(self, expected=ExpectationType.observed, data=None):
        """Hessian wrapped in the foreign-framework array type."""
        current = self._background if expected == ExpectationType.apriori else self._data
        current = current if data is None else np.asarray(data, dtype=float)
        return lambda pars: self._wrapper(self._blocks(pars, current)[2])

    def expected_data(self, pars, **kwargs):
        """Expected yields wrapped in the foreign-framework array type."""
        return self._wrapper(self._derivs(*pars)[0])


def _local_derivs_a(mu, c1, c2):
    """SR_A derivatives restricted to its own three local parameters."""
    lam, grad, hess = _derivs_a(mu, c1, c2)
    keep = np.array([0, 1, 2])
    return lam, grad[keep], hess[np.ix_(keep, keep)]


def _local_derivs_b(mu, c1, c3):
    """SR_B derivatives restricted to its own three local parameters."""
    lam, grad, hess = _derivs_b(mu, c1, c3)
    keep = np.array([0, 1, 3])
    return lam, grad[keep], hess[np.ix_(keep, keep)]


def _local_derivs_c(mu, c2):
    """SR_C derivatives restricted to its own two local parameters."""
    lam, grad, hess = _derivs_c(mu, c2)
    keep = np.array([0, 2])
    return lam, grad[keep], hess[np.ix_(keep, keep)]


class TestForeignAutodiffFrameworks:
    """
    Backends may compute derivatives with ``jax``, ``tensorflow`` or ``pytorch``.

    The combiner must consume whatever those frameworks return without ever tracing
    through them, and must keep the combined log-pdf differentiable by
    :mod:`autograd` regardless.
    """

    @staticmethod
    def _combiner(wrapper_a, wrapper_b, wrapper_c):
        models = [
            StatisticalModel(
                backend=_ForeignFrameworkBackend(
                    _local_derivs_a, BKG_A, SIG_A, OBS_A, 3, wrapper_a
                ),
                analysis="SR_A",
            ),
            StatisticalModel(
                backend=_ForeignFrameworkBackend(
                    _local_derivs_b, BKG_B, SIG_B, OBS_B, 3, wrapper_b
                ),
                analysis="SR_B",
            ),
            StatisticalModel(
                backend=_ForeignFrameworkBackend(
                    _local_derivs_c, BKG_C, SIG_C, OBS_C, 2, wrapper_c
                ),
                analysis="SR_C",
            ),
        ]
        return CorrelatedStatisticsCombiner(
            models,
            [
                "mu",
                {"name": "c1", "members": {"SR_A": 1, "SR_B": 1}},
                {"name": "c2", "members": {"SR_A": 2, "SR_C": 1}},
            ],
        )

    @pytest.mark.parametrize(
        "wrapper",
        [_JaxLikeArray, _TensorFlowLikeTensor, _TorchLikeTensor],
        ids=["jax-like", "tensorflow-like", "pytorch-like"],
    )
    def test_single_framework(self, wrapper):
        backend = self._combiner(wrapper, wrapper, wrapper)
        assert backend.is_differentiable and backend.is_hessian_available
        assert np.array_equal(backend.index_map["SR_B"], IDX_B)

        for pars in TEST_POINTS:
            value, grad, hess = analytic(pars)
            # float32 inputs: compare at single precision tolerance
            assert backend.get_logpdf_func()(pars) == pytest.approx(value, rel=1e-5)
            computed_value, computed_grad = backend.get_objective_function(do_grad=True)(
                pars
            )
            assert computed_value == pytest.approx(-value, rel=1e-5)
            assert np.allclose(-computed_grad, grad, rtol=1e-4, atol=1e-4)
            assert np.allclose(
                backend.get_hessian_logpdf_func()(pars), hess, rtol=1e-4, atol=1e-4
            )
            assert np.allclose(
                backend.expected_data(pars), analytic_expected_data(pars), rtol=1e-5
            )

    def test_mixed_frameworks(self):
        """A combination may mix frameworks — and spey's own autograd backends."""
        backend = self._combiner(_JaxLikeArray, _TorchLikeTensor, _TensorFlowLikeTensor)
        pars = TEST_POINTS[1]
        value, grad, hess = analytic(pars)
        computed_value, computed_grad = backend.get_objective_function(do_grad=True)(pars)
        assert computed_value == pytest.approx(-value, rel=1e-5)
        assert np.allclose(-computed_grad, grad, rtol=1e-4, atol=1e-4)
        assert np.allclose(backend.get_hessian_logpdf_func()(pars), hess, rtol=1e-4)

    def test_returns_are_float64_numpy(self):
        """Framework objects must never leak out of the combiner."""
        backend = self._combiner(_JaxLikeArray, _JaxLikeArray, _JaxLikeArray)
        pars = TEST_POINTS[0]
        _, grad = backend.get_objective_function(do_grad=True)(pars)
        hess = backend.get_hessian_logpdf_func()(pars)
        assert isinstance(grad, np.ndarray) and grad.dtype == np.float64
        assert isinstance(hess, np.ndarray) and hess.dtype == np.float64
        assert isinstance(backend.get_logpdf_func()(pars), float)

    def test_autograd_still_differentiates_the_combination(self):
        """`find_contour`'s access pattern works on non-autograd backends."""
        autograd = pytest.importorskip("autograd")
        anp = pytest.importorskip("autograd.numpy")

        backend = self._combiner(_JaxLikeArray, _TensorFlowLikeTensor, _TorchLikeTensor)
        logpdf = backend.get_logpdf_func()

        def nll(signal_pars):
            return -logpdf(anp.concatenate([anp.array([1.0]), signal_pars]))

        point = np.array([-0.4, 0.9, 1.3])
        _, expected_grad, expected_hess = analytic(np.concatenate([[1.0], point]))
        assert np.allclose(autograd.grad(nll)(point), -expected_grad[1:], rtol=1e-4)
        assert np.allclose(
            autograd.hessian(nll)(point), -expected_hess[1:, 1:], rtol=1e-4, atol=1e-4
        )

    def test_fit_converges_with_foreign_gradients(self):
        backend = self._combiner(_JaxLikeArray, _JaxLikeArray, _JaxLikeArray)
        model = StatisticalModel(backend=backend, analysis="foreign")
        _, nll = model.maximize_likelihood(return_nll=True)
        assert np.isfinite(nll)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------
class _LogPdfOnlyBackend(BackendBase):
    """Minimal backend: a log-pdf, nothing else — no gradient, Hessian or sampler."""

    name = "test.logpdf_only"
    version = "0.0.1"
    author = "test"
    spey_requires = spey.__version__

    def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
        """Two parameters: ``mu`` and one nuisance."""
        return ModelConfig(
            poi_index=0,
            minimum_poi=-10.0,
            suggested_init=[1.0, 0.0],
            suggested_bounds=[(-10.0, poi_upper_bound), (None, None)],
            parameter_names=["mu", "theta"],
        )

    def get_logpdf_func(self, expected=ExpectationType.observed, data=None):
        """A simple, smooth, strictly concave log-pdf."""
        return lambda pars: -0.5 * (pars[0] - 0.5) ** 2 - 0.5 * pars[1] ** 2


class TestDegradedCapabilities:
    @staticmethod
    def _mixed():
        model_a, _, _ = build_models()
        poor = StatisticalModel(backend=_LogPdfOnlyBackend(), analysis="poor")
        return CorrelatedStatisticsCombiner([model_a, poor], ["mu"])

    def test_capability_flags(self):
        backend = self._mixed()
        assert not backend.is_differentiable
        assert not backend.is_hessian_available

    def test_gradient_raises_with_informative_message(self):
        backend = self._mixed()
        with pytest.raises(NotImplementedError, match="poor"):
            backend.get_objective_function(do_grad=True)

    def test_hessian_raises_with_informative_message(self):
        backend = self._mixed()
        with pytest.raises(NotImplementedError, match="poor"):
            backend.get_hessian_logpdf_func()

    def test_gradient_free_objective_still_works(self):
        backend = self._mixed()
        objective = backend.get_objective_function(do_grad=False)
        pars = np.array([1.0, 0.3, -0.2, 0.4])
        model_a, _, _ = build_models()
        manual = -float(
            model_a.backend.get_logpdf_func()(pars[backend.index_map["SR_A"]])
        ) - float(_LogPdfOnlyBackend().get_logpdf_func()(pars[backend.index_map["poor"]]))
        assert objective(pars) == pytest.approx(manual, rel=1e-12)

    def test_fit_falls_back_to_the_numerical_optimiser(self):
        model = StatisticalModel(backend=self._mixed(), analysis="mixed")
        fit_options = model.prepare_for_fit()
        assert fit_options["do_grad"] is False
        _, nll = model.maximize_likelihood(return_nll=True)
        assert np.isfinite(nll)

    def test_expected_data_propagates_not_implemented(self):
        backend = self._mixed()
        with pytest.raises(NotImplementedError):
            backend.expected_data(np.array([1.0, 0.3, -0.2, 0.4]))

    def test_sampler_propagates_not_implemented(self):
        backend = self._mixed()
        with pytest.raises(NotImplementedError):
            backend.get_sampler(np.array([1.0, 0.3, -0.2, 0.4]))


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
class TestDataHandling:
    def test_expected_data_round_trips_through_logpdf(self):
        backend = build_combiner()
        asimov = backend.expected_data(np.array([0.0, 1.0, 1.0, 1.0]))
        logpdf = backend.get_logpdf_func(data=asimov)
        for pars in TEST_POINTS:
            assert logpdf(pars) == pytest.approx(
                analytic(pars, data=asimov)[0], rel=1e-12
            )

    def test_gradient_and_hessian_with_explicit_data(self):
        backend = build_combiner()
        asimov = backend.expected_data(np.array([0.0, 1.0, 1.0, 1.0]))
        pars = TEST_POINTS[1]
        _, expected_grad, expected_hess = analytic(pars, data=asimov)
        _, grad = backend.get_objective_function(data=asimov, do_grad=True)(pars)
        assert np.allclose(-grad, expected_grad, atol=1e-9)
        assert np.allclose(
            backend.get_hessian_logpdf_func(data=asimov)(pars), expected_hess, atol=1e-9
        )

    def test_wrong_data_length(self):
        backend = build_combiner()
        with pytest.raises(InvalidInput, match="combined dataset of length 7"):
            backend.get_logpdf_func(data=np.ones(5))

    def test_sampler_shape_and_layout(self):
        backend = build_combiner()
        sampler = backend.get_sampler(np.array([1.0, 1.0, 1.0, 1.0]))
        sample = sampler(16)
        assert sample.shape == (16, len(BKG_A) + len(BKG_B) + len(BKG_C))
        # a sample must be usable as data, i.e. share expected_data's layout
        assert np.isfinite(backend.get_logpdf_func(data=sample[0])(TEST_POINTS[0]))

    def test_asimov_data_via_statistical_model(self):
        model = build_combined_model()
        asimov = model.generate_asimov_data()
        assert len(asimov) == 7
        # mu = 0 for 'qtilde' => the Asimov dataset is the background prediction
        assert np.allclose(asimov, np.concatenate([BKG_A, BKG_B, BKG_C]))


class TestStatisticalModelIntegration:
    def test_hypothesis_testing_runs(self):
        model = build_combined_model()
        assert np.isfinite(model.likelihood(1.0))
        assert np.isfinite(model.chi2(1.0))
        assert 0.0 <= model.exclusion_confidence_level()[0] <= 1.0

    def test_likelihood_with_named_multi_poi(self):
        model = build_combined_model()
        fixed = model.likelihood({"mu": 1.0, "c1": 0.5, "c2": 0.5}, return_nll=True)
        # fixing three of the four parameters can not beat the global optimum
        assert fixed >= model.maximize_likelihood(return_nll=True)[1]

    def test_sigma_mu_from_hessian(self):
        r"""
        The combined Hessian feeds :func:`~spey.StatisticalModel.sigma_mu_from_hessian`.

        Evaluated at :math:`\hat\mu`, where the observed information is positive
        definite.  Away from the optimum this strongly non-linear signal model has an
        indefinite information matrix and ``spey`` returns ``nan`` — that happens for
        the *constituent* models on their own too, so it is not a property of the
        combination.
        """
        model = build_combined_model()
        muhat, _ = model.maximize_likelihood()
        sigma_mu = model.sigma_mu_from_hessian(muhat)
        assert np.isfinite(sigma_mu) and sigma_mu > 0.0

    def test_sigma_mu_with_shared_nuisance(self):
        """A near-linear combination gives a finite ``sigma_mu`` at any ``mu``."""
        first, second = TestSharedNuisanceParameters._models()
        model = statistical_model_wrapper(CorrelatedStatisticsCombiner)(
            statistical_models=[first, second],
            shared_parameters=["mu", {"name": "theta", "members": {"A": 1, "B": 1}}],
            analysis="AB",
        )
        assert np.isfinite(model.sigma_mu_from_hessian(1.0))

    def test_find_contour(self):
        """The combined model is accepted by the multi-parameter contour finder."""
        find_contour = pytest.importorskip("spey.multiparameter").find_contour
        model = build_combined_model()
        result = find_contour(
            model,
            confidence_level=0.95,
            n_radial=40,
            n_hmc_chains=2,
            n_hmc_steps=20,
            n_gap_candidates=100,
            random_seed=1,
            n_jobs=1,
        )
        assert result.dof == 3
        assert result.parameter_names == ["c1", "c2", "signal_par_1"]
        assert len(result.contour_points) > 0

        logpdf = model.backend.get_logpdf_func()
        residuals = np.array(
            [
                -float(logpdf(np.concatenate([[1.0], point]))) - result.threshold
                for point in result.contour_points
            ]
        )
        assert np.abs(residuals).max() < 1e-5


class TestPluginRegistration:
    def test_metadata(self):
        assert CorrelatedStatisticsCombiner.name == "default.correlated_combiner"
        assert issubclass(CorrelatedStatisticsCombiner, BackendBase)
        assert not getattr(CorrelatedStatisticsCombiner, "__abstractmethods__", False)

    def test_wrapper_produces_statistical_model(self):
        assert isinstance(build_combined_model(), StatisticalModel)

    def test_entry_point_is_declared(self):
        """Discoverable through ``spey.get_backend`` once the package is installed."""
        if "default.correlated_combiner" not in spey.AvailableBackends():
            pytest.skip(
                "entry points are stale in this environment; reinstall with `pip install -e .`"
            )
        wrapper = spey.get_backend("default.correlated_combiner")
        model = wrapper(
            statistical_models=list(build_models()),
            shared_parameters=SHARED,
            analysis="combined",
        )
        assert isinstance(model, StatisticalModel)
        assert model.backend.parameter_names == ["mu", "c1", "c2", "signal_par_1"]


class TestCapabilityReporting:
    """The combination must advertise only the calculators it can really run."""

    def test_full_combination_offers_every_calculator(self):
        model = build_combined_model()
        assert model.is_asymptotic_calculator_available
        assert model.is_toy_calculator_available
        assert sorted(model.available_calculators) == [
            "asymptotic",
            "chi_square",
            "toy",
        ]

    def test_degraded_combination_offers_only_chi_square(self):
        """One incapable constituent disables Asimov data and sampling for all."""
        model = StatisticalModel(
            backend=TestDegradedCapabilities._mixed(), analysis="mixed"
        )
        assert model.is_asymptotic_calculator_available is False
        assert model.is_toy_calculator_available is False
        assert model.available_calculators == ["chi_square"]
