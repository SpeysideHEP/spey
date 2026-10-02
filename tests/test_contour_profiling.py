"""
Tests for profiling nuisance parameters inside :func:`spey.multiparameter.find_contour`.

When ``poi_indices`` selects a subset of the parameters, every other non-``mu``
parameter is minimised at each NLL evaluation.  Backends hand their optimiser
constraints over either as scipy-style dicts or as
:class:`scipy.optimize.NonlinearConstraint` objects; both must be restricted to the
profile subspace.
"""

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize_scalar

import spey
from spey.multiparameter import find_contour
from spey.multiparameter.contour import _restrict_constraint

# Full parameter vector [mu, c1, c2, theta]; contour (c1, c2) fixed, theta profiled.
_CP = np.array([0.3, -0.7])
_PROFILE = np.array([3])


def _embed(pp):
    return np.concatenate([[1.0], _CP, pp])


def _fun(pars):
    return np.array([pars[0] + 2.0 * pars[1] - pars[2] + 4.0 * pars[3]])


def _jac(pars):
    return np.array([[1.0, 2.0, -1.0, 4.0]])


class TestRestrictConstraint:
    def test_dict_constraint(self):
        restricted = _restrict_constraint(
            {"type": "ineq", "fun": _fun, "jac": _jac}, _embed, _PROFILE
        )
        pp = np.array([0.25])
        assert restricted["type"] == "ineq"
        assert np.allclose(restricted["fun"](pp), _fun(_embed(pp)))
        assert np.allclose(restricted["jac"](pp), [[4.0]])

    def test_nonlinear_constraint(self):
        restricted = _restrict_constraint(
            NonlinearConstraint(_fun, 0.0, np.inf, jac=_jac), _embed, _PROFILE
        )
        pp = np.array([0.25])
        assert isinstance(restricted, NonlinearConstraint)
        assert restricted.lb == 0.0 and restricted.ub == np.inf
        assert np.allclose(restricted.fun(pp), _fun(_embed(pp)))
        assert np.allclose(restricted.jac(pp), [[4.0]])

    def test_finite_difference_jacobian_is_kept(self):
        restricted = _restrict_constraint(
            NonlinearConstraint(_fun, 0.0, np.inf), _embed, _PROFILE
        )
        assert restricted.jac == "2-point"

    def test_unsupported_constraint_raises(self):
        with pytest.raises(TypeError, match="LinearConstraint"):
            _restrict_constraint(
                LinearConstraint(np.ones((1, 4)), 0.0, 1.0), _embed, _PROFILE
            )


def test_find_contour_profiles_nonlinear_constraints():
    """A backend with NonlinearConstraint objects can be profiled over."""
    coefficients = np.array([2.5]), np.array([3.7])

    def signal(pars):
        return coefficients[0] * pars[0] ** 2 + coefficients[1] * pars[1] ** 2

    stat_model = spey.get_backend("default.uncorrelated_background")(
        signal_yields=signal,
        background_yields=[10.6],
        data=[6],
        absolute_uncertainties=[4.8],
        n_signal_parameters=2,
        analysis="profiled",
    )
    assert stat_model.backend.config().parameter_names == [
        "mu",
        "signal_par_0",
        "signal_par_1",
        "theta_bkg_0",
    ]
    assert stat_model.backend.constraints and all(
        isinstance(c, NonlinearConstraint) for c in stat_model.backend.constraints
    )

    result = find_contour(
        stat_model,
        poi_indices=[1, 2],  # theta_bkg_0 is profiled
        n_radial=40,
        n_hmc_chains=1,
        n_hmc_steps=20,
        n_gap_candidates=200,
        n_multistart=1,
        n_coverage_passes=1,
        random_seed=7,
        n_jobs=1,
    )
    assert result.dof == 2
    assert result.parameter_names == ["signal_par_0", "signal_par_1"]
    assert np.allclose(result.theta_mle, 0.0, atol=1e-3)

    # Every boundary point must sit on the *profiled* threshold.
    logpdf = stat_model.backend.get_logpdf_func()
    for point in result.contour_points[result.from_radial][:10]:
        profiled_nll = minimize_scalar(
            lambda theta: -logpdf(np.array([1.0, *point, theta])),
            bounds=(-5.0, 5.0),
            method="bounded",
            options={"xatol": 1e-10},
        ).fun
        assert profiled_nll == pytest.approx(result.threshold, abs=1e-4)
