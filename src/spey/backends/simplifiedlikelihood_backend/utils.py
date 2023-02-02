"""
This file contains necessary functions to compute negative log-likelihood for a given theta

compute_negloglikelihood_theta:     compute the negative log-likelihood for a given theta
_common_gradient_computation:       function to compute commonalities between first and
                                    second gradient
compute_dnegloglikelihood_dtheta:   compute first order derivative of negative log-likelihood
                                    for a given theta
compute_d2negloglikelihood_dtheta2: compute second order derivative of negative log-likelihood
                                    for a given theta

for details see https://arxiv.org/abs/1809.05548
"""
import scipy, warnings
from typing import Optional, Tuple, List

from .data import Data, expansion_output

from autograd.scipy.stats.poisson import logpmf
from autograd.scipy.special import gammaln
from autograd import numpy as np
from autograd import grad, hessian
from scipy.optimize import minimize

__all__ = ["twice_nll", "fit", "compute_sigma_mu"]


def twice_nll(
    mu: np.ndarray,
    theta: np.ndarray,
    signal: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
) -> float:
    """
    Compute the log value of the full density.

    :param mu: POI (signal strength)
    :param model: statistical model
    :param theta: nuisance parameters
    :param third_moment_expansion: computed results for the third moment
           expansion of the statistical model
    :return: (float) negative log-likelihood
    """

    if third_moment_expansion.A is None:
        lmbda = background + mu * signal + theta
    else:
        lmbda = (
            mu * signal
            + third_moment_expansion.A
            + theta
            + third_moment_expansion.C * np.square(theta) / np.square(third_moment_expansion.B)
        )
    lmbda = np.clip(lmbda, 1e-5, None)

    # scipy.stats.poisson.logpmf is faster than computing by hand
    if observed.dtype in [np.int32, np.int16, np.int64]:
        poisson = logpmf(observed, lmbda)
    else:
        poisson = -lmbda + observed * np.log(lmbda) - gammaln(observed + 1)

    logcoeff = (
        -len(observed) / 2.0 * np.log(2.0 * np.pi)
        - 0.5 * third_moment_expansion.logdet_covariance[1]
    )
    gaussian = -0.5 * np.dot(np.dot(theta, third_moment_expansion.inv_covariance), theta) + logcoeff

    return -2.0 * (gaussian + np.sum(poisson))


def _combined_nuisance(pars, signal, background, observed, third_moment_expansion):
    return twice_nll(pars[0], pars[1:], signal, background, observed, third_moment_expansion)


_dtwice_nll = grad(twice_nll, argnum=[0, 1])

_twice_nll_hessian = hessian(_combined_nuisance, argnum=0)


def compute_sigma_mu(
    model: Data, pars: np.ndarray, third_moment_expansion: Optional[expansion_output] = None
) -> float:
    """
    Compute uncertainty on parameter of interest

    :param model: description of the statistical model
    :param pars: nuisance parameters
    :param third_moment_expansion: computed results for the third moment
                                    expansion of the statistical model
    :return: sigma mu
    """

    if third_moment_expansion is None:
        third_moment_expansion = model.compute_expansion()

    hessian = _twice_nll_hessian(
        pars, model.signal, model.background, model.observed, third_moment_expansion
    )
    return np.clip(np.sqrt(1.0 / hessian[0, 0]), 1e-5, None)


def fit(
    model: Data,
    init_pars: List[float],
    par_bounds: List[Tuple[float, float]],
    fixed_poi: Optional[float] = None,
    third_moment_expansion: Optional[expansion_output] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute minimum of -logpdf

    :param model: description of the statistical model
    :param init_pars: initial parameters
    :param par_bounds: parameter bounds
    :param fixed_poi: if a value is given, fixed poi optimisation will be performed
    :param third_moment_expansion: computed results for the third moment
                                    expansion of the statistical model
    :return: -logpdf and fit parameters
    """
    assert len(init_pars) == len(model) + 1, (
        "Dimensionality of initialization parameters does "
        "not match the dimensionality of the model."
    )

    if third_moment_expansion is None:
        third_moment_expansion = model.compute_expansion()

    twice_nll_func = lambda pars: twice_nll(
        pars[0], pars[1:], model.signal, model.background, model.observed, third_moment_expansion
    )

    grad_func = lambda pars: np.hstack(
        _dtwice_nll(
            pars[0],
            pars[1:],
            model.signal,
            model.background,
            model.observed,
            third_moment_expansion,
        )
    )

    constraints = None
    if fixed_poi is not None:
        init_pars[0] = fixed_poi
        constraints = [{"type": "eq", "fun": lambda v: v[0] - fixed_poi}]
        assert (
            par_bounds[0][0] <= fixed_poi <= par_bounds[0][1]
        ), "POI is beyond the requested limits."

    opt = minimize(
        twice_nll_func,
        np.array(init_pars),
        method="SLSQP",
        jac=grad_func,
        bounds=par_bounds,
        constraints=constraints,
        tol=1e-6,
        options={"maxiter": 10000},
    )

    if not opt.success:
        warnings.warn(message=opt.message, category=RuntimeWarning)

    return opt.fun / 2.0, opt.x
