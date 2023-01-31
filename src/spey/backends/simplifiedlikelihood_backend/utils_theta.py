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
from typing import Optional, Tuple

import numpy as np

from .data import Data, expansion_output

__all__ = [
    "minus_logpdf",
    "compute_dnegloglikelihood_dtheta",
    "compute_d2negloglikelihood_dtheta2",
    "fixed_poi_fit",
]


def minus_logpdf(
    mu: float,
    model: Data,
    theta: np.ndarray,
    third_moment_expansion: Optional[expansion_output] = None,
) -> float:
    """
    Compute likelihood of the statistical model with respect to given theta at a POI

    :param mu: POI (signal strength)
    :param model: statistical model
    :param theta: nuisance parameters
    :param third_moment_expansion: computed results for the third moment
           expansion of the statistical model
    :return: (float) negative log-likelihood
    """
    signal_yields = mu * model

    if third_moment_expansion is None:
        third_moment_expansion = model.compute_expansion()

    if model.isLinear:
        lmbda = model.background + signal_yields + theta
    else:
        lmbda = (
            signal_yields
            + third_moment_expansion.A
            + theta
            + third_moment_expansion.C * np.square(theta) / np.square(third_moment_expansion.B)
        )
    lmbda = np.clip(lmbda, 1e-30, None)

    # scipy.stats.poisson.logpmf is faster than computing by hand
    if model.observed.dtype in [np.int32, np.int16, np.int64]:
        poisson = scipy.stats.poisson.logpmf(model.observed, lmbda)
    else:
        poisson = (
            -lmbda + model.observed * np.log(lmbda) - scipy.special.loggamma(model.observed + 1)
        )

    logcoeff = (
        -len(model) / 2.0 * np.log(2.0 * np.pi) - 0.5 * third_moment_expansion.logdet_covariance[1]
    )
    gaussian = -0.5 * np.dot(np.dot(theta, third_moment_expansion.inv_covariance), theta) + logcoeff

    return -gaussian - np.sum(poisson)


def _common_gradient_computation(
    mu: float,
    model: Data,
    theta: np.ndarray,
    third_moment_expansion: expansion_output,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute the commonalities in the gradient computation. The purpose of this function
    is to simplify the gradient computation given below.

    :param mu: POI (signal strength)
    :param model: statistical model
    :param theta: nuisance parameters
    :param third_moment_expansion: computed results for the third moment
           expansion of the statistical model
    """
    signal_yields = mu * model

    if model.isLinear:
        total_expected = np.clip(theta + model.background + signal_yields, 1e-30, None)
        return total_expected, None

    lmbda = (
        signal_yields
        + third_moment_expansion.A
        + theta
        + third_moment_expansion.C * np.square(theta) / np.square(third_moment_expansion.B)
    )
    lmbda = np.where(lmbda <= 0.0, 1e-30, lmbda)
    # TODO what is T here???
    T = 1.0 + 2.0 * third_moment_expansion.C / np.square(third_moment_expansion.B) * theta
    return T, lmbda


def compute_dnegloglikelihood_dtheta(
    mu: float,
    model: Data,
    theta: np.ndarray,
    third_moment_expansion: Optional[expansion_output] = None,
) -> float:
    """
    Compute the first derivative of the negloglikelihood with respect to theta

    :param mu: POI (signal strength)
    :param model: statistical model
    :param theta: nuisance parameters
    :param third_moment_expansion: computed results for the third moment
           expansion of the statistical model
    :return: (float) derivative of negative log-likelihood
    """
    if third_moment_expansion is None:
        third_moment_expansion = model.compute_expansion()

    result, lmbda = _common_gradient_computation(mu, model, theta, third_moment_expansion)

    if model.isLinear:
        return 1.0 - model.observed / result + np.dot(theta, third_moment_expansion.inv_covariance)

    return (
        result
        - model.observed / lmbda * result
        + np.dot(theta, third_moment_expansion.inv_covariance)
    )


def compute_d2negloglikelihood_dtheta2(
    mu: float,
    model: Data,
    theta: np.ndarray,
    third_moment_expansion: Optional[expansion_output] = None,
) -> float:
    """
    Compute the second derivative of the negloglikelihood with respect to theta

    :param mu: POI (signal strength)
    :param model: statistical model
    :param theta: nuisance parameters
    :param third_moment_expansion: computed results for the third moment
           expansion of the statistical model
    :return: (float) second derivative of negative log-likelihood
    """
    if third_moment_expansion is None:
        third_moment_expansion = model.compute_expansion()

    result, lmbda = _common_gradient_computation(mu, model, theta, third_moment_expansion)

    if model.isLinear:
        return third_moment_expansion.inv_covariance + np.diag(model.observed / np.square(result))

    return (
        third_moment_expansion.inv_covariance
        + np.diag(model.observed * np.square(result) / np.square(lmbda))
        - np.diag(
            model.observed
            / lmbda
            * 2.0
            * third_moment_expansion.C
            / np.square(third_moment_expansion.B)
        )
        + np.diag(2.0 * third_moment_expansion.C / np.square(third_moment_expansion.B))
    )


def fixed_poi_fit(
    mu: float,
    model: Data,
    third_moment_expansion: Optional[expansion_output] = None,
) -> Tuple[float, np.ndarray]:
    """
    compute minimum of negative log-likelihood with respect to nuisanse parameters

    :param mu: POI (signal strength)
    :param model: statistical model
    :param third_moment_expansion: computed results for the third moment
                                   expansion of the statistical model
    :return: negative log-likelihood, theta
    """

    if third_moment_expansion is None:
        third_moment_expansion = model.compute_expansion()

    initial_theta = model.suggested_theta_init(mu)

    nll_theta = lambda theta: minus_logpdf(
        mu = mu, model = model, theta = theta, third_moment_expansion = third_moment_expansion
        )
    dnll_dtheta = lambda theta: compute_dnegloglikelihood_dtheta(
        mu=mu, model=model, theta=theta, third_moment_expansion=third_moment_expansion
    )
    d2nll_dtheta2 = lambda theta: compute_d2negloglikelihood_dtheta2(
        mu=mu, model=model, theta=theta, third_moment_expansion=third_moment_expansion
    )

    res = scipy.optimize.fmin_ncg(
        f=nll_theta,
        x0=initial_theta,
        fprime=dnll_dtheta,
        fhess=d2nll_dtheta2,
        full_output=True,
        disp=0,
    )
    bounds = [[-10 * obs, 10 * obs] for obs in model.observed]

    x, nfeval, rc = scipy.optimize.fmin_tnc(
        func=nll_theta, x0=res[0], fprime=dnll_dtheta, disp=0, bounds=bounds
    )
    if not 0 <= rc <= 2:
        return_code = {
            -1: "Infeasible (lower bound > upper bound)",
            0: "Local minimum reached (|pg| ~= 0)",
            1: "Converged (|f_n-f_(n-1)| ~= 0)",
            2: "Converged (|x_n-x_(n-1)| ~= 0)",
            3: "Max. number of function evaluations reached",
            4: "Linear search failed",
            5: "All lower bounds are equal to the upper bounds",
            6: "Unable to progress",
            7: "User requested end of minimization",
        }
        warnings.warn(
            message=f"Can not converge within {nfeval} iterations. {return_code[rc]}",
            category=RuntimeWarning,
        )

    return nll_theta(x), x
