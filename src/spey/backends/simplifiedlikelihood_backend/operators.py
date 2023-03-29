"""
This file contains necessary functions to compute negative log-likelihood for simplified likelihoods

for details see https://arxiv.org/abs/1809.05548
"""
from typing import Optional, Callable, Tuple, Union

from autograd.scipy.stats.poisson import logpmf
from autograd.scipy.special import gammaln
from autograd import numpy as np
from autograd import grad, hessian

from .sldata import expansion_output

__all__ = [
    "logpdf",
    "twice_nll_func",
    "hessian_logpdf_func",
    "objective_wrapper",
]

# pylint: disable=E1101


def logpdf(
    pars: np.ndarray,
    signal: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    third_moment_expansion: expansion_output,
) -> float:
    """
    Compute twice negative log-likelihood

    :param pars: nuisance parameters
    :param signal: signal yields
    :param background: expected background yields
    :param observed: observations
    :param third_moment_expansion: third moment expansion
    :return: twice negative log-likelihood
    """
    if third_moment_expansion.A is None:
        lmbda = background + pars[1:] + pars[0] * signal
    else:
        lmbda = (
            pars[0] * signal
            + third_moment_expansion.A
            + pars[1:]
            + third_moment_expansion.C * np.square(pars[1:]) / np.square(third_moment_expansion.B)
        )
    lmbda = np.clip(lmbda, 1e-5, None)

    # scipy.stats.poisson.logpmf is faster than computing by hand
    if observed.dtype in [np.int32, np.int16, np.int64]:
        poisson = logpmf(observed, lmbda)
    else:
        poisson = -lmbda + observed * np.log(lmbda) - gammaln(observed + 1)

    # NOTE: autograd.scipy.stats.multivariate_normal.logpdf is too slow!!
    # logpdf(pars[1:], mean=np.zeros(len(observed)), cov=third_moment_expansion.V)
    logcoeff = (
        -len(observed) / 2.0 * np.log(2.0 * np.pi)
        - 0.5 * third_moment_expansion.logdet_covariance[1]
    )
    gaussian = -0.5 * (pars[1:] @ third_moment_expansion.inv_covariance @ pars[1:]) + logcoeff

    return (gaussian + np.sum(poisson)).astype(np.float64)


def twice_nll_func(
    signal: np.ndarray,
    background: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Function to generate a callable for twice negative log-likelihood

    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): background yields
    :param third_moment_expansion (`Optional[expansion_output]`): computed third momenta expansion
    :return `Callable[[np.ndarray], np.ndarray]`: function to compute twice negative log-likelihood
    """

    def twice_nll(pars: np.ndarray, data: np.ndarray) -> float:
        """
        Compute twice negative log likelihood

        :param pars (`np.ndarray`): poi and nuisance parameters
        :param data (`np.ndarray`): observations
        :return `np.ndarray`: twice negative log likelihood value
        """
        return -2.0 * logpdf(pars, signal, background, data, third_moment_expansion)

    return twice_nll


def hessian_logpdf_func(
    signal: np.ndarray,
    background: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Function to generate a callable for the hessian of twice negative log-likelihood

    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): background yields
    :param third_moment_expansion (`Optional[expansion_output]`): computed third momenta expansion
    :return `Callable[[np.ndarray], np.ndarray]`: function to compute hessian of twice negative log-likelihood
    """

    def func(pars: np.ndarray, data: np.ndarray) -> np.ndarray:
        return logpdf(pars, signal, background, data, third_moment_expansion)

    # pylint: disable=E1120
    return hessian(func, argnum=0)


def objective_wrapper(
    signal: np.ndarray,
    background: np.ndarray,
    data: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
    do_grad: bool,
) -> Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]:
    """
    Retreive objective function (twice negative log-likelihood) and its gradient

    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): background yields
    :param third_moment_expansion (`Optional[expansion_output]`): computed third momenta expansion
    :return `Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]`: function to compute objective
        function and its gradient
    """
    twice_nll = twice_nll_func(signal, background, third_moment_expansion)
    if do_grad:
        # pylint: disable=E1120
        grad_twice_nll = grad(twice_nll, argnum=0)

        def func(pars: np.ndarray) -> Tuple[float, np.ndarray]:
            pars = np.array(pars).astype(np.float64)
            return twice_nll(pars, data), grad_twice_nll(pars, data)

    else:

        def func(pars: np.ndarray) -> float:
            return twice_nll(pars, data)

    return func
