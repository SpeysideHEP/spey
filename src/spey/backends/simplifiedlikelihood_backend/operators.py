"""
This file contains necessary functions to compute negative log-likelihood for simplified likelihoods

for details see https://arxiv.org/abs/1809.05548
"""
from typing import Optional, Callable, Tuple, Union, Any

from autograd import numpy as np
from autograd import grad, hessian
from .distributions import Poisson

from .sldata import expansion_output

__all__ = [
    "logpdf",
    "twice_nll_func",
    "hessian_logpdf_func",
    "objective_wrapper",
]

# pylint: disable=E1101
Distribution = Any


def logpdf(
    pars: np.ndarray,
    signal: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    third_moment_expansion: expansion_output,
    gaussian: Distribution,
) -> float:
    """
    Compute log-probability of the simplified likelihood

    Args:
        pars (``np.ndarray``): nuisance parameters
        signal (``np.ndarray``): signal yields
        background (``np.ndarray``): background yields
        observed (``np.ndarray``): data
        third_moment_expansion (``expansion_output``): third moment expansion
        gaussian (``Distribution``): Normal or Multivariate normal distribution

    Returns:
        ``float``:
        log-probability value
    """
    if third_moment_expansion.A is None:
        lam = background + pars[1:] + pars[0] * signal
    else:
        lam = (
            pars[0] * signal
            + third_moment_expansion.A
            + pars[1:]
            + third_moment_expansion.C
            * np.square(pars[1:])
            / np.square(third_moment_expansion.B)
        )
    lam = np.clip(lam, 1e-5, None)

    poisson = Poisson(lam)

    return gaussian.log_prob(pars[1:]) + poisson.log_prob(observed)


def twice_nll_func(
    signal: np.ndarray,
    background: np.ndarray,
    third_moment_expansion: expansion_output,
    gaussian: Distribution,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    r"""
    Retreive twice negative log-likelihood function

    Args:
        signal (``np.ndarray``): signal yields
        background (``np.ndarray``): background yields
        third_moment_expansion (``expansion_output``): third moment expansion
        gaussian (``Distribution``): Normal or Multivariate normal distribution

    Returns:
        ``Callable[[np.ndarray, np.ndarray], np.ndarray]``:
        Function that takes POI and nuisance parameters, :math:`\mu` and :math:`\theta`
        to compute twice negative log-likelihood
    """

    def twice_nll(pars: np.ndarray, data: np.ndarray) -> float:
        """
        Compute twice negative log likelihood

        :param pars (`np.ndarray`): poi and nuisance parameters
        :param data (`np.ndarray`): observations
        :return `np.ndarray`: twice negative log likelihood value
        """
        return -2.0 * logpdf(
            pars, signal, background, data, third_moment_expansion, gaussian
        )

    return twice_nll


def hessian_logpdf_func(
    signal: np.ndarray,
    background: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
    gaussian: Distribution,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Function to generate a callable for the hessian of twice negative log-likelihood

    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): background yields
    :param third_moment_expansion (`Optional[expansion_output]`): computed third momenta expansion
    :return `Callable[[np.ndarray], np.ndarray]`: function to compute hessian of twice negative log-likelihood
    """

    def func(pars: np.ndarray, data: np.ndarray) -> np.ndarray:
        return logpdf(pars, signal, background, data, third_moment_expansion, gaussian)

    # pylint: disable=E1120
    return hessian(func, argnum=0)


def objective_wrapper(
    signal: np.ndarray,
    background: np.ndarray,
    data: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
    gaussian: Distribution,
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
    twice_nll = twice_nll_func(signal, background, third_moment_expansion, gaussian)
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
