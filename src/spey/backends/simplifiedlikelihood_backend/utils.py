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
from typing import Optional, Callable

from autograd.scipy.stats.poisson import logpmf
from autograd.scipy.special import gammaln
from autograd import numpy as np
from autograd import grad, hessian

from .sldata import expansion_output

__all__ = ["twice_nll", "twice_nll_func", "gradient_twice_nll_func", "hessian_twice_nll_func"]

# pylint: disable=E1101


def twice_nll(
    pars: np.ndarray,
    signal: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
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

    return -2.0 * (gaussian + np.sum(poisson))


def twice_nll_func(
    signal: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Function to generate a callable for twice negative log-likelihood

    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): background yields
    :param observed (`np.ndarray`): observed yields
    :param third_moment_expansion (`Optional[expansion_output]`): computed third momenta expansion
    :return `Callable[[np.ndarray], np.ndarray]`: function to compute twice negative log-likelihood
    """
    return lambda pars: twice_nll(pars, signal, background, observed, third_moment_expansion)


def gradient_twice_nll_func(
    signal: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Function to generate a callable for the gradient of twice negative log-likelihood

    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): background yields
    :param observed (`np.ndarray`): observed yields
    :param third_moment_expansion (`Optional[expansion_output]`): computed third momenta expansion
    :return `Callable[[np.ndarray], np.ndarray]`: function to compute gradient of twice negative log-likelihood
    """
    # pylint: disable=E1120
    _grad = grad(twice_nll, argnum=0)
    return lambda pars: _grad(pars, signal, background, observed, third_moment_expansion)


def hessian_twice_nll_func(
    signal: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    third_moment_expansion: Optional[expansion_output],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Function to generate a callable for the hessian of twice negative log-likelihood

    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): background yields
    :param observed (`np.ndarray`): observed yields
    :param third_moment_expansion (`Optional[expansion_output]`): computed third momenta expansion
    :return `Callable[[np.ndarray], np.ndarray]`: function to compute hessian of twice negative log-likelihood
    """
    # pylint: disable=E1120
    _hessian = hessian(twice_nll, argnum=0)
    return lambda pars: _hessian(pars, signal, background, observed, third_moment_expansion)


# def fit(
#     model: SLData,
#     init_pars: List[float],
#     par_bounds: List[Tuple[float, float]],
#     fixed_poi: Optional[float] = None,
#     third_moment_expansion: Optional[expansion_output] = None,
# ) -> Tuple[float, np.ndarray]:
#     """
#     Compute minimum of -logpdf

#     :param model: description of the statistical model
#     :param init_pars: initial parameters
#     :param par_bounds: parameter bounds
#     :param fixed_poi: if a value is given, fixed poi optimisation will be performed
#     :param third_moment_expansion: computed results for the third moment
#                                     expansion of the statistical model
#     :return: -logpdf and fit parameters
#     """
#     assert len(init_pars) == len(model) + 1, (
#         "Dimensionality of initialization parameters does "
#         "not match the dimensionality of the model."
#     )

#     if third_moment_expansion is None:
#         third_moment_expansion = model.compute_expansion()

#     twice_nll_func = lambda pars: twice_nll(
#         pars, model.signal, model.background, model.observed, third_moment_expansion
#     )

#     _grad = grad(twice_nll, argnum=0)

#     grad_func = lambda pars: _grad(
#         pars, model.signal, model.background, model.observed, third_moment_expansion
#     )

#     _hessian = hessian(twice_nll, argnum=0)

#     hess_func = lambda pars: _hessian(
#         pars, model.signal, model.background, model.observed, third_moment_expansion
#     )

#     constraints = None
#     if fixed_poi is not None:
#         init_pars[0] = fixed_poi
#         constraints = [{"type": "eq", "fun": lambda v: v[0] - fixed_poi}]

#     opt = minimize(
#         twice_nll_func,
#         np.array(init_pars),
#         method="SLSQP",
#         jac=grad_func,
#         # hess=hess_func,
#         bounds=par_bounds,
#         constraints=constraints,
#         tol=1e-6,
#         options=dict(maxiter=10000, disp=0),
#     )

#     if not opt.success:
#         warnings.warn(message=opt.message, category=RuntimeWarning)

#     return opt.fun / 2.0, opt.x
