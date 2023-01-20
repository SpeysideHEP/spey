"""This file contains functions to compute marginalised likelihood"""
from typing import Optional
import numpy as np
import scipy, math

from .data import Data, expansion_output


def marginalised_negloglikelihood(
    mu: float,
    model: Data,
    third_moment_expansion: Optional[expansion_output] = None,
    ntoys: Optional[int] = 30000,
) -> float:
    """
    Compute marginalised negative log-likelihood for the statistical model

    :param mu: POI (signal strength)
    :param model: statistical model
    :param third_moment_expansion: third moment expansion
    :param ntoys: number of toy examples
    :return: negative log-likelihood
    """
    if model.is_single_region:
        return marginalised_negloglikelihood_singleregion(mu, model)

    if third_moment_expansion is None:
        third_moment_expansion = model.compute_expansion()

    signal_yields = mu * model

    thetas = scipy.stats.multivariate_normal.rvs(
        mean=np.zeros(shape=len(model)),
        cov=third_moment_expansion.V,
        size=ntoys,
    )

    values = []
    for theta in thetas:
        if model.isLinear:
            lmbda = signal_yields + model.background + theta
        else:
            lmbda = (
                signal_yields
                + third_moment_expansion.A
                + theta
                + third_moment_expansion.C * np.square(theta) / np.square(third_moment_expansion.B)
            )
        lmbda = np.where(lmbda <= 0.0, 1e-30, lmbda)

        poisson = model.observed * np.log(lmbda) - lmbda - scipy.special.gammaln(model.observed + 1)
        values.append(np.exp(sum(poisson)))

    mean = np.mean(values)
    return -np.log(mean if mean != 0.0 else 1e-99)


def marginalised_negloglikelihood_singleregion(mu: float, model: Data) -> float:
    """
    Return the likelihood (of 1 signal region) to observe nobs events given the
    predicted background nb, error on this background (deltab),
    signal strength of mu and the relative error on the signal (deltas_rel).

    :param mu: POI (signal strength)
    :param model: statistical model
    :return: negative log-likelihood
    :raises TimeoutError: if integration takes longer than 10 iterations
    """
    signal_yields = mu * model

    sigma2 = model.covariance + model.var_smu(mu)
    sigma_tot = np.sqrt(sigma2)
    lngamma = math.lgamma(model.observed[0] + 1)
    #     Why not a simple gamma function for the factorial:
    #     -----------------------------------------------------
    #     The scipy.stats.poisson.pmf probability mass function
    #     for the Poisson distribution only works for discrete
    #     numbers. The gamma distribution is used to create a
    #     continuous Poisson distribution.
    #
    #     Why not a simple gamma function for the factorial:
    #     -----------------------------------------------------
    #     The gamma function does not yield results for integers
    #     larger than 170. Since the expression for the Poisson
    #     probability mass function as a whole should not be huge,
    #     the exponent of the log of this expression is calculated
    #     instead to avoid using large numbers.

    # Define integrand (gaussian_(bg+signal)*poisson(nobs)):
    def prob(x: np.ndarray, signal: np.ndarray) -> np.ndarray:
        poisson = np.exp(model.observed * np.log(x) - x - lngamma)
        gaussian = scipy.stats.norm.pdf(x, loc=model.background + signal, scale=sigma_tot)
        return poisson * gaussian

    # Compute maximum value for the integrand:
    # If nb + nsig = sigma2, shift the values slightly:
    xm = model.background + signal_yields - sigma2
    xm = xm if xm != 0.0 else 0.001
    xmax = xm * (1.0 + np.sign(xm) * np.sqrt(1.0 + 4.0 * model.observed * sigma2 / xm**2)) / 2.0

    # Define initial integration range:
    nrange = 5.0
    a = max(0.0, xmax - nrange * np.sqrt(sigma2))
    b = xmax + nrange * sigma_tot
    like = scipy.integrate.quad(prob, a, b, (signal_yields), epsabs=0.0, epsrel=1e-3)[0]
    if like == 0.0:
        return 0.0

    # Increase integration range until integral converges
    err, ctr = 1.0, 0
    while err > 0.01:
        ctr += 1
        if ctr > 10.0:
            raise TimeoutError("Could not compute likelihood within required precision")

        like_old = like
        nrange *= 2.0
        a = max(0.0, (xmax - nrange * sigma_tot)[0][0])
        b = (xmax + nrange * sigma_tot)[0][0]
        like = scipy.integrate.quad(prob, a, b, (signal_yields), epsabs=0.0, epsrel=1e-3)[0]
        if like == 0.0:
            continue
        err = abs(like_old - like) / like

    # Renormalize the likelihood to account for the cut at x = 0.
    # The integral of the gaussian from 0 to infinity gives:
    # (1/2)*(1 + Erf(mu/sqrt(2*sigma2))), so we need to divide by it
    # (for mu - sigma >> 0, the normalization gives 1.)
    norm = (1.0 / 2.0) * (
        1.0 + scipy.special.erf((model.background + signal_yields) / np.sqrt(2.0 * sigma2))
    )
    return -np.log(like / norm)[0][0]
