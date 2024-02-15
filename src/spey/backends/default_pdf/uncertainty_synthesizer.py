from typing import Any, Dict, List, Text, Tuple

import autograd.numpy as np

from spey.helper_functions import correlation_to_covariance

from .third_moment import third_moment_expansion

# pylint: disable=E1101,E1120


def constraint_from_corr(
    correlation_matrix: List[List[float]], size: int, domain: slice
) -> List[Dict[Text, Any]]:
    """
    Derive constraints from inputs

    Args:
        correlation_matrix (``List[List[float]]``): correlation matrix
        size (``int``): size of the signal vector
        domain (``slice``): domain of the nuisances

    Returns:
        ``List[Dict[Text, Any]]``:
    """
    if correlation_matrix is not None:
        corr = np.array(correlation_matrix)
        constraint_term = [
            {
                "distribution_type": "multivariatenormal",
                "args": [np.zeros(size), corr],
                "kwargs": {"domain": domain},
            }
        ]
    else:
        constraint_term = [
            {
                "distribution_type": "normal",
                "args": [np.zeros(size), np.ones(size)],
                "kwargs": {"domain": domain},
            }
        ]

    return constraint_term


def signal_uncertainty_synthesizer(
    signal_yields: List[float],
    absolute_uncertainties: List[float] = None,
    absolute_uncertainty_envelops: List[Tuple[float, float]] = None,
    correlation_matrix: List[List[float]] = None,
    third_moments: List[float] = None,
    domain: slice = None,
) -> Dict[Text, np.ndarray]:
    """
    Synthesize signal uncertainties

    Args:
        signal_yields (``List[float]``): signal yields
        absolute_uncertainties (``List[float]``, default ``None``): absolute uncertainties
        absolute_uncertainty_envelops (``List[Tuple[float, float]]``, default ``None``):
            asymmetric uncertainty envelops (upper, lower)
        correlation_matrix (``List[List[float]]``, default ``None``): correlation matrix
        third_moments (``List[float]``, default ``None``): third moments
        domain (``slice``, default ``None``): domain of the nuisances

    Raises:
        ``ValueError``: if inconsistent number of input has been provided

    Returns:
        ``Dict[Text, np.ndarray]``:
        Inputs for the main model and the constraint term
    """
    assert domain is not None, "Invalid domain"
    signal_yields = np.array(signal_yields)

    if absolute_uncertainties is not None and third_moments is None:
        absolute_uncertainties = np.array(absolute_uncertainties)

        def lam_signal(pars: np.ndarray) -> np.ndarray:
            return absolute_uncertainties * pars[domain]

        constraint_term = constraint_from_corr(
            correlation_matrix, len(absolute_uncertainties), domain
        )

    elif absolute_uncertainty_envelops is not None:
        sigma_plus, sigma_minus = [], []
        for upper, lower in absolute_uncertainty_envelops:
            sigma_plus.append(abs(upper))
            sigma_minus.append(abs(lower))
        sigma_plus = np.array(sigma_plus)
        sigma_minus = np.array(sigma_minus)

        # arXiv:pyhsics/0406120 eq. 18-19
        def effective_sigma(pars: np.ndarray) -> np.ndarray:
            """Compute effective sigma"""
            return np.sqrt(
                sigma_plus * sigma_minus
                + (sigma_plus - sigma_minus) * (pars[domain] - signal_yields)
            )

        def lam_signal(pars: np.ndarray) -> np.ndarray:
            """Compute lambda for Main model"""
            return effective_sigma(pars) * pars[domain]

        constraint_term = constraint_from_corr(
            correlation_matrix, len(sigma_plus), domain
        )

    elif all(
        x is not None for x in [third_moments, correlation_matrix, absolute_uncertainties]
    ):
        correlation_matrix = np.array(correlation_matrix)
        absolute_uncertainties = np.array(absolute_uncertainties)
        third_moments = np.array(third_moments)
        cov = correlation_to_covariance(correlation_matrix, absolute_uncertainties)

        A, B, C, corr = third_moment_expansion(signal_yields, cov, third_moments, True)

        def lam_signal(pars: np.ndarray) -> np.ndarray:
            """
            Compute lambda for Main model with third moment expansion.
            For details see above eq 2.6 in :xref:`1809.05548`

            Args:
                pars (``np.ndarray``): nuisance parameters

            Returns:
                ``np.ndarray``:
                expectation value of the poisson distribution with respect to
                nuisance parameters.
            """
            return A + B * pars[domain] + C * np.square(pars[domain])

        constraint_term = [
            {
                "distribution_type": "multivariatenormal",
                "args": [np.zeros(len(signal_yields)), corr],
                "kwargs": {"domain": domain},
            }
        ]

    else:
        raise ValueError("Inconsistent input.")

    return {"lambda": lam_signal, "constraint": constraint_term}
