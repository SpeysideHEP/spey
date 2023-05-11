"""Various helper functions"""

import numpy as np


def correlation_to_covariance(
    correlation_matrix: np.ndarray, standard_deviations: np.ndarray
) -> np.ndarray:
    """
    Convert correlation matrix into covariance matrix

    Args:
        correlation_matrix (``np.ndarray``): a real NxN matrix
        standard_deviations (``np.ndarray``): a real N-dimensional
          vector representing standard deviations.

    Returns:
        ``np.ndarray``:
        Covariance matrix
    """
    sigma = np.diag(standard_deviations)
    return sigma @ correlation_matrix @ sigma


def covariance_to_correlation(covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix into correlation matrix.

    Args:
        covariance_matrix (``np.ndarray``): a real NxN matrix

    Returns:
        ``np.ndarray``:
        Correlation matrix
    """
    sigma = np.linalg.inv(np.diag(np.sqrt(np.diag(covariance_matrix))))
    return sigma @ covariance_matrix @ sigma
