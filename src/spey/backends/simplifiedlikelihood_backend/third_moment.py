"""Tools for computing third moment expansion"""
from typing import Tuple, Optional

import autograd.numpy as np

# pylint: disable=E1101,E1120


def third_moment_expansion(
    expectation_value: np.ndarray,
    covariance_matrix: np.ndarray,
    third_moment: Optional[np.ndarray] = None,
    return_correlation_matrix: bool = False,
) -> Tuple:
    """
    Construct the terms for third moment expansion. For details see :xref:`1809.05548`.

    Args:
        expectation_value (``np.ndarray``): expectation value of the background
        covariance_matrix (``np.ndarray``): covariance matrix
        third_moment (``np.ndarray``): Diagonal components of the third moment
        return_correlation_matrix (``bool``, default ``False``): If true reconstructs
          and returns correlation matrix.

    Returns:
        ``np.ndarray``:
        A, B, C terms from :xref:`1809.05548` eqns 2.9, 2.10, 2.11. if
        ``return_correlation_matrix`` is ``True`` it also returns correlation matrix.
    """
    cov_diag = np.diag(covariance_matrix)

    assert np.all(8.0 * cov_diag**3 >= third_moment**2), (
        "Given covariance matrix and diagonal terms of the third moment does not "
        + "satisfy the condition: 8 * diag(cov)**3 >= third_moment**2."
    )

    # arXiv:1809.05548 eq. 2.9
    C = (
        -np.sign(third_moment)
        * np.sqrt(2.0 * cov_diag)
        * np.cos(
            (4.0 * np.pi / 3.0)
            + (1.0 / 3.0)
            * np.arctan(np.sqrt(((8.0 * cov_diag**3) / third_moment**2) - 1.0))
        )
    )

    # arXiv:1809.05548 eq. 2.10
    B = np.sqrt(cov_diag - 2 * C**2)

    # arXiv:1809.05548 eq. 2.11
    A = expectation_value - C

    # arXiv:1809.05548 eq. 2.12
    eps = 1e-5
    if return_correlation_matrix:
        corr = np.zeros((C.shape[0], C.shape[0]))
        for i in range(corr.shape[0]):
            for j in range(corr.shape[0]):
                ci = C[i] + eps if C[i] >= 0 else C[i] - eps
                cj = C[j] + eps if C[j] >= 0 else C[j] - eps
                cicj = ci * cj
                bibj = B[i] * B[j]

                discr1 = bibj**2
                discr2 = 8 * cicj * covariance_matrix[i, j]
                discr = discr1 + discr2

                corr[i, j] = (np.sqrt(abs(discr)) - bibj) / 4 / cicj

                if i != j:
                    corr[j, i] = corr[i, j]

        return A, B, C, corr

    return A, B, C
