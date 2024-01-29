"""Tools for computing third moment expansion"""
import warnings
from typing import Optional, Tuple, Union
import logging
import autograd.numpy as np
from scipy import integrate
from scipy.stats import norm

# pylint: disable=E1101,E1120,W1203
log = logging.getLogger("Spey")


def third_moment_expansion(
    expectation_value: np.ndarray,
    covariance_matrix: np.ndarray,
    third_moment: Optional[np.ndarray] = None,
    return_correlation_matrix: bool = False,
) -> Tuple:
    r"""
    Construct the terms for third moment expansion. For details see :xref:`1809.05548`.

    .. attention::

        This function expects :math:`8\Sigma_{ii}^3 \geq (m^{(3)}_i)^2`. In case its not
        satisfied, `NaN` values will be replaced with zero.

    Args:
        expectation_value (``np.ndarray``): expectation value of the background
        covariance_matrix (``np.ndarray``): covariance matrix
        third_moment (``np.ndarray``): Diagonal components of the third moment, :math:`m^{(3)}`
        return_correlation_matrix (``bool``, default ``False``): If true reconstructs
          and returns correlation matrix.

    Returns:
        ``np.ndarray``:
        A, B, C terms from :xref:`1809.05548` eqns 2.9, 2.10, 2.11. if
        ``return_correlation_matrix`` is ``True`` it also returns correlation matrix.
    """
    cov_diag = np.diag(covariance_matrix)

    if not np.all(8.0 * cov_diag**3 >= third_moment**2):
        log.warning(
            r"Third moments does not satisfy the following condition: $8\Sigma_{ii}^3 \geq (m^{(3)}_i)^2$"
        )
        log.warning("The values that do not satisfy this condition will be set to zero.")

    # arXiv:1809.05548 eq. 2.9
    with warnings.catch_warnings(record=True) as w:
        C = (
            -np.sign(third_moment)
            * np.sqrt(2.0 * cov_diag)
            * np.cos(
                (4.0 * np.pi / 3.0)
                + (1.0 / 3.0)
                * np.arctan(np.sqrt(((8.0 * cov_diag**3) / third_moment**2) - 1.0))
            )
        )
    if len(w) > 0:
        warnings.warn(
            "8 * diag(cov)**3 >= third_moment**2 condition is not satisfied,"
            " setting nan values to zero.",
            category=RuntimeWarning,
        )
        C = np.where(np.isnan(C), 0.0, C)
    log.debug(f"C: {C}")

    # arXiv:1809.05548 eq. 2.10
    B = np.sqrt(cov_diag - 2 * C**2)
    log.debug(f"B: {B}")

    # arXiv:1809.05548 eq. 2.11
    A = expectation_value - C
    log.debug(f"A: {A}")

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

        log.debug(f"rho: {corr}")
        return A, B, C, corr

    return A, B, C


def compute_third_moments(
    absolute_upper_uncertainties: np.ndarray,
    absolute_lower_uncertainties: np.ndarray,
    return_integration_error: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    r"""
    Assuming that the uncertainties are modelled as Gaussian, it computes third moments
    using Bifurcated Gaussian with asymmetric uncertainties.

    .. math::

        m^{(3)} = \frac{2}{\sigma^+ + \sigma^-} \left[ \sigma^-\int_{-\infty}^0 x^3
        \mathcal{N}(x|0,\sigma^-)dx + \sigma^+ \int_0^{\infty} x^3
        \mathcal{N}(x|0,\sigma^+)dx \right]

    .. note::

        Recall that expectation value of the :math:`k` th moment of a function :math:`f(x)`
        can be calculated as

        .. math::

            \mathbb{E}[(\mathbf{X} - c)^k] = \int_{-\infty}^\infty(x-c)^kf(x)dx


    .. attention::

        :func:`~spey.backends.default_pdf.third_moment.third_moment_expansion`
        function expects :math:`8\Sigma_{ii}^3 \geq (m^{(3)}_i)^2` since this function is
        constructed with upper and lower uncertainty envelops independent of covariance matrix,
        it does not guarantee that the condition will be satisfied. This depends on the
        covariance matrix.

    Args:
        absolute_upper_uncertainties (``np.ndarray``): absolute value of the upper uncertainties
        absolute_lower_uncertainties (``np.ndarray``): absolute value of the lower uncertainties
        return_integration_error (``bool``, default ``False``): If true returns integration error

    Returns:
        ``Tuple[np.ndarray, np.ndarray]`` or ``np.ndarray``:
        Diagonal elements of the third moments and integration error.
    """

    def compute_x3BifurgatedGaussian(x: float, upper: float, lower: float) -> float:
        Norm = 2.0 / (upper + lower)
        if x >= 0.0:
            # integrate from 0 to inf
            return Norm * upper * x**3 * norm.pdf(x, 0.0, upper)
        # integrate from -inf to 0
        return Norm * lower * x**3 * norm.pdf(x, 0.0, lower)

    third_moment, error = [], []
    for upper, lower in zip(absolute_upper_uncertainties, absolute_lower_uncertainties):
        third_moment_tmp, error_tmp = integrate.quad(
            compute_x3BifurgatedGaussian, -np.inf, np.inf, args=(abs(upper), abs(lower))
        )
        third_moment.append(third_moment_tmp)
        error.append(error_tmp)

    if return_integration_error:
        return np.array(third_moment), np.array(error)
    log.debug(f"Error: {error}")

    return np.array(third_moment)
