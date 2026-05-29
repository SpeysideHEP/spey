r"""
Third-Moment Expansion Tools
==============================

This module provides the mathematical machinery for the **third-moment expansion**
of the simplified likelihood, following :xref:`1809.05548` Sec. 2.

Motivation
----------

The standard simplified likelihood models background uncertainties as a multivariate
normal distribution centred at zero with covariance :math:`\Sigma`.  This is exact
only when the background fluctuations are themselves Gaussian.  When significant
asymmetry (skewness) is present — as is common for small background yields or
log-normal-shaped systematic uncertainties — the Gaussian approximation can over-
or under-cover.

The third-moment expansion extends the simplified likelihood by incorporating the
diagonal elements of the third central moment tensor :math:`m^{(3)}_i`, which
characterise the skewness of the background distribution in each bin.

Mathematical background
-----------------------

Given the first three moments of the background:

* :math:`m^{(1)}_i` — expected value (mean),
* :math:`m^{(2)}_{ij}` — covariance matrix (:math:`\Sigma_{ij}`),
* :math:`m^{(3)}_i` — diagonal third central moments (skewness per bin),

one defines per-bin coefficients via eqs. 2.9–2.11 of :xref:`1809.05548`:

.. math::

    C_i &= -\mathrm{sign}(m^{(3)}_i)\,\sqrt{2\,\Sigma_{ii}}
           \cos\!\left(\frac{4\pi}{3}
           + \frac{1}{3}\arctan\!\sqrt{\frac{8\,\Sigma_{ii}^3}{(m^{(3)}_i)^2} - 1}
           \right), \\[4pt]
    B_i &= \sqrt{\Sigma_{ii} - 2C_i^2}, \\[4pt]
    A_i &= m^{(1)}_i - C_i.

These coefficients enter the quadratic :math:`\lambda` function (eq. 2.6 of
:xref:`1809.05548`):

.. math::

    \lambda_i(\mu, \theta_i) = \mu\, n^s_i + A_i + B_i\,\theta_i + C_i\,\theta_i^2,

and a modified inter-bin correlation matrix (eq. 2.12 of :xref:`1809.05548`):

.. math::

    \rho_{ij} = \frac{1}{4C_i C_j}
    \left(\sqrt{(B_i B_j)^2 + 8\,C_i C_j\,\Sigma_{ij}} - B_i B_j\right).

**Validity condition**: the expansion requires :math:`8\,\Sigma_{ii}^3 \geq (m^{(3)}_i)^2`
for each bin.  Bins that violate this condition are treated with :math:`C_i = 0`
(reverting to the standard simplified likelihood for that bin).

When :math:`m^{(3)}_i = 0` for all bins the expansion reduces identically to the
standard simplified likelihood (:class:`~spey.backends.default_pdf.CorrelatedBackground`).

.. autosummary::
    :toctree: ../_generated/

    third_moment_expansion

Computing third moments from asymmetric uncertainties
------------------------------------------------------

When only asymmetric uncertainty envelopes :math:`(\sigma^+_i, \sigma^-_i)` are
available, the third moment can be estimated analytically by modelling the
background as a **bifurcated Gaussian** — a piecewise normal distribution that
uses :math:`\sigma^-_i` for :math:`\theta < 0` and :math:`\sigma^+_i` for
:math:`\theta \geq 0`:

.. math::

    p(\theta) = \frac{2}{\sigma^+_i + \sigma^-_i}
    \begin{cases}
      \mathcal{N}(\theta \mid 0, \sigma^-_i), & \theta < 0, \\
      \mathcal{N}(\theta \mid 0, \sigma^+_i), & \theta \geq 0.
    \end{cases}

The third central moment of this distribution is

.. math::

    m^{(3)}_i = \frac{2}{\sigma^+_i + \sigma^-_i}
    \left[\sigma^-_i \int_{-\infty}^{0} \theta^3\,
          \mathcal{N}(\theta\mid 0,\sigma^-_i)\,d\theta
    + \sigma^+_i \int_{0}^{\infty} \theta^3\,
          \mathcal{N}(\theta\mid 0,\sigma^+_i)\,d\theta\right],

which is computed numerically by :func:`compute_third_moments`.
See :xref:`2602.18611` for a usecase.

.. autosummary::
    :toctree: ../_generated/

    compute_third_moments

References
----------
* CMS Collaboration, *Simplified likelihood for the re-interpretation of public CMS
  results*, :xref:`1809.05548`, Sec. 2.
* CMS Collaboration, *Combined measurements and interpretations of Higgs boson
  production and decay in proton-proton collisions at* :math:`\sqrt{s} = 13` TeV,
  :xref:`2602.18611`
"""
import logging
from typing import Optional, Tuple, Union

import autograd.numpy as np
from scipy import integrate
from scipy.stats import norm

from spey.system.exceptions import warning_tracker

# pylint: disable=E1101,E1120,W1203
log = logging.getLogger("Spey")


@warning_tracker
def third_moment_expansion(
    expectation_value: np.ndarray,
    covariance_matrix: np.ndarray,
    third_moment: Optional[np.ndarray] = None,
    return_correlation_matrix: bool = False,
) -> Tuple:
    r"""
    Compute the :math:`A`, :math:`B`, :math:`C` coefficients and optional modified
    correlation matrix for the third-moment expansion of the simplified likelihood.

    Given the background expectation values :math:`m^{(1)}_i`, covariance matrix
    :math:`\Sigma_{ij}`, and diagonal third moments :math:`m^{(3)}_i`, the function
    implements eqs. 2.9–2.12 of :xref:`1809.05548`:

    .. math::

        C_i &= -\mathrm{sign}(m^{(3)}_i)\,\sqrt{2\,\Sigma_{ii}}
               \cos\!\left(\frac{4\pi}{3}
               + \frac{1}{3}\arctan\!\sqrt{\frac{8\,\Sigma_{ii}^3}{(m^{(3)}_i)^2} - 1}
               \right), \\[4pt]
        B_i &= \sqrt{\Sigma_{ii} - 2C_i^2}, \\[4pt]
        A_i &= m^{(1)}_i - C_i.

    When ``return_correlation_matrix=True``, the modified inter-bin correlation
    matrix is also computed (eq. 2.12):

    .. math::

        \rho_{ij} = \frac{1}{4C_i C_j}
        \left(\sqrt{(B_i B_j)^2 + 8\,C_i C_j\,\Sigma_{ij}} - B_i B_j\right).

    A small regularisation :math:`\varepsilon = 10^{-5}` is added to :math:`C_i`
    in the denominator to avoid division by zero when :math:`C_i = 0`.

    .. attention::

        The expansion requires :math:`8\,\Sigma_{ii}^3 \geq (m^{(3)}_i)^2` for every
        bin.  Bins that violate this condition produce ``NaN`` values for :math:`C_i`,
        which are silently replaced by zero, effectively reverting to the standard
        simplified likelihood (:math:`\lambda_i = \mu n^s_i + n^b_i + B_i\,\theta_i`)
        for those bins.  A warning is emitted when this occurs.

    Args:
        expectation_value (``np.ndarray``): Per-bin background expectation values
          :math:`\{m^{(1)}_i\}` (shape :math:`N`).
        covariance_matrix (``np.ndarray``): :math:`N \times N` background covariance
          matrix :math:`\Sigma`.
        third_moment (``np.ndarray``): Diagonal elements of the third-moment tensor
          :math:`\{m^{(3)}_i\}` (shape :math:`N`).
        return_correlation_matrix (``bool``, default ``False``): If ``True``, also
          compute and return the modified correlation matrix :math:`\rho`.

    Returns:
        ``Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]``:
        The triple :math:`(A, B, C)` (each of shape :math:`N`), and additionally
        :math:`\rho` (shape :math:`N \times N`) when ``return_correlation_matrix=True``.
    """
    cov_diag = np.diag(covariance_matrix)

    if not np.all(8.0 * cov_diag**3 >= third_moment**2):
        log.warning(
            r"Third moments does not satisfy the following condition: $8\Sigma_{ii}^3 \geq (m^{(3)}_i)^2$"
        )
        log.warning("The values that do not satisfy this condition will be set to zero.")

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
    log.debug(f"C: {C}")
    C = np.where(np.isnan(C), 0.0, C)

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
    Estimate the diagonal third central moments :math:`\{m^{(3)}_i\}` from
    asymmetric uncertainty envelopes using a bifurcated Gaussian model.

    When only the upper and lower uncertainty envelopes
    :math:`(\sigma^+_i, \sigma^-_i)` are available (rather than the full
    background distribution), the background fluctuation in bin :math:`i` is
    modelled as a **bifurcated Gaussian**:

    .. math::

        p_i(\theta) = \frac{2}{\sigma^+_i + \sigma^-_i}
        \begin{cases}
          \mathcal{N}(\theta \mid 0,\, \sigma^-_i), & \theta < 0, \\
          \mathcal{N}(\theta \mid 0,\, \sigma^+_i), & \theta \geq 0.
        \end{cases}

    The third central moment of this distribution is

    .. math::

        m^{(3)}_i = \mathbb{E}_i[\theta^3]
        = \frac{2}{\sigma^+_i + \sigma^-_i}
          \left[
            \sigma^-_i \int_{-\infty}^{0} \theta^3\,
            \mathcal{N}(\theta \mid 0, \sigma^-_i)\,d\theta
          + \sigma^+_i \int_{0}^{\infty} \theta^3\,
            \mathcal{N}(\theta \mid 0, \sigma^+_i)\,d\theta
          \right],

    which is evaluated numerically via :func:`scipy.integrate.quad`.

    .. note::

        The :math:`k`-th central moment of a distribution :math:`p` is defined as

        .. math::

            m^{(k)} = \mathbb{E}[(X - c)^k]
            = \int_{-\infty}^{\infty} (x - c)^k\, p(x)\, dx,

        with :math:`c = 0` for the bifurcated Gaussian (which is already centred).

    .. attention::

        :func:`third_moment_expansion` requires :math:`8\,\Sigma_{ii}^3 \geq (m^{(3)}_i)^2`.
        Because this function derives :math:`m^{(3)}_i` independently from the covariance
        matrix, the condition is not guaranteed to hold.  Whether it is satisfied depends
        on the interplay between the envelopes and the diagonal of :math:`\Sigma`.

    Args:
        absolute_upper_uncertainties (``np.ndarray``): Per-bin upper absolute
          uncertainties :math:`\{\sigma^+_i\}` (positive values).
        absolute_lower_uncertainties (``np.ndarray``): Per-bin lower absolute
          uncertainties :math:`\{\sigma^-_i\}` (positive values; signs are taken
          as absolute internally).
        return_integration_error (``bool``, default ``False``): If ``True``, also
          return the per-bin numerical integration errors from
          :func:`scipy.integrate.quad`.

    Returns:
        ``np.ndarray`` or ``Tuple[np.ndarray, np.ndarray]``:
        Array of diagonal third moments :math:`\{m^{(3)}_i\}` (shape :math:`N`).
        When ``return_integration_error=True``, a 2-tuple
        ``(third_moments, errors)`` is returned.
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
