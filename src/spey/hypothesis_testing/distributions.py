"""Test statistic distributions"""

from scipy.stats import multivariate_normal
import numpy as np

__all__ = ["AsymptoticTestStatisticsDistribution", "EmpricTestStatisticsDistribution"]


class AsymptoticTestStatisticsDistribution:
    """
    The distribution the test statistic in the asymptotic case.

    Args:
        shift (``float``): The shift in test statistic distribution.
    """

    __slots__ = ["shift", "cutoff"]

    def __init__(self, shift: float, cutoff: float = -np.inf):
        self.shift = shift
        self.cutoff = cutoff

    def pvalue(self, value: float) -> float:
        r"""
        Compute the p-value for a given value of the test statistic.

        Args:
            value (``float``): The test statistic value.

        Returns:
            ``float``:
            p-value
        """
        return_value = multivariate_normal.cdf(self.shift - value)
        return return_value if return_value >= self.cutoff else np.nan

    def expected_value(self, nsigma: float) -> float:
        """
        Compute expected value of the test statistic with respec to the level of standard deviations

        Args:
            nsigma (``float``): level of standard deviations.

        Returns:
            ``float``:
            expected value of test statistic.
        """
        tot = nsigma + self.shift
        return tot if tot > self.cutoff else self.cutoff


class EmpricTestStatisticsDistribution:
    """
    Create emprical distribution. The p-values are computed from sampled distribution.

    Args:
        samples (``np.ndarray``): input samples
    """

    __slots__ = ["samples"]

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples

    def pvalue(self, value: float) -> float:
        """
        Compute the p-value for a given test statistic.

        Args:
            value (``float``): test statistic value

        Returns:
            ``float``:
            p-value
        """

        return np.sum(np.where(self.samples >= value, 1.0, 0.0)) / self.samples.shape[0]

    def expected_value(self, nsigma: float) -> float:
        """
        Compute expected value of the test statistic with respec to the level of standard deviations

        Args:
            nsigma (``float``): level of standard deviations.

        Returns:
            ``float``:
            expected value of test statistic.
        """
        return np.percentile(
            self.samples, multivariate_normal.cdf(nsigma) * 100.0, interpolation="linear"
        )
