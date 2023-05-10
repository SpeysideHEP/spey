"""Autograd based distribution classes for simplified likelihood interface"""

from typing import Optional
from autograd.scipy.special import gammaln
from autograd.scipy.stats.poisson import logpmf
import autograd.numpy as np
from scipy.stats import poisson, norm, multivariate_normal

# pylint: disable=E1101

__all__ = ["Poisson", "Normal", "MultivariateNormal"]


def __dir__():
    return __all__


class Poisson:
    """Poisson distribution"""

    def __init__(self, lam: np.ndarray):
        self.lam = lam

    @property
    def shape(self):
        """Return sample shape"""
        if isinstance(self.lam, np.ndarray):
            return len(self.lam)
        return 1

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Poisson distribution."""
        return np.array(self.lam)

    def sample(self, sample_size: int) -> np.ndarray:
        """Generate samples"""
        shape = [sample_size]
        if isinstance(self.lam, np.ndarray):
            shape += [len(self.lam)]
        return poisson(self.lam).rvs(size=shape)

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log-probability"""
        # for code efficiency
        if np.array(value).dtype in [np.int32, np.int16, np.int64]:
            return np.sum(logpmf(value, self.lam)).astype(np.float64)

        return np.sum(value * np.log(self.lam) - self.lam - gammaln(value + 1.0)).astype(
            np.float64
        )


class Normal:
    """
    Normal distribution

    Args:
        loc (``np.ndarray``): Mean of the distribution.
        scale (``np.ndarray``): standard deviation.
    """

    def __init__(self, loc: np.ndarray, scale: np.ndarray):
        self.loc = loc
        self.scale = scale

    @property
    def shape(self):
        """Return sample shape"""
        if isinstance(self.loc, np.ndarray):
            return len(self.loc)
        return 1

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Normal distribution."""
        return np.array(self.loc)

    def sample(self, sample_size: int) -> np.ndarray:
        """Generate samples"""
        shape = [sample_size]
        if isinstance(self.loc, np.ndarray):
            shape += [len(self.loc)]
        return norm(self.loc, self.scale).rvs(size=shape)

    def log_prob(self, value: float) -> np.ndarray:
        """Compute log-probability"""
        return np.sum(
            -np.log(self.scale)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * np.square(np.divide(value - self.loc, self.scale))
        ).astype(np.float64)


class MultivariateNormal:
    """
    Multivariate normal distribution

    Args:
        mean (``np.ndarray``): Mean of the distribution.
        cov (``np.ndarray``): Symmetric positive (semi)definite
          covariance matrix of the distribution.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        """Mean of the distribution."""
        self.cov = cov
        """Symmetric positive (semi)definite covariance matrix of the distribution."""

        if callable(cov):
            self._inv_cov = lambda val: np.linalg.inv(cov(val))
            self._det_cov = lambda val: np.linalg.det(cov(val))
        else:
            # for code efficiency
            inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            self._inv_cov = lambda val: inv
            self._det_cov = lambda val: det

    @property
    def shape(self):
        """Return sample shape"""
        return len(self.mean)

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Multivariate Normal distribution."""
        return self.mean

    def sample(self, sample_size: int) -> np.ndarray:
        """Generate samples"""
        return multivariate_normal(self.mean, self.cov).rvs(size=(sample_size,))

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log-probability"""
        var = value - self.mean
        return (
            -0.5 * (var @ self._inv_cov(value) @ var)
            - 0.5 * (len(value) * np.log(2.0 * np.pi) + np.log(self._det_cov(value)))
        ).astype(np.float64)


class MixtureModel:
    """
    Generate probability distribution from combination of different distributions

    Args:
        args: Distributions

    .. warning::

        All distributions have been assumed to have same shape.
    """

    def __init__(self, *args):
        self.distributions = [d for d in args if hasattr(d, "log_prob")]

    def __iter__(self):
        yield from self.distributions

    def sample(self, sample_shape: int) -> np.ndarray:
        """Generate samples"""
        data = np.zeros(
            (sample_shape, self.distributions[-1].shape, len(self.distributions))
        )
        for idx, dist in enumerate(self):
            data[:, ..., idx] = dist.sample(sample_shape)
        random_idx = np.random.choice(
            np.arange(len(self.distributions)),
            size=(sample_shape,),
            p=[1.0 / len(self.distributions)] * len(self.distributions),
        )
        return data[np.arange(sample_shape), ..., random_idx]

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log-probability"""
        return np.sum([dist.log_prob(value) for dist in self.distributions])
