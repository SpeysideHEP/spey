"""Autograd based distribution classes for simplified likelihood interface"""

from typing import Callable, Text, Union
from autograd.scipy.special import gammaln
from autograd.scipy.stats.poisson import logpmf
import autograd.numpy as np
from scipy.stats import poisson, norm, multivariate_normal


# pylint: disable=E1101

__all__ = ["Poisson", "Normal", "MultivariateNormal", "MainModel", "ConstraintModel"]


def __dir__():
    return __all__


class Poisson:
    """Poisson distribution"""

    def __init__(self, lam: np.ndarray):
        self.lam = lam

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
            return logpmf(value, self.lam).astype(np.float64)

        return (value * np.log(self.lam) - self.lam - gammaln(value + 1.0)).astype(
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

        if callable(scale):
            self.scale = scale
        else:
            self.scale = lambda pars: scale

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Normal distribution."""
        return np.array(self.loc)

    def sample(self, value: np.ndarray, sample_size: int) -> np.ndarray:
        """Generate samples"""
        shape = [sample_size]
        if isinstance(self.loc, np.ndarray):
            shape += [len(self.loc)]

        return norm(self.loc, self.scale(value)).rvs(size=shape)

    def log_prob(self, value: float) -> np.ndarray:
        """Compute log-probability"""
        return (
            -np.log(self.scale(value))
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * np.square(np.divide(value - self.loc, self.scale(value)))
        ).astype(np.float64)


class GeneralisedPoisson:
    """Generalised Poisson distribution. See :xref:`physics/0406120` eq. 10a"""

    def __init__(self, best_fit: np.ndarray, alpha: np.ndarray, nu: np.ndarray):
        self.alpha = alpha
        self.nu = nu
        self.best_fit = best_fit

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Normal distribution."""
        return np.array(self.best_fit)

    def log_prob(self, value: float) -> np.ndarray:
        """Compute log-probability"""
        return (
            -self.alpha * (value - self.best_fit)
            + self.nu
            * (np.log(self.nu + self.alpha * (value - self.best_fit)) - np.log(self.nu))
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
        self.cov = cov if callable(cov) else lambda val: cov
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

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Multivariate Normal distribution."""
        return self.mean

    def sample(self, value: np.ndarray, sample_size: int) -> np.ndarray:
        """Generate samples"""
        return multivariate_normal(self.mean, self.cov(value)).rvs(size=(sample_size,))

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log-probability"""
        var = value - self.mean
        return (
            -0.5 * (var @ self._inv_cov(value) @ var)
            - 0.5 * (len(value) * np.log(2.0 * np.pi) + np.log(self._det_cov(value)))
        ).astype(np.float64)


class MainModel:
    """
    Main statistical model, modelled as Poisson distribution which has a
    variable lambda.

    Args:
        lam (``Callable[[np.ndarray], np.ndarray]``): callable function that represents
          lambda values of poisson distribution. It takes nuisance parameters as input.
    """

    def __init__(self, lam: Callable[[np.ndarray], np.ndarray]):
        self._pdf = lambda pars: Poisson(lam(pars))

    def expected_data(self, pars: np.ndarray) -> np.ndarray:
        """The expectation value of the main model."""
        return self._pdf(pars).lam

    def sample(self, pars: np.ndarray, sample_size: int) -> np.ndarray:
        r"""
        Generate samples

        Args:
            pars (``np.ndarray``): parameter of interest and nuisance parameters
              :math:`\mu` and :math:`\theta` combined.
            sample_size (``int``): size of the sample to return

        Returns:
            ``np.ndarray``:
            sampled data
        """
        return self._pdf(pars).sample(sample_size)

    def log_prob(self, pars: np.ndarray, data: np.ndarray) -> np.ndarray:
        r"""
        Compute log-probability

        Args:
            pars (``np.ndarray``): parameter of interest and nuisance parameters
              :math:`\mu` and :math:`\theta` combined.
            data (``np.ndarray``): actual data

        Returns:
            ``np.ndarray``:
            log-probability of the main model
        """
        return np.sum(self._pdf(pars).log_prob(data))


class ConstraintModel:
    """
    Constraint term modelled as a Gaussian distribution.

    Args:
        distribution_type (``Text``): ``"normal"`` or ``"multivariatenormal"``
        args: Input arguments for the distribution
    """

    def __init__(
        self,
        distribution_type: Text,
        *args,
    ):
        assert distribution_type.lower() in [
            "normal",
            "multivariatenormal",
            "generalisedpoisson",
        ], "Unknown distribution type"

        self._pdf = {
            "normal": Normal,
            "multivariatenormal": MultivariateNormal,
            "generalisedpoisson": GeneralisedPoisson,
        }[distribution_type.lower()](*args)

    def expected_data(self) -> np.ndarray:
        """The expectation value of the constraint model."""
        return self._pdf.expected_data()

    def sample(self, pars: np.ndarray, sample_size: int) -> np.ndarray:
        r"""
        Generate samples

        Args:
            pars (``np.ndarray``): parameter of interest and nuisance parameters
              :math:`\mu` and :math:`\theta` combined.
            sample_size (``int``): size of the sample to return

        Returns:
            ``np.ndarray``:
            sampled data
        """
        return self._pdf.sample(pars, sample_size)

    def log_prob(self, pars: np.ndarray) -> np.ndarray:
        r"""
        Compute log-probability

        Args:
            pars (``np.ndarray``): parameter of interest and nuisance parameters
              :math:`\mu` and :math:`\theta` combined.
            data (``np.ndarray``): actual data

        Returns:
            ``np.ndarray``:
            log-probability of the main model
        """
        return np.sum(self._pdf.log_prob(pars))


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
