"""Autograd based differentiable distribution classes"""

import logging
import warnings
from typing import Any, Callable, Dict, List, Literal, Text, Union

import autograd.numpy as np
from autograd.scipy.special import gammaln
from autograd.scipy.stats.poisson import logpmf
from scipy.stats import multivariate_normal, norm, poisson

from spey.system.exceptions import DistributionError

# pylint: disable=E1101, W1203, E1121

log = logging.getLogger("Spey")

__all__ = ["Poisson", "Normal", "MultivariateNormal", "MainModel", "ConstraintModel"]


def __dir__():
    return __all__


class Poisson:
    """Poisson distribution"""

    def __init__(self, loc: np.ndarray):
        # ! Clip for numeric stability, poisson can not take negative values
        self.loc = np.clip(loc, 1e-20, None)

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Poisson distribution."""
        return np.array(self.loc)

    def sample(self, sample_size: int) -> np.ndarray:
        """Generate samples"""
        shape = [sample_size]
        if isinstance(self.loc, np.ndarray):
            shape += [len(self.loc)]
        return poisson(self.loc).rvs(size=shape)

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log-probability"""
        # for code efficiency
        if np.array(value).dtype in [np.int32, np.int16, np.int64]:
            return logpmf(value, self.loc).astype(np.float64)

        return (value * np.log(self.loc) - self.loc - gammaln(value + 1.0)).astype(
            np.float64
        )


class Normal:
    """
    Normal distribution

    Args:
        loc (``np.ndarray``): Mean of the distribution.
        scale (``np.ndarray``): standard deviation.
        weight (``Callable[[np.ndarray], float]`` or ``float``, default ``1.0``): weight of
          the distribution.
        domain (``slice``, default ``slice(None, None)``): set of parameters to be used within
          the distribution.
    """

    def __init__(
        self,
        loc: np.ndarray,
        scale: np.ndarray,
        weight: Union[Callable[[np.ndarray], float], float] = 1.0,
        domain: slice = slice(None, None),
    ):
        self.loc = loc
        self.weight = weight if callable(weight) else lambda pars: weight
        """Weight of the distribution"""
        self.domain = domain
        """Which parameters should be used during the computation of the pdf"""

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

        return norm(self.loc, self.scale(value[self.domain])).rvs(size=shape)

    def log_prob(self, value: float) -> np.ndarray:
        """Compute log-probability"""
        return self.weight(value) * (
            -np.log(self.scale(value[self.domain]))
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5
            * np.square(
                np.divide(value[self.domain] - self.loc, self.scale(value[self.domain]))
            )
        ).astype(np.float64)


class MultivariateNormal:
    """
    Multivariate normal distribution

    Args:
        mean (``np.ndarray``): Mean of the distribution.
        cov (``np.ndarray``): Symmetric positive (semi)definite
          covariance matrix of the distribution.
        weight (``Callable[[np.ndarray], float]`` or ``float``, default ``1.0``): weight of
          the distribution.
        domain (``slice``, default ``slice(None, None)``): set of parameters to be used within
          the distribution.
    """

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        weight: Union[Callable[[np.ndarray], float], float] = 1.0,
        domain: slice = slice(None, None),
    ):
        self.mean = mean
        """Mean of the distribution."""
        self.cov = cov if callable(cov) else lambda val: cov
        """Symmetric positive (semi)definite covariance matrix of the distribution."""
        self.weight = weight if callable(weight) else lambda pars: weight
        """Weight of the distribution"""
        self.domain = domain
        """Which parameters should be used during the computation of the pdf"""

        # ! the min determinant value of the covariance matrix is artificially set
        # ! to 1e-10 this might cause problems in the future!!!

        if callable(cov):
            self._inv_cov = lambda val: np.linalg.inv(cov(val))
            self._det_cov = lambda val: np.clip(np.linalg.det(cov(val)), 1e-20, None)
        else:
            # for code efficiency
            inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            if det <= 0.0:
                warnings.warn(
                    "det(rho) <= 0, this might cause numeric problems. "
                    "The value of the determinant will be limited to 1e-20. "
                    "This might be due to non-positive definite correlation matrix input."
                )
                det = np.clip(det, 1e-20, None)
            self._inv_cov = lambda val: inv
            self._det_cov = lambda val: det

    def expected_data(self) -> np.ndarray:
        """The expectation value of the Multivariate Normal distribution."""
        return self.mean

    def sample(self, value: np.ndarray, sample_size: int) -> np.ndarray:
        """Generate samples"""
        return multivariate_normal(self.mean, self.cov(value[self.domain])).rvs(
            size=(sample_size,)
        )

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log-probability"""
        var = value[self.domain] - self.mean
        return self.weight(value) * (
            -0.5 * (var @ self._inv_cov(value[self.domain]) @ var)
            - 0.5
            * (
                len(value[self.domain]) * np.log(2.0 * np.pi)
                + np.log(self._det_cov(value[self.domain]))
            )
        ).astype(np.float64)


class MainModel:
    """
    Main statistical model, modelled as Poisson distribution which has a
    variable lambda.

    Args:
        loc (``Callable[[np.ndarray], np.ndarray]``): callable function that represents
          lambda values of poisson distribution. It takes nuisance parameters as input.
    """

    def __init__(
        self,
        loc: Callable[[np.ndarray], np.ndarray],
        cov: np.ndarray = None,
        pdf_type: Literal["poiss", "gauss", "multivariategauss"] = "poiss",
    ):
        self.pdf_type = pdf_type
        """Type of the PDF"""
        if pdf_type == "poiss":
            self._pdf = lambda pars: Poisson(loc(pars))
        elif pdf_type == "gauss" and cov is not None:
            self._pdf = lambda pars: Normal(loc=loc(pars), scale=cov)
        elif pdf_type == "multivariategauss" and cov is not None:
            self._pdf = lambda pars: MultivariateNormal(mean=loc(pars), cov=cov)
        else:
            raise DistributionError("Unknown pdf type or associated input.")

    def expected_data(self, pars: np.ndarray) -> np.ndarray:
        """The expectation value of the main model."""
        return self._pdf(pars).expected_data()

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
        if self.pdf_type == "poiss":
            return self._pdf(pars).sample(sample_size)

        return self._pdf(pars).sample(pars, sample_size)

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
        pdf_descriptions (``List[Dict[Text, Any]]``): description of the pdf component.
            Dictionary elements should contain two keywords

              * ``"distribution_type"`` (``Text``): ``"normal"`` or ``"multivariatenormal"``
              * ``"args"``: Input arguments for the distribution
              * ``"kwargs"``: Input keyword arguments for the distribution
    """

    def __init__(self, pdf_descriptions: List[Dict[Text, Any]]):
        self._pdfs = []
        distributions = {"normal": Normal, "multivariatenormal": MultivariateNormal}

        log.debug("Adding constraint terms:")
        for desc in pdf_descriptions:
            assert desc["distribution_type"].lower() in [
                "normal",
                "multivariatenormal",
            ], f"Unknown distribution type: {desc['distribution_type']}"
            log.debug(f"{desc}")
            self._pdfs.append(
                distributions[desc["distribution_type"]](
                    *desc.get("args", []), **desc.get("kwargs", {})
                )
            )

    def __len__(self):
        return len(self._pdfs)

    def expected_data(self) -> np.ndarray:
        """The expectation value of the constraint model."""
        if len(self) > 1:
            return np.hstack([pdf.expected_data() for pdf in self._pdfs])

        return self._pdfs[0].expected_data()

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
        if len(self) > 1:
            return np.hstack([pdf.sample(pars, sample_size) for pdf in self._pdfs])

        return self._pdfs[0].sample(pars, sample_size)

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
        return np.sum([pdf.log_prob(pars) for pdf in self._pdfs])


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
