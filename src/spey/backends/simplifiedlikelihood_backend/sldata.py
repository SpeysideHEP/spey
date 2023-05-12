"""Data structure for Simplified Likelihood interface"""

from dataclasses import dataclass
from typing import Text, Optional, Union, Callable

import autograd.numpy as np

from spey.system.exceptions import NegativeExpectedYields
from spey.base import ModelConfig
from .third_moment import third_moment_expansion

# pylint: disable=E1101

__all__ = ["SLData"]


def __dir__():
    return __all__


@dataclass(frozen=True)
class SLData:
    """
    Data container for simplified likelihoods

    Args:
        observed (``np.ndarray``): observed yields
        signal (``np.ndarray``): signal yields
        background (``np.ndarray``): simulated background yields
        covariance_matrix (``np.ndarray``): covariance matrix
        third_moment (``np.ndarray``, default ``None``): diagonal elements of the third moment
        name (``str``): name of the dataset.

    Raises:
        ``TypeError``: If the data is not ``numpy`` array.
    """

    observed: np.ndarray
    signal: np.ndarray
    background: np.ndarray
    covariance_matrix: Optional[
        Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
    ] = None
    third_moment: Optional[np.ndarray] = None
    name: Text = "__unknown_model__"

    def __post_init__(self):
        cov = (
            self.covariance_matrix
            if not callable(self.covariance_matrix)
            else self.covariance(np.random.uniform(0, 1, (len(self.background))))
        )

        if cov is not None:
            assert isinstance(cov, np.ndarray), "Covariance has to be numpy array"
            assert (
                cov.shape[0] == cov.shape[1] == len(self.observed)
            ), "Dimensionality of covariance matrix does not match."

        # validate inputs
        if not (
            isinstance(self.observed, np.ndarray)
            and isinstance(self.signal, np.ndarray)
            and isinstance(self.background, np.ndarray)
            and (self.third_moment is None or isinstance(self.third_moment, np.ndarray))
        ):
            raise TypeError(
                f"Invalid type.\nobserved: {type(self.observed)}, "
                f"\nsignal: {type(self.signal)}, "
                f"\nbackground: {type(self.background)}, "
                f"\nthird_moment: {type(self.third_moment)}"
            )

        assert (
            len(self.observed) == len(self.signal) == len(self.background) >= 1
        ), "Input shapes does not match"

    def __repr__(self):
        return (
            f"SLData(\n    name='{self.name}',"
            f"\n    data structure that represents {len(self)} regions,"
            f"\n    isLinear={self.isLinear}\n)"
        )

    def reset_observations(self, observations: np.ndarray, name: Text):
        """
        Create an new dataset by overwriding observed yields.

        Args:
            observations (``np.ndarray``): observed yields
            name (``Text``): name of the dataset.

        Returns:
            ~spey.backends.simplifiedlikelihood_backend.sldata.SLData:
            updated data container with new observed yields.
        """
        assert len(observations) == len(
            self
        ), "Dimensionality of the input does not match the statistical model."
        return SLData(
            observations,
            self.signal,
            self.background,
            self.covariance_matrix,
            self.third_moment,
            name,
        )

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 40.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``40.0``): upper bound for parameter of interest,
              :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """
        minimum_poi = self.minimum_poi
        return ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[1] * (len(self) + 1),
            suggested_bounds=[
                (minimum_poi if allow_negative_signal else 0.0, poi_upper_bound)
            ]
            + [(-5.0, 5.0)] * len(self),
        )

    @property
    def expected_dataset(self):
        """Retrieve expected dataset"""
        return self.reset_observations(self.background, f"{self.name}_exp")

    def __len__(self) -> int:
        return len(self.observed)

    @property
    def isAlive(self) -> bool:
        """Is there any region with non-zero signal yields"""
        return len(self.signal[self.signal > 0.0]) != 0

    @property
    def isLinear(self) -> bool:
        """Is statistical model linear? i.e. no quadratic term in the poissonians"""
        return self.third_moment is None

    @property
    def is_single_region(self) -> bool:
        """Is the statistical model has only one region"""
        return len(self) == 1

    @property
    def diag_cov(self) -> np.ndarray:
        """Retrieve diagonal terms of the covariance matrix"""
        return np.diag(self.covariance_matrix)

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Compute correlation matrix computed from covariance matrix"""
        if self.third_moment is None:
            inv_sqrt_diag = np.linalg.inv(np.sqrt(np.diag(self.diag_cov)))
            return inv_sqrt_diag @ self.covariance_matrix @ inv_sqrt_diag

        return third_moment_expansion(
            self.background, self.covariance_matrix, self.third_moment, True
        )[-1]

    @property
    def minimum_poi(self) -> float:
        """Find minimum POI test that can be applied to this statistical model"""
        if np.all(self.signal == 0.0):
            return -np.inf
        return -np.min(
            np.true_divide(
                self.background[self.signal != 0.0], self.signal[self.signal != 0.0]
            )
        )

    def __mul__(self, signal_strength: float) -> np.ndarray:
        """
        Multiply signal yields with signal strength

        :param signal_strength: POI test
        :return: scaled signal yields
        :raises NegativeExpectedYields: if any bin has negative expected yields.
        """
        signal_yields = signal_strength * self.signal
        if signal_strength < 0.0:
            bin_values = signal_yields + self.background
            if np.any(bin_values < 0.0):
                raise NegativeExpectedYields(
                    "SimplifiedLikelihoodInterface::Statistical model involves negative "
                    "expected bin yields'. Bin values: "
                    + ", ".join([f"{x:.3f}" for x in bin_values])
                )
        return signal_strength * self.signal

    def __rmul__(self, signal_strength: float) -> np.ndarray:
        """Multiply signal yields with signal strength"""
        return self.__mul__(signal_strength)
