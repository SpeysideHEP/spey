"""Data structure for Simplified Likelihood interface"""

from dataclasses import dataclass
from typing import Text, Optional
from collections import namedtuple

import numpy as np

from spey.system.exceptions import NegativeExpectedYields
from spey.base import ModelConfig

__all__ = ["SLData", "expansion_output"]

expansion_output = output = namedtuple(
    "expansion", ["A", "B", "C", "rho", "V", "logdet_covariance", "inv_covariance"]
)


@dataclass(frozen=True)
class SLData:
    """
    Data container for simplified likelihoods

    Args:
        observed (``np.ndarray``): observed yields
        signal (``np.ndarray``): signal yields
        background (``np.ndarray``): simulated background yields
        covariance (``np.ndarray``): covariance matrix
        delta_sys (``float``, default ``0.0``): systematic uncertainty on signal yields
        third_moment (``np.ndarray``, default ``None``): third moment for skewed gaussian
        name (``str``): name of the dataset.

    Raises:
        ``TypeError``: If the data is not ``numpy`` array.
    """

    observed: np.ndarray
    signal: np.ndarray
    background: np.ndarray
    covariance: np.ndarray
    delta_sys: Optional[float] = 0.0
    third_moment: Optional[np.ndarray] = None
    name: Text = "__unknown_model__"

    def __post_init__(self):
        # validate inputs
        if not (
            isinstance(self.observed, np.ndarray)
            and isinstance(self.signal, np.ndarray)
            and isinstance(self.background, np.ndarray)
            and isinstance(self.covariance, np.ndarray)
            and isinstance(self.delta_sys, float)
            and (self.third_moment is None or isinstance(self.third_moment, np.ndarray))
        ):
            raise TypeError(
                f"Invalid type.\nobserved: {type(self.observed)}, "
                f"\nsignal: {type(self.signal)}, "
                f"\nbackground: {type(self.background)}, "
                f"\ncovariance: {type(self.covariance)}, "
                f"\ndelta_sys: {type(self.delta_sys)}, "
                f"\nthird_moment: {type(self.third_moment)}"
            )

        assert len(self.covariance.shape) == 2, "Covariance input has to be matrix."

        assert (
            len(self.observed)
            == len(self.signal)
            == len(self.background)
            == self.covariance.shape[0]
            == self.covariance.shape[1]
            >= 1
        ), "Input shapes does not match"

        if self.third_moment is not None:
            assert len(self.diag_cov) == len(
                self.third_moment
            ), "Dimensionality of the third moment does not match with covariance matrix."
            assert np.all(
                8.0 * np.power(self.diag_cov, 3) / np.power(self.third_moment, 2) >= 1.0
            ), "Inequality for third moment has not been satisfied."

    def __repr__(self):
        return (
            f"SLData(\n    name='{self.name}',"
            f"\n    data structure that represents {len(self)} regions,"
            f"\n    delta_sys={self.delta_sys:.1f},"
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
            self.covariance,
            self.delta_sys,
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
            suggested_bounds=[(minimum_poi if allow_negative_signal else 0.0, poi_upper_bound)]
            + [(-5.0, 5.0)] * len(self),
        )

    @property
    def expected_dataset(self):
        """Retrieve expected dataset"""
        return self.reset_observations(self.background, f"{self.name}_exp")

    def __len__(self) -> int:
        return len(self.observed)

    @property
    def var_s(self) -> np.ndarray:
        """Variance of the signal yields"""
        return self.var_smu(1.0)

    def var_smu(self, signal_strength: float) -> np.ndarray:
        """Variance of the signal yields for a given signal strength"""
        return np.diag(np.square(signal_strength * self * self.delta_sys))

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
        return np.diag(self.covariance)

    def compute_expansion(self) -> expansion_output:
        """Compute the terms described in :xref:`1809.05548` eqs. 3.10, 3.11, 3.12, 3.13"""
        if self.isLinear:
            return expansion_output(
                None,
                None,
                None,
                None,
                self.covariance,
                np.linalg.slogdet(self.covariance),
                np.linalg.inv(self.covariance),
            )

        diag_cov: np.ndarray = self.diag_cov

        # arXiv:1809.05548 eq. 3.13
        C: np.ndarray = np.zeros(shape=diag_cov.shape)
        for idx, (m2, m3) in enumerate(zip(diag_cov, self.third_moment)):
            m3 = np.clip(m3, 1e-10, None)
            k = -np.sign(m3) * np.sqrt(2.0 * m2)
            dm = np.sqrt(8.0 * m2**3 / m3**2 - 1.0)
            C[idx] = k * np.cos(4.0 * np.pi / 3.0 + np.arctan(dm) / 3.0)

        B: np.ndarray = np.sqrt(diag_cov - 2.0 * np.square(C))  # B, as defined in Eq. 3.11
        A: np.ndarray = self.background - C  # A, Eq. 1.30
        Cmat: np.ndarray = C.reshape(-1, 1) @ C.reshape(1, -1)
        Bmat: np.ndarray = B.reshape(-1, 1) @ B.reshape(1, -1)
        rho: np.ndarray = np.power(4.0 * Cmat, -1) * (
            np.sqrt(np.square(Bmat) + 8.0 * Cmat * self.covariance) - Bmat
        )
        rho = np.tril(rho) + np.triu(rho.T, 1)

        V: np.ndarray = np.zeros(shape=(len(B), len(B)))
        # pylint: disable=C0200
        for idx in range(len(B)):
            for idy in range(idx, len(B)):
                T = B[idx] * B[idy] * rho[idx][idy]
                V[idx][idy] = T
                V[idy][idx] = T

        return expansion_output(A, B, C, rho, V, np.linalg.slogdet(V), np.linalg.inv(V))

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Compute correlation matrix computed from covariance matrix"""
        inv_sqrt_diag = np.linalg.inv(np.sqrt(np.diag(self.diag_cov)))
        return inv_sqrt_diag @ self.covariance @ inv_sqrt_diag

    @property
    def minimum_poi(self) -> float:
        """Find minimum POI test that can be applied to this statistical model"""
        if np.all(self.signal == 0.0):
            return -np.inf
        return -np.min(
            np.true_divide(self.background[self.signal != 0.0], self.signal[self.signal != 0.0])
        )

    def suggested_theta_init(self, poi_test: float = 1.0) -> np.ndarray:
        """
        Compute nuisance parameter theta that minimizes the negative log-likelihood by setting
        dNLL / dtheta = 0

        :param poi_test: POI (signal strength)
        :return:
        """
        diag_cov = np.diag(self.var_smu(poi_test) + self.covariance)
        total_expected = self.background + poi_test * self

        q = diag_cov * (total_expected - self.observed)
        p = total_expected + diag_cov

        return -p / 2.0 + np.sign(p) * np.sqrt(np.square(p) / 4.0 - q)

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
