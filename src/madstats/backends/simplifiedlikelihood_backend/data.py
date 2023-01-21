import numpy as np
from dataclasses import dataclass
from typing import Text, Optional
from collections import namedtuple

expansion_output = output = namedtuple(
    "expansion", ["A", "B", "C", "rho", "V", "logdet_covariance", "inv_covariance"]
)


@dataclass(frozen=True)
class Data:
    """
    Data structure for simplified likelihoods

    :param observed: number of observed events.
    :param signal: number of signal events.
    :param background: number of expected background yields.
    :param covariance: Covariance matrix or single region uncertainty
    :param delta_sys: systematic uncertainty on signal.
    :param third_moment: third moment.
    :param name: name of the statistical model.

    :raises TypeError: If the types of the inputs does not match the expected types
    :raises AssertionError: If the dimensionality of the inputs are wrong.
    """

    observed: np.ndarray
    signal: np.ndarray
    background: np.ndarray
    covariance: np.ndarray
    delta_sys: Optional[float] = 0.2
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
            raise TypeError("Invalid type.")

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

    def __repr__(self):
        return (
            f"Data(\n    name='{self.name}',"
            f"\n    data structure that represents {len(self)} regions,"
            f"\n    delta_sys={self.delta_sys:.1f},"
            f"\n    isLinear={self.isLinear}\n)"
        )

    def reset_observations(self, observations: np.ndarray, name: Text):
        """
        Create the same statistical model with different observed yields

        :param observations: new observed yields
        :param name: name of the statistical model.
        :return: creates a new dataset by replacing the observations
        :raises AssertionError: if dimensionality of input does not match the
                                current statistical model
        """
        assert len(observations) == len(
            self
        ), "Dimensionality of the input does not match the statistical model."
        return Data(
            observations,
            self.signal,
            self.background,
            self.covariance,
            self.delta_sys,
            self.third_moment,
            name,
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
        return self.isLinear and len(self) == 1

    @property
    def diag_cov(self) -> np.ndarray:
        """Retrieve diagonal terms of the covariance matrix"""
        return np.diag(self.covariance)

    def compute_expansion(self) -> expansion_output:
        """Compute the terms described in arXiv:1809.05548 eqs. 3.10, 3.11, 3.12, 3.13"""
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

        diag_cov = self.diag_cov

        # arXiv:1809.05548 eq. 3.13
        C: np.ndarray = np.zeros(shape=diag_cov.shape)
        for idx, (m2, m3) in enumerate(zip(diag_cov, self.third_moment)):
            m3 = m3 if m3 != 0 else 1e-30
            k = -np.sign(m3) * np.sqrt(2.0 * m2)
            dm = np.sqrt(8.0 * m2**3 / m3**2 - 1.0)
            C[idx] = k * np.cos(4.0 * np.pi / 3.0 + np.arctan(dm) / 3.0)

        B: np.ndarray = np.sqrt(diag_cov - 2.0 * np.square(C))  # B, as defined in Eq. 3.11
        A: np.ndarray = self.background - C  # A, Eq. 1.30
        rho = np.zeros(shape=(len(self), len(self)))
        for idx in range(len(self)):
            for idy in range(len(self)):
                bxby = B[idx] * B[idy]
                cxcy = C[idx] * C[idy]
                e = (4.0 * cxcy) ** (-1) * (
                    np.sqrt(bxby**2.0 + 8.0 * cxcy * self.covariance[idx][idy]) - bxby
                )
                rho[idx][idy] = e
                rho[idy][idx] = e

        V = np.zeros(shape=(len(B), len(B)))
        for idx in range(len(B)):
            for idy in range(idx, len(B)):
                T = B[idx] * B[idy] * rho[idx][idy]
                V[idx][idy] = T
                V[idy][idx] = T

        return expansion_output(A, B, C, rho, V, np.linalg.slogdet(V), np.linalg.inv(V))

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Compute correlation matrix computed from covariance matrix"""
        corr = np.zeros(shape=self.covariance.shape)
        for idx in range(self.covariance.shape[0]):
            corr[idx][idx] = 1.0
            for idy in range(idx + 1, self.covariance.shape[0]):
                rho = self.covariance[idx][idy] / np.sqrt(
                    self.covariance[idx][idx] * self.covariance[idy][idy]
                )
                corr[idx][idy] = rho
                corr[idy][idx] = rho
        return corr

    def __mul__(self, signal_strength: float) -> np.ndarray:
        """Multiply signal yields with signal strength"""
        return signal_strength * self.signal

    def __rmul__(self, signal_strength: float) -> np.ndarray:
        """Multiply signal yields with signal strength"""
        return self.__mul__(signal_strength)
