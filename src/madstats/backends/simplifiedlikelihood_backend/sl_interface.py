from madstats.base.backend_base import BackendBase
from madstats.utils import ExpectationType

from .data import Data
from .likelihood_computer import LikelihoodComputer
from .upper_limit_computer import UpperLimitComputer

from typing import Union, Dict, Optional, Tuple
import numpy as np


class SimplifiedLikelihoodInterface(BackendBase):
    """
    Simplified Likelihood Interface

    :param signal: signal events
    :param background: observed events
    :param covariance: covariance matrix. In case of single region this acts as delta_nb
    :param nb: expected number of background events.
    :param third_moment:
    :param delta_sys: systematic uncertainty on signal.
    """

    # This is temporary definition, will be changed as the SL backend fixed
    EXPECTATION = {
        str(ExpectationType.observed): False,
        str(ExpectationType.apriori): True,
        str(ExpectationType.aposteriori): "posteriori",
    }

    def __init__(
        self,
        signal: Union[float, np.ndarray],
        background: Union[float, np.ndarray],
        covariance: Union[float, np.ndarray],
        nb: Optional[np.ndarray] = None,
        third_moment: Optional[np.ndarray] = None,
        delta_sys: float = 0.2,
    ):
        super().__init__(signal, background)
        self.nb = nb

        assert (
            len(signal) == len(background)
            and len(background) == covariance.shape[0]
            and len(nb) == len(background)
        ), "Incorrect dimensionality"

        self.data = Data(
            background,
            nb,
            covariance,
            third_moment,
            signal,
            "model",
            delta_sys,
        )

    def computeCLs(
        self,
        mu: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        **kwargs,
    ) -> Union[float, Dict]:
        """
        Compute exclusion confidence level.

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :return: 1 - CLs value
        """
        computer = UpperLimitComputer()
        return computer.computeCLs(
            self.data, expected=SimplifiedLikelihoodInterface.EXPECTATION[str(expected)]
        )

    def computeUpperLimitOnMu(
        self, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """

        :param expected:
        :return:
        """
        interface = UpperLimitComputer()
        return interface.getUpperLimitOnMu(self.data, marginalize=False, expected=expected)

    def likelihood(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: bool = False,
        return_nll: Optional[bool] = False,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: if true, POI can get negative values
        :param return_nll: if true returns negative log-likelihood value
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: (float) likelihood
        """
        # TODO add Asimov construction
        data = self.data
        if expected == ExpectationType.apriori:
            data = self.data.get_expected()
        computer = LikelihoodComputer(data)
        return computer.likelihood(mu, marginalize=False, nll=return_nll)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = False,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param return_nll: if true, likelihood will be returned
        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: allow negative POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs:
        :return: muhat, maximum of the likelihood
        """
        data = self.data
        if expected == ExpectationType.apriori:
            data = self.data.get_expected()
        computer = LikelihoodComputer(data)
        res = computer.findMuHat(
            allowNegativeSignals=allow_negative_signal,
            nll=return_nll,
            extended_output=True,
        )
        return res["muhat"], res["lmax"]
