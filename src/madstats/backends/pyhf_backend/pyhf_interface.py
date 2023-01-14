import logging, scipy
from typing import Dict, Union, Optional, Tuple, List
from numpy import warnings, isnan
import numpy as np

import pyhf
from pyhf.infer.calculators import generate_asimov_data

from madstats.utils import ExpectationType
from madstats.base.backend_base import BackendBase
from .utils import compute_negloglikelihood, initialise_workspace, compute_min_negloglikelihood

pyhf.pdf.log.setLevel(logging.CRITICAL)
pyhf.workspace.log.setLevel(logging.CRITICAL)
pyhf.set_backend("numpy", precision="64b")


class PyhfInterface(BackendBase):
    """
    Pyhf Interface

    :param signal: either histfactory type signal patch or float value of number of events
    :param background: either background only JSON histfactory or float value of observed data
    :param nb: expected number of background events. In case of statistical model it is not needed
    :param delta_nb: uncertainty on backgorund. In case of statistical model it is not needed
    """

    def __init__(
        self,
        signal: Union[List, float],
        background: Union[Dict, float],
        nb: Optional[float] = None,
        delta_nb: Optional[float] = None,
    ):
        super().__init__(signal, background)
        self.model, self.data, self.workspace, self._exp = None, None, None, None

        self.nb = nb
        self.delta_nb = delta_nb
        self._expectation_fixed = False

    @classmethod
    def fixed_expectation(
        cls,
        signal: Union[List, float],
        background: Union[Dict, float],
        nb: Optional[float] = None,
        delta_nb: Optional[float] = None,
        expected: Optional[ExpectationType] = ExpectationType.observed,
    ):
        """
        Fixing the expectation prevents reinitialising the
        statistical model and allows faster computation.

        :param signal: either histfactory type signal patch or float value of number of events
        :param background: either background only JSON histfactory or float value of observed data
        :param nb: expected number of background events. In case of statistical model it is not needed
        :param delta_nb: uncertainty on backgorund. In case of statistical model it is not needed
        :param expected: observed, expected (true, apriori) or aposteriori
        """
        interface = cls(signal, background, nb, delta_nb)
        interface._initialize_statistical_model(expected)
        interface._expectation_fixed = True
        return interface

    def _initialize_statistical_model(
        self, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> None:
        """
        Initialize the statistical model
        :param expected: observed, expected (true, apriori) or aposteriori
        """
        if (self._exp != expected or self._exp is None) and not self._expectation_fixed:
            self._exp = expected
            self.workspace, self.model, self.data = initialise_workspace(
                self.signal,
                self.background,
                self.nb,
                self.delta_nb,
                expected,
            )

    def computeCLs(
        self,
        mu: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        iteration_threshold: int = 3,
        **kwargs,
    ) -> Union[float, Dict]:
        """
        Compute exclusion confidence level.

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :param iteration_threshold: sets threshold on when to stop
        :return: CLs values {"CLs_obs": xx, "CLs_exp": [xx] * 5} or single CLs value
        """
        expected = ExpectationType.as_expectationtype(expected)
        self._initialize_statistical_model(expected)

        if self.model is None or self.data is None:
            if "CLs_exp" in kwargs.keys() or "CLs_obs" in kwargs.keys():
                return -1
            else:
                return {"CLs_obs": -1, "CLs_exp": [-1] * 5}

        def get_CLs(model, data, **keywordargs):
            try:
                CLs_obs, CLs_exp = pyhf.infer.hypotest(
                    mu,
                    data,
                    model,
                    test_stat=keywordargs.get("stats", "qtilde"),
                    par_bounds=keywordargs.get("bounds", model.config.suggested_bounds()),
                    return_expected_set=True,
                )

            except (AssertionError, pyhf.exceptions.FailedMinimization, ValueError) as err:
                logging.getLogger("MA5").debug(str(err))
                # dont use false here 1.-CLs = 0 can be interpreted as false
                return "update bounds"

            # if isnan(float(CLs_obs)) or any([isnan(float(x)) for x in CLs_exp]):
            #     return "update mu"
            CLs_obs = float(CLs_obs[0]) if isinstance(CLs_obs, (list, tuple)) else float(CLs_obs)

            return {
                "CLs_obs": 1.0 - CLs_obs,
                "CLs_exp": list(map(lambda x: float(1.0 - x), CLs_exp)),
            }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # pyhf can raise an error if the poi_test bounds are too stringent
            # they need to be updated dynamically.
            arguments = dict(bounds=self.model.config.suggested_bounds(), stats="qtilde")
            it = 0
            while True:
                CLs = get_CLs(self.model, self.data, **arguments)
                if CLs == "update bounds":
                    arguments["bounds"][self.model.config.poi_index] = (
                        arguments["bounds"][self.model.config.poi_index][0],
                        2 * arguments["bounds"][self.model.config.poi_index][1],
                    )
                    logging.getLogger("MA5").debug(
                        "Hypothesis test inference integration bounds has been increased to "
                        + str(arguments["bounds"][self.model.config.poi_index])
                    )
                    it += 1
                elif isinstance(CLs, dict):
                    if isnan(CLs["CLs_obs"]) or any([isnan(x) for x in CLs["CLs_exp"]]):
                        arguments["stats"] = "q"
                        arguments["bounds"][self.model.config.poi_index] = (
                            arguments["bounds"][self.model.config.poi_index][0] - 5,
                            arguments["bounds"][self.model.config.poi_index][1],
                        )
                        logging.getLogger("MA5").debug(
                            "Hypothesis test inference integration bounds has been increased to "
                            + str(arguments["bounds"][self.model.config.poi_index])
                        )
                    else:
                        break
                else:
                    it += 1
                # hard limit on iteration required if it exceeds this value it means
                # Nsig >>>>> Nobs
                if it >= iteration_threshold:
                    if "CLs_exp" in kwargs.keys() or "CLs_obs" in kwargs.keys():
                        return 1
                    return {"CLs_obs": 1.0, "CLs_exp": [1.0] * 5}

        if kwargs.get("CLs_exp", False):
            return CLs["CLs_exp"][2]
        elif kwargs.get("CLs_obs", False):
            return CLs["CLs_obs"]

        return CLs

    def likelihood(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: bool = False,
        return_nll: Optional[bool] = False,
        isAsimov: Optional[bool] = False,
        mu_lim: Optional[Tuple[float, float]] = (-20.0, 40.0),
        iteration_threshold: Optional[int] = 3,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: if true, POI can get negative values
        :param return_nll: if true returns negative log-likelihood value
        :param isAsimov: if true, computes likelihood for Asimov data
        :param mu_lim: boundaries for mu
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: (float) likelihood
        """
        expected = ExpectationType.as_expectationtype(expected)

        self._initialize_statistical_model(expected)
        data = self.data

        if self.model is None or self.data is None:
            return -1
        # set a threshold for mu
        if not isinstance(mu, float):
            mu = 1.0
        # protection during the scan
        if mu < mu_lim[0]:
            mu = mu_lim[0]
        elif mu > mu_lim[1]:
            mu = mu_lim[1]

        if isAsimov:
            data = generate_asimov_data(
                0.0,
                self.data,
                self.model,
                self.model.config.suggested_init(),
                self.model.config.suggested_bounds(),
                self.model.config.suggested_fixed(),
                return_fitted_pars=False,
            )

        negloglikelihood = compute_negloglikelihood(
            mu,
            data,
            self.model,
            allow_negative_signal,
            iteration_threshold,
        )

        return negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = False,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        iteration_threshold: Optional[int] = 3,
    ) -> Tuple[float, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param return_nll: if true, likelihood will be returned
        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: allow negative POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: muhat, maximum of the likelihood
        """
        self._initialize_statistical_model(expected)
        data = self.data
        if isAsimov:
            data = generate_asimov_data(
                0.0,
                self.data,
                self.model,
                self.model.config.suggested_init(),
                self.model.config.suggested_bounds(),
                self.model.config.suggested_fixed(),
                return_fitted_pars=False,
            )

        muhat, negloglikelihood = compute_min_negloglikelihood(
            data, self.model, allow_negative_signal, iteration_threshold
        )

        return muhat, negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def chi2(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
    ) -> float:
        """
        Compute $$\chi^2$$

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}_{\mu = 1}}{\mathcal{L}_{max}}\right)

        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: allow negative POI
        :return: chi^2
        """
        return 2.0 * (
            self.likelihood(1.0, expected, allow_negative_signal, True)
            - self.maximize_likelihood(True, expected, allow_negative_signal)[1]
        )

    def computeUpperLimitOnMu(
        self, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, expected (true, apriori) or aposteriori
        :return: mu
        """
        expected = ExpectationType.as_expectationtype(expected)

        kwargs = dict(
            expected=expected,
            CLs_obs=expected in [ExpectationType.apriori, ExpectationType.observed],
            CLs_exp=expected == ExpectationType.aposteriori,
        )
        computer = lambda mu: self.computeCLs(mu=mu, **kwargs) - 0.95

        low, hig = 1.0, 1.0
        while computer(low) > 0.95:
            low *= 0.1
            if low < 1e-10:
                break
        while computer(hig) < 0.95:
            hig *= 10.0
            if hig > 1e10:
                break

        return scipy.optimize.brentq(computer, low, hig, xtol=low / 100.0)
