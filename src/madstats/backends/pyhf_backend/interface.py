import copy, logging, scipy, warnings
from typing import Dict, Union, Optional, Tuple
import numpy as np

import pyhf
from pyhf.infer.calculators import generate_asimov_data

from madstats.utils import ExpectationType
from madstats.base.backend_base import BackendBase
from .utils import compute_negloglikelihood, compute_min_negloglikelihood
from .data import Data
from madstats.backends import AvailableBackends

pyhf.pdf.log.setLevel(logging.CRITICAL)
pyhf.workspace.log.setLevel(logging.CRITICAL)
pyhf.set_backend("numpy", precision="64b")


class PyhfInterface(BackendBase):
    """
    Pyhf Interface

    :param model: contains all the information regarding the regions, yields
    :raises AssertionError: if the input type is wrong.
    """

    __slots__ = "_model"

    def __init__(self, model: Data):
        assert isinstance(model, Data), "Invalid statistical model."
        self._model = model

    @property
    def model(self) -> Data:
        """Retrieve statistical model"""
        return self._model

    @property
    def type(self) -> AvailableBackends:
        return AvailableBackends.pyhf

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
        :param kwargs:
            :param CLs_exp: if true return expected of the posterior fit
            :param CLs_obs: if true return observed or apriori expectation depending on the
                            expected flag.
        :return: 1 - CLs values {"CLs_obs": xx, "CLs_exp": [xx] * 5} or a single 1 - CLs value

        Note CLs_exp output is the expected of the posterior fit and comes with mean,
        1sigma and 2sigma expected exclusion limits. If `kwargs = {"CLs_exp" : True}` only the mean
        of CLs_exp will be returned, if `kwargs = {"CLs_obs":True}` only CLs_obs value will be
        returned. For the expected of the prefit simply set `expected = ExpectationType.apriori`
        and `kwargs = {"CLs_obs":True}`.
        """
        if not self.model.isAlive:
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

        _, model, data = self.model(mu=1.0, expected=expected)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # pyhf can raise an error if the poi_test bounds are too stringent
            # they need to be updated dynamically.
            arguments = dict(bounds=model.config.suggested_bounds(), stats="qtilde")
            it = 0
            while True:
                CLs = get_CLs(model, data, **arguments)
                if CLs == "update bounds":
                    arguments["bounds"][model.config.poi_index] = (
                        arguments["bounds"][model.config.poi_index][0],
                        2 * arguments["bounds"][model.config.poi_index][1],
                    )
                    logging.getLogger("MA5").debug(
                        "Hypothesis test inference integration bounds has been increased to "
                        + str(arguments["bounds"][model.config.poi_index])
                    )
                    it += 1
                elif isinstance(CLs, dict):
                    if np.isnan(CLs["CLs_obs"]) or any([np.isnan(x) for x in CLs["CLs_exp"]]):
                        arguments["stats"] = "q"
                        arguments["bounds"][model.config.poi_index] = (
                            arguments["bounds"][model.config.poi_index][0] - 5,
                            arguments["bounds"][model.config.poi_index][1],
                        )
                        logging.getLogger("MA5").debug(
                            "Hypothesis test inference integration bounds has been increased to "
                            + str(arguments["bounds"][model.config.poi_index])
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
        allow_negative_signal: bool = True,
        return_nll: Optional[bool] = False,
        isAsimov: Optional[bool] = False,
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

        _, model, data = self.model(mu=1.0, expected=expected)

        # if not self.model.isAlive:
        #     return np.inf if return_nll else 0.0


        if isAsimov:
            data = generate_asimov_data(
                0.0,
                data,
                model,
                model.config.suggested_init(),
                model.config.suggested_bounds(),
                model.config.suggested_fixed(),
                return_fitted_pars=False,
            )

        # CHECK THE MODEL BOUNDS!!
        # POI Test needs to be adjusted according to the boundaries for sake of convergence
        # see issue https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579235311
        # comment https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579299831
        poi_test = copy.deepcopy(mu)
        bounds = model.config.suggested_bounds()[model.config.poi_index]
        if not bounds[0] <= poi_test <= bounds[1]:
            _, model, data = self.model(mu = mu, expected = expected)
            poi_test = 1.0

        negloglikelihood = compute_negloglikelihood(
            poi_test,
            data,
            model,
            allow_negative_signal,
            iteration_threshold,
        )

        if np.isnan(negloglikelihood):
            return np.inf if return_nll else 0.0

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
        _, model, data = self.model(mu=1.0, expected=expected)
        if isAsimov:
            data = generate_asimov_data(
                0.0,
                data,
                model,
                model.config.suggested_init(),
                model.config.suggested_bounds(),
                model.config.suggested_fixed(),
                return_fitted_pars=False,
            )

        muhat, negloglikelihood = compute_min_negloglikelihood(
            data, model, allow_negative_signal, iteration_threshold
        )

        return muhat, negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def chi2(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
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
            self.likelihood(
                mu=1.0,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                return_nll=True,
                isAsimov=isAsimov,
            )
            - self.maximize_likelihood(
                return_nll=True,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                isAsimov=isAsimov,
            )[1]
        )

    def computeUpperLimitOnMu(
        self, expected: Optional[ExpectationType] = ExpectationType.observed, **kwargs
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, expected (true, apriori) or aposteriori
        :return: mu
        """
        kwargs.update(
            dict(
                expected=expected,
                CLs_obs=expected in [ExpectationType.apriori, ExpectationType.observed],
                CLs_exp=expected == ExpectationType.aposteriori,
            )
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
