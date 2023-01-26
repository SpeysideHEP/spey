import warnings, scipy
import numpy as np
from typing import Optional, List, Text, Union, Generator, Any

from madstats.interface.statistical_model import StatisticalModel
from madstats.utils import ExpectationType
from madstats.tools.utils_cls import compute_confidence_level
from madstats.system.exceptions import AnalysisQueryError, NegativeExpectedYields


class PredictionCombiner:
    """
    Statistical model combination routine

    :param args: Statistical models
    """

    __slots__ = ["_statistical_models"]

    def __init__(self, *args):
        self._statistical_models = list()
        for arg in args:
            self.append(arg)

    def append(self, statistical_model: StatisticalModel) -> None:
        """
        Add new analysis to the statistical model stack

        :param statistical_model: statistical model to be added to the stack
        :raises AnalysisQueryError: if analysis name matches with another analysis within the stack
        """
        if isinstance(statistical_model, StatisticalModel):
            if statistical_model.analysis in self.analyses:
                raise AnalysisQueryError(f"{statistical_model.analysis} already exists.")
            self._statistical_models.append(statistical_model)
        else:
            raise TypeError(f"Can not append type {type(statistical_model)}.")

    def remove(self, analysis: Text) -> None:
        """Remove a specific analysis from the model list"""
        to_remove = None
        for name, model in self.items():
            if name == analysis:
                to_remove = model
        if to_remove is None:
            raise AnalysisQueryError(f"'{analysis}' is not among the analyses.")
        else:
            self._statistical_models.remove(to_remove)

    @property
    def statistical_models(self) -> List[StatisticalModel]:
        """Retreive the list of statistical models"""
        return self._statistical_models

    @property
    def analyses(self) -> List[Text]:
        """List of analyses that are included in combiner database"""
        return [model.analysis for model in self]

    @property
    def minimum_poi_test(self) -> float:
        """Find minimum POI test that can be applied to this statistical model"""
        return max([model.backend.model.minimum_poi_test for model in self])

    def __getitem__(self, item: Union[Text, int]) -> StatisticalModel:
        """Retrieve a statistical model"""
        if isinstance(item, int):
            if item < len(self):
                return self.statistical_models[item]
            else:
                raise AnalysisQueryError(f"Request exceeds number of statistical models available.")

        for model in self:
            if model.analysis == item:
                return model
        raise AnalysisQueryError(f"'{item}' is not among the analyses.")

    def __iter__(self):
        """Iterate over statistical models"""
        for model in self._statistical_models:
            yield model

    def __len__(self):
        """Number of statistical models within the stack"""
        return len(self._statistical_models)

    def items(self) -> Generator[tuple[Any, Any], Any, None]:
        """Returns a generator for analysis name and corresponding statistical model"""
        return ((model.analysis, model) for model in self)

    def likelihood(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood for the statistical model with a given POI

        :param mu: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3

            kwargs = {
                str(AvailableBackends.pyhf): {"iteration_threshold": 3},
                str(AvailableBackends.simplified_likelihoods): {"marginalize": False},
            }

        This will allow keyword arguments to be chosen with respect to specific backend.

        :return: likelihood value
        """

        nll = 0.0
        for statistical_model in self:

            current_kwargs = {}
            current_kwargs.update(kwargs.get(str(statistical_model.backend_type), {}))

            try:
                nll += statistical_model.backend.likelihood(
                    mu=mu, expected=expected, return_nll=True, **current_kwargs
                )
            except NegativeExpectedYields as err:
                warnings.warn(err.args[0] + f" Setting NLL({mu:3f}) = inf", category=RuntimeWarning)
                nll = np.inf

            if np.isinf(nll):
                break

        return nll if return_nll else np.exp(-nll)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        iteration_threshold: Optional[int] = 10000,
        **kwargs,
    ):
        """
        Minimize negative log-likelihood of the statistical model with respect to POI

        :param return_nll: if true returns negative log-likelihood value
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3
            import madstats
            combiner = madstats.PredictionCombiner(stat_model1, stat_model2)
            kwargs = {
                str(AvailableBackends.pyhf): {"iteration_threshold": 20},
                str(AvailableBackends.simplified_likelihoods): {"marginalize": False},
            }
            muhat_apri, nll_min_apri = combiner.maximize_likelihood(
                return_nll=True,
                allow_negative_signal=True,
                expected=madstats.ExpectationType.apriori,
                **kwargs
            )

        This will allow keyword arguments to be chosen with respect to specific backend.
        :return: POI that minimizes the negative log-likelihood, minimum negative log-likelihood
        """
        negloglikelihood = lambda mu: self.likelihood(
            mu[0], expected=expected, return_nll=True, **kwargs
        )

        muhat_init = np.random.uniform(
            self.minimum_poi_test if allow_negative_signal else 0.0, 10.0, (1,)
        )

        # It is possible to allow user to modify the optimiser properties in the future
        opt = scipy.optimize.minimize(
            negloglikelihood,
            muhat_init,
            method="SLSQP",
            bounds=[(self.minimum_poi_test if allow_negative_signal else 0.0, 40.0)],
            tol=1e-6,
            options={"maxiter": iteration_threshold},
        )

        if not opt.success:
            raise RuntimeWarning("Optimiser was not able to reach required precision.")

        nll, muhat = opt.fun, opt.x[0]
        if not allow_negative_signal and muhat < 0.0:
            muhat, nll = 0.0, negloglikelihood([0.0])

        return muhat, nll if return_nll else np.exp(-nll)

    def chi2(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ):
        """
        Compute $$\chi^2$$

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}_{\mu = 1}}{\mathcal{L}_{max}}\right)

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3

            kwargs = {
                str(AvailableBackends.pyhf): {"iteration_threshold": 20},
                str(AvailableBackends.simplified_likelihoods): {"marginalize": False},
            }

        This will allow keyword arguments to be chosen with respect to specific backend.
        :return: chi^2
        """
        return 2.0 * (
            self.likelihood(mu=1.0, expected=expected, return_nll=True, **kwargs)
            - self.maximize_likelihood(
                return_nll=True,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                **kwargs,
            )[1]
        )

    def _exclusion_tools(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
        **kwargs,
    ):
        """
        Compute tools needed for exclusion limit computation

        :param expected: observed, apriori or aposteriori
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param allow_negative_signal: if true, allow negative mu
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3

            kwargs = {
                str(AvailableBackends.pyhf): {"iteration_threshold": 20},
                str(AvailableBackends.simplified_likelihoods): {"marginalize": False},
            }

        This will allow keyword arguments to be chosen with respect to specific backend.
        """
        muhat, min_nll = self.maximize_likelihood(
            return_nll=True,
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
            **kwargs,
        )

        muhat_asimov, min_nll_asimov = self.maximize_likelihood(
            return_nll=True,
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
            isAsimov=True,
            **kwargs,
        )

        negloglikelihood = lambda mu: self.likelihood(
            mu[0], expected=expected, return_nll=True, **kwargs
        )
        negloglikelihood_asimov = lambda mu: self.likelihood(
            mu[0],
            expected=expected,
            return_nll=True,
            isAsimov=True,
            **kwargs,
        )

        return min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood

    def computeCLs(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        iteration_threshold: Optional[int] = 10000,
        **kwargs,
    ) -> float:
        """
        Compute 1 - CLs value

        :param mu: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3

            kwargs = {
                str(AvailableBackends.pyhf): {"iteration_threshold": 20},
                str(AvailableBackends.simplified_likelihoods): {"marginalize": False},
            }

        This will allow keyword arguments to be chosen with respect to specific backend.
        :return: 1 - CLs
        """
        min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood = self._exclusion_tools(
            expected=expected,
            allow_negative_signal=False,
            iteration_threshold=iteration_threshold,
            **kwargs,
        )

        return 1.0 - compute_confidence_level(
            1.0, negloglikelihood_asimov, min_nll_asimov, negloglikelihood, min_nll
        )

    def computeUpperLimitOnMu(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        iteration_threshold: Optional[int] = 10000,
        **kwargs,
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, apriori or aposteriori
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3

            kwargs = {
                str(AvailableBackends.pyhf): {"iteration_threshold": 20},
                str(AvailableBackends.simplified_likelihoods): {"marginalize": False},
            }

        This will allow keyword arguments to be chosen with respect to specific backend.
        :return: excluded POI value at 95% CLs
        """

        min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood = self._exclusion_tools(
            expected=expected,
            allow_negative_signal=False,
            iteration_threshold=iteration_threshold,
            **kwargs,
        )

        computer = lambda mu: 0.05 - compute_confidence_level(
            mu, negloglikelihood_asimov, min_nll_asimov, negloglikelihood, min_nll
        )

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
