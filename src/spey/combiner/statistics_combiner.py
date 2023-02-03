import warnings, scipy
import numpy as np
from typing import Optional, List, Text, Union, Generator, Any

from spey.interface.statistical_model import StatisticalModel
from spey.utils import ExpectationType
from spey.system.exceptions import AnalysisQueryError, NegativeExpectedYields
from spey.base.recorder import Recorder
from spey.base.hypotest_base import HypothesisTestingBase

__all__ = ["StatisticsCombiner"]


class StatisticsCombiner(HypothesisTestingBase):
    """
    Statistical model combination routine

    :param args: Statistical models
    """

    __slots__ = ["_statistical_models", "_recorder"]

    def __init__(self, *args):
        self._statistical_models = list()
        self._recorder = Recorder()
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

    def __iter__(self) -> StatisticalModel:
        """Iterate over statistical models"""
        yield from self._statistical_models

    def __len__(self):
        """Number of statistical models within the stack"""
        return len(self._statistical_models)

    def items(self) -> Generator[tuple[Any, Any], Any, None]:
        """Returns a generator for analysis name and corresponding statistical model"""
        return ((model.analysis, model) for model in self)

    def find_most_sensitive(self) -> StatisticalModel:
        """
        Find the most sensitive statistical model which will return
        the model with minimum expected excluded cross-section
        """
        results = [model.s95exp for model in self]
        return self[results.index(min(results))]

    def likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood for the statistical model with a given POI

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3

            >>> from spey import AvailableBackends
            >>> kwargs = {
            >>>     str(AvailableBackends.pyhf): {"iteration_threshold": 3},
            >>>     str(AvailableBackends.simplified_likelihoods): {"marginalize": False},
            >>> }

        This will allow keyword arguments to be chosen with respect to specific backend.

        :return: likelihood value
        """
        if self._recorder.get_poi_test(expected, poi_test) is not False and not isAsimov:
            nll = self._recorder.get_poi_test(expected, poi_test)
        else:
            nll = 0.0
            for statistical_model in self:

                current_kwargs = {}
                current_kwargs.update(kwargs.get(str(statistical_model.backend_type), {}))

                try:
                    nll += statistical_model.backend.likelihood(
                        poi_test=poi_test,
                        expected=expected,
                        return_nll=True,
                        isAsimov=isAsimov,
                        **current_kwargs,
                    )
                except NegativeExpectedYields as err:
                    warnings.warn(
                        err.args[0] + f"\nSetting NLL({poi_test:.3f}) = inf",
                        category=RuntimeWarning,
                    )
                    nll = np.nan

                if np.isnan(nll):
                    break

            if not isAsimov:
                self._recorder.record_poi_test(expected, poi_test, nll)

        return nll if return_nll or np.isnan(nll) else np.exp(-nll)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        poi_upper_bound: Optional[float] = 10.0,
        isAsimov: Optional[bool] = False,
        maxiter: Optional[int] = 200,
        **kwargs,
    ):
        """
        Minimize negative log-likelihood of the statistical model with respect to POI

        :param return_nll: if true returns negative log-likelihood value
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param poi_upper_bound: Set upper bound for POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param maxiter: number of iterations to be held for convergence of the fit.
        :param kwargs: model dependent arguments. In order to specify backend specific inputs
                       provide the input in the following format

        .. code-block:: python3

            >>> import spey
            >>> combiner = spey.StatisticsCombiner(stat_model1, stat_model2)
            >>> kwargs = {
            >>>     str(spey.AvailableBackends.pyhf): {"iteration_threshold": 20},
            >>>     str(spey.AvailableBackends.simplified_likelihoods): {"marginalize": False},
            >>> }
            >>> muhat_apri, nll_min_apri = combiner.maximize_likelihood(
            >>>     return_nll=True,
            >>>     expected=spey.ExpectationType.apriori,
            >>>     allow_negative_signal=True,
            >>>     **kwargs
            >>> )

        This will allow keyword arguments to be chosen with respect to specific backend.
        :return: POI that minimizes the negative log-likelihood, minimum negative log-likelihood
        """
        if self._recorder.get_maximum_likelihood(expected) is not False and not isAsimov:
            muhat, nll = self._recorder.get_maximum_likelihood(expected)
        else:
            for current_model in self:
                current_model.backend._recorder.pause()

            twice_nll = lambda mu: np.array(
                [
                    2
                    * self.likelihood(
                        mu[0], expected=expected, return_nll=True, isAsimov=isAsimov, **kwargs
                    )
                ],
                dtype=np.float64,
            )

            # It is possible to allow user to modify the optimiser properties in the future
            opt = scipy.optimize.minimize(
                twice_nll,
                [0.0],
                method="SLSQP",
                bounds=[(self.minimum_poi_test if allow_negative_signal else 0.0, poi_upper_bound)],
                tol=1e-6,
                options={"maxiter": maxiter},
            )

            if not opt.success:
                raise RuntimeWarning("Optimiser was not able to reach required precision.")

            nll, muhat = opt.fun / 2.0, opt.x[0]

            if not isAsimov:
                self._recorder.record_maximum_likelihood(expected, muhat, nll)

            for current_model in self:
                current_model.backend._recorder.play()

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
            self.likelihood(poi_test=1.0, expected=expected, return_nll=True, **kwargs)
            - self.maximize_likelihood(
                return_nll=True,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                **kwargs,
            )[1]
        )
