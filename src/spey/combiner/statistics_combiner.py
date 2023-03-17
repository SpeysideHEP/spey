"""
Statistical Model combiner class: this class combines likelihoods 
of different statistical models for hypothesis testing
"""

from typing import List, Text, Generator, Any, Tuple, Union, Dict, Optional
import warnings
import numpy as np

from spey.interface.statistical_model import StatisticalModel
from spey.utils import ExpectationType
from spey.system.exceptions import AnalysisQueryError, NegativeExpectedYields
from spey.base.recorder import Recorder
from spey.base.hypotest_base import HypothesisTestingBase
from spey.optimizer.core import fit
from spey.base.model_config import ModelConfig

__all__ = ["StatisticsCombiner"]


class StatisticsCombiner(HypothesisTestingBase):
    """
    Statistical model combination routine

    :param args: Statistical models
    """

    __slots__ = ["_statistical_models", "_recorder"]

    def __init__(self, *args):
        self._statistical_models = []
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
    def minimum_poi(self) -> float:
        """Find minimum POI test that can be applied to this statistical model"""
        return max(model.backend.model.minimum_poi for model in self)

    @property
    def isAlive(self) -> bool:
        """Is there any statistical model with non-zero signal yields in any region"""
        return any(model.isAlive for model in self)

    def __getitem__(self, item: Union[Text, int]) -> StatisticalModel:
        """Retrieve a statistical model"""
        if isinstance(item, int):
            if item < len(self):
                return self.statistical_models[item]
            raise AnalysisQueryError("Request exceeds number of statistical models available.")
        if isinstance(item, slice):
            return self.statistical_models[item]

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
        return self[np.argmin([model.s95exp for model in self])]

    def likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **kwargs,
    ) -> float:
        """
        Compute the combined likelihood of the statistical model collection

        :param poi_test (`float`, default `1.0`): parameter of interest (signal strength).
        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param return_nll (`bool`, default `True`): if false returns likelihood value, if true will return negative
            log-likelihood value.
        :param statistical_model_options (`Optional[Dict[Text, Dict]]`, default `None`): statistical model specific options.
            should be in the following form:

        ..code-block:: python3
            >>> statistical_model_options = {
            >>>     str(spey.AvailableBackends.pyhf): {"opt1": value},
            >>>     str(spey.AvailableBackends.simplified_likelihoods): {"opt1": value},
            >>> }

        note that each type of dictionary will be fed with respect to the backend type.

        :param kwargs: keyword arguments for optimiser.
        :return `float`: likelihood value
        """
        statistical_model_options = statistical_model_options or {}

        nll = 0.0
        for statistical_model in self:

            current_kwargs = {}
            current_kwargs.update(
                statistical_model_options.get(str(statistical_model.backend_type), {})
            )

            try:
                nll += statistical_model.likelihood(
                    poi_test=poi_test,
                    expected=expected,
                    **current_kwargs,
                    **kwargs,
                )
            except NegativeExpectedYields as err:
                warnings.warn(
                    err.args[0] + f"\nSetting NLL({poi_test:.3f}) = nan",
                    category=RuntimeWarning,
                )
                nll = np.nan

            if np.isnan(nll):
                break

        return nll if return_nll or np.isnan(nll) else np.exp(-nll)

    def asimov_likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: Text = "qtilde",
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **kwargs,
    ) -> float:
        """
        Compute the combined likelihood of the statistical model collection for Asimov data.
        Asimov data for each statistical model is computed independently.

        :param poi_test (`float`, default `1.0`): parameter of interest (signal strength).
        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param return_nll (`bool`, default `True`): if false returns likelihood value, if true will return negative
            log-likelihood value.
        :param test_statistics (`Text`, default `"qtilde"`): test statistics, `q`, `qtilde` or `q0`
        :param statistical_model_options (`Optional[Dict[Text, Dict]]`, default `None`): statistical model specific options.
            should be in the following form:

        ..code-block:: python3
            >>> statistical_model_options = {
            >>>     str(spey.AvailableBackends.pyhf): {"opt1": value},
            >>>     str(spey.AvailableBackends.simplified_likelihoods): {"opt1": value},
            >>> }

        note that each type of dictionary will be fed with respect to the backend type.
        :param kwargs: keyword arguments for optimiser.
        :return `float`: likelihood value for Asimov data
        """
        statistical_model_options = statistical_model_options or {}

        nll = 0.0
        for statistical_model in self:

            current_kwargs = {}
            current_kwargs.update(
                statistical_model_options.get(str(statistical_model.backend_type), {})
            )

            nll += statistical_model.asimov_likelihood(
                poi_test=poi_test,
                expected=expected,
                test_statistics=test_statistics,
                **current_kwargs,
                **kwargs,
            )

            if np.isnan(nll):
                break

        return nll if return_nll or np.isnan(nll) else np.exp(-nll)

    def maximize_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        initial_muhat_value: Optional[float] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **optimiser_options,
    ) -> Tuple[float, float]:
        r"""
        Minimize negative log-likelihood of the combined statistical model with respect to POI

        :param return_nll (`bool`, default `True`): if true returns negative log-likelihood value.
        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param allow_negative_signal (`bool`, default `True`): if true, $\hat\mu$ is allowed to be negative.
            Note that this has been superseeded by the `par_bounds` option defined by the user.
        :param initial_muhat_value (`float`, default `None`): Initial value for muhat. If None,
            an initial value will be estimated with respect to $\hat\mu_i$ weighted by $\sigma_{\hat\mu}$.

        ..math::
            \tilde{\mu} = \mathcal{N}\sum \frac{\hat\mu_i}{\sigma^2_{\hat\mu}}\quad , \quad \mathcal{N} = \sum\frac{1}{\sigma^2_{\hat\mu}}

        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): User defined upper and lower limits for muhat.
            If none the lower limit will be set as the maximum $\mu$ value that the statistical model ensample can take and the max
            value will be set to 10.
        :param statistical_model_options (`Optional[Dict[Text, Dict]]`, default `None`): options for the likelihood computation of the where user
            can define individual options for each statistical model type.

        .. code-block:: python3

            >>> import spey
            >>> combiner = spey.StatisticsCombiner(stat_model1, stat_model2)
            >>> statistical_model_options = {
            >>>     str(spey.AvailableBackends.pyhf): {"init_pars": [1,1,0.5]},
            >>>     str(spey.AvailableBackends.simplified_likelihoods): {"init_pars": [1,2,0.3,0.5]},
            >>> }
            >>> muhat, nll_min = combiner.maximize_likelihood(
            >>>     statistical_model_options=statistical_model_options
            >>> )

        :param kwargs: Additional optimizer specific options.
        :return `Tuple[float, float]`: $\hat\mu$ value and minimum negative log-likelihood
        """
        statistical_model_options = statistical_model_options or {}

        # muhat initial value estimation
        mu_init = initial_muhat_value or 0.0
        if initial_muhat_value is None:
            _mu, _sigma_mu = np.zeros(len(self)), np.ones(len(self))
            for idx, stat_model in enumerate(self):

                current_kwargs = {}
                current_kwargs.update(
                    statistical_model_options.get(str(stat_model.backend_type), {})
                )

                _mu[idx] = stat_model.maximize_likelihood(
                    expected=expected, **current_kwargs, **optimiser_options
                )[0]
                _sigma_mu[idx] = stat_model.sigma_mu(
                    poi_test=_mu[idx], expected=expected, **current_kwargs, **optimiser_options
                )
            mu_init = np.sum(np.power(_sigma_mu, -2)) * np.sum(
                np.true_divide(_mu, np.square(_sigma_mu))
            )

        config: ModelConfig = ModelConfig(
            poi_index=0,
            minimum_poi=self.minimum_poi,
            suggested_init=[float(mu_init)],
            suggested_bounds=(
                par_bounds or [(self.minimum_poi if allow_negative_signal else 0.0, 10.0)]
            ),
        )

        def twice_nll(poi_test: Union[float, np.ndarray]) -> float:
            """Function to compute twice negative log-likelihood for a given poi test"""
            return 2.0 * self.likelihood(
                poi_test if isinstance(poi_test, float) else poi_test[0],
                expected=expected,
                return_nll=True,
                **statistical_model_options,
                **optimiser_options,
            )

        twice_nll, fit_params = fit(
            func=twice_nll,
            model_configuration=config,
            **optimiser_options,
        )

        return fit_params[0], twice_nll / 2.0 if return_nll else np.exp(-twice_nll / 2.0)

    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        initial_muhat_value: Optional[float] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **optimiser_options,
    ) -> Tuple[float, float]:
        """
        Find maximum of the likelihood for the asimov data

        :param expected (`ExpectationType`): observed, apriori or aposteriori,.
            (default `ExpectationType.observed`)
        :param return_nll (`bool`): if false, likelihood value is returned.
            (default `True`)
        :param test_statistics (`Text`): test statistics. `"qmu"` or `"qtilde"` for exclusion
                                     tests `"q0"` for discovery test. (default `"qtilde"`)
        :return `Tuple[float, float]`: muhat, negative log-likelihood
        """
        allow_negative_signal: bool = True if test_statistics in ["q", "qmu"] else False

        config: ModelConfig = ModelConfig(
            poi_index=0,
            minimum_poi=self.minimum_poi,
            suggested_init=[initial_muhat_value or 0.0],
            suggested_bounds=(
                par_bounds or [(self.minimum_poi if allow_negative_signal else 0.0, 10.0)]
            ),
        )

        statistical_model_options = statistical_model_options or {}

        def twice_nll(poi_test: Union[float, np.ndarray]) -> float:
            """Function to compute twice negative log-likelihood for a given poi test"""
            return 2.0 * self.asimov_likelihood(
                poi_test if isinstance(poi_test, float) else poi_test[0],
                expected=expected,
                **statistical_model_options,
                **optimiser_options,
            )

        twice_nll, fit_params = fit(
            func=twice_nll,
            model_configuration=config,
            **optimiser_options,
        )

        return fit_params[0], twice_nll / 2.0 if return_nll else np.exp(-twice_nll / 2.0)
