"""
Statistical Model combiner class: this class combines likelihoods 
of different statistical models for hypothesis testing
"""

import warnings
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

import numpy as np

from spey.base.hypotest_base import HypothesisTestingBase
from spey.base.model_config import ModelConfig
from spey.interface.statistical_model import StatisticalModel
from spey.optimizer.core import fit
from spey.system.exceptions import AnalysisQueryError, NegativeExpectedYields
from spey.utils import ExpectationType

__all__ = ["UnCorrStatisticsCombiner"]


class UnCorrStatisticsCombiner(HypothesisTestingBase):
    """
    Module to combine **uncorrelated** statistical models. It takes serries of
    :obj:`~spey.StatisticalModel` object as input. These statistical models does not
    need to have same backend. However, this class assumes that all the input
    statistical model's are completely independent from each other.

    .. warning::

        :obj:`~spey.UnCorrStatisticsCombiner` assumes that all input are uncorrelated and
        non of the statistical models posesses the same set of nuisance parameters. Thus
        each statistical model is optimised independently.

    Args:
        input arguments (:obj:`~spey.StatisticalModel`): Uncorrelated statistical models
        ntoys (``int``, default ``1000``): Number of toy samples for hypothesis testing.
          (Only used for toy-based hypothesis testing)

    Raises:
        :obj:`~spey.system.exceptions.AnalysisQueryError`: If multiple :class:`~spey.StatisticalModel`
          has the same :attr:`~spey.StatisticalModel.analysis` attribute.
        :obj:`TypeError`: If the input type is not :class:`~spey.StatisticalModel`.
    """

    __slots__ = ["_statistical_models"]

    def __init__(self, *args, ntoys: int = 1000):
        super().__init__(ntoys=ntoys)
        self._statistical_models = []
        for arg in args:
            self.append(arg)

    def append(self, statistical_model: StatisticalModel) -> None:
        """
        Append new independent :class:`~spey.StatisticalModel` to the stack.

        Args:
            statistical_model (:class:`~spey.StatisticalModel`): new statistical model
              to be added to the stack.

        Raises:
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If multiple :class:`~spey.StatisticalModel`
              has the same :attr:`~spey.StatisticalModel.analysis` attribute.
            :obj:`TypeError`: If the input type is not :class:`~spey.StatisticalModel`.
        """
        if isinstance(statistical_model, StatisticalModel):
            if statistical_model.analysis in self.analyses:
                raise AnalysisQueryError(f"{statistical_model.analysis} already exists.")
            self._statistical_models.append(statistical_model)
        else:
            raise TypeError(f"Can not append type {type(statistical_model)}.")

    def remove(self, analysis: Text) -> None:
        """
        Remove an analysis from the stack.

        Args:
            analysis (``Text``): unique identifier of the analysis to be removed.

        Raises:
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If the unique identifier does not match
              any of the statistical models in the stack.
        """
        to_remove = None
        for name, model in self.items():
            if name == analysis:
                to_remove = model
        if to_remove is None:
            raise AnalysisQueryError(f"'{analysis}' is not among the analyses.")
        self._statistical_models.remove(to_remove)

    @property
    def statistical_models(self) -> Tuple[StatisticalModel]:
        """
        Accessor to the statistical model stack

        Returns:
            ``Tuple[StatisticalModel]``:
            statistical model stack as an ntuple.
        """
        return tuple(self._statistical_models)

    @property
    def analyses(self) -> List[Text]:
        """
        List of the unique identifiers of the statistical models within the stack.

        Returns:
            ``List[Text]``:
            List of analysis names.
        """
        return [model.analysis for model in self]

    @property
    def minimum_poi(self) -> float:
        r"""
        Find the maximum minimum poi, :math:`\mu` value within the analysis stack.

        Returns:
            ``float``:
            minimum poi value that the combined stack can take.
        """
        return max(model.backend.config().minimum_poi for model in self)

    @property
    def is_alive(self) -> bool:
        """
        Returns true if there is at least one statistical model with at least
        one non zero bin with signal yield.

        Returns:
            ``bool``
        """
        return any(model.is_alive for model in self)

    @property
    def is_asymptotic_calculator_available(self) -> bool:
        """Check if Asymptotic calculator is available for the backend"""
        return all(model.is_asymptotic_calculator_available for model in self)

    @property
    def is_toy_calculator_available(self) -> bool:
        """Check if Toy calculator is available for the backend"""
        return False

    @property
    def is_chi_square_calculator_available(self) -> bool:
        r"""Check if :math:`\chi^2` calculator is available for the backend"""
        return all(model.is_chi_square_calculator_available for model in self)

    def __getitem__(self, item: Union[Text, int]) -> StatisticalModel:
        """Retrieve a statistical model"""
        if isinstance(item, int):
            if item < len(self):
                return self.statistical_models[item]
            raise AnalysisQueryError(
                "Request exceeds number of statistical models available."
            )
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

    def items(self) -> Iterator[Tuple[Text, StatisticalModel]]:
        """
        Returns a generator that returns analysis name and :obj:`~spey.StatisticalModel`
        every iteration.

        Returns:
            ``Iterator[Tuple[Text, StatisticalModel]]``
        """
        return ((model.analysis, model) for model in self)

    def find_most_sensitive(self) -> StatisticalModel:
        """
        If the cross sections are defined, finds the statistical model with minimum prefit
        excluded cross section value. See :attr:`~spey.StatisticalModel.s95exp`.

        Returns:
            :obj:`~spey.StatisticalModel`:
            Statistical model with minimum prefit excluded cross section value.
        """
        return self[np.argmin([model.s95exp for model in self])]

    def likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        data: Optional[Dict[Text, List[float]]] = None,
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **kwargs,
    ) -> float:
        r"""
        Compute the likelihood of the statistical model stack at a fixed parameter of interest

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest or signal strength,
              :math:`\mu`.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            data (``Dict[Text, List[float]]``, default ``None``): Input data to be used for fit. It needs
              to be given as a dictionary where the key argument is the name of the analysis and item is
              the statistical model specific data.
            statistical_model_options (``Optional[Dict[Text, Dict]]``, default ``None``): backend specific
              options. The dictionary key needs to be the backend name and the item needs to be the dictionary
              holding the keyword arguments specific to that particular backend.

              .. code-block:: python3

                >>> statistical_model_options = {"default_pdf.uncorrelated_background" : {"init_pars" : [1., 3., 4.]}}

            kwargs: keyword arguments for the optimiser.

        Returns:
            ``float``:
            value of the likelihood of the stacked statistical model.
        """
        statistical_model_options = statistical_model_options or {}
        data = data or {}

        nll = 0.0
        for statistical_model in self:

            current_kwargs = {}
            current_kwargs.update(
                statistical_model_options.get(str(statistical_model.backend_type), {})
            )
            current_data = data.get(statistical_model.analysis, None)

            try:
                nll += statistical_model.likelihood(
                    poi_test=poi_test,
                    expected=expected,
                    data=current_data,
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

    def generate_asimov_data(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistic: Text = "qtilde",
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **kwargs,
    ) -> Dict[Text, List[float]]:
        r"""
        Generate Asimov data for the statistical model. This function calls
        :func:`~spey.StatisticalModel.generate_asimov_data` function for each statistical model with
        appropriate ``statistical_model_options`` and generates Asimov data for each statistical
        model independently.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            test_statistic (``Text``, default ``"qtilde"``): test statistics.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence ``allow_negative_signal``
                    assumed to be ``False``. If this function has been executed by user, ``spey``
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through ``spey`` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            statistical_model_options (``Dict[Text, Dict]``, default ``None``): backend specific
              options. The dictionary key needs to be the backend name and the item needs to be the dictionary
              holding the keyword arguments specific to that particular backend.

              .. code-block:: python3

                >>> statistical_model_options = {"default_pdf.uncorrelated_background" : {"init_pars" : [1., 3., 4.]}}

            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Dict[Text, List[float]]``:
            Returns a dictionary for data specific to each analysis. keywords will be analysis names
            and the items are data.
        """
        statistical_model_options = statistical_model_options or {}

        data = {}
        for statistical_model in self:
            current_kwargs = {}
            current_kwargs.update(
                statistical_model_options.get(str(statistical_model.backend_type), {})
            )

            data.update(
                {
                    statistical_model.analysis: statistical_model.generate_asimov_data(
                        expected=expected,
                        test_statistic=test_statistic,
                        **current_kwargs,
                        **kwargs,
                    )
                }
            )

        return data

    def asimov_likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: Text = "qtilde",
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **kwargs,
    ) -> float:
        r"""
        Compute likelihood of the statistical model stack generated with the Asimov data.

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest, :math:`\mu`.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            test_statistics (``Text``, default ``"qtilde"``): test statistics.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence ``allow_negative_signal``
                    assumed to be ``False``. If this function has been executed by user, ``spey``
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through ``spey`` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            statistical_model_options (``Dict[Text, Dict]``, default ``None``): backend specific
              options. The dictionary key needs to be the backend name and the item needs to be the dictionary
              holding the keyword arguments specific to that particular backend.

              .. code-block:: python3

                >>> statistical_model_options = {"default_pdf.uncorrelated_background" : {"init_pars" : [1., 3., 4.]}}

            kwargs: keyword arguments for the optimiser.

        Returns:
            ``float``:
            likelihood computed for asimov data
        """
        return self.likelihood(
            poi_test=poi_test,
            expected=expected,
            return_nll=return_nll,
            data=self.generate_asimov_data(
                expected=expected,
                test_statistic=test_statistics,
                statistical_model_options=statistical_model_options,
                **kwargs,
            ),
            statistical_model_options=statistical_model_options,
            **kwargs,
        )

    def maximize_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        data: Optional[Dict[Text, List[float]]] = None,
        initial_muhat_value: Optional[float] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        statistical_model_options: Optional[Dict[Text, Dict]] = None,
        **optimiser_options,
    ) -> Tuple[float, float]:
        r"""
        Find the maximum of the likelihood.

        Args:
            return_nll (``bool``, default ``True``): If  ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            data (``Dict[Text, List[float]]``, default ``None``): Input data to be used for fit. It needs
              to be given as a dictionary where the key argument is the name of the analysis and item is
              the statistical model specific data.
            initial_muhat_value (``float``, default ``None``): Initialisation for the :math:`\hat\mu` for
              the optimiser. If ``None`` the initial value will be estimated by weighted combination of
              :math:`\hat\mu_i` where :math:`i` stands for the statistical models within the stack.

              .. math::

                    \mu_{\rm init} = \frac{1}{\sum_i \sigma_{\hat\mu, i}^2} \sum_i \frac{\hat\mu_i}{\sigma_{\hat\mu, i}^2}

              where the value of :math:`\sigma_{\hat\mu}` has been estimated by
              :func:`~spey.base.hypotest_base.HypothesisTestingBase.sigma_mu` function.
            par_bounds (``List[Tuple[float, float]]``, default  ``None``): parameter bounds for
              the optimiser.
            statistical_model_options (``Dict[Text, Dict]``, default ``None``): backend specific
              options. The dictionary key needs to be the backend name and the item needs to be the dictionary
              holding the keyword arguments specific to that particular backend.

              .. code-block:: python3

                >>> statistical_model_options = {"default_pdf.uncorrelated_background" : {"init_pars" : [1., 3., 4.]}}

            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Tuple[float, float]``:
            :math:`\hat\mu` value and maximum value of the likelihood.
        """
        statistical_model_options = statistical_model_options or {}
        data = data or {}

        # muhat initial value estimation in gaussian limit
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
                    poi_test=_mu[idx],
                    expected=expected,
                    **current_kwargs,
                    **optimiser_options,
                )
            norm = np.sum(np.power(_sigma_mu, -2))
            mu_init = np.true_divide(1.0, norm) * np.sum(
                np.true_divide(_mu, np.square(_sigma_mu))
            )

        config: ModelConfig = ModelConfig(
            poi_index=0,
            minimum_poi=self.minimum_poi,
            suggested_init=[float(mu_init)],
            suggested_bounds=(
                par_bounds
                or [
                    (
                        self.minimum_poi if allow_negative_signal else 0.0,
                        max(10.0, mu_init),
                    )
                ]
            ),
        )

        def twice_nll(poi_test: Union[float, np.ndarray]) -> float:
            """Function to compute twice negative log-likelihood for a given poi test"""
            return 2.0 * self.likelihood(
                poi_test if isinstance(poi_test, float) else poi_test[0],
                expected=expected,
                return_nll=True,
                data=data,
                statistical_model_options=statistical_model_options,
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
        r"""
        Find the maximum of the likelihood which computed with respect to Asimov data.

        Args:
            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            test_statistics (``Text``, default ``"qtilde"``): test statistic.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence :obj:`allow_negative_signal`
                    assumed to be :obj:`False`. If this function has been executed by user, :obj:`spey`
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through :obj:`spey` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            initial_muhat_value (``float``, default ``None``): Initial value for :math:`\hat\mu`.
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            statistical_model_options (``Dict[Text, Dict]``, default ``None``): backend specific
              options. The dictionary key needs to be the backend name and the item needs to be the dictionary
              holding the keyword arguments specific to that particular backend.

              .. code-block:: python3

                >>> statistical_model_options = {"default_pdf.uncorrelated_background" : {"init_pars" : [1., 3., 4.]}}

            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Tuple[float, float]``:
            :math:`\hat\mu` value and maximum value of the likelihood.
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

        data = self.generate_asimov_data(
            expected=expected,
            test_statistic=test_statistics,
            statistical_model_options=statistical_model_options,
            **optimiser_options,
        )

        def twice_nll(poi_test: Union[float, np.ndarray]) -> float:
            """Function to compute twice negative log-likelihood for a given poi test"""
            return 2.0 * self.likelihood(
                poi_test if isinstance(poi_test, float) else poi_test[0],
                expected=expected,
                data=data,
                statistical_model_options=statistical_model_options,
                **optimiser_options,
            )

        twice_nll, fit_params = fit(
            func=twice_nll,
            model_configuration=config,
            **optimiser_options,
        )

        return fit_params[0], twice_nll / 2.0 if return_nll else np.exp(-twice_nll / 2.0)
