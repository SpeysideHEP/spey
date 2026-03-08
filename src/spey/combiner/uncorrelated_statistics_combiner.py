r"""
Uncorrelated Statistical Model Combiner
========================================

This module provides :class:`UnCorrStatisticsCombiner`, which combines an arbitrary
collection of :class:`~spey.StatisticalModel` objects under the assumption that they
are statistically **independent** (uncorrelated).  Because the analyses share no
nuisance parameters, the joint likelihood factorises into a simple product, which
makes the combination both exact and computationally efficient.

Mathematical Background
-----------------------

**Factorised likelihood**

Given :math:`N` independent statistical models indexed by :math:`i`, each with its
own set of nuisance parameters :math:`\boldsymbol{\theta}_i`, the joint likelihood is

.. math::

    \mathcal{L}_{\rm comb}(\mu, \{\boldsymbol{\theta}_i\})
    = \prod_{i=1}^{N} \mathcal{L}_i(\mu, \boldsymbol{\theta}_i),

where :math:`\mu` is the single, shared parameter of interest (signal strength).
Taking the negative logarithm converts the product into a sum,

.. math::

    -\ln\mathcal{L}_{\rm comb}(\mu, \{\boldsymbol{\theta}_i\})
    = \sum_{i=1}^{N} \left[-\ln\mathcal{L}_i(\mu, \boldsymbol{\theta}_i)\right],

so the combined NLL is simply the sum of the individual NLLs evaluated at the same
:math:`\mu`.

**Profile likelihood ratio**

The profile likelihood ratio is

.. math::

    \lambda(\mu)
    = \frac{\mathcal{L}_{\rm comb}(\mu,\,\hat{\hat{\boldsymbol{\theta}}}(\mu))}
           {\mathcal{L}_{\rm comb}(\hat\mu,\,\hat{\boldsymbol{\theta}})},

where :math:`(\hat\mu, \hat{\boldsymbol{\theta}})` denote the global maximum and
:math:`\hat{\hat{\boldsymbol{\theta}}}(\mu)` denotes the conditional maximum for
fixed :math:`\mu`.  Because the analyses are independent the nuisance parameters
of each model are profiled *separately*, so

.. math::

    -2\ln\lambda_{\rm comb}(\mu)
    = \sum_{i=1}^{N} \left[-2\ln\lambda_i(\mu)\right].

This means the combined test statistic is the sum of the individual test statistics,
and standard asymptotic formulae (Wald approximation, Asimov data) apply without
modification.

**Initialisation of** :math:`\hat\mu`

When no initial value is provided, :class:`UnCorrStatisticsCombiner` estimates a
starting point for the optimiser using the Gaussian (inverse-variance) weighted
combination of the per-model best-fit signal strengths,

.. math::

    \mu_{\rm init}
    = \frac{\displaystyle\sum_i \hat\mu_i\,\sigma_{\hat\mu,i}^{-2}}
           {\displaystyle\sum_i \sigma_{\hat\mu,i}^{-2}},

where :math:`\hat\mu_i` is the best-fit signal strength for model :math:`i` and
:math:`\sigma_{\hat\mu,i}` is the corresponding uncertainty estimated via
:meth:`~spey.base.hypotest_base.HypothesisTestingBase.sigma_mu`.

**Asimov data**

Asimov data are generated independently for each constituent model and stored in a
dictionary keyed by analysis name.  The combined Asimov likelihood is then computed
by evaluating the factorised NLL at those synthetic observations.

References
----------
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for
  likelihood-based tests of new physics*, Eur. Phys. J. C **71** (2011) 1554,
  :xref:`1007.1727`.
"""

import logging
import warnings
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from spey.base.hypotest_base import HypothesisTestingBase
from spey.base.model_config import ModelConfig
from spey.interface.statistical_model import PoiTest, StatisticalModel
from spey.optimizer.core import fit
from spey.system.exceptions import AnalysisQueryError, NegativeExpectedYields
from spey.utils import ExpectationType

__all__ = ["UnCorrStatisticsCombiner"]

log = logging.getLogger("Spey")


class UnCorrStatisticsCombiner(HypothesisTestingBase):
    r"""
    Combine **uncorrelated** (independent) statistical models.

    This class accepts an arbitrary number of :class:`~spey.StatisticalModel` instances
    and treats them as statistically independent analyses.  Because the models share no
    nuisance parameters, the joint likelihood factorises into a product of individual
    likelihoods,

    .. math::

        \mathcal{L}_{\rm comb}(\mu) = \prod_{i} \mathcal{L}_i(\mu),

    and the combined negative log-likelihood (NLL) is therefore just the sum,

    .. math::

        \mathrm{NLL}_{\rm comb}(\mu) = \sum_{i} \mathrm{NLL}_i(\mu).

    The constituent models are not required to use the same backend — any mix of
    registered backends is supported.

    The combined stack is mutable: models can be added with :meth:`append` (or the
    ``@`` operator, :meth:`__matmul__`) and removed with :meth:`remove`.  Models are
    identified by their unique :attr:`~spey.StatisticalModel.analysis` string; duplicate
    analysis names are rejected.

    .. warning::

        :class:`~spey.UnCorrStatisticsCombiner` assumes that **none** of the constituent
        statistical models share nuisance parameters.  Violating this assumption leads to
        incorrect (over-confident) results because the nuisance parameters of each model
        are profiled independently rather than jointly.

    Args:
        *args (:class:`~spey.StatisticalModel`): One or more independent statistical
          models to include in the initial stack.  Additional models can be added later
          via :meth:`append`.
        ntoys (``int``, default ``1000``): Number of pseudo-experiments used when a
          toy-based calculator is requested.  This argument is passed through to the
          base class and is otherwise not used by the uncorrelated combiner itself
          (see :attr:`is_toy_calculator_available`).

    Raises:
        :obj:`~spey.system.exceptions.AnalysisQueryError`: If two or more of the
          supplied :class:`~spey.StatisticalModel` objects share the same
          :attr:`~spey.StatisticalModel.analysis` identifier.
        :obj:`TypeError`: If any positional argument is not a
          :class:`~spey.StatisticalModel` instance.

    Examples:
        >>> import spey
        >>> pdf_wrapper = spey.get_backend("default.poisson")
        >>> model_A = pdf_wrapper(signal_yields=[3.0], background_yields=[50.0],
        ...                       data=[52], analysis="SR_A")
        >>> model_B = pdf_wrapper(signal_yields=[1.5], background_yields=[20.0],
        ...                       data=[18], analysis="SR_B")
        >>> combiner = spey.UnCorrStatisticsCombiner(model_A, model_B)
        >>> combiner.exclusion_confidence_level()  # combined CLs
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

    def remove(self, analysis: str) -> None:
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
        Immutable snapshot of the current model stack.

        Returns:
            ``Tuple[StatisticalModel]``:
            All :class:`~spey.StatisticalModel` instances currently registered in the
            combiner, in insertion order.
        """
        return tuple(self._statistical_models)

    @property
    def analyses(self) -> List[str]:
        """
        Unique analysis identifiers of all models currently in the stack.

        Returns:
            ``List[str]``:
            Analysis names in the same order as :attr:`statistical_models`.
        """
        return [model.analysis for model in self]

    @property
    def minimum_poi(self) -> float:
        r"""
        Lower bound on the parameter of interest :math:`\mu` for the combined stack.

        Because all models share a single :math:`\mu`, the combined lower bound is the
        *maximum* of the per-model lower bounds: the signal strength must simultaneously
        satisfy every individual model's constraint.

        Returns:
            ``float``:
            Maximum of the individual minimum-POI values across all models in the stack.
        """
        return max(model.backend.config().minimum_poi for model in self)

    @property
    def is_alive(self) -> bool:
        """
        Whether the combined model carries any non-zero signal.

        Returns ``True`` if at least one model in the stack has at least one bin
        with a non-zero signal yield.  A combiner that is not alive cannot produce
        a meaningful exclusion limit.

        Returns:
            ``bool``:
            ``True`` if any constituent model is alive, ``False`` otherwise.
        """
        return any(model.is_alive for model in self)

    @property
    def is_asymptotic_calculator_available(self) -> bool:
        """
        Whether the asymptotic calculator is available for the combined stack.

        Returns ``True`` only when *every* constituent model supports the asymptotic
        calculator, because the combination must evaluate all individual likelihoods.

        Returns:
            ``bool``
        """
        return all(model.is_asymptotic_calculator_available for model in self)

    @property
    def is_toy_calculator_available(self) -> bool:
        """
        Whether a toy (pseudo-experiment) calculator is available.

        Toy-based combination across independent analyses would require sampling from
        the joint distribution, which is not yet implemented.  This property therefore
        always returns ``False``.

        Returns:
            ``bool``: Always ``False``.
        """
        return False

    @property
    def is_chi_square_calculator_available(self) -> bool:
        r"""
        Whether the :math:`\chi^2` calculator is available for the combined stack.

        Returns ``True`` only when *every* constituent model supports the
        :math:`\chi^2` calculator.

        Returns:
            ``bool``
        """
        return all(model.is_chi_square_calculator_available for model in self)

    def __getitem__(self, item: Union[str, int]) -> StatisticalModel:
        """
        Retrieve a constituent statistical model by index, slice, or analysis name.

        Args:
            item (``int | slice | str``): Positional index, slice object, or the
              :attr:`~spey.StatisticalModel.analysis` string of the desired model.

        Returns:
            :class:`~spey.StatisticalModel`:
            The requested model (or tuple of models for a slice).

        Raises:
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If an integer index
              is out of range or the analysis name is not found in the stack.
        """
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
        """
        Iterate over the constituent statistical models in insertion order.

        Yields:
            :class:`~spey.StatisticalModel`: Next model in the stack.
        """
        yield from self._statistical_models

    def __len__(self) -> int:
        """
        Number of statistical models currently registered in the stack.

        Returns:
            ``int``
        """
        return len(self._statistical_models)

    def items(self) -> Iterator[Tuple[str, StatisticalModel]]:
        """
        Iterate over ``(analysis_name, model)`` pairs, analogous to ``dict.items()``.

        Yields:
            ``Tuple[str, StatisticalModel]``:
            The :attr:`~spey.StatisticalModel.analysis` identifier and the corresponding
            :class:`~spey.StatisticalModel` for each entry in the stack.
        """
        return ((model.analysis, model) for model in self)

    def find_most_sensitive(self) -> StatisticalModel:
        """
        Return the constituent model with the smallest expected 95 % CL exclusion cross section.

        "Most sensitive" is defined as the analysis that places the tightest expected upper
        limit on the signal cross section (:attr:`~spey.StatisticalModel.s95exp`), i.e. the
        model for which the expected excluded signal strength is smallest.

        .. note::

            Cross-section information must be attached to each model for this method to
            work.  See :attr:`~spey.StatisticalModel.s95exp` for details.

        Returns:
            :class:`~spey.StatisticalModel`:
            The model with the minimum value of :attr:`~spey.StatisticalModel.s95exp`.
        """
        return self[np.argmin([model.s95exp for model in self])]

    def likelihood(
        self,
        poi_test: PoiTest = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        data: Optional[Dict[str, List[float]]] = None,
        statistical_model_options: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> float:
        r"""
        Evaluate the combined (profile) likelihood at a fixed signal strength :math:`\mu`.

        Under the uncorrelated assumption, the nuisance parameters of each model are
        profiled independently and the combined NLL is the sum of the individual NLLs:

        .. math::

            \mathrm{NLL}_{\rm comb}(\mu) = \sum_{i=1}^{N} \mathrm{NLL}_i(\mu).

        If any individual model raises :obj:`~spey.system.exceptions.NegativeExpectedYields`
        (which can happen at very large or very small :math:`\mu` values), a
        :obj:`RuntimeWarning` is issued and ``NaN`` is returned immediately without
        evaluating the remaining models.

        Args:
            poi_test (``float``, default ``1.0``): Signal strength :math:`\mu` at which
              to evaluate the combined likelihood.
            expected (:class:`~spey.ExpectationType`): Controls which dataset is treated
              as observed when profiling nuisance parameters and computing p-values.

              * :obj:`~spey.ExpectationType.observed`: Use the real experimental data
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the post-fit expected
                dataset (data-driven Asimov).
              * :obj:`~spey.ExpectationType.apriori`: Use the pre-fit expected dataset
                (SM Asimov; nuisance parameters set to their nominal values).

            return_nll (``bool``, default ``True``): If ``True``, return the combined
              negative log-likelihood :math:`\mathrm{NLL}_{\rm comb}(\mu)`.  If
              ``False``, return the likelihood
              :math:`\mathcal{L}_{\rm comb}(\mu) = e^{-\mathrm{NLL}_{\rm comb}(\mu)}`.
            data (``Dict[str, List[float]]``, default ``None``): Per-analysis observed
              data to override the defaults stored in each model.  Keys are analysis
              names; values are lists of bin counts or equivalent data understood by the
              respective backend.  Analyses not present in the dictionary fall back to
              their internally stored data.
            statistical_model_options (``Dict[str, Dict]``, default ``None``): Backend-
              specific keyword arguments, keyed by backend type string.  Each value is
              forwarded as ``**kwargs`` to the corresponding model's
              :meth:`~spey.StatisticalModel.likelihood` call.

              .. code-block:: python3

                >>> statistical_model_options = {
                ...     "default.uncorrelated_background": {"init_pars": [1.0, 3.0, 4.0]}
                ... }

            **kwargs: Additional keyword arguments forwarded to every individual model's
              likelihood evaluation (e.g. optimiser settings).

        Returns:
            ``float``:
            Combined NLL (if ``return_nll=True``) or combined likelihood value.
            Returns ``NaN`` if any individual model cannot be evaluated at the requested
            :math:`\mu`.
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
        test_statistic: str = "qtilde",
        statistical_model_options: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
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

                >>> statistical_model_options = {"default.uncorrelated_background" : {"init_pars" : [1., 3., 4.]}}

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
        poi_test: PoiTest = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: str = "qtilde",
        statistical_model_options: Optional[Dict[str, Dict]] = None,
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

                >>> statistical_model_options = {"default.uncorrelated_background" : {"init_pars" : [1., 3., 4.]}}

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
        data: Optional[Dict[str, List[float]]] = None,
        initial_muhat_value: Optional[float] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        statistical_model_options: Optional[Dict[str, Dict]] = None,
        poi_indices: Optional[List[Union[int, str]]] = None,
        **optimiser_options,
    ) -> Tuple[Union[float, Dict[Union[int, str], float]], float]:
        r"""
        Find the global maximum of the combined likelihood over :math:`\mu`.

        The optimiser minimises :math:`-2\ln\mathcal{L}_{\rm comb}(\mu)` with respect
        to the single shared signal strength :math:`\mu`, while the nuisance parameters
        of each constituent model are profiled out independently inside
        :meth:`likelihood`.

        **Initialisation heuristic**

        If ``initial_muhat_value`` is not provided, the starting point for the optimiser
        is estimated in the Gaussian limit using the inverse-variance weighted mean of
        the per-model best-fit values,

        .. math::

            \mu_{\rm init}
            = \frac{\sum_i \hat\mu_i\,/\,\sigma_{\hat\mu,i}^2}
                   {\sum_i 1\,/\,\sigma_{\hat\mu,i}^2},

        where :math:`\hat\mu_i` is obtained from
        :meth:`~spey.StatisticalModel.maximize_likelihood` and
        :math:`\sigma_{\hat\mu,i}` from
        :meth:`~spey.base.hypotest_base.HypothesisTestingBase.sigma_mu`.  If this
        estimation fails for any reason (e.g. a model is not yet converging), the
        initialisation falls back to :math:`\mu_{\rm init} = 0`.

        Args:
            return_nll (``bool``, default ``True``): If ``True``, return the minimised
              NLL; if ``False``, return the likelihood value at the maximum.
            expected (:class:`~spey.ExpectationType`): Dataset prescription used when
              profiling nuisance parameters.

              * :obj:`~spey.ExpectationType.observed`: Real data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected data.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit SM Asimov data.

            allow_negative_signal (``bool``, default ``True``): If ``True``,
              :math:`\hat\mu` is unconstrained from below.  If ``False``, the lower
              bound is set to zero (i.e. :math:`\hat\mu \geq 0`).
            data (``Dict[str, List[float]]``, default ``None``): Per-analysis data
              overrides; see :meth:`likelihood` for the expected format.
            initial_muhat_value (``float``, default ``None``): Explicit starting value
              for the optimiser.  When ``None`` the weighted-mean heuristic described
              above is used.
            par_bounds (``List[Tuple[float, float]]``, default ``None``): Explicit
              bounds ``[(mu_min, mu_max)]`` for the optimiser.  When ``None`` the
              bounds are derived from :attr:`minimum_poi` and ``initial_muhat_value``.
            statistical_model_options (``Dict[str, Dict]``, default ``None``): Backend-
              specific keyword arguments forwarded to each model's calls.

              .. code-block:: python3

                >>> statistical_model_options = {
                ...     "default.uncorrelated_background": {"init_pars": [1.0, 3.0, 4.0]}
                ... }

            poi_indices (``List[int | str]``, default ``None``): If provided, the
              returned :math:`\hat\mu` is wrapped in a dictionary keyed by the entries
              of this list (useful when downstream code expects a mapping of POI names
              to values).
            **optimiser_options: Additional keyword arguments forwarded to the
              :func:`~spey.optimizer.core.fit` routine.

        Returns:
            ``Tuple[float | Dict, float]``:
            A 2-tuple of (:math:`\hat\mu`, :math:`\mathrm{NLL}_{\rm min}`).
            :math:`\hat\mu` is a plain ``float`` when ``poi_indices`` is ``None``,
            or a ``Dict`` mapping each entry of ``poi_indices`` to the same
            :math:`\hat\mu` value otherwise.
        """
        statistical_model_options = statistical_model_options or {}
        data = data or {}

        # muhat initial value estimation in gaussian limit
        mu_init = initial_muhat_value or 0.0
        if initial_muhat_value is None:
            try:
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
            except Exception as err:
                log.debug(str(err))
                mu_init = 0.0

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

        nll = twice_nll / 2.0 if return_nll else np.exp(-twice_nll / 2.0)
        if poi_indices is None:
            return fit_params[0], nll
        return {key: float(fit_params[0]) for key in poi_indices}, nll

    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: str = "qtilde",
        initial_muhat_value: Optional[float] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        statistical_model_options: Optional[Dict[str, Dict]] = None,
        poi_indices: Optional[List[Union[int, str]]] = None,
        **optimiser_options,
    ) -> Tuple[Union[float, Dict[Union[int, str], float]], float]:
        r"""
        Find the maximum of the combined likelihood evaluated on Asimov data.

        This method first generates per-model Asimov datasets via
        :meth:`generate_asimov_data`, then maximises the combined NLL with respect to
        :math:`\mu` using those synthetic observations.  The result is used to compute
        the expected (median) test statistic under the signal or background hypothesis,
        which is the basis of the asymptotic CLs calculation.

        The choice of ``test_statistics`` controls both the Asimov dataset generation
        and the signal-strength constraint applied during minimisation:

        * ``'qtilde'`` / ``'q0'``: require :math:`\hat\mu \geq 0`
          (``allow_negative_signal = False``).
        * ``'q'`` / ``'qmu'``: allow :math:`\hat\mu < 0`
          (``allow_negative_signal = True``).

        Args:
            return_nll (``bool``, default ``True``): If ``True``, return the minimised
              NLL; if ``False``, return the likelihood value at the maximum.
            expected (:class:`~spey.ExpectationType`): Dataset prescription used when
              generating Asimov data and profiling nuisance parameters.

              * :obj:`~spey.ExpectationType.observed`: Use the real data to set nuisance
                parameters before generating Asimov data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected dataset.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit SM Asimov data.

            test_statistics (``str``, default ``"qtilde"``): Test statistic used for
              Asimov data generation and to determine the :math:`\mu` constraint.

              * ``'qtilde'``: Alternative test statistic :math:`\tilde{q}_{\mu}`, see
                eq. (62) of :xref:`1007.1727`.  Assumes :math:`\hat\mu \geq 0`.

                .. warning::

                    When ``test_statistics='qtilde'`` is used, ``allow_negative_signal``
                    is set to ``False`` internally.  If you call this method directly,
                    ensure that your external code handles this constraint consistently.
                    When p-values or upper limits are computed through ``spey`` this is
                    managed automatically.

              * ``'q'``: Test statistic :math:`q_{\mu}`, see eq. (54) of
                :xref:`1007.1727`.  Allows :math:`\hat\mu < 0`.
              * ``'q0'``: Discovery test statistic :math:`q_{0}`, see eq. (47) of
                :xref:`1007.1727`.  Assumes :math:`\hat\mu \geq 0`.

            initial_muhat_value (``float``, default ``None``): Starting value for the
              optimiser.  Defaults to ``0.0`` when ``None``.
            par_bounds (``List[Tuple[float, float]]``, default ``None``): Explicit
              ``[(mu_min, mu_max)]`` bounds for the optimiser.
            statistical_model_options (``Dict[str, Dict]``, default ``None``): Backend-
              specific keyword arguments forwarded to each model.

              .. code-block:: python3

                >>> statistical_model_options = {
                ...     "default.uncorrelated_background": {"init_pars": [1.0, 3.0, 4.0]}
                ... }

            poi_indices (``List[int | str]``, default ``None``): If provided, the
              returned :math:`\hat\mu` is wrapped in a dictionary keyed by these values.
            **optimiser_options: Additional keyword arguments forwarded to
              :func:`~spey.optimizer.core.fit`.

        Returns:
            ``Tuple[float | Dict, float]``:
            A 2-tuple of (:math:`\hat\mu`, :math:`\mathrm{NLL}_{\rm min}`), where
            :math:`\hat\mu` is a ``float`` or a ``Dict`` depending on ``poi_indices``.
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

        nll = twice_nll / 2.0 if return_nll else np.exp(-twice_nll / 2.0)
        if poi_indices is None:
            return fit_params[0], nll
        return {key: float(fit_params[0]) for key in poi_indices}, nll

    def __matmul__(
        self, other: Union[StatisticalModel, "UnCorrStatisticsCombiner"]
    ) -> "UnCorrStatisticsCombiner":
        """
        Merge two combiners (or a combiner and a single model) using the ``@`` operator.

        Returns a **new** :class:`UnCorrStatisticsCombiner` that contains all models from
        ``self`` followed by the model(s) from ``other``.  The original combiner is not
        modified.

        Args:
            other (:class:`~spey.StatisticalModel` | :class:`UnCorrStatisticsCombiner`):
              A single statistical model or another combiner whose models are to be
              appended to the new stack.

        Returns:
            :class:`UnCorrStatisticsCombiner`:
            A new combiner containing the union of models from both operands.

        Raises:
            :obj:`ValueError`: If ``other`` is neither a :class:`~spey.StatisticalModel`
              nor an :class:`UnCorrStatisticsCombiner`.
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If any analysis name in
              ``other`` already exists in ``self``.

        Examples:
            >>> combined = combiner_A @ combiner_B
            >>> combined = combiner_A @ single_model
        """
        new_model = UnCorrStatisticsCombiner(*self._statistical_models)
        if isinstance(other, StatisticalModel):
            new_model.append(other)
        elif isinstance(other, UnCorrStatisticsCombiner):
            for model in other:
                new_model.append(model)
        else:
            raise ValueError(
                f"Can not combine type<{type(other)}> with UnCorrStatisticsCombiner"
            )
        return new_model
