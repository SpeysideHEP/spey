r"""
Deprecated Uncorrelated Combiner
================================

:class:`UnCorrStatisticsCombiner` is the historical, mutable interface for combining
independent statistical models.  It is kept for backwards compatibility only: every
likelihood evaluation is delegated to the ``default.uncorrelated_combiner`` plug-in
(:class:`~spey.combiner.UncorrelatedStatisticsCombiner`).

.. deprecated:: 0.2.8

    Use the plug-in instead:

    .. code-block:: python3

        >>> combiner = spey.get_backend("default.uncorrelated_combiner")
        >>> combined = combiner(statistical_models=[model_a, model_b], analysis="combined")
"""

import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from spey.base.hypotest_base import HypothesisTestingBase
from spey.interface.statistical_model import PoiTest, StatisticalModel
from spey.system.exceptions import AnalysisQueryError, NegativeExpectedYields
from spey.utils import ExpectationType

from .uncorrelated_statistics_combiner import UncorrelatedStatisticsCombiner

__all__ = ["UnCorrStatisticsCombiner"]

log = logging.getLogger("Spey")

_DEPRECATION_MESSAGE = (
    "`spey.UnCorrStatisticsCombiner` is deprecated and will not be available in future "
    "versions of spey. Please use the `default.uncorrelated_combiner` plug-in instead: "
    '`spey.get_backend("default.uncorrelated_combiner")(statistical_models=[...], '
    'analysis="...")`.'
)


def __dir__():
    return __all__


class UnCorrStatisticsCombiner(HypothesisTestingBase):
    r"""
    Combine **uncorrelated** (independent) statistical models.

    .. deprecated:: 0.2.8

        This class will not be available in future versions of ``spey``.  It is now a
        thin wrapper around the ``default.uncorrelated_combiner`` plug-in
        (:class:`~spey.combiner.UncorrelatedStatisticsCombiner`); use the plug-in
        directly:

        .. code-block:: python3

            >>> combiner = spey.get_backend("default.uncorrelated_combiner")
            >>> combined = combiner(statistical_models=[model_a, model_b], analysis="AB")

    The combined negative log-likelihood is the sum of the individual ones at a common
    signal strength :math:`\mu`,

    .. math::

        \mathrm{NLL}_{\rm comb}(\mu) = \sum_{i} \mathrm{NLL}_i(\mu),

    see :mod:`spey.combiner.uncorrelated_statistics_combiner` for the derivation.

    The combined stack is mutable: models can be added with :meth:`append` (or the
    ``@`` operator, :meth:`__matmul__`) and removed with :meth:`remove`.  Models are
    identified by their unique :attr:`~spey.StatisticalModel.analysis` string; duplicate
    analysis names are rejected.

    .. warning::

        :class:`~spey.UnCorrStatisticsCombiner` assumes that **none** of the constituent
        statistical models share nuisance parameters.  Violating this assumption leads to
        incorrect (over-confident) results.

    Args:
        *args (:class:`~spey.StatisticalModel`): One or more independent statistical
          models to include in the initial stack.  Additional models can be added later
          via :meth:`append`.
        ntoys (``int``, default ``1000``): Number of pseudo-experiments, passed through
          to the base class.  The toy calculator is not available through this class
          (see :attr:`is_toy_calculator_available`); use the plug-in instead.

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

    __slots__ = ["_statistical_models", "_combined"]

    def __init__(self, *args, ntoys: int = 1000):
        warnings.warn(_DEPRECATION_MESSAGE, category=FutureWarning, stacklevel=2)
        super().__init__(ntoys=ntoys)
        self._statistical_models: List[StatisticalModel] = []
        self._combined: Optional[StatisticalModel] = None
        for arg in args:
            self.append(arg)

    # ------------------------------------------------------------------
    # Stack management
    # ------------------------------------------------------------------
    def append(self, statistical_model: StatisticalModel) -> None:
        """
        Append new independent :class:`~spey.StatisticalModel` to the stack.

        Args:
            statistical_model (:class:`~spey.StatisticalModel`): new statistical model
              to be added to the stack.

        Raises:
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If multiple
              :class:`~spey.StatisticalModel` has the same
              :attr:`~spey.StatisticalModel.analysis` attribute.
            :obj:`TypeError`: If the input type is not :class:`~spey.StatisticalModel`.
        """
        if not isinstance(statistical_model, StatisticalModel):
            raise TypeError(f"Can not append type {type(statistical_model)}.")
        if statistical_model.analysis in self.analyses:
            raise AnalysisQueryError(f"{statistical_model.analysis} already exists.")
        self._statistical_models.append(statistical_model)
        self._combined = None

    def remove(self, analysis: str) -> None:
        """
        Remove an analysis from the stack.

        Args:
            analysis (``str``): unique identifier of the analysis to be removed.

        Raises:
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If the unique identifier
              does not match any of the statistical models in the stack.
        """
        for idx, model in enumerate(self._statistical_models):
            if model.analysis == analysis:
                self._statistical_models.pop(idx)
                self._combined = None
                return
        raise AnalysisQueryError(f"'{analysis}' is not among the analyses.")

    @property
    def combined_model(self) -> StatisticalModel:
        """
        The current stack as a ``default.uncorrelated_combiner`` statistical model.

        This is the object every likelihood evaluation is delegated to, and the
        recommended replacement for this class.  It is rebuilt whenever the stack
        changes.

        Raises:
            :obj:`~spey.system.exceptions.InvalidInput`: If the stack is empty.

        Returns:
            :class:`~spey.StatisticalModel`:
            Statistical model wrapping
            :class:`~spey.combiner.UncorrelatedStatisticsCombiner`.
        """
        if self._combined is None:
            self._combined = StatisticalModel(
                backend=UncorrelatedStatisticsCombiner(self._statistical_models),
                analysis="+".join(self.analyses),
                ntoys=self.ntoys,
            )
        return self._combined

    @property
    def statistical_models(self) -> Tuple[StatisticalModel, ...]:
        """
        Immutable snapshot of the current model stack.

        Returns:
            ``Tuple[~spey.StatisticalModel, ...]``:
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
        *maximum* of the per-model lower bounds.

        Returns:
            ``float``:
            Maximum of the individual minimum-POI values across all models in the stack.
        """
        return max(model.backend.config().minimum_poi for model in self)

    @property
    def is_alive(self) -> bool:
        """
        Whether at least one model in the stack carries a non-zero signal yield.

        Returns:
            ``bool``
        """
        return any(model.is_alive for model in self)

    @property
    def is_asymptotic_calculator_available(self) -> bool:
        """
        Whether *every* constituent model supports the asymptotic calculator.

        Returns:
            ``bool``
        """
        return all(model.is_asymptotic_calculator_available for model in self)

    @property
    def is_toy_calculator_available(self) -> bool:
        """
        Whether a toy (pseudo-experiment) calculator is available.

        Always ``False`` for this deprecated class; the
        ``default.uncorrelated_combiner`` plug-in supports toys whenever every
        constituent model does.

        Returns:
            ``bool``: Always ``False``.
        """
        return False

    @property
    def is_chi_square_calculator_available(self) -> bool:
        r"""
        Whether *every* constituent model supports the :math:`\chi^2` calculator.

        Returns:
            ``bool``
        """
        return all(model.is_chi_square_calculator_available for model in self)

    def __getitem__(self, item: Union[str, int, slice]) -> StatisticalModel:
        """
        Retrieve a constituent statistical model by index, slice, or analysis name.

        Args:
            item (``Union[str, int, slice]``): Positional index, slice object, or the
              :attr:`~spey.StatisticalModel.analysis` string of the desired model.

        Raises:
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If an integer index
              is out of range or the analysis name is not found in the stack.

        Returns:
            :class:`~spey.StatisticalModel`:
            The requested model (or tuple of models for a slice).
        """
        if isinstance(item, (int, np.integer)):
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

    def __iter__(self) -> Iterator[StatisticalModel]:
        """Iterate over the constituent statistical models in insertion order."""
        yield from self._statistical_models

    def __len__(self) -> int:
        """Number of statistical models currently registered in the stack."""
        return len(self._statistical_models)

    def items(self) -> Iterator[Tuple[str, StatisticalModel]]:
        """
        Iterate over ``(analysis_name, model)`` pairs, analogous to ``dict.items()``.

        Returns:
            ``Iterator[Tuple[str, ~spey.StatisticalModel]]``
        """
        return ((model.analysis, model) for model in self)

    def find_most_sensitive(self) -> StatisticalModel:
        """
        Return the constituent model with the smallest expected excluded cross section.

        .. note::

            Cross-section information must be attached to each model, see
            :attr:`~spey.StatisticalModel.s95exp`.

        Returns:
            :class:`~spey.StatisticalModel`:
            The model with the minimum value of :attr:`~spey.StatisticalModel.s95exp`.
        """
        return self[int(np.argmin([model.s95exp for model in self]))]

    def __matmul__(
        self, other: Union[StatisticalModel, "UnCorrStatisticsCombiner"]
    ) -> "UnCorrStatisticsCombiner":
        """
        Return a **new** combiner holding the models of ``self`` followed by ``other``.

        Args:
            other (:class:`~spey.StatisticalModel` | :class:`UnCorrStatisticsCombiner`):
              A single statistical model or another combiner.

        Raises:
            :obj:`ValueError`: If ``other`` is neither a :class:`~spey.StatisticalModel`
              nor an :class:`UnCorrStatisticsCombiner`.
            :obj:`~spey.system.exceptions.AnalysisQueryError`: If any analysis name in
              ``other`` already exists in ``self``.

        Returns:
            :class:`UnCorrStatisticsCombiner`:
            A new combiner containing the union of models from both operands.
        """
        if isinstance(other, StatisticalModel):
            others = [other]
        elif isinstance(other, UnCorrStatisticsCombiner):
            others = list(other)
        else:
            raise ValueError(
                f"Can not combine type<{type(other)}> with UnCorrStatisticsCombiner"
            )
        with warnings.catch_warnings():
            # The user has already been warned when constructing `self`.
            warnings.simplefilter("ignore", FutureWarning)
            new_model = UnCorrStatisticsCombiner(
                *self._statistical_models, ntoys=self.ntoys
            )
        for model in others:
            new_model.append(model)
        return new_model

    # ------------------------------------------------------------------
    # Delegation to the plug-in
    # ------------------------------------------------------------------
    @staticmethod
    def _ignore_options(statistical_model_options: Optional[Dict[str, Dict]]) -> None:
        """Warn that per-backend options no longer apply to the joint fit."""
        if statistical_model_options:
            log.warning(
                "`statistical_model_options` is ignored: the models are now fitted "
                "jointly by the `default.uncorrelated_combiner` plug-in."
            )

    def _nan_on_negative_yields(self, poi_test: PoiTest, evaluate) -> float:
        """Evaluate a likelihood, returning ``nan`` if negative yields are encountered."""
        try:
            return evaluate()
        except NegativeExpectedYields as err:
            poi = f"{poi_test:.3f}" if isinstance(poi_test, float) else poi_test
            warnings.warn(
                err.args[0] + f"\nSetting NLL({poi}) = nan", category=RuntimeWarning
            )
            return np.nan

    def _fit_options(
        self,
        allow_negative_signal: bool,
        initial_muhat_value: Optional[float],
        par_bounds: Optional[List[Tuple[float, float]]],
        optimiser_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Translate the POI-only fit options of this class to the joint parameter vector.

        ``initial_muhat_value`` and a single ``(low, high)`` entry in ``par_bounds``
        refer to :math:`\\mu`; they are inserted into the plug-in's suggested
        initialisation and bounds.  Full-length ``init_pars`` / ``par_bounds`` are
        passed through untouched.
        """
        config = self.combined_model.backend.config(
            allow_negative_signal=allow_negative_signal
        )
        poi = config.poi_index

        init_pars = optimiser_options.pop("init_pars", None)
        if init_pars is None and initial_muhat_value is not None:
            init_pars = list(config.suggested_init)
            init_pars[poi] = float(initial_muhat_value)

        if par_bounds is not None and len(par_bounds) == 1 and config.npar != 1:
            bounds = list(config.suggested_bounds)
            bounds[poi] = tuple(par_bounds[0])
            par_bounds = bounds

        return {"init_pars": init_pars, "par_bounds": par_bounds, **optimiser_options}

    @staticmethod
    def _poi_output(
        muhat: float, poi_indices: Optional[List[Union[int, str]]]
    ) -> Union[float, Dict[Union[int, str], float]]:
        """Map :math:`\\hat\\mu` onto every requested key, as the legacy class did."""
        if poi_indices is None:
            return muhat
        return {key: float(muhat) for key in poi_indices}

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

        Delegates to :func:`~spey.StatisticalModel.likelihood` of
        :attr:`combined_model`.  If a model raises
        :obj:`~spey.system.exceptions.NegativeExpectedYields`, a :obj:`RuntimeWarning`
        is issued and ``NaN`` is returned.

        Args:
            poi_test (``float``, default ``1.0``): Signal strength :math:`\mu`.
            expected (:class:`~spey.ExpectationType`): Dataset prescription.

              * :obj:`~spey.ExpectationType.observed`: Real experimental data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected dataset.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit (SM) expected dataset.

            return_nll (``bool``, default ``True``): If ``True``, return the negative
              log-likelihood, otherwise the likelihood.
            data (``Dict[str, List[float]]``, default ``None``): Per-analysis data
              overrides keyed by analysis name; analyses that are not listed use their
              own data.
            statistical_model_options (``Dict[str, Dict]``, default ``None``): Ignored;
              kept for backwards compatibility.
            kwargs: Forwarded to :func:`~spey.StatisticalModel.likelihood`.

        Returns:
            ``float``:
            Combined NLL (if ``return_nll=True``) or likelihood value, ``NaN`` if a model
            cannot be evaluated at the requested :math:`\mu`.
        """
        self._ignore_options(statistical_model_options)
        nll = self._nan_on_negative_yields(
            poi_test,
            lambda: self.combined_model.likelihood(
                poi_test=poi_test,
                expected=expected,
                return_nll=True,
                data=data if not isinstance(data, dict) or data else None,
                **kwargs,
            ),
        )
        return nll if return_nll or np.isnan(nll) else np.exp(-nll)

    def generate_asimov_data(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistic: str = "qtilde",
        statistical_model_options: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        r"""
        Generate Asimov data for every model in the stack.

        Args:
            expected (~spey.ExpectationType): Dataset prescription, see :meth:`likelihood`.
            test_statistic (``str``, default ``"qtilde"``): Test statistic.

              * ``'qtilde'``: :math:`\tilde{q}_{\mu}`, eq. (62) of :xref:`1007.1727`.
              * ``'q'``: :math:`q_{\mu}`, eq. (54) of :xref:`1007.1727`.
              * ``'q0'``: :math:`q_{0}`, eq. (47) of :xref:`1007.1727`.

            statistical_model_options (``Dict[str, Dict]``, default ``None``): Ignored;
              kept for backwards compatibility.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Dict[str, List[float]]``:
            Asimov data keyed by analysis name.
        """
        self._ignore_options(statistical_model_options)
        data = self.combined_model.generate_asimov_data(
            expected=expected, test_statistic=test_statistic, **kwargs
        )
        return {
            analysis: chunk.tolist()
            for analysis, chunk in self.combined_model.backend.split_data(data).items()
        }

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
        Compute the likelihood of the stack on Asimov data.

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest, :math:`\mu`.
            expected (~spey.ExpectationType): Dataset prescription, see :meth:`likelihood`.
            return_nll (``bool``, default ``True``): If ``True``, returns negative
              log-likelihood value, otherwise the likelihood.
            test_statistics (``str``, default ``"qtilde"``): Test statistic used to
              generate the Asimov data, see :meth:`generate_asimov_data`.
            statistical_model_options (``Dict[str, Dict]``, default ``None``): Ignored;
              kept for backwards compatibility.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``float``:
            likelihood computed for asimov data
        """
        self._ignore_options(statistical_model_options)
        nll = self._nan_on_negative_yields(
            poi_test,
            lambda: self.combined_model.asimov_likelihood(
                poi_test=poi_test,
                expected=expected,
                return_nll=True,
                test_statistics=test_statistics,
                **kwargs,
            ),
        )
        return nll if return_nll or np.isnan(nll) else np.exp(-nll)

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
        Find the global maximum of the combined likelihood.

        Args:
            return_nll (``bool``, default ``True``): If ``True``, return the minimised
              NLL; if ``False``, return the likelihood value at the maximum.
            expected (:class:`~spey.ExpectationType`): Dataset prescription, see
              :meth:`likelihood`.
            allow_negative_signal (``bool``, default ``True``): If ``False``,
              :math:`\hat\mu \geq 0` is enforced.
            data (``Dict[str, List[float]]``, default ``None``): Per-analysis data
              overrides; see :meth:`likelihood`.
            initial_muhat_value (``float``, default ``None``): Starting value of
              :math:`\mu` for the optimiser.
            par_bounds (``List[Tuple[float, float]]``, default ``None``): Either
              ``[(mu_min, mu_max)]`` or bounds for the full combined parameter vector.
            statistical_model_options (``Dict[str, Dict]``, default ``None``): Ignored;
              kept for backwards compatibility.
            poi_indices (``List[Union[int, str]]``, default ``None``): If provided,
              :math:`\hat\mu` is returned as a dictionary keyed by these values.
            optimiser_options: Forwarded to the optimiser.

        Returns:
            ``Tuple[Union[float, Dict], float]``:
            :math:`\hat\mu` (a ``dict`` if ``poi_indices`` is given) and the minimum
            NLL (or maximum likelihood).
        """
        self._ignore_options(statistical_model_options)
        muhat, nll = self.combined_model.maximize_likelihood(
            return_nll=return_nll,
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            data=data or None,
            **self._fit_options(
                allow_negative_signal, initial_muhat_value, par_bounds, optimiser_options
            ),
        )
        return self._poi_output(muhat, poi_indices), nll

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

        Args:
            return_nll (``bool``, default ``True``): If ``True``, return the minimised
              NLL; if ``False``, return the likelihood value at the maximum.
            expected (:class:`~spey.ExpectationType`): Dataset prescription, see
              :meth:`likelihood`.
            test_statistics (``str``, default ``"qtilde"``): Test statistic used to
              generate the Asimov data, see :meth:`generate_asimov_data`.
              ``'qtilde'`` and ``'q0'`` enforce :math:`\hat\mu \geq 0`.
            initial_muhat_value (``float``, default ``None``): Starting value of
              :math:`\mu` for the optimiser.
            par_bounds (``List[Tuple[float, float]]``, default ``None``): Either
              ``[(mu_min, mu_max)]`` or bounds for the full combined parameter vector.
            statistical_model_options (``Dict[str, Dict]``, default ``None``): Ignored;
              kept for backwards compatibility.
            poi_indices (``List[Union[int, str]]``, default ``None``): If provided,
              :math:`\hat\mu` is returned as a dictionary keyed by these values.
            optimiser_options: Forwarded to the optimiser.

        Returns:
            ``Tuple[Union[float, Dict], float]``:
            :math:`\hat\mu` (a ``dict`` if ``poi_indices`` is given) and the minimum
            NLL (or maximum likelihood) on Asimov data.
        """
        self._ignore_options(statistical_model_options)
        allow_negative_signal = test_statistics in ["q", "qmu"]
        muhat, nll = self.combined_model.maximize_asimov_likelihood(
            return_nll=return_nll,
            expected=expected,
            test_statistics=test_statistics,
            **self._fit_options(
                allow_negative_signal, initial_muhat_value, par_bounds, optimiser_options
            ),
        )
        return self._poi_output(muhat, poi_indices), nll
