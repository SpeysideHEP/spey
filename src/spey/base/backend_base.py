"""
Abstract base classes for ``spey`` statistical-model backends.

This module defines two abstract base classes:

* :class:`~spey.BackendBase` — the interface every statistical-model backend must
  implement to integrate with ``spey``'s hypothesis-testing machinery.
* :class:`~spey.base.backend_base.ConverterBase` — a lightweight interface for
  objects that convert one statistical model representation into a
  :class:`~spey.BackendBase` instance.

For a step-by-step guide on writing, registering, and packaging a new backend see
the :ref:`sec_new_plugin` tutorial in the documentation.
"""

import types
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spey.base.model_config import ModelConfig
from spey.system.cache import cache_results, _PerInstanceCacheDescriptor
from spey.utils import ExpectationType

__all__ = ["BackendBase"]


def __dir__():
    return __all__


class BackendBase(ABC):
    r"""
    Abstract base class that every ``spey`` statistical-model backend must inherit.

    ``spey`` relies on a plugin system to support multiple likelihood prescriptions.
    Any new prescription is expressed as a Python class that inherits
    :class:`~spey.BackendBase` and implements, at minimum,
    :func:`~spey.BackendBase.config` and :func:`~spey.BackendBase.get_logpdf_func`.
    The framework then automatically enables hypothesis testing, upper-limit
    computation, and Asimov-data generation for the new prescription through
    :class:`~spey.StatisticalModel`.

    **Required class-level metadata**

    Each backend class must expose the following attributes so that ``spey``'s plugin
    registry can identify, version-check, and cite it:

    .. code-block:: python

        class MyBackend(spey.BackendBase):
            name          = "my_package.my_model"   # unique entry-point name
            version       = "1.0.0"                 # backend version string
            author        = "Jane Doe <jane@example.com>"
            spey_requires = ">=0.1.0"               # minimum compatible spey version
            doi           = []                       # optional list of citable DOIs
            arXiv         = []                       # optional list of arXiv IDs

    **Required methods**

    Subclasses *must* implement:

    * :func:`~spey.BackendBase.config` — returns a
      :class:`~spey.base.model_config.ModelConfig` describing the parameter
      structure (number of parameters, POI index, suggested initial values,
      and parameter bounds).
    * :func:`~spey.BackendBase.get_logpdf_func` — returns a callable
      ``f(pars: np.ndarray) -> float`` that evaluates
      :math:`\log\mathcal{L}(\mu, \theta)` for a given parameter vector.

    **Optional methods**

    Each optional method unlocks additional capabilities in the ``spey`` interface:

    .. list-table::
        :header-rows: 1
        :widths: 40 60

        * - Method
          - Capability unlocked
        * - :func:`~spey.BackendBase.is_alive`
          - Quick validity check; defaults to ``True``.
        * - :func:`~spey.BackendBase.expected_data`
          - Required for the **asymptotic** calculator and Asimov-data generation.
        * - :func:`~spey.BackendBase.get_objective_function`
          - Override to supply analytical gradients for the optimiser.
        * - :func:`~spey.BackendBase.get_hessian_logpdf_func`
          - Enables :func:`~spey.StatisticalModel.sigma_mu_from_hessian`.
        * - :func:`~spey.BackendBase.get_sampler`
          - Required for the **toy** (pseudo-experiment) calculator.
        * - :func:`~spey.BackendBase.combine`
          - Enables model combination via :func:`~spey.StatisticalModel.combine`
            and the ``@`` operator.
        * - :func:`~spey.BackendBase.negative_loglikelihood` and variants
          - Optional fast-path overrides that bypass the generic ``spey`` optimiser.

    **Minimal working example**

    The example below implements a simple Poisson counting model,
    :math:`\mathcal{L}(\mu) = \prod_i \mathrm{Poiss}(n^i \mid \mu s^i + b^i)`,
    and registers it directly without a ``setup.py``:

    .. code-block:: python

        import numpy as np
        import spey
        from spey.base.model_config import ModelConfig

        @spey.register_backend
        class PoissonModel(spey.BackendBase):
            name          = "my_package.poisson"
            version       = "1.0.0"
            author        = "Jane Doe"
            spey_requires = ">=0.1.0"

            def __init__(self, signal, background, data):
                self._signal     = np.array(signal,     dtype=float)
                self._background = np.array(background, dtype=float)
                self._data       = np.array(data,       dtype=float)

            @property
            def is_alive(self):
                return bool(np.any(self._signal > 0.0))

            def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
                minimum_poi = -10.0 if allow_negative_signal else 0.0
                return ModelConfig(
                    poi_index=0,
                    minimum_poi=minimum_poi,
                    suggested_init=[1.0],
                    suggested_bounds=[(minimum_poi, poi_upper_bound)],
                )

            def get_logpdf_func(
                self, expected=spey.ExpectationType.observed, data=None
            ):
                obs = self._data if data is None else np.array(data)
                if expected is spey.ExpectationType.apriori:
                    obs = self._background

                def logpdf(pars):
                    mu   = pars[0]
                    rate = mu * self._signal + self._background
                    return float(np.sum(obs * np.log(rate) - rate))

                return logpdf

            def expected_data(self, pars):
                mu = pars[0]
                return list(mu * self._signal + self._background)

        # Use the model
        model = spey.get_backend("my_package.poisson")(
            signal=[5.0, 3.0],
            background=[10.0, 8.0],
            data=[12, 9],
            analysis="example",
            xsection=0.05,
        )
        print(model.exclusion_confidence_level())

    .. seealso::

        :ref:`sec_new_plugin` — full tutorial on writing, registering, and packaging
        a ``spey`` plugin, including entry-point installation via ``setup.py`` /
        ``pyproject.toml`` and citation metadata.
    """

    @property
    def is_alive(self) -> bool:
        """
        Whether the model has at least one bin with a non-zero signal yield.

        The default implementation always returns ``True``.  Override this to
        short-circuit expensive calculations for signal hypotheses that are
        effectively empty.

        Returns:
            ``bool``:
            ``True`` if at least one signal bin is non-zero; ``False`` otherwise.
        """
        return True

    @abstractmethod
    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ) -> ModelConfig:
        r"""
        Return the model configuration used by the optimiser.

        This **abstract** method must be implemented by every backend.  It communicates
        the parameter structure of the model to the ``spey`` optimiser: how many
        parameters there are, which index belongs to the parameter of interest (POI)
        :math:`\mu`, what sensible initial values are, and what bounds to apply.

        Args:
            allow_negative_signal (``bool``, default ``True``): When ``True``, the lower
              bound of the POI is set to
              :attr:`~spey.base.model_config.ModelConfig.minimum_poi`; when ``False``
              the lower bound is forced to ``0.0`` so that :math:`\hat\mu \geq 0`.
            poi_upper_bound (``float``, default ``10.0``): Upper bound applied to the
              POI :math:`\mu` during optimisation.

        Returns:
            ~spey.base.model_config.ModelConfig:
            Configuration object containing the POI index, minimum POI value,
            suggested initialisation parameters, and suggested parameter bounds
            for the optimiser.
        """

    @abstractmethod
    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Return a callable that evaluates :math:`\log\mathcal{L}(\mu, \theta)`.

        This **abstract** method must be implemented by every backend.  The returned
        function is the primary input to ``spey``'s optimiser and hypothesis-testing
        machinery.

        The ``expected`` argument selects which dataset is used when ``data`` is
        ``None``:

        * :obj:`~spey.ExpectationType.observed` — use the observed experimental counts.
        * :obj:`~spey.ExpectationType.apriori` — use the background-only prediction
          (SM hypothesis), giving the *expected* (pre-fit) likelihood.

        When ``data`` is explicitly provided it always takes precedence over
        ``expected`` (this is used internally for Asimov-data computations).

        Args:
            expected (~spey.ExpectationType): Selects which dataset to use when
              ``data`` is ``None``.

              * :obj:`~spey.ExpectationType.observed`: Use the observed data
                (post-fit, default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use the background-only
                prediction (pre-fit / SM hypothesis).

            data (``Union[List[float], np.ndarray]``, default ``None``): Explicit
              dataset to condition on.  When provided, overrides ``expected``.

        Returns:
            ``Callable[[np.ndarray], float]``:
            A function ``logpdf(pars) -> float`` where ``pars`` is a 1-D array of
            fit parameters :math:`(\mu, \theta_1, \theta_2, \ldots)` and the return
            value is :math:`\log\mathcal{L}(\mu, \theta)`.
        """

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]:
        r"""
        Return the objective function (and optionally its gradient) for the optimiser.

        The objective is the negative log-likelihood,
        :math:`-\log\mathcal{L}(\mu, \theta)`.  When ``do_grad=True`` the returned
        callable should also return the gradient with respect to all parameters as a
        second element of a tuple, enabling gradient-based optimisers.

        The default implementation raises :obj:`NotImplementedError` for
        ``do_grad=True`` and falls back to negating the value of
        :func:`~spey.BackendBase.get_logpdf_func` for ``do_grad=False``.  Override
        this method to provide analytical or auto-differentiation-based gradients,
        which can substantially improve optimisation speed and stability.

        Args:
            expected (~spey.ExpectationType): Selects which dataset to use when
              ``data`` is ``None``.

              * :obj:`~spey.ExpectationType.observed`: Use the observed data
                (post-fit, default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use the background-only
                prediction (pre-fit / SM hypothesis).

            data (``Union[List[float], np.ndarray]``, default ``None``): Explicit
              dataset to condition on.  When provided, overrides ``expected``.
            do_grad (``bool``, default ``True``): If ``True``, return a callable that
              yields ``(objective, gradient)``; if ``False``, return a callable that
              yields only the scalar objective.

        Raises:
            :obj:`NotImplementedError`: When ``do_grad=True`` and the backend has not
              overridden this method.

        Returns:
            ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
            A function ``objective(pars)`` that returns either a scalar
            :math:`-\log\mathcal{L}` (``do_grad=False``) or a tuple
            ``(-logL, gradient)`` (``do_grad=True``), where ``gradient`` is a 1-D
            array of the same length as ``pars``.
        """
        if do_grad:
            raise NotImplementedError("Gradient is not implemented by default.")

        logpdf = self.get_logpdf_func(expected=expected, data=data)
        return lambda pars: -logpdf(pars)

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Return a callable that evaluates the Hessian of :math:`\log\mathcal{L}(\mu, \theta)`.

        The Hessian is used by :func:`~spey.StatisticalModel.sigma_mu_from_hessian`
        to estimate the variance on the parameter of interest :math:`\mu` via the
        inverse of the observed information matrix (see eqs. 27–28 of
        :xref:`1007.1727`).

        The default implementation raises :obj:`NotImplementedError`.  Override this
        method when an analytical or auto-differentiation Hessian is available, as it
        is considerably more accurate than a finite-difference approximation.

        Args:
            expected (~spey.ExpectationType): Selects which dataset to use when
              ``data`` is ``None``.

              * :obj:`~spey.ExpectationType.observed`: Use the observed data
                (post-fit, default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use the background-only
                prediction (pre-fit / SM hypothesis).

            data (``Union[List[float], np.ndarray]``, default ``None``): Explicit
              dataset to condition on.  When provided, overrides ``expected``.

        Raises:
            :obj:`NotImplementedError`: If the backend has not implemented the Hessian.

        Returns:
            ``Callable[[np.ndarray], np.ndarray]``:
            A function ``hessian(pars) -> np.ndarray`` where ``pars`` is a 1-D
            parameter array and the return value is the square Hessian matrix of
            :math:`\log\mathcal{L}` with shape ``(n_pars, n_pars)``.
        """
        raise NotImplementedError("This method has not been implemented")

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Return a callable that draws pseudo-data from the model at fixed parameters.

        Implementing this method enables the **toy** (pseudo-experiment) calculator
        for hypothesis testing.  The returned sampler is conditioned on the supplied
        fit parameters ``pars``; ``spey`` typically calls this after fitting the
        nuisance parameters for a given :math:`\mu`.

        The default implementation raises :obj:`NotImplementedError`.

        Args:
            pars (:obj:`np.ndarray`): 1-D array of fit parameters
              :math:`(\mu, \theta_1, \theta_2, \ldots)` at which to condition the
              sampler.

        Raises:
            :obj:`NotImplementedError`: If the backend has not implemented a sampler.

        Returns:
            ``Callable[[int], np.ndarray]``:
            A function ``sampler(n) -> np.ndarray`` that draws ``n`` independent
            pseudo-datasets from the model, returned as an array of shape
            ``(n, n_bins)``.
        """
        raise NotImplementedError("This method has not been implemented")

    def expected_data(self, pars: List[float]) -> List[float]:
        r"""
        Return the expected bin counts for a given parameter vector.

        This method is used internally by
        :func:`~spey.StatisticalModel.generate_asimov_data` to produce Asimov
        datasets, and is therefore required for the **asymptotic** calculator.  Without
        it, only the :math:`\chi^2` calculator is available.

        The default implementation raises :obj:`NotImplementedError`.

        Args:
            pars (``List[float]``): 1-D array or list of fit parameters
              :math:`(\mu, \theta_1, \theta_2, \ldots)`.

        Raises:
            :obj:`NotImplementedError`: If the backend has not implemented this method.

        Returns:
            ``List[float]``:
            Expected bin counts :math:`\langle n^i \rangle = \mu s^i + b^i` (or the
            model-specific equivalent) evaluated at ``pars``.
        """
        raise NotImplementedError("This method has not been implemented")

    def combine(self, other, **kwargs):
        """
        Combine this statistical model with another backend instance.

        Implementing this method enables model combination via
        :func:`~spey.StatisticalModel.combine` and the ``@`` operator on
        :class:`~spey.StatisticalModel`.  The returned object must itself be a
        :class:`~spey.BackendBase` instance so that ``spey`` can wrap it in a new
        :class:`~spey.StatisticalModel`.

        .. note::

            This method is optional and only needs to be implemented if the backend
            supports a specific combination routine (e.g. merging bin lists, combining
            workspaces, or constructing a joint likelihood).

        Args:
            other (:obj:`~spey.BackendBase`): The backend instance to combine with.
            kwargs: Backend-specific keyword arguments forwarded to the combination
              routine.

        Raises:
            :obj:`NotImplementedError`: If the backend does not implement combination.

        Returns:
            :obj:`~spey.BackendBase`:
            A new backend instance representing the combined statistical model.
        """
        raise NotImplementedError("This method does not have combination implementation.")

    def negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Compute the profiled negative log-likelihood at a fixed :math:`\mu`.

        This is an **optional fast-path override**.  ``spey`` first tries to call this
        method; if it raises :obj:`NotImplementedError`, the interface falls back to
        minimising the objective function from :func:`~spey.BackendBase.get_objective_function`
        using the built-in optimiser.  Implementing this method is only worthwhile when
        the backend has an efficient internal minimiser for the nuisance parameters.

        Args:
            poi_test (``float``, default ``1.0``): Fixed value of the parameter of
              interest :math:`\mu` at which to evaluate the profiled likelihood.
            expected (~spey.ExpectationType): Selects which dataset to condition on.

              * :obj:`~spey.ExpectationType.observed`: Use the observed data
                (post-fit, default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use the background-only
                prediction (pre-fit / SM hypothesis).

            kwargs: Additional keyword arguments forwarded to the backend's internal
              optimiser.

        Raises:
            :obj:`NotImplementedError`: If the backend has not implemented this method
              (the ``spey`` interface will then use the generic optimiser).

        Returns:
            ``Tuple[float, np.ndarray]``:
            A tuple ``(nll, pars)`` where ``nll`` is the profiled negative
            log-likelihood :math:`-\log\mathcal{L}(\mu, \hat{\theta}_\mu)` and
            ``pars`` is the 1-D array of all fit parameters at the optimum.
        """
        raise NotImplementedError("This method has not been implemented")

    def asimov_negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: str = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Compute the profiled negative log-likelihood at fixed :math:`\mu` on Asimov data.

        This is an **optional fast-path override** for backends that can compute the
        Asimov likelihood more efficiently than the generic ``spey`` pathway (which
        first generates Asimov data via :func:`~spey.BackendBase.expected_data` and
        then calls the standard optimiser).

        Args:
            poi_test (``float``, default ``1.0``): Fixed value of the parameter of
              interest :math:`\mu`.
            expected (~spey.ExpectationType): Selects which dataset to condition on
              when constructing the Asimov dataset.

              * :obj:`~spey.ExpectationType.observed`: Use the observed data
                (post-fit, default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use the background-only
                prediction (pre-fit / SM hypothesis).

            test_statistics (``str``, default ``"qtilde"``): Test statistic that
              determines the signal strength used to generate the Asimov dataset
              (``"q0"`` → :math:`\mu=1`; all others → :math:`\mu=0`).

              * ``'qtilde'``: Alternative test statistic :math:`\tilde{q}_\mu`,
                eq. (62) of :xref:`1007.1727`.

                .. warning::

                    This test statistic assumes :math:`\hat\mu \geq 0`
                    (``allow_negative_signal=False``).  When called through ``spey``'s
                    public interface this constraint is enforced automatically.

              * ``'q'``: Standard test statistic :math:`q_\mu`,
                eq. (54) of :xref:`1007.1727`.
              * ``'q0'``: Discovery test statistic :math:`q_0`,
                eq. (47) of :xref:`1007.1727`.

            kwargs: Additional keyword arguments forwarded to the backend's internal
              optimiser.

        Raises:
            :obj:`NotImplementedError`: If the backend has not implemented this method.

        Returns:
            ``Tuple[float, np.ndarray]``:
            A tuple ``(nll, pars)`` where ``nll`` is the profiled negative
            log-likelihood evaluated on the Asimov dataset and ``pars`` is the 1-D
            array of all fit parameters at the optimum.
        """
        raise NotImplementedError("This method has not been implemented")

    def minimize_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Find the global minimum of the negative log-likelihood (free fit).

        This is an **optional fast-path override**.  ``spey`` first tries to call this
        method; if it raises :obj:`NotImplementedError`, the interface falls back to
        minimising the objective function from
        :func:`~spey.BackendBase.get_objective_function` using the built-in optimiser.
        Implement this method when the backend has a more efficient internal minimiser.

        Args:
            expected (~spey.ExpectationType): Selects which dataset to condition on.

              * :obj:`~spey.ExpectationType.observed`: Use the observed data
                (post-fit, default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use the background-only
                prediction (pre-fit / SM hypothesis).

            allow_negative_signal (``bool``, default ``True``): When ``True``,
              :math:`\hat\mu` is unconstrained; when ``False`` the fit enforces
              :math:`\hat\mu \geq 0`.
            kwargs: Additional keyword arguments forwarded to the backend's internal
              optimiser.

        Raises:
            :obj:`NotImplementedError`: If the backend has not implemented this method.

        Returns:
            ``Tuple[float, np.ndarray]``:
            A tuple ``(nll, pars)`` where ``nll`` is the minimum negative
            log-likelihood :math:`-\log\mathcal{L}(\hat\mu, \hat\theta)` and ``pars``
            is the 1-D array of all fit parameters at the global optimum.
        """
        raise NotImplementedError("This method has not been implemented")

    def minimize_asimov_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: str = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Find the global minimum of the negative log-likelihood on Asimov data (free fit).

        This is an **optional fast-path override** complementing
        :func:`~spey.BackendBase.asimov_negative_loglikelihood`.  Together they allow
        the asymptotic calculator to bypass ``spey``'s generic optimisation loop.
        If this method raises :obj:`NotImplementedError`, the interface falls back to
        the standard pipeline.

        Args:
            expected (~spey.ExpectationType): Selects which dataset to condition on
              when constructing the Asimov dataset.

              * :obj:`~spey.ExpectationType.observed`: Use the observed data
                (post-fit, default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use the background-only
                prediction (pre-fit / SM hypothesis).

            test_statistics (``str``, default ``"qtilde"``): Test statistic that
              determines the signal strength used to generate the Asimov dataset
              (``"q0"`` → :math:`\mu=1`; all others → :math:`\mu=0`).

              * ``'qtilde'``: Alternative test statistic :math:`\tilde{q}_\mu`,
                eq. (62) of :xref:`1007.1727`.

                .. warning::

                    This test statistic assumes :math:`\hat\mu \geq 0`
                    (``allow_negative_signal=False``).  When called through ``spey``'s
                    public interface this constraint is enforced automatically.

              * ``'q'``: Standard test statistic :math:`q_\mu`,
                eq. (54) of :xref:`1007.1727`.
              * ``'q0'``: Discovery test statistic :math:`q_0`,
                eq. (47) of :xref:`1007.1727`.

            kwargs: Additional keyword arguments forwarded to the backend's internal
              optimiser.

        Raises:
            :obj:`NotImplementedError`: If the backend has not implemented this method.

        Returns:
            ``Tuple[float, np.ndarray]``:
            A tuple ``(nll, pars)`` where ``nll`` is the minimum negative
            log-likelihood on the Asimov dataset and ``pars`` is the 1-D array of
            all fit parameters at the global optimum.
        """
        raise NotImplementedError("This method has not been implemented")


class ConverterBase(ABC):
    """
    Abstract base class for objects that convert one statistical model into another.

    A ``ConverterBase`` subclass acts as a stateless callable that accepts a
    :class:`~spey.StatisticalModel` (or any other representation) and returns a new
    :class:`~spey.BackendBase` instance.  This is useful for translating between
    different likelihood prescriptions without exposing construction details to the
    user.

    Subclasses must expose the same class-level metadata as :class:`~spey.BackendBase`
    (``name``, ``version``, ``author``, ``spey_requires``) so that the plugin registry
    can identify them, and must override :func:`__call__` to perform the actual
    conversion.

    .. note::

        ``ConverterBase`` subclasses are **not** expected to accept arguments in
        ``__init__``.  All conversion logic should live in :func:`__call__`.

    Example:

    .. code-block:: python

        import spey
        from spey.base.backend_base import BackendBase, ConverterBase

        class MyStatConverter(ConverterBase):
            name          = "example.converter"
            version       = "0.0.1"
            author        = "Tom Bombadil"
            spey_requires = ">=0.1.0"

            def __call__(self, stat_model: spey.StatisticalModel) -> BackendBase:
                # Extract information from the input model and build a new backend
                signal     = stat_model.backend._signal
                background = stat_model.backend._background
                data       = stat_model.backend._data
                return UncorrelatedBackground(signal, background, data)
    """

    def __call__(self, *args, **kwargs) -> BackendBase:
        """
        Convert the input representation into a :class:`~spey.BackendBase` object.

        Subclasses **must** override this method.  It may accept any positional or
        keyword arguments that are meaningful for the specific conversion (e.g. a
        :class:`~spey.StatisticalModel`, a workspace dictionary, or raw arrays).

        Raises:
            :obj:`NotImplementedError`: If the subclass has not implemented
              :func:`__call__`.

        Returns:
            :obj:`~spey.BackendBase`:
            A new backend instance compatible with the ``spey`` interface.
        """
        raise NotImplementedError("Invalid implementation of ConverterBase object")
