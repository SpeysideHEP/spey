"""Abstract Methods for backend objects"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Text, Tuple, Union

import numpy as np

from spey.base.model_config import ModelConfig
from spey.utils import ExpectationType

__all__ = ["BackendBase"]


def __dir__():
    return __all__


class BackendBase(ABC):
    """
    An abstract class construction to enforce certain behaviour on statistical model backend.
    In order to perform certain computations, ``spey`` needs to have access to specific
    function constructions such as precsription to form likelihood. Hence, each backend is
    required to inherit :obj:`~spey.BackendBase`.
    """

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return True

    @abstractmethod
    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``10.0``): upper bound for parameter
              of interest, :math:`\mu`.

        Returns:
            ~spey.base.model_config.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """

    @abstractmethod
    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Generate function to compute :math:`\log\mathcal{L}(\mu, \theta)` where :math:`\mu` is the
        parameter of interest and :math:`\theta` are nuisance parameters.

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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and computes
            :math:`\log\mathcal{L}(\mu, \theta)`.
        """

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]:
        r"""
        Objective function is the function to perform the optimisation on. This function is
        expected to be negative log-likelihood, :math:`-\log\mathcal{L}(\mu, \theta)`.
        Additionally, if available it canbe bundled with the gradient of negative log-likelihood.

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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit
            do_grad (``bool``, default ``True``): If ``True`` return objective and its gradient
              as ``tuple`` (subject to availablility) if ``False`` only returns objective function.

        Returns:
            ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
            Function which takes fit parameters (:math:`\mu` and :math:`\theta`) and returns either
            objective or objective and its gradient.
        """
        if do_grad:
            raise NotImplementedError("Gradient is not implemented by default.")

        logpdf = self.get_logpdf_func(expected=expected, data=data)
        return lambda pars: -logpdf(pars)

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Currently Hessian of :math:`\log\mathcal{L}(\mu, \theta)` is only used to compute
        variance on :math:`\mu`. This method returns a callable function which takes fit
        parameters (:math:`\mu` and :math:`\theta`) and returns Hessian.

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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit

        Raises:
            :obj:`NotImplementedError`: If the Hessian of the backend has not been implemented.

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and
            returns Hessian of :math:`\log\mathcal{L}(\mu, \theta)`.
        """
        raise NotImplementedError("This method has not been implemented")

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Retreives the function to sample from.

        Args:
            pars (:obj:`np.ndarray`): fit parameters (:math:`\mu` and :math:`\theta`)

        Raises:
            :obj:`NotImplementedError`: If the sampler for the backend has not been implemented.

        Returns:
            ``Callable[[int], np.ndarray]``:
            Function that takes ``number_of_samples`` as input and draws as many samples
            from the statistical model.
        """
        raise NotImplementedError("This method has not been implemented")

    def expected_data(self, pars: List[float]) -> List[float]:
        r"""
        Compute the expected value of the statistical model. This function is mainly used to
        generate Asimov data within the package, see
        :func:`~spey.StatisticalModel.generate_asimov_data`.

        Args:
            pars (``List[float]``): nuisance, :math:`\theta` and parameter of interest,
              :math:`\mu`.

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        raise NotImplementedError("This method has not been implemented")

    def combine(self, other, **kwargs):
        """
        A routine to combine to statistical models.

        .. note::

            This function is only available if the backend has a specific routine
            for combination between same or other backends.

        Args:
            other (:obj:`~spey.BackendBase`): Statistical model object to be combined.

        Raises:
            ``NotImplementedError``: If the backend does not have a combination scheme.

        Returns:
            :obj:`~spey.BackendBase`:
            Create a new statistical model from combination of this and other one.
        """
        raise NotImplementedError("This method does not have combination implementation.")

    def negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Backend specific method to compute negative log-likelihood for a parameter of interest
        :math:`\mu`.

        .. note::

            Interface first calls backend specific methods to compute likelihood. If they are not
            implemented, it optimizes objective function through ``spey`` interface. Either prescription
            to optimizing the likelihood or objective function must be available for a backend to
            be sucessfully integrated to the ``spey`` interface.

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

            kwargs: keyword arguments for the optimiser.

        Raises:
            :obj:`NotImplementedError`: If the method is not available for the backend.

        Returns:
            ``Tuple[float, np.ndarray]``:
            value of negative log-likelihood at POI of interest and fit parameters
            (:math:`\mu` and :math:`\theta`).
        """
        raise NotImplementedError("This method has not been implemented")

    def asimov_negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Compute negative log-likelihood at fixed :math:`\mu` for Asimov data.

        .. note::

            Interface first calls backend specific methods to compute likelihood. If they are not
            implemented, it optimizes objective function through ``spey`` interface. Either prescription
            to optimizing the likelihood or objective function must be available for a backend to
            be sucessfully integrated to the ``spey`` interface.

        Args:
            poi_test (``float``, default ``1.0``\): parameter of interest, :math:`\mu`.
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

              * ``'q'``\: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``\: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            kwargs: keyword arguments for the optimiser.

        Raises:
            :obj:`NotImplementedError`: If the method is not available for the backend.

        Returns:
            ``Tuple[float, np.ndarray]``\:
            value of negative log-likelihood at POI of interest and fit parameters
            (:math:`\mu` and :math:`\theta`).
        """
        raise NotImplementedError("This method has not been implemented")

    def minimize_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        A backend specific method to minimize negative log-likelihood.

        .. note::

            Interface first calls backend specific methods to compute likelihood. If they are not
            implemented, it optimizes objective function through ``spey`` interface. Either prescription
            to optimizing the likelihood or objective function must be available for a backend to
            be sucessfully integrated to the ``spey`` interface.

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

            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            kwargs: keyword arguments for the optimiser.

        Raises:
            :obj:`NotImplementedError`: If the method is not available for the backend.

        Returns:
            ``Tuple[float, np.ndarray]``:
            value of negative log-likelihood and fit parameters (:math:`\mu` and :math:`\theta`).
        """
        raise NotImplementedError("This method has not been implemented")

    def minimize_asimov_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        A backend specific method to minimize negative log-likelihood for Asimov data.

        .. note::

            Interface first calls backend specific methods to compute likelihood. If they are not
            implemented, it optimizes objective function through ``spey`` interface. Either prescription
            to optimizing the likelihood or objective function must be available for a backend to
            be sucessfully integrated to the ``spey`` interface.

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

            kwargs: keyword arguments for the optimiser.

        Raises:
            :obj:`NotImplementedError`: If the method is not available for the backend.

        Returns:
            ``Tuple[float, np.ndarray]``:
            value of negative log-likelihood and fit parameters (:math:`\mu` and :math:`\theta`).
        """
        raise NotImplementedError("This method has not been implemented")


class ConverterBase(ABC):
    """
    An abstract class construction to enforce certain behaviour on statistical model backend.
    This base class is used to act as a midle function where the function can act as a converter
    between two statistical models. It has to have a call function which needs to return a
    :obj:`~spey.BackendBase` object.

    .. note::

        ``ConverterBase`` object is not expected to have an ``__init__`` method that expect
        arguments or keyword arguments. This function should have a ``__call__`` method that
        returns :obj:`~spey.BackendBase` object.

    Example:

    .. code:: python3

        >>> class MyStatConverter(ConverterBase):
        >>>     name= "example.converter"
        >>>     version="0.0.1"
        >>>     author="Tom Bombadil"
        >>>     spey_requires=">0.1.0"

        >>>     def __call__(self, stat_model: spey.StatisticalModel) -> BackendBase:
        >>>         # do something
        >>>         return UncorrelatedBackground(...)
    """

    def __call__(self, *args, **kwargs) -> BackendBase:
        """
        Function that compiles the :obj:`~spey.BackendBase` object.

        Raises:
            ``NotImplementedError``: If ``__call__`` function is not implemented for the class.

        Returns:
            :obj:`~spey.BackendBase`:
            This function should return any :obj:`~spey.BackendBase` object.
        """
        raise NotImplementedError("Invalid implementation of ConverterBase object")
