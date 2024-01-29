from typing import Callable, Optional, Tuple, List

from autograd import numpy as np

from .interface.statistical_model import StatisticalModel
from .utils import ExpectationType

# pylint: disable=E1101

__all__ = ["value_and_grad", "hessian"]


def __dir__():
    return __all__


def value_and_grad(
    statistical_model: StatisticalModel,
    expected: ExpectationType = ExpectationType.observed,
    data: Optional[List[float]] = None,
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Retreive function to compute negative log-likelihood and its gradient.

    .. versionadded:: 0.1.6

    Args:
        statistical_model (~spey.StatisticalModel): statistical model to be used.
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

        data (``List[float]``, default ``None``): input data that to fit. If `None` observed
          data will be used.

    Returns:
        ``Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]``:
        negative log-likelihood and its gradient with respect to nuisance parameters
    """
    val_and_grad = statistical_model.backend.get_objective_function(
        expected=expected, data=None if data is None else np.array(data), do_grad=True
    )
    return lambda pars: val_and_grad(np.array(pars))


def hessian(
    statistical_model: StatisticalModel,
    expected: ExpectationType = ExpectationType.observed,
    data: Optional[List[float]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Retreive the function to compute Hessian of negative log-likelihood

    .. math::

        {\rm Hessian} = -\frac{\partial^2\mathcal{L}(\theta)}{\partial\theta_i\partial\theta_j}

    .. versionadded:: 0.1.6

    Args:
        statistical_model (~spey.StatisticalModel): statistical model to be used.
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

        data (``List[float]``, default ``None``): input data that to fit. If `None` observed
          data will be used.

    Returns:
        ``Callable[[np.ndarray], np.ndarray]``:
        function to compute hessian of negative log-likelihood
    """
    hess = statistical_model.backend.get_hessian_logpdf_func(
        expected=expected, data=None if data is None else np.array(data)
    )
    return lambda pars: -1.0 * hess(np.array(pars))
