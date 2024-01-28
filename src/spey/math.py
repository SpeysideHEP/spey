from typing import Callable, Optional, Tuple

from autograd import numpy

from .interface.statistical_model import StatisticalModel
from .utils import ExpectationType

# pylint: disable=E1101

__all__ = ["value_and_grad", "hessian"]


def __dir__():
    return __all__


def value_and_grad(
    statistical_model: StatisticalModel,
    expected: ExpectationType = ExpectationType.observed,
    data: Optional[numpy.ndarray] = None,
) -> Callable[[numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]:
    """
    Retreive function to compute negative log-likelihood and its gradient.

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

        data (``numpy.ndarray``, default ``None``): input data that to fit. If `None` observed
          data will be used.

    Returns:
        ``Callable[[numpy.ndarray], numpy.ndarray, numpy.ndarray]``:
        negative log-likelihood and its gradient with respect to nuisance parameters
    """
    val_and_grad = statistical_model.backend.get_objective_function(
        expected=expected, data=data, do_grad=True
    )
    return lambda pars: val_and_grad(numpy.array(pars))


def hessian(
    statistical_model: StatisticalModel,
    expected: ExpectationType = ExpectationType.observed,
    data: Optional[numpy.ndarray] = None,
) -> Callable[[numpy.ndarray], numpy.ndarray]:
    r"""
    Retreive the function to compute Hessian of negative log-likelihood

    .. math::

        {\rm Hessian} = -\frac{\partial^2\mathcal{L}(\theta)}{\partial\theta_i\partial\theta_j}

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

        data (``numpy.ndarray``, default ``None``): input data that to fit. If `None` observed
          data will be used.

    Returns:
        ``Callable[[numpy.ndarray], numpy.ndarray]``:
        function to compute hessian of negative log-likelihood
    """
    hess = statistical_model.backend.get_hessian_logpdf_func(expected=expected, data=data)
    return lambda pars: -1.0 * hess(numpy.array(pars))
