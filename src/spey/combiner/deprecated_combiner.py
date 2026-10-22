r"""
Deprecated Uncorrelated Combiner
================================

:func:`UnCorrStatisticsCombiner` is the historical entry point for combining independent
statistical models.  It is kept for backwards compatibility only: it takes the same
inputs as before and returns the statistical model built by the
``default.uncorrelated_combiner`` plug-in
(:class:`~spey.combiner.UncorrelatedStatisticsCombiner`).

.. deprecated:: 0.2.8

    Use the plug-in instead:

    .. code-block:: python3

        >>> combiner = spey.get_backend("default.uncorrelated_combiner")
        >>> combined = combiner(statistical_models=[model_a, model_b], analysis="combined")
"""

import warnings

from spey.interface.statistical_model import StatisticalModel

__all__ = ["UnCorrStatisticsCombiner"]

_DEPRECATION_MESSAGE = (
    "`spey.UnCorrStatisticsCombiner` is deprecated and will not be available in future "
    "versions of spey. Please use the `default.uncorrelated_combiner` plug-in instead: "
    '`spey.get_backend("default.uncorrelated_combiner")(statistical_models=[...], '
    'analysis="...")`.'
)


def __dir__():
    return __all__


def UnCorrStatisticsCombiner(  # pylint: disable=invalid-name
    *statistical_models: StatisticalModel, ntoys: int = 1000
) -> StatisticalModel:
    r"""
    Combine **uncorrelated** (independent) statistical models.

    .. deprecated:: 0.2.8

        This function will not be available in future versions of ``spey``; it only
        calls the ``default.uncorrelated_combiner`` plug-in
        (:class:`~spey.combiner.UncorrelatedStatisticsCombiner`):

        .. code-block:: python3

            >>> combiner = spey.get_backend("default.uncorrelated_combiner")
            >>> combined = combiner(statistical_models=[model_a, model_b], analysis="AB")

    The combined negative log-likelihood is the sum of the individual ones at a common
    signal strength :math:`\mu`, see :mod:`spey.combiner.uncorrelated_statistics_combiner`.

    .. warning::

        The models must not share nuisance parameters; violating this assumption leads to
        incorrect (over-confident) results.

    .. versionchanged:: 0.2.8

        This used to be a class with a mutable stack (``append``, ``remove``, ``@``,
        indexing) and options of its own.  It now returns an ordinary
        :class:`~spey.StatisticalModel`: the stack is fixed, build a new combination from
        the desired models instead, and combine two combinations with ``@``.

    Args:
        *statistical_models (:class:`~spey.StatisticalModel`): independent statistical
          models, each with a unique :attr:`~spey.StatisticalModel.analysis` name.
        ntoys (``int``, default ``1000``): number of pseudo-experiments of the
          combined model.

    Raises:
        :obj:`~spey.system.exceptions.InvalidInput`: If the models are not valid inputs of
          the plug-in, see :class:`~spey.combiner.UncorrelatedStatisticsCombiner`.

    Returns:
        :class:`~spey.StatisticalModel`:
        The combined model, with analysis name ``"+".join(analyses)``.

    Examples:
        >>> import spey
        >>> pdf_wrapper = spey.get_backend("default.poisson")
        >>> model_A = pdf_wrapper(signal_yields=[3.0], background_yields=[50.0],
        ...                       data=[52], analysis="SR_A")
        >>> model_B = pdf_wrapper(signal_yields=[1.5], background_yields=[20.0],
        ...                       data=[18], analysis="SR_B")
        >>> combined = spey.UnCorrStatisticsCombiner(model_A, model_B)
        >>> combined.exclusion_confidence_level()  # combined CLs
    """
    warnings.warn(_DEPRECATION_MESSAGE, category=FutureWarning, stacklevel=2)
    import spey  # pylint: disable=import-outside-toplevel,cyclic-import

    return spey.get_backend("default.uncorrelated_combiner")(
        statistical_models=list(statistical_models),
        analysis="+".join(
            str(getattr(model, "analysis", model)) for model in statistical_models
        ),
        ntoys=ntoys,
    )
