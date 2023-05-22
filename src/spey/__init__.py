from typing import Text, Union, List, Dict, Optional, Tuple, Callable, Any
import numpy as np
import pkg_resources
from semantic_version import Version, SimpleSpec

from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.base import BackendBase
from spey.combiner import UnCorrStatisticsCombiner
from spey.system.exceptions import PluginError
from .utils import ExpectationType
from ._version import __version__

__all__ = [
    "version",
    "StatisticalModel",
    "UnCorrStatisticsCombiner",
    "ExpectationType",
    "AvailableBackends",
    "get_correlated_nbin_statistical_model",
    "get_uncorrelated_nbin_statistical_model",
    "get_backend",
    "get_backend_metadata",
    "reset_backend_entries",
    "BackendBase",
]


def __dir__():
    return __all__


def version() -> Text:
    """
    Version of ``spey`` package

    Returns:
        ``Text``: version in X.Y.Z format
    """
    return __version__


def _get_backend_entrypoints() -> Dict[Text, pkg_resources.EntryPoint]:
    """Collect plugin entries"""
    return {
        entry.name: entry
        for entry in pkg_resources.iter_entry_points("spey.backend.plugins")
    }


_backend_entries: Dict[Text, pkg_resources.EntryPoint] = _get_backend_entrypoints()
# ! Preinitialise backends, it might be costly to scan the system everytime


def reset_backend_entries() -> None:
    """Scan the system for backends and reset the entries"""
    _backend_entries = _get_backend_entrypoints()


def AvailableBackends() -> List[Text]:
    """
    Returns a list of available backends. The default backends are automatically installed
    with ``spey`` package. To enable other backends, please see the relevant section
    in the documentation.

    .. note::

        This function does not validate the backend. For backend validation please use
        :func:`~spey.get_backend` function.

    Returns:
        ``List[Text]``: list of names of available backends.
    """
    return [*_backend_entries.keys()]


def get_backend(name: Text) -> Callable[[Any, ...], StatisticalModel]:
    """
    Statistical model backend retreiver. Available backend names can be found via
    :func:`~spey.AvailableBackends` function.

    Args:
        name (``Text``): backend identifier. This backend refers to different packages
          that prescribes likelihood function.

    Raises:
        :obj:`~spey.system.exceptions.PluginError`: If the backend is not available
          or the available backend requires different version of ``spey``.
        :obj:`AssertionError`: If the backend does not have necessary metadata.

    Returns:
        ``Callable[[Any, ...], StatisticalModel]``:
        A callable function that takes backend specific arguments and two additional
        keyword arguments ``analysis`` (which is a unique identifier of analysis name as :obj:`str`)
        and ``xsection`` (which is cross section value with a.u.). Details about the function can be
        found in :func:`~spey.statistical_model_wrapper`. This wrapper
        returns a :obj:`~spey.StatisticalModel` object.

    Example:

    .. code-block:: python3
        :linenos:

        >>> import spey; import numpy as np
        >>> stat_wrapper = spey.get_backend("simplified_likelihoods")

        >>> data = np.array([1])
        >>> signal = np.array([0.5])
        >>> background = np.array([2.])
        >>> background_unc = np.array([[1.1]])

        >>> stat_model = stat_wrapper(
        ...     signal_yields=signal,
        ...     background_yields=background,
        ...     data=data,
        ...     covariance_matrix=background_unc,
        ...     analysis="simple_sl",
        ...     xsection=0.123
        ... )
        >>> stat_model.exclusion_confidence_level() # [0.4022058844566345]

    .. note::

        The documentation of the ``stat_wrapper`` defined above includes the docstring
        of the backend as well. Hence typing ``stat_wrapper?`` in terminal will result with
        complete documentation for the :func:`~spey.statistical_model_wrapper` and
        the backend it self which is in this particular example
        :obj:`~spey.backends.simplifiedlikelihood_backend.interface.SimplifiedLikelihoodInterface`.
    """
    backend = _backend_entries.get(name, False)

    if backend:
        statistical_model = backend.load()

        assert hasattr(
            statistical_model, "spey_requires"
        ), "Backend does not have `'spey_requires'` attribute."

        if getattr(statistical_model, "name", False):
            assert (
                statistical_model.name == name
            ), "The identity of the statistical model is wrongly set."
        else:
            setattr(statistical_model, "name", name)

        if Version(version()) not in SimpleSpec(statistical_model.spey_requires):
            raise PluginError(
                f"The backend {name}, requires spey version {statistical_model.spey_requires}. "
                f"However the current spey version is {__version__}."
            )
        return statistical_model_wrapper(statistical_model)

    raise PluginError(
        f"The backend {name} is unavailable. Available backends are "
        + ", ".join(AvailableBackends())
        + "."
    )


def get_backend_metadata(name: Text) -> Dict[Text, Any]:
    """
    Retreive metadata about the backend. This includes citation information,
    doi, author names etc. Available backend names can be found via
    :func:`~spey.AvailableBackends` function.

    Args:
        name (``Text``): backend identifier. This backend refers to different packages
          that prescribes likelihood function.

    Raises:
        ~spey.system.exceptions.PluginError: If the backend does not exists.

    Returns:
        ``Dict[Text, Text]``:
        Metadata about the backend.

    Example:

    .. code-block:: python3

        >>> spey.get_backend_metadata("simplified_likelihoods")

    will return the following

    .. code-block:: python3

        >>> {'name': 'simplified_likelihoods',
        ... 'author': 'SpeysideHEP',
        ... 'version': '0.0.1',
        ... 'spey_requires': '0.0.1',
        ... 'doi': ['10.1007/JHEP04(2019)064'],
        ... 'arXiv': ['1809.05548']}
    """
    backend = _backend_entries.get(name, False)

    if backend:
        statistical_model = backend.load()
        return {
            "name": getattr(statistical_model, "name", "__unknown_model__"),
            "author": getattr(statistical_model, "author", "__unknown_author__"),
            "version": getattr(statistical_model, "version", "__unknown_version__"),
            "spey_requires": statistical_model.spey_requires,
            "doi": list(getattr(statistical_model, "doi", [])),
            "arXiv": list(getattr(statistical_model, "arXiv", [])),
        }

    raise PluginError(
        f"The backend {name} is unavailable. Available backends are "
        + ", ".join(AvailableBackends())
        + "."
    )


def get_uncorrelated_nbin_statistical_model(
    data: Union[float, int, np.ndarray, List[float]],
    backgrounds: Union[float, np.ndarray, List[float]],
    background_uncertainty: Union[float, np.ndarray, List[float]],
    signal_yields: Union[float, np.ndarray, List[float]],
    xsection: Union[float, np.ndarray, List[float]],
    analysis: Text,
    backend: Text,
) -> StatisticalModel:
    """
    Create a statistical model from uncorrelated bins.

    Args:
        data (``float, int, np.ndarray, List[float]``): data yields
        backgrounds (``float, np.ndarray, List[float]``): background yields
        background_uncertainty (``float, np.ndarray, List[float]``): absolute background uncertainty
        signal_yields (``float, np.ndarray, List[float]``): signal yields
        xsection (``float, np.ndarray, List[float]``): cross section value, unit determined by the user.
        analysis (``Text``): unique analysis name for the statistical model.
        backend (``Text``): statistical model backend. Currently available backend names can be
          retreived via :func:`~spey.AvailableBackends` function.

    Raises:
        `NotImplementedError`: If the backend is not implemented.

    Returns:
        :class:`~spey.StatisticalModel`:
        Statistical model object.

    Example:

    A single bin example can be initiated via

    .. code-block:: python3

        >>> import spey
        >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
        ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
        ... )
        >>> statistical_model.exclusion_confidence_level() # [0.4014584422111511]

    And a multi-bin structure can be embeded via simple :obj:`List[float]` inputs

    .. code-block:: python3

        >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
        ...     [1, 3], [2.0, 2.8], [1.1, 0.8], [0.5, 2.0], 0.123, "simple_sl", "simplified_likelihoods"
        ... )
        >>> statistical_model.exclusion_confidence_level() # [0.7016751766204834]
    """
    if backend == "simplified_likelihoods":
        backend = "simplified_likelihoods.uncorrelated_background"
    if backend == "pyhf":
        backend = "pyhf.uncorrelated_background"

    statistical_model = get_backend(backend)

    if backend == "pyhf.uncorrelated_background":
        return statistical_model(
            signal_yields=signal_yields,
            background_yields=backgrounds,
            data=data,
            absolute_uncertainties=background_uncertainty,
            xsection=xsection,
            analysis=analysis,
        )

    if backend == "simplified_likelihoods.uncorrelated_background":
        # Convert everything to numpy array
        background_uncertainty = (
            np.array(background_uncertainty).reshape(-1)
            if isinstance(background_uncertainty, (list, float))
            else background_uncertainty
        )
        signal_yields = (
            np.array(signal_yields).reshape(-1)
            if isinstance(signal_yields, (list, float))
            else signal_yields
        )
        nobs = (
            np.array(data).reshape(-1) if isinstance(data, (list, float, int)) else data
        )
        nb = (
            np.array(backgrounds).reshape(-1)
            if isinstance(backgrounds, (list, float))
            else backgrounds
        )

        return statistical_model(
            signal_yields=signal_yields,
            background_yields=nb,
            data=nobs,
            absolute_uncertainties=background_uncertainty,
            xsection=xsection,
            analysis=analysis,
        )

    raise NotImplementedError(
        "Requested backend has not been implemented to this helper function."
    )


def get_correlated_nbin_statistical_model(
    data: Union[np.ndarray, Dict[Text, List], List[float]],
    signal_yields: Union[np.ndarray, List[Dict[Text, List]], List[float]],
    covariance_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None,
    backgrounds: Optional[Union[np.ndarray, List[float]]] = None,
    xsection: float = np.nan,
    analysis: Text = "__unknown_analysis__",
) -> StatisticalModel:
    """
    Create a statistical model from a correlated multi-bin data structure.

    Args:
        data (``Union[np.ndarray, Dict[Text, List], List[float]]``): data yields. In order to activate
          :xref:`pyhf` plugin ``JSON`` type of input should be used. For details about the dictionary
          structure please refer to :xref:`pyhf` documentation `in this link <https://pyhf.readthedocs.io/>`_.
          Additionally analysis specific, **background only** ``JSON`` files can be found through
          :xref:`HEPData`.
        signal_yields (``Union[np.ndarray, List[Dict[Text, List]], List[float]]``): signal yields. To
          activate :xref:`pyhf` plugin input needs to have a ``JSONPATCH`` structure consistent with
          the input data.
        covariance_matrix (``Optional[Union[np.ndarray, List[List[float]]]]``, default ``None``):
          Simplified likelihoods are constructed via covariance matrices. Input should have a matrix structure
          with each axis have the same dimensionality as number of reqions included in data input.
          This input is only used for ``"simplified_likelihoods"`` backend.
        backgrounds (``Optional[Union[np.ndarray, List[float]]]``, default ``None``): The SM backgrounds
          for simplified likelihood backend. These are combined background only yields and the size of the input
          vector should be the same as data input. This input is only used for ``"simplified_likelihoods"`` backend.
        xsection (``float``, default ``np.nan``): cross section value. unit is determined by the user.
        analysis (``Text``, default ``"__unknown_analysis__"``): unique analysis identifier.

    Raises:
        `NotImplementedError`: If the plugin does not exist or the inputs are not consistently matching
          to a particular plugin.

    Returns:
        :class:`~spey.StatisticalModel`:
        Model formed with correlated multi-bin structure.

    Example:

    ``"simplified_likelihoods"`` backend can be invoked via the following input structure

    .. code-block:: python3
        :linenos:

        >>> import spey
        >>> statistical_model = spey.get_correlated_nbin_statistical_model(
        ...     analysis="simple_sl_test",
        ...     signal_yields=[12.0, 11.0],
        ...     data=[51.0, 48.0],
        ...     covariance_matrix=[[3.,0.5], [0.6,7.]],
        ...     backgrounds=[50.0, 52.0],
        ...     third_moment=[0.2, 0.1],
        ...     xsection=0.5
        ... )
        >>> statistical_model.exclusion_confidence_level() # [0.9733284916728735]
        >>> statistical_model.backend_type # 'simplified_likelihoods'
    """

    if (
        isinstance(signal_yields, list)
        and isinstance(signal_yields[0], dict)
        and isinstance(data, dict)
    ):
        PyhfInterface = get_backend("pyhf")
        return PyhfInterface(
            signal_patch=signal_yields,
            background_only_model=data,
            xsection=xsection,
            analysis=analysis,
        )

    if (
        covariance_matrix is not None
        and isinstance(signal_yields, (list, np.ndarray))
        and isinstance(data, (list, np.ndarray))
    ):
        SimplifiedLikelihoodInterface = get_backend("simplified_likelihoods")

        # Convert everything to numpy array
        covariance_matrix = (
            np.array(covariance_matrix)
            if isinstance(covariance_matrix, list)
            else covariance_matrix
        )
        signal_yields = (
            np.array(signal_yields) if isinstance(signal_yields, list) else signal_yields
        )
        data = np.array(data) if isinstance(data, list) else data
        backgrounds = (
            np.array(backgrounds) if isinstance(backgrounds, list) else backgrounds
        )

        return SimplifiedLikelihoodInterface(
            signal_yields=signal_yields,
            background_yields=backgrounds,
            data=data,
            covariance_matrix=covariance_matrix,
            xsection=xsection,
            analysis=analysis,
        )

    raise NotImplementedError(
        "Requested backend has not been implemented to this helper function."
    )
