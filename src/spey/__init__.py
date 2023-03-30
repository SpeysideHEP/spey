from typing import Text, Union, List, Dict, Optional, Tuple, Callable
import numpy as np
import pkg_resources
from semantic_version import Version, SimpleSpec

from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.base import BackendBase, DataBase
from spey.combiner import StatisticsCombiner
from spey.system.exceptions import PluginError
from .utils import ExpectationType
from ._version import __version__

__all__ = [
    "version",
    "StatisticalModel",
    "StatisticsCombiner",
    "ExpectationType",
    "AvailableBackends",
    "get_correlated_nbin_statistical_model",
    "get_uncorrelated_nbin_statistical_model",
    "get_backend",
    "get_backend_metadata",
    "BackendBase",
    "DataBase",
]


def __dir__():
    return __all__


def version() -> Text:
    """
    Version of :obj:`spey` package

    Returns:
        :obj:`Text`: version in X.Y.Z format
    """
    return __version__


def _get_backend_entrypoints() -> Dict:
    """Collect plugin entries"""
    return {entry.name: entry for entry in pkg_resources.iter_entry_points("spey.backend.plugins")}


def AvailableBackends() -> List[Text]:
    """
    Returns a list of available backends. The default backends are automatically installed
    with :obj:`spey` package. However there are plugins available to interface for other
    packages that has different likelihood prescription such as `pyhf <https://pyhf.readthedocs.io/en/v0.7.0/>`_.

    Returns:
        :obj:`List[Text]`: list of names of available backends.
    """
    return [*_get_backend_entrypoints().keys()]


def get_backend(name: Text) -> Tuple[Union[Callable, DataBase], StatisticalModel]:
    """
    Statistical model backend retreiver. Available backend names can be found via
    :func:`~spey.AvailableBackends` function.

    Args:
        name (:obj:`Text`): backend identifier. This backend refers to different packages
          that prescribes likelihood function.

    Raises:
        ~spey.system.exception.PluginError: If the backend is not available or the available backend requires
          different version of :obj:`spey`.
        :obj:`AssertionError`: If the backend does not have necessary metadata.

    Returns:
        :obj:`Tuple[Union[Callable, DataBase], StatisticalModel]`:
        statistical model wraped with :func:`~spey.interface.statistiacl_model.statistical_model_wrapper`
        and backend specific data handler.

    Example:

    .. code-block:: python3

        >>> import spey; import numpy as np
        >>> data_handler, stat_wrapper = spey.get_backend("simplified_likelihoods")
        >>> data = np.array([1])
        >>> signal = np.array([0.5])
        >>> background = np.array([2.])
        >>> background_unc = np.array([[1.1]])
        >>> stat_model = stat_wrapper(
        ...    data_handler(data, signal, background, background_unc),
        ...    analysis="simple_sl",
        ...    xsection=0.123
        ... )
        >>> stat_model.exclusion_confidence_level() # [0.4022058844566345]
    """
    backend = _get_backend_entrypoints().get(name, False)

    if backend:
        statistical_model = backend.load()

        for meta in ["name", "spey_requires", "datastructure", "author", "version"]:
            assert hasattr(statistical_model, meta), f"Backend does not have {meta} attribute."

        if Version(version()) not in SimpleSpec(statistical_model.spey_requires):
            raise PluginError(
                f"The backend {name}, requires spey version {statistical_model.spey_requires}. "
                f"However the current spey version is {__version__}."
            )
        return statistical_model.datastructure, statistical_model_wrapper(statistical_model)

    raise PluginError(
        f"The backend {name} is unavailable. Available backends are "
        + ", ".join(AvailableBackends())
        + "."
    )


def get_backend_metadata(name: Text) -> Dict[Text, Text]:
    """
    Retreive metadata about the backend. This includes citation information,
    doi, author names etc. Available backend names can be found via
    :func:`~spey.AvailableBackends` function.

    Args:
        name (:obj:`Text`): backend identifier. This backend refers to different packages
          that prescribes likelihood function.

    Raises:
        ~spey.system.exceptions.PluginError: If the backend does not exists.

    Returns:
        :obj:`Dict[Text, Text]`:
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
    backend = _get_backend_entrypoints().get(name, False)

    if backend:
        statistical_model = backend.load()
        return {
            "name": statistical_model.name,
            "author": statistical_model.author,
            "version": statistical_model.version,
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
        data (:obj:`float, int, np.ndarray, List[float]`): data yields
        backgrounds (:obj:`float, np.ndarray, List[float]`): background yields
        background_uncertainty (:obj:`float, np.ndarray, List[float]`): absolute background uncertainty
        signal_yields (:obj:`float, np.ndarray, List[float]`): signal yields
        xsection (:obj:`float, np.ndarray, List[float]`): cross section value, unit determined by the user.
        analysis (:obj:`Text`): unique analysis name for the statistical model.
        backend (:obj:`Text`): statistical model backend. Currently available backend names can be
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
    datastructure, statistical_model = get_backend(backend)

    if backend == "pyhf":
        model = datastructure(
            signal=signal_yields,
            background=data,
            nb=backgrounds,
            delta_nb=background_uncertainty,
            name="pyhfModel",
        )

        return statistical_model(model=model, xsection=xsection, analysis=analysis)

    if backend == "simplified_likelihoods":
        # Convert everything to numpy array
        covariance = (
            np.array(background_uncertainty).reshape(-1)
            if isinstance(background_uncertainty, (list, float))
            else background_uncertainty
        )
        signal_yields = (
            np.array(signal_yields).reshape(-1)
            if isinstance(signal_yields, (list, float))
            else signal_yields
        )
        nobs = np.array(data).reshape(-1) if isinstance(data, (list, float, int)) else data
        nb = (
            np.array(backgrounds).reshape(-1)
            if isinstance(backgrounds, (list, float))
            else backgrounds
        )
        covariance = np.square(covariance) * np.eye(len(covariance))

        model = datastructure(
            signal=signal_yields,
            observed=nobs,
            covariance=covariance,
            background=nb,
            delta_sys=0.0,
            name="SLModel",
        )

        return statistical_model(model=model, xsection=xsection, analysis=analysis)

    raise NotImplementedError(
        f"Requested backend ({backend}) has not been implemented. "
        f"Currently available backends are " + ", ".join(AvailableBackends()) + "."
    )


def get_correlated_nbin_statistical_model(
    data: Union[np.ndarray, Dict[Text, List], List[float]],
    signal_yields: Union[np.ndarray, List[Dict[Text, List]], List[float]],
    covariance_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None,
    backgrounds: Optional[Union[np.ndarray, List[float]]] = None,
    third_moment: Optional[Union[np.ndarray, List[float]]] = None,
    delta_sys: float = 0.0,
    xsection: float = np.nan,
    analysis: Text = "__unknown_analysis__",
) -> StatisticalModel:
    """
    Create a statistical model from a correlated multi-bin data structure.

    Args:
        data (:obj:`Union[np.ndarray, Dict[Text, List], List[float]]`): data yields. In order to activate
          :xref:`pyhf` plugin :obj:`JSON` type of input should be used. For details about the dictionary
          structure please refer to :xref:`pyhf` documentation `in this link <https://pyhf.readthedocs.io/>`_.
          Additionally analysis specific, **background only** :obj:`JSON` files can be found through
          :xref:`HEPData`.
        signal_yields (:obj:`Union[np.ndarray, List[Dict[Text, List]], List[float]]`): signal yields. To
          activate :xref:`pyhf` plugin input needs to have a :obj:`JSONPATCH` structure consistent with
          the input data.
        covariance_matrix (:obj:`Optional[Union[np.ndarray, List[List[float]]]]`, default :obj:`None`):
          Simplified likelihoods are constructed via covariance matrices. Input should have a matrix structure
          with each axis have the same dimensionality as number of reqions included in data input.
          This input is only used for :obj:`simplified_likelihoods` backend.
        backgrounds (:obj:`Optional[Union[np.ndarray, List[float]]]`, default :obj:`None`): The SM backgrounds
          for simplified likelihood backend. These are combined background only yields and the size of the input
          vector should be the same as data input. This input is only used for :obj:`simplified_likelihoods` backend.
        third_moment (:obj:`Optional[Union[np.ndarray, List[float]]]`, default :obj:`None`): Third moment information
          for the scewed gaussian formation in simplified likelihoods. This input is only used for
          :obj:`simplified_likelihoods` backend.
        delta_sys (:obj:`float`, default :obj:`0.0`): Systematic uncertainties on signal.
          This input is only used for :obj:`simplified_likelihoods` backend.
        xsection (:obj:`float`, default :obj:`np.nan`): cross section value. unit is determined by the user.
        analysis (:obj:`Text`, default :obj:`"__unknown_analysis__"`): unique analysis identifier.

    Raises:
        `NotImplementedError`: If the plugin does not exist or the inputs are not consistently matching
          to a particular plugin.

    Returns:
        :obj:`StatisticalModel`:
        Model formed with correlated multi-bin structure.

    Example:

    :obj:`simplified_likelihoods` backend can be invoked via the following input structure

    .. code-block:: python3

        >>> import spey
        >>> statistical_model = spey.get_correlated_nbin_statistical_model(
        ...     analysis="simple_sl_test",
        ...     signal_yields=[12.0, 11.0],
        ...     data=[51.0, 48.0],
        ...     covariance_matrix=[[3.,0.5], [0.6,7.]],
        ...     backgrounds=[50.0, 52.0],
        ...     delta_sys=0.,
        ...     third_moment=[0.2, 0.1],
        ...     xsection=0.5
        ... )
        >>> statistical_model.exclusion_confidence_level() # [0.9733284916728735]
        >>> statistical_model.backend_type # 'simplified_likelihoods'

    Assuming that :xref:`pyhf` plugin for spey has been installed via ``pip install spey-pyhf`` command,
    :obj:`JSON` type of input can be given to the function as follows;

    .. code-block:: python3

        >>> import spey
        >>> background_only = {
        ...   "channels": [
        ...     { "name": "singlechannel",
        ...       "samples": [
        ...         { "name": "background",
        ...           "data": [50.0, 52.0],
        ...           "modifiers": [{ "name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0]}]
        ...         }
        ...       ]
        ...     }
        ...   ],
        ...   "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
        ...   "measurements": [{"name": "Measurement", "config": { "poi": "mu", "parameters": []} }],
        ...   "version": "1.0.0"
        ... }
        >>> signal = [{"op": "add",
        ...     "path": "/channels/0/samples/1",
        ...     "value": {"name": "signal", "data": [12.0, 11.0],
        ...     "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]}}]
        >>> statistical_model = spey.get_correlated_nbin_statistical_model(
        ...     analysis="simple_pyhf",
        ...     data=background_only,
        ...     signal_yields=signal,
        ... )
        >>> statistical_model.exclusion_confidence_level() # [0.9474850259721279]
        >>> statistical_model.backend_type # 'pyhf'
    """

    if (
        isinstance(signal_yields, list)
        and isinstance(signal_yields[0], dict)
        and isinstance(data, dict)
    ):
        PyhfDataWrapper, PyhfInterface = get_backend("pyhf")
        model = PyhfDataWrapper(signal=signal_yields, background=data, name="pyhfModel")
        return PyhfInterface(model=model, xsection=xsection, analysis=analysis)

    if (
        covariance_matrix is not None
        and isinstance(signal_yields, (list, np.ndarray))
        and isinstance(data, (list, np.ndarray))
    ):
        SLData, SimplifiedLikelihoodInterface = get_backend("simplified_likelihoods")

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
        backgrounds = np.array(backgrounds) if isinstance(backgrounds, list) else backgrounds
        third_moment = np.array(third_moment) if isinstance(third_moment, list) else third_moment

        model = SLData(
            observed=data,
            signal=signal_yields,
            background=backgrounds,
            covariance=covariance_matrix,
            delta_sys=delta_sys,
            third_moment=third_moment,
            name="SLModel",
        )

        # pylint: disable=E1123
        return SimplifiedLikelihoodInterface(model=model, xsection=xsection, analysis=analysis)

    raise NotImplementedError("Requested backend has not been recognised.")
