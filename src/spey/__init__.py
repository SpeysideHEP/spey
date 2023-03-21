from typing import Text, Union, List, Dict, Optional, Tuple, Callable
import numpy as np
import pkg_resources

from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.base import BackendBase
from spey.combiner import StatisticsCombiner
from spey.base.recorder import Recorder
from spey.system.exceptions import PluginError
from .utils import ExpectationType
from ._version import __version__

__all__ = [
    "version",
    "StatisticalModel",
    "StatisticsCombiner",
    "ExpectationType",
    "AvailableBackends",
    "get_multi_region_statistical_model",
    "get_uncorrelated_region_statistical_model",
    "Recorder",
    "get_backend",
]


def version() -> Text:
    """Version of the package"""
    return __version__


def _resolve_backends() -> Dict:
    """Collect plugin entries"""
    return {entry.name: entry for entry in pkg_resources.iter_entry_points("spey.plugins")}


def AvailableBackends() -> List[Text]:
    """
    Return list of available backends

    :return `List[Text]`: List of backend names
    """
    return [*_resolve_backends().keys()]


def get_backend(name: Text) -> Tuple[Callable, StatisticalModel]:
    """
    Retreive backend by name.

    :param name (`Text`): backend identifier
    :raises `PluginError`: if backend is not available in the current system
    :return `Tuple[Callable, StatisticalModel]`: Function to setup model
                    specific data structure and statistical model backend.
    """
    backend = _resolve_backends().get(name, False)

    if backend:
        statistical_model = backend.load()
        return statistical_model.datastructure(), statistical_model_wrapper(statistical_model)

    raise PluginError(
        f"Unknown backend: {name}. Available backends are " + ", ".join(AvailableBackends())
    )


def get_uncorrelated_region_statistical_model(
    observations: Union[int, np.ndarray],
    backgrounds: Union[float, np.ndarray],
    background_uncertainty: Union[float, np.ndarray],
    signal_yields: Union[float, np.ndarray],
    xsection: Union[float, np.ndarray],
    analysis: Text,
    backend: Text,
) -> StatisticalModel:
    """
    Create statistical model from a single bin or multiple uncorrelated regions

    :param observations: number of observed events
    :param backgrounds: number of expected background events
    :param background_uncertainty: uncertainty on background
    :param signal_yields: signal yields
    :param xsection: cross-section
    :param analysis: name of the analysis
    :param backend: pyhf or simplified_likelihoods
    :return: Statistical model

    :raises NotImplementedError: If requested backend has not been recognised.
    """
    datastructure, statistical_model = get_backend(backend)

    if backend == "pyhf":
        model = datastructure(
            signal=signal_yields,
            background=observations,
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
        nobs = (
            np.array(observations).reshape(-1)
            if isinstance(observations, (list, float))
            else observations
        )
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


def get_multi_region_statistical_model(
    analysis: Text,
    signal: Union[np.ndarray, List[Dict[Text, List]], List[float]],
    observed: Union[np.ndarray, Dict[Text, List], List[float]],
    covariance: Optional[Union[np.ndarray, List[List[float]]]] = None,
    nb: Optional[Union[np.ndarray, List[float]]] = None,
    third_moment: Optional[Union[np.ndarray, List[float]]] = None,
    delta_sys: float = 0.0,
    xsection: float = np.nan,
) -> StatisticalModel:
    """
    Create a statistical model from multibin data.

    :param signal: number of signal events. For simplified likelihood backend this input can
                   contain `np.array` or `List[float]` which contains signal yields per region.
                   For `pyhf` backend this input expected to be a JSON-patch i.e. `List[Dict]`,
                   see `pyhf` documentation for details on JSON-patch format.
    :param observed: number of observed events. For simplified likelihood backend this input can
                       be `np.ndarray` or `List[float]` which contains observations. For `pyhf`
                       backend this contains **background only** JSON sterilized HistFactory
                       i.e. `Dict[List]`.
    :param covariance: Covariance matrix either in the form of `List` or NumPy array. Only used for
                       simplified likelihood backend.
    :param nb: number of expected background yields. Only used for simplified likelihood backend.
    :param third_moment: third moment. Only used for simplified likelihood backend.
    :param delta_sys: systematic uncertainty on signal. Only used for simplified likelihood backend.
    :param xsection: cross-section in pb
    :param analysis: name of the analysis
    :return: Statistical model

    :raises NotImplementedError: if input patter does not match to any backend specific input option

    `pyhf` interface example

    .. code-block:: python3

        >>> import spey
        >>> background = {
        >>>   "channels": [
        >>>     { "name": "singlechannel",
        >>>       "samples": [
        >>>         { "name": "background",
        >>>           "data": [50.0, 52.0],
        >>>           "modifiers": [{ "name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0]}]
        >>>         }
        >>>       ]
        >>>     }
        >>>   ],
        >>>   "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
        >>>   "measurements": [{"name": "Measurement", "config": { "poi": "mu", "parameters": []} }],
        >>>   "version": "1.0.0"
        >>> }
        >>> signal = [{"op": "add",
        >>>     "path": "/channels/0/samples/1",
        >>>     "value": {"name": "signal", "data": [12.0, 11.0],
        >>>       "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]}}]
        >>> multi_bin = spey.get_multi_region_statistical_model(
        >>>     "simpleanalysis", signal, background, xsection=1.
        >>> )
        >>> print(multi_bin)
        >>> # StatisticalModel(analysis='simpleanalysis', xsection=1.000e+00 [pb], backend=pyhf)
        >>> multi_bin.exclusion_confidence_level()
        >>> # [0.9474850257628679] # 1-CLs
        >>> multi_bin.s95exp
        >>> # 1.0685773410460155 # prefit excluded cross section in pb
        >>> multi_bin.maximize_likelihood()
        >>> # (-0.0669277855002002, 12.483595567080783) # muhat and maximum negative log-likelihood
        >>> multi_bin.likelihood(poi_test=1.5)
        >>> # 16.59756909879556
        >>> multi_bin.exclusion_confidence_level(expected=spey.ExpectationType.aposteriori)
        >>> # [0.9973937390501324, 0.9861799464393675, 0.9355467946443513, 0.7647435613928496, 0.4269637940897122]

    Simplified Likelihood interface example

    .. code-block:: python3

        >>> stat_model_sl = spey.get_multi_region_statistical_model(
        >>>     "simple_sl_test",
        >>>     signal=[12.0, 11.0],
        >>>     observed=[51.0, 48.0],
        >>>     covariance=[[3.,0.5], [0.6,7.]],
        >>>     nb=[50.0, 52.0],
        >>>     delta_sys=0.,
        >>>     third_moment=[0.2, 0.1],
        >>>     xsection=0.5
        >>> )
        >>> stat_model_sl.chi2(poi_test=2.5)
        >>> # 24.80950457177922
        >>> stat_model_sl.s95exp, stat_model_sl.s95obs
        >>> # 0.47739909991661555, 0.4351657698811163

    """

    if isinstance(signal, list) and isinstance(signal[0], dict) and isinstance(observed, dict):
        PyhfDataWrapper, PyhfInterface = get_backend("pyhf")
        model = PyhfDataWrapper(signal=signal, background=observed, name="pyhfModel")
        return PyhfInterface(model=model, xsection=xsection, analysis=analysis)

    if (
        covariance is not None
        and isinstance(signal, (list, np.ndarray))
        and isinstance(observed, (list, np.ndarray))
    ):
        SLData, SimplifiedLikelihoodInterface = get_backend("simplified_likelihoods")

        # Convert everything to numpy array
        covariance = np.array(covariance) if isinstance(covariance, list) else covariance
        signal = np.array(signal) if isinstance(signal, list) else signal
        observed = np.array(observed) if isinstance(observed, list) else observed
        nb = np.array(nb) if isinstance(nb, list) else nb
        third_moment = np.array(third_moment) if isinstance(third_moment, list) else third_moment

        model = SLData(
            observed=observed,
            signal=signal,
            background=nb,
            covariance=covariance,
            delta_sys=delta_sys,
            third_moment=third_moment,
            name="SLModel",
        )

        # pylint: disable=E1123
        return SimplifiedLikelihoodInterface(model=model, xsection=xsection, analysis=analysis)

    raise NotImplementedError("Requested backend has not been recognised.")
