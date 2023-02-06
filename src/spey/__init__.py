from .utils import ExpectationType
from typing import Text, Union, List, Dict, Optional
import numpy as np

from .backends import AvailableBackends
from spey.interface.statistical_model import StatisticalModel
from spey.combiner.statistics_combiner import StatisticsCombiner
from spey.base.recorder import Recorder
from ._version import __version__

__all__ = [
    "__version__",
    "StatisticalModel",
    "StatisticsCombiner",
    "ExpectationType",
    "AvailableBackends",
    "get_multi_region_statistical_model",
    "get_uncorrelated_region_statistical_model",
    "Recorder",
]


def get_uncorrelated_region_statistical_model(
    nobs: Union[int, np.ndarray],
    nb: Union[float, np.ndarray],
    deltanb: Union[float, np.ndarray],
    signal_yields: Union[float, np.ndarray],
    xsection: Union[float, np.ndarray],
    analysis: Text,
    backend: AvailableBackends,
) -> StatisticalModel:
    """
    Create statistical model from a single bin or multiple uncorrelated regions

    :param nobs: number of observed events
    :param nb: number of expected background events
    :param deltanb: uncertainty on background
    :param signal_yields: signal yields
    :param xsection: cross-section
    :param analysis: name of the analysis
    :param backend: pyhf or simplified_likelihoods
    :return: Statistical model

    :raises NotImplementedError: If requested backend has not been recognised.
    """
    if backend == AvailableBackends.pyhf:
        from spey.backends.pyhf_backend import PyhfInterface, PyhfDataWrapper

        model = PyhfDataWrapper(signal=signal_yields, background=nobs, nb=nb, delta_nb=deltanb)
        return PyhfInterface(model=model, xsection=xsection, analysis=analysis)

    elif backend == AvailableBackends.simplified_likelihoods:
        from spey.backends.simplifiedlikelihood_backend import SLData, SimplifiedLikelihoodInterface

        # Convert everything to numpy array
        covariance = np.array(deltanb).reshape(-1) if isinstance(deltanb, (list, float)) else deltanb
        signal_yields = (
            np.array(signal_yields).reshape(-1) if isinstance(signal_yields, (list, float)) else signal_yields
        )
        nobs = np.array(nobs).reshape(-1) if isinstance(nobs, (list, float)) else nobs
        nb = np.array(nb).reshape(-1) if isinstance(nb, (list, float)) else nb
        covariance = covariance * np.eye(len(covariance))

        model = SLData(
            signal=signal_yields,
            observed=nobs,
            covariance=covariance,
            background=nb,
            delta_sys=0.0,
            name="model",
        )
        return SimplifiedLikelihoodInterface(model=model, xsection=xsection, analysis=analysis)

    else:
        raise NotImplementedError(
            f"Requested backend ({backend}) has not been implemented. "
            f"Currently available backends are " + ", ".join(AvailableBackends) + "."
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
        from spey.backends.pyhf_backend import PyhfDataWrapper, PyhfInterface

        model = PyhfDataWrapper(signal=signal, background=observed)
        return PyhfInterface(model=model, xsection=xsection, analysis=analysis)

    elif (
        covariance is not None
        and isinstance(signal, (list, np.ndarray))
        and isinstance(observed, (list, np.ndarray))
    ):
        from spey.backends.simplifiedlikelihood_backend import SLData, SimplifiedLikelihoodInterface

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
            name="model",
        )

        return SimplifiedLikelihoodInterface(model=model, xsection=xsection, analysis=analysis)

    else:
        raise NotImplementedError("Requested backend has not been recognised.")
