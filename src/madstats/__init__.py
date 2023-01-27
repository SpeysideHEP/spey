from .utils import ExpectationType, Units
from .base.backend_base import BackendBase
from typing import Text, Union, List, Dict, Optional
import numpy as np

from madstats.backends import AvailableBackends
from madstats.interface.statistical_model import StatisticalModel
from madstats.combiner.prediction_combiner import PredictionCombiner
from ._version import __version__

__all__ = [
    "__version__",
    "StatisticalModel",
    "PredictionCombiner",
    "ExpectationType",
    "AvailableBackends",
    "get_multi_region_statistical_model",
    "get_single_region_statistical_model"
]


def get_single_region_statistical_model(
    nobs: int,
    nb: float,
    deltanb: float,
    signal_eff: float,
    xsection: float,
    lumi: float,
    analysis: Text,
    backend: AvailableBackends,
) -> StatisticalModel:
    """
    Create statistical model from a single bin

    :param nobs: number of observed events
    :param nb: number of expected background events
    :param deltanb: uncertainty on background
    :param signal_eff: signal efficiency
    :param xsection: cross-section in pb
    :param lumi: luminosity in 1/fb
    :param analysis: name of the analysis
    :param backend: pyhf or simplified_likelihoods
    :return: Statistical model

    :raises NotImplementedError: If requested backend has not been recognised.
    """
    if backend == AvailableBackends.pyhf:
        from madstats.backends.pyhf_backend.interface import PyhfInterface, Data

        model = Data(
            signal=signal_eff * xsection * 1000.0 * lumi, background=nobs, nb=nb, delta_nb=deltanb
        )
        return StatisticalModel(
            backend=PyhfInterface(model=model), xsection=xsection, analysis=analysis
        )

    elif backend == AvailableBackends.simplified_likelihoods:
        from madstats.backends.simplifiedlikelihood_backend.interface import (
            SimplifiedLikelihoodInterface,
            Data,
        )

        model = Data(
            signal=np.array([signal_eff * xsection * 1000.0 * lumi]),
            observed=np.array([nobs]),
            covariance=np.array([deltanb]),
            background=np.array([nb]),
            delta_sys=0.0,
            name="model",
        )
        return StatisticalModel(
            backend=SimplifiedLikelihoodInterface(model=model), xsection=xsection, analysis=analysis
        )

    else:
        raise NotImplementedError(
            f"Requested backend ({backend}) has not been implemented. "
            f"Currently available backends are " + ", ".join(AvailableBackends) + "."
        )


def get_multi_region_statistical_model(
    analysis: Text,
    signal: Union[np.ndarray, List[Dict[Text, List]], List[float]],
    background: Union[np.ndarray, Dict[Text, List], List[float]],
    covariance: Optional[Union[np.ndarray, List[List[float]]]] = None,
    nb: Optional[np.ndarray] = None,
    third_moment: Optional[np.ndarray] = None,
    delta_sys: float = 0.2,
    xsection: float = 1.0,
) -> StatisticalModel:
    """
    Create a statistical model from multibin data.

    :param signal: number of signal events. For simplified likelihood backend this input can
                   contain `np.array` or `List[float]` which contains signal yields per region.
                   For `pyhf` backend this input expected to be a JSON-patch i.e. `List[Dict]`,
                   see `pyhf` documentation for details on JSON-patch format.
    :param background: number of observed events. For simplified likelihood backend this input can
                       be `np.ndarray` or `List[float]` which contains observations. For `pyhf`
                       backend this contains background only JSON sterilized HistFactory
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
    """
    assert len(signal) > 1, "Incorrect input shape."

    if isinstance(signal, list) and isinstance(signal[0], dict) and isinstance(background, dict):
        from madstats.backends.pyhf_backend.interface import PyhfInterface
        from madstats.backends.pyhf_backend.data import Data

        model = Data(signal=signal, background=background)
        return StatisticalModel(
            backend=PyhfInterface(model=model), xsection=xsection, analysis=analysis
        )

    elif (
        covariance is not None
        and isinstance(signal, (list, np.ndarray))
        and isinstance(background, (list, np.ndarray))
    ):
        from madstats.backends.simplifiedlikelihood_backend.interface import (
            SimplifiedLikelihoodInterface,
            Data,
        )

        # Convert everything to numpy array
        covariance = np.array(covariance) if isinstance(covariance, list) else covariance
        signal = np.array(signal) if isinstance(signal, list) else signal
        background = np.array(background) if isinstance(background, list) else background
        nb = np.array(nb) if isinstance(nb, list) else nb

        model = Data(
            observed=background,
            signal=signal,
            background=nb,
            covariance=covariance,
            delta_sys=delta_sys,
            third_moment=third_moment,
            name="model",
        )

        return StatisticalModel(
            backend=SimplifiedLikelihoodInterface(model=model), xsection=xsection, analysis=analysis
        )

    else:
        raise NotImplementedError("Requested backend has not been recognised.")
