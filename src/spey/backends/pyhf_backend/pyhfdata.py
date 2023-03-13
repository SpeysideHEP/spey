from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List, Text, Tuple

import pyhf, copy
import numpy as np
from pyhf import Workspace, Model

from spey.utils import ExpectationType
from spey.system.exceptions import NegativeExpectedYields
from spey.base import DataBase
from spey.base import ModelConfig
from .utils import initialise_workspace

__all__ = ["PyhfData", "PyhfDataWrapper"]


def PyhfDataWrapper(
    signal: Union[List[Dict], float],
    background: Union[Dict, float],
    nb: Optional[float] = None,
    delta_nb: Optional[float] = None,
    default_expectation: Optional[ExpectationType] = ExpectationType.observed,
    name: Text = "__unknown_model__",
):
    """
    Dataclass for pyhf interface

    :param signal: either histfactory type signal patch or float value of number of events
    :param background: either background only JSON histfactory or float value of observed data
    :param nb: expected number of background events. In case of statistical model it is not needed
    :param delta_nb: uncertainty on backgorund. In case of statistical model it is not needed
    :param default_expectation: observed, aprioti, aposteriori
    :param name: name of the statistical model.
    :raises AssertionError: if the statistical model is not valid
    """
    (
        new_signal,
        new_background,
        new_nb,
        new_delta_nb,
        workspace,
        model,
        data,
        minimum_poi,
    ) = initialise_workspace(
        signal, background, nb, delta_nb, expected=default_expectation, return_full_data=True
    )
    return PyhfData(
        signal=new_signal,
        background=new_background,
        nb=new_nb,
        delta_nb=new_delta_nb,
        default_expectation=default_expectation,
        name=name,
        _workspace=workspace,
        _model=model,
        _data=data,
        _minimum_poi=minimum_poi,
    )


@dataclass(frozen=True)
class PyhfData(DataBase):
    """
    Dataclass for pyhf interface

    :param signal: either histfactory type signal patch or float value of number of events
    :param background: either background only JSON histfactory or float value of observed data
    :param nb: expected number of background events. In case of statistical model it is not needed
    :param delta_nb: uncertainty on backgorund. In case of statistical model it is not needed
    :param default_expectation: observed, aprioti, aposteriori
    :param name: name of the statistical model.
    :param _workspace: pyhf.Workpace
    :param _model: pyhf.pdf.Model
    :param _data: data combined with auxiliary data
    :param _minimum_poi: minimum value that POI can take.
    :raises AssertionError: if the statistical model is not valid
    """

    signal: Union[List[Dict], np.ndarray]
    background: Union[Dict, np.ndarray]
    nb: Optional[np.ndarray] = None
    delta_nb: Optional[np.ndarray] = None
    default_expectation: Optional[ExpectationType] = field(
        default=ExpectationType.observed, repr=False
    )
    name: Text = "__unknown_model__"
    _workspace: Optional[pyhf.Workspace] = field(default=None, init=True, repr=False)
    _model: Optional[pyhf.pdf.Model] = field(default=None, init=True, repr=False)
    _data: Optional[List[float]] = field(default=None, init=True, repr=False)
    _minimum_poi: float = field(default=-np.inf, init=True, repr=False)

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: Optional[float] = None
    ) -> ModelConfig:
        """
        Configuration of the statistical model. This class contains information
        regarding how the fit should evolve.

        :param allow_negative_signal (`bool`, default `True`): if the negative POI is allowed during fits.
        :param poi_upper_bound (` Optional[float]`, default `None`): sets the upper bound for POI
        :return `ModelConfig`: Configuration information of the model.
        """
        bounds = self._model.config.suggested_bounds
        bounds[self._model.config.poi_index] = (
            self._minimum_poi if allow_negative_signal else 0.0,
            bounds[self._model.config.poi_index][1] if not poi_upper_bound else poi_upper_bound,
        )
        return ModelConfig(
            poi_index=self._model.config.poi_index,
            minimum_poi=self._minimum_poi,
            suggested_init=self._model.config.suggested_init,
            suggested_bounds=bounds,
        )

    def __call__(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
    ) -> Union[
        tuple[Optional[Workspace], Optional[Model], Optional[List[float]]],
        tuple[Workspace, Model, Union[np.ndarray, np.ndarray]],
    ]:
        """
        Create pyhf workspace with respect to given POI and expectation

        :param poi_test: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :return: workspace, statistical model and data
        :raises NegativeExpectedYields: if `mu * signal + background` becomes
                                        negative for the requested POI test.
        """
        if poi_test == 1.0 and expected == self.default_expectation:
            return (
                copy.deepcopy(self._workspace),
                copy.deepcopy(self._model),
                copy.deepcopy(self._data),
            )

        signal = copy.deepcopy(self.signal)
        if poi_test != 1.0:
            if isinstance(self.signal, np.ndarray):
                signal *= poi_test
            else:
                for ids, channel in enumerate(signal):
                    if signal[ids].get("value", False):
                        signal[ids]["value"]["data"] = np.array(
                            [nsig * poi_test for nsig in signal[ids]["value"]["data"]],
                            dtype=np.float32,
                        ).tolist()

        workspace, model, data = initialise_workspace(
            signal,
            copy.deepcopy(self.background),
            copy.deepcopy(self.nb),
            copy.deepcopy(self.delta_nb),
            expected=expected,
        )

        # Check if there is any negative number of events
        if poi_test < 0.0:
            if isinstance(signal, np.ndarray):
                if np.any(signal + self.background < 0.0):
                    raise NegativeExpectedYields(
                        "PyhfInterface::Statistical model involves negative expected bin yields. "
                        "Bin value: "
                        + ", ".join([f"{x:.3f}" for x in (signal + self.background).tolist()])
                    )
            else:
                for channel in model.spec.get("channels", []):
                    current: Union[np.ndarray, None] = None
                    for ch in channel.get("samples", []):
                        is_valid: bool = "data" in ch.keys()
                        if current is None and is_valid:
                            current = np.array(ch["data"], dtype=np.float32)
                        elif current is not None and is_valid:
                            current += np.array(ch["data"], dtype=np.float32)
                    if current is not None:
                        if np.any(current < 0.0):
                            raise NegativeExpectedYields(
                                f"PyhfInterface::Statistical model involves negative expected "
                                f"bin yields in region '{channel['name']}'. Bin values: "
                                + ", ".join([f"{x:.3f}" for x in current])
                            )

        return workspace, model, data

    @property
    def isAlive(self) -> bool:
        """Does the statitical model has any non-zero signal events?"""
        if isinstance(self.signal, np.ndarray):
            return np.any(self.signal > 0.0)

        for channel in self.signal:
            if channel.get("value", False):
                if np.any([nsig > 0.0 for nsig in channel["value"].get("data", list())]):
                    return True
        return False

    @property
    def minimum_poi(self):
        """Find minimum POI test that can be applied to this statistical model"""
        return self._minimum_poi
