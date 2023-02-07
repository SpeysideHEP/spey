import pyhf, copy
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List, Text, Tuple
import numpy as np
from pyhf import Workspace, Model

from spey.utils import ExpectationType
from .utils import initialise_workspace
from spey.system.exceptions import NegativeExpectedYields
from spey.base.backend_base import DataBase

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

    @property
    def npar(self) -> int:
        """Number of nuisance parameters except poi"""
        return self._model.config.npars - 1

    @property
    def poi_index(self) -> int:
        return self._model.config.poi_index

    @property
    def suggested_init(self) -> np.ndarray:
        """Suggested initial nuisance parameters (except poi)"""
        pars = self._model.config.suggested_init()
        return pars[: self.poi_index] + pars[self.poi_index + 1 :]

    @property
    def suggested_bounds(self) -> List[Tuple[float, float]]:
        """Suggested bounds for nuisance parameters (except poi)"""
        pars = self._model.config.suggested_bounds()
        return pars[: self.poi_index] + pars[self.poi_index + 1 :]

    @property
    def suggested_fixed(self) -> List[bool]:
        """Suggested fixed nuisance parameters (except poi)"""
        pars = self._model.config.suggested_fixed()
        return pars[: self.poi_index] + pars[self.poi_index + 1 :]

    @property
    def suggested_poi_init(self) -> np.ndarray:
        """Suggested initial nuisance parameters (except poi)"""
        return self._model.config.suggested_init()[self.poi_index]

    @property
    def suggested_poi_bounds(self) -> Tuple[float, float]:
        """Suggested bounds for nuisance parameters (except poi)"""
        return self._model.config.suggested_bounds()[self.poi_index]

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
            if isinstance(signal, float):
                if signal + self.background < 0.0:
                    raise NegativeExpectedYields(
                        f"PyhfInterface::Statistical model involves negative expected bin yields. "
                        f"Bin value: {signal + self.background:.3f}"
                    )
            else:
                for channel in model.spec.get("channels", []):
                    current: Union[np.ndarray, None] = None
                    for idx, ch in enumerate(channel.get("samples", [])):
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
        else:
            for channel in self.signal:
                if channel.get("value", False):
                    if np.any([nsig > 0.0 for nsig in channel["value"].get("data", list())]):
                        return True
            return False

    @property
    def minimum_poi(self):
        """Find minimum POI test that can be applied to this statistical model"""
        return self._minimum_poi
