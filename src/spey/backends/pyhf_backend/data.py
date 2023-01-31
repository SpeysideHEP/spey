import pyhf, copy
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List, Text, Tuple
import numpy as np
from pyhf import Workspace, Model

from spey.utils import ExpectationType
from .utils import initialise_workspace
from spey.system.exceptions import NegativeExpectedYields
from spey.base.backend_base import DataBase

__all__ = ["Data"]


@dataclass(frozen=True)
class Data(DataBase):
    """
    Dataclass for pyhf interface

    :param signal: either histfactory type signal patch or float value of number of events
    :param background: either background only JSON histfactory or float value of observed data
    :param nb: expected number of background events. In case of statistical model it is not needed
    :param delta_nb: uncertainty on backgorund. In case of statistical model it is not needed
    :param name: name of the statistical model.
    :raises AssertionError: if the statistical model is not valid
    """

    signal: Union[List[Dict], float]
    background: Union[Dict, float]
    nb: Optional[float] = None
    delta_nb: Optional[float] = None
    default_expectation: Optional[ExpectationType] = field(
        default=ExpectationType.observed, repr=False
    )
    name: Text = "__unknown_model__"
    _workspace: Optional[pyhf.Workspace] = field(default=None, init=False, repr=False)
    _model: Optional[pyhf.pdf.Model] = field(default=None, init=False, repr=False)
    _data: Optional[List[float]] = field(default=None, init=False, repr=False)
    _minimum_poi_test: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        workspace, model, data = initialise_workspace(
            self.signal, self.background, self.nb, self.delta_nb, expected=self.default_expectation
        )
        assert model is not None and data is not None, "Invalid statistical model."
        object.__setattr__(self, "_workspace", workspace)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_data", data)

        # Find minimum POI test that can be applied to this statistical model
        if isinstance(self.signal, float):
            if self.signal > 0.0:
                object.__setattr__(
                    self, "_minimum_poi_test", -np.true_divide(self.background, self.signal)
                )
            else:
                object.__setattr__(self, "_minimum_poi_test", -np.inf)
        else:
            min_ratio = []
            for idc, channel in enumerate(self.background.get("channels", [])):
                current_signal = []
                for sigch in self.signal:
                    if idc == int(sigch["path"].split("/")[2]):
                        current_signal = np.array(
                            sigch.get("value", {}).get("data", []), dtype=np.float32
                        )
                        break
                if len(current_signal) == 0:
                    continue
                current_bkg = []
                for ch in channel["samples"]:
                    if len(current_bkg) == 0:
                        current_bkg = np.zeros(shape=(len(ch["data"]),), dtype=np.float32)
                    current_bkg += np.array(ch["data"], dtype=np.float32)
                min_ratio.append(
                    np.min(
                        np.true_divide(
                            current_bkg[current_signal != 0.0],
                            current_signal[current_signal != 0.0],
                        )
                    )
                    if np.any(current_signal != 0.0)
                    else np.inf
                )
            if len(min_ratio) > 0:
                object.__setattr__(self, "_minimum_poi_test", -np.min(min_ratio).astype(np.float32))
            else:
                object.__setattr__(self, "_minimum_poi_test", -np.inf)

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
        tuple[Optional[Workspace], Optional[Model], Optional[list[float]]],
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
            if isinstance(self.signal, float):
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
        if isinstance(self.signal, float):
            return self.signal > 0.0
        else:
            for channel in self.signal:
                if channel.get("value", False):
                    if np.any([nsig > 0.0 for nsig in channel["value"].get("data", list())]):
                        return True
            return False

    @property
    def minimum_poi_test(self):
        """Find minimum POI test that can be applied to this statistical model"""
        return self._minimum_poi_test
