import pyhf, copy
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List
import numpy as np
from pyhf import Workspace, Model

from madstats.utils import ExpectationType
from .utils import initialise_workspace
from madstats.system.exceptions import NegativeExpectedYields


@dataclass(frozen=True)
class Data:
    """
    Dataclass for pyhf interface

    :param signal: either histfactory type signal patch or float value of number of events
    :param background: either background only JSON histfactory or float value of observed data
    :param nb: expected number of background events. In case of statistical model it is not needed
    :param delta_nb: uncertainty on backgorund. In case of statistical model it is not needed
    :raises AssertionError: if the statistical model is not valid
    """

    signal: Union[List[Dict], float]
    background: Union[Dict, float]
    nb: Optional[float] = None
    delta_nb: Optional[float] = None
    default_expectation: Optional[ExpectationType] = field(
        default=ExpectationType.observed, repr=False
    )
    _workspace: Optional[pyhf.Workspace] = field(default=None, init=False, repr=False)
    _model: Optional[pyhf.pdf.Model] = field(default=None, init=False, repr=False)
    _data: Optional[List[float]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        workspace, model, data = initialise_workspace(
            self.signal, self.background, self.nb, self.delta_nb, expected=self.default_expectation
        )
        assert model is not None and data is not None, "Invalid statistical model."
        object.__setattr__(self, "_workspace", workspace)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_data", data)

    def __call__(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
    ) -> Union[
        tuple[Optional[Workspace], Optional[Model], Optional[list[float]]],
        tuple[Workspace, Model, Union[np.ndarray, np.ndarray]],
    ]:
        """
        Create pyhf workspace with respect to given POI and expectation

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :return: workspace, statistical model and data
        """
        if mu == 1.0 and expected == self.default_expectation:
            return self._workspace, self._model, self._data

        signal = copy.deepcopy(self.signal)
        if mu != 1.0:
            if isinstance(self.signal, float):
                signal *= mu
            else:
                for ids, channel in enumerate(signal):
                    if signal[ids].get("value", False):
                        signal[ids]["value"]["data"] = [
                            nsig * mu for nsig in signal[ids]["value"]["data"]
                        ]

        workspace, model, data = initialise_workspace(
            signal,
            copy.deepcopy(self.background),
            copy.deepcopy(self.nb),
            copy.deepcopy(self.delta_nb),
            expected=expected,
        )

        # Check if there is any negative number of events
        if mu < 0.0:
            if isinstance(signal, float):
                if signal + self.background < 0.0:
                    raise NegativeExpectedYields(
                        f"Statistical model involves negative expected bin yields. "
                        f"Bin values: {signal + self.background:.3f}"
                    )
            else:
                for channel in model.spec.get("channels", []):
                    current = []
                    for ch in channel["samples"]:
                        if len(current) == 0:
                            current = np.zeros((len(ch["data"]),))
                        current += np.array(ch["data"])
                    if np.any(np.array(current) < 0.0):
                        raise NegativeExpectedYields(
                            f"Statistical model involves negative expected "
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
