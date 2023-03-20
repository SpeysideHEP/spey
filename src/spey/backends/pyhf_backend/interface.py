"""Pyhf interface"""

from typing import Optional, Tuple, List, Text
import copy, logging, pyhf
import numpy as np

from pyhf.infer.calculators import generate_asimov_data

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase, DataBase
from spey.backends import AvailableBackends
from spey.base.recorder import Recorder
from spey.interface.statistical_model import statistical_model_wrapper
from spey.optimizer import fit
from .utils import twice_nll_func
from .pyhfdata import PyhfData

__all__ = ["PyhfInterface"]

pyhf.pdf.log.setLevel(logging.CRITICAL)
pyhf.workspace.log.setLevel(logging.CRITICAL)
pyhf.set_backend("numpy", precision="64b")


@statistical_model_wrapper
class PyhfInterface(BackendBase):
    """
    Pyhf Interface. This is object has been wrapped with `StatisticalModel` class to ensure
    universality across all platforms.

    :param model: contains all the information regarding the regions, yields
    :param analysis: a unique name for the analysis (default `"__unknown_analysis__"`)
    :param xsection: cross section value for the signal, only used to compute excluded cross section
                     value. Default `NaN`
    :raises AssertionError: if the input type is wrong.

    .. code-block:: python3

        >>> from spey.backends.pyhf_backend.data import SLData
        >>> from spey.backends.pyhf_backend.interface import PyhfInterface
        >>> from spey import ExpectationType
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
        >>> model = SLData(signal=signal, background=background)
        >>> statistical_model = PyhfInterface(model=model, xsection=1.0, analysis="my_analysis")
        >>> print(statistical_model)
        >>> # StatisticalModel(analysis='my_analysis', xsection=1.000e+00 [pb], backend=pyhf)
        >>> statistical_model.exclusion_confidence_level()
        >>> # [0.9474850257628679] # 1-CLs
        >>> statistical_model.s95exp
        >>> # 1.0685773410460155 # prefit excluded cross section in pb
        >>> statistical_model.maximize_likelihood()
        >>> # (-0.0669277855002002, 12.483595567080783) # muhat and maximum negative log-likelihood
        >>> statistical_model.likelihood(poi_test=1.5)
        >>> # 16.59756909879556
        >>> statistical_model.exclusion_confidence_level(expected=ExpectationType.aposteriori)
        >>> # [0.9973937390501324, 0.9861799464393675, 0.9355467946443513, 0.7647435613928496, 0.4269637940897122]
    """

    __slots__ = ["_model", "_recorder", "_asimov_nuisance"]

    def __init__(self, model: PyhfData):
        assert isinstance(model, PyhfData) and isinstance(
            model, DataBase
        ), "Invalid statistical model."
        self._model = model
        self._recorder = Recorder()
        self._asimov_nuisance = {
            str(ExpectationType.observed): False,
            str(ExpectationType.apriori): False,
        }

    @property
    def model(self) -> PyhfData:
        """Retrieve statistical model"""
        return self._model

    @property
    def type(self) -> AvailableBackends:
        return AvailableBackends.pyhf

    def generate_asimov_data(
        self,
        model: pyhf.pdf.Model,
        data: np.ndarray,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        test_statistics: Text = "qtilde",
    ) -> np.ndarray:
        """
        Generate asimov data for the statistical model (only valid for teststat = qtilde)

        :param model: statistical model
        :param data: data
        :param expected: observed, apriori or aposteriori
        :return: asimov data
        """
        # asimov_nuisance_key = (
        #     str(ExpectationType.apriori)
        #     if expected == ExpectationType.apriori
        #     else str(ExpectationType.observed)
        # )
        asimov_data = False #self._asimov_nuisance.get(asimov_nuisance_key, False)
        if asimov_data is False:
            asimov_data = generate_asimov_data(
                1.0 if test_statistics == "q0" else 0.0,
                data,
                model,
                model.config.suggested_init(),
                model.config.suggested_bounds(),
                model.config.suggested_fixed(),
                return_fitted_pars=False,
            )
            # self._asimov_nuisance[asimov_nuisance_key] = copy.deepcopy(asimov_data)
        return asimov_data

    def negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute negative log-likelihood of the statistical model

        :param poi_test (`float`, default `1.0`): parameter of interest.
        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type.
            observed, apriori or aposteriori
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: optimizer parameters see `spey.optimizer.scipy_tools.minimize`
        :return `Tuple[float, np.ndarray]`: negative log-likelihood and fit parameters
        """
        # CHECK THE MODEL BOUNDS!!
        # POI Test needs to be adjusted according to the boundaries for sake of convergence
        # see issue https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579235311
        # comment https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579299831
        # NOTE During tests we observed that shifting poi with respect to bounds is not needed.
        _, model, data = self.model(expected=expected)

        twice_nll, fit_pars = fit(
            func=twice_nll_func(model, data),
            model_configuration=self.model.config(),
            initial_parameters=init_pars,
            bounds=par_bounds,
            fixed_poi_value=poi_test,
            **kwargs,
        )

        return twice_nll / 2.0, fit_pars

    def asimov_negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute negative log-likelihood of the statistical model for Asimov data

        :param poi_test (`float`, default `1.0`): parameter of interest.
        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type.
            observed, apriori or aposteriori
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: optimizer parameters see `spey.optimizer.scipy_tools.minimize`
        :return `Tuple[float, np.ndarray]`: negative log-likelihood and fit parameters
        """
        # Asimov llhd is only computed for poi = 1 or 0, control is not necessary
        _, model, data = self.model(expected=expected)
        data = self.generate_asimov_data(
            model=model, data=data, expected=expected, test_statistics=test_statistics
        )

        twice_nll, fit_pars = fit(
            func=twice_nll_func(model, data),
            model_configuration=self.model.config(),
            initial_parameters=init_pars,
            bounds=par_bounds,
            fixed_poi_value=poi_test,
            **kwargs,
        )

        return twice_nll / 2.0, fit_pars

    def minimize_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Compute minimum of negative log-likelihood for a given statistical model

        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type.
            observed, apriori or aposteriori.
        :param allow_negative_signal (`bool`, default `True`): if True $\hat\mu$ values will allowed to be negative.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: optimizer parameters see `spey.optimizer.scipy_tools.minimize`
        :return `Tuple[float, np.ndarray]`: minimum of negative log-likelihood and fit parameters
        """
        _, model, data = self.model(expected=expected)

        twice_nll, fit_pars = fit(
            func=twice_nll_func(model, data),
            model_configuration=self.model.config(allow_negative_signal=allow_negative_signal),
            initial_parameters=init_pars,
            bounds=par_bounds,
            **kwargs,
        )

        return twice_nll / 2.0, fit_pars

    def minimize_asimov_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Compute minimum of negative log-likelihood for a given statistical model for the Asimov data

        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type.
            observed, apriori or aposteriori.
        :param allow_negative_signal (`bool`, default `True`): if True $\hat\mu$ values will allowed to be negative.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: optimizer parameters see `spey.optimizer.scipy_tools.minimize`
        :return `Tuple[float, np.ndarray]`: minimum of negative log-likelihood and fit parameters
        """
        _, model, data = self.model(poi_test=1.0, expected=expected)
        data = self.generate_asimov_data(
            model, data=data, expected=expected, test_statistics=test_statistics
        )

        twice_nll, fit_pars = fit(
            func=twice_nll_func(model, data),
            model_configuration=self.model.config(
                allow_negative_signal=test_statistics in ["q", "qmu", "q0"]
            ),
            initial_parameters=init_pars,
            bounds=par_bounds,
            **kwargs,
        )

        return twice_nll / 2.0, fit_pars
