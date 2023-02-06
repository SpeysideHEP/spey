import copy, logging, scipy, warnings, pyhf
from typing import Dict, Union, Optional, Tuple, List, Text
import numpy as np

from pyhf.infer.calculators import generate_asimov_data

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase, DataBase
from .utils import fixed_poi_fit, compute_min_negloglikelihood
from .pyhfdata import PyhfData
from spey.backends import AvailableBackends
from spey.system.exceptions import NegativeExpectedYields
from spey.hypothesis_testing.utils import find_root_limits
from spey.base.recorder import Recorder
from spey.interface.statistical_model import statistical_model_wrapper

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
        assert isinstance(model, PyhfData) and isinstance(model, DataBase), "Invalid statistical model."
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

    def _get_asimov_data(
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
        asimov_nuisance_key = (
            str(ExpectationType.apriori)
            if expected == ExpectationType.apriori
            else str(ExpectationType.observed)
        )
        asimov_data = self._asimov_nuisance.get(asimov_nuisance_key, False)
        if asimov_data is False:
            asimov_data = generate_asimov_data(
                0.0 if test_statistics in ["q", "qmu", "qtilde"] else 1.0,
                data,
                model,
                model.config.suggested_init(),
                model.config.suggested_bounds(),
                model.config.suggested_fixed(),
                return_fitted_pars=False,
            )
            self._asimov_nuisance[asimov_nuisance_key] = copy.deepcopy(asimov_data)
        return asimov_data

    def logpdf(
        self,
        poi_test: float,
        nuisance_parameters: np.ndarray,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        isAsimov: Optional[bool] = False,
    ) -> float:
        """
        Compute the log value of the full density.

        :param poi_test: parameter of interest
        :param nuisance_parameters: nuisance parameters
        :param expected: observed, apriori or aposteriori
        :param isAsimov:
        :return: negative log-likelihood
        """
        assert len(nuisance_parameters) == self.model.npar, (
            f"Parameters must be {self.model.npar} dimensional vector, "
            f"{len(nuisance_parameters)} dimensions has been given."
        )
        _, model, data = self.model(poi_test=1.0, expected=expected)
        poi_bounds = self.model.suggested_bounds[self.model.poi_index]

        if isAsimov:
            data = self._get_asimov_data(model, data, expected)

        if not poi_bounds[0] <= poi_test <= poi_bounds[1]:
            _, model, _ = self.model(poi_test=poi_test, expected=expected)
            poi_test = 1.0

        complete_pars = np.array(
            nuisance_parameters.tolist()[: model.config.poi_index]
            + [poi_test]
            + nuisance_parameters.tolist()[model.config.poi_index :]
        )

        return model.logpdf(complete_pars, data).astype(np.float32)[0]

    def sigma_mu(
        self, pars: np.ndarray, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """Currently not implemented"""
        return 1.0

    def likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        isAsimov: Optional[bool] = False,
        return_theta: Optional[bool] = False,
        iteration_threshold: Optional[int] = 3,
        options: Optional[Dict] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute  likelihood of the given statistical model

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param return_theta: return fitted parameters
        :param isAsimov: if true, computes likelihood for Asimov data
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :param options: optimizer options where the default values are
                :param maxiter: maximum iterations (default 200)
                :param verbose: verbosity (default False)
                :param tolerance: Tolerance for termination. See specific optimizer
                                  for detailed meaning. (default None)
                :param solver_options: (dict) additional solver options. See
                                :func:`scipy.optimize.show_options` for additional options of
                                optimization solvers. (default {})
                :param method: optimisation method (default SLSQP)
                        Available methods are:
                        - 'Nelder-Mead' :ref:`(see here) <scipy.optimize.minimize-neldermead>`
                        - 'Powell'      :ref:`(see here) <scipy.optimize.minimize-powell>`
                        - 'CG'          :ref:`(see here) <scipy.optimize.minimize-cg>`
                        - 'BFGS'        :ref:`(see here) <scipy.optimize.minimize-bfgs>`
                        - 'Newton-CG'   :ref:`(see here) <scipy.optimize.minimize-newtoncg>`
                        - 'L-BFGS-B'    :ref:`(see here) <scipy.optimize.minimize-lbfgsb>`
                        - 'TNC'         :ref:`(see here) <scipy.optimize.minimize-tnc>`
                        - 'COBYLA'      :ref:`(see here) <scipy.optimize.minimize-cobyla>`
                        - 'SLSQP'       :ref:`(see here) <scipy.optimize.minimize-slsqp>`
                        - 'trust-constr':ref:`(see here) <scipy.optimize.minimize-trustconstr>`
                        - 'dogleg'      :ref:`(see here) <scipy.optimize.minimize-dogleg>`
                        - 'trust-ncg'   :ref:`(see here) <scipy.optimize.minimize-trustncg>`
                        - 'trust-exact' :ref:`(see here) <scipy.optimize.minimize-trustexact>`
                        - 'trust-krylov' :ref:`(see here) <scipy.optimize.minimize-trustkrylov>`
        :return: (float) likelihood
        """
        _, model, data = self.model(poi_test=1.0, expected=expected)

        if isAsimov:
            data = self._get_asimov_data(model, data, expected)

        # CHECK THE MODEL BOUNDS!!
        # POI Test needs to be adjusted according to the boundaries for sake of convergence
        # see issue https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579235311
        # comment https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579299831
        new_poi_test = copy.deepcopy(poi_test)
        execute = True
        bounds = model.config.suggested_bounds()[model.config.poi_index]
        if not bounds[0] <= new_poi_test <= bounds[1]:
            try:
                _, model, _ = self.model(poi_test=new_poi_test, expected=expected)
                new_poi_test = 1.0
            except NegativeExpectedYields as err:
                warnings.warn(
                    err.args[0] + f"\nSetting NLL({poi_test:.3f}) = inf.",
                    category=RuntimeWarning,
                )
                execute = False

        negloglikelihood, fit_param = np.nan, np.nan
        if execute:
            negloglikelihood, fit_param = fixed_poi_fit(
                new_poi_test, data, model, iteration_threshold, options
            )

        return negloglikelihood, fit_param

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = False,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        iteration_threshold: Optional[int] = 3,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param return_nll: if true, likelihood will be returned
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: allow negative POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: muhat, maximum of the likelihood, sigma mu
        """
        _, model, data = self.model(poi_test=1.0, expected=expected)
        if isAsimov:
            data = self._get_asimov_data(model, data, expected)

        fit_param, negloglikelihood = compute_min_negloglikelihood(
            data, model, allow_negative_signal, iteration_threshold, self.model.minimum_poi_test
        )

        return negloglikelihood, fit_param, self.sigma_mu(fit_param, expected)

    def exclusion_confidence_level(
        self,
        poi_test: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        iteration_threshold: int = 3,
    ) -> List[float]:
        """
        Compute exclusion confidence level.

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param iteration_threshold: sets threshold on when to stop
        :return: 1 - CLs values
        """
        if not self.model.isAlive:
            return [-1] if expected == ExpectationType.observed else [-1] * 5

        def get_CLs(poi: float, model: pyhf.pdf.Model, data: np.ndarray, **keywordargs):
            try:
                CLs_obs, CLs_exp = pyhf.infer.hypotest(
                    poi,
                    data,
                    model,
                    test_stat="q" if allow_negative_signal else "qtilde",
                    par_bounds=keywordargs.get("bounds", model.config.suggested_bounds()),
                    return_expected_set=True,
                )

            except (AssertionError, pyhf.exceptions.FailedMinimization, ValueError) as err:
                logging.getLogger("MA5").debug(str(err))
                # dont use false here 1.-CLs = 0 can be interpreted as false
                return "update bounds"

            # if isnan(float(CLs_obs)) or any([isnan(float(x)) for x in CLs_exp]):
            #     return "update mu"
            CLs_obs = float(CLs_obs[0]) if isinstance(CLs_obs, (list, tuple)) else float(CLs_obs)

            return {
                "CLs_obs": [1.0 - CLs_obs],
                "CLs_exp": list(map(lambda x: float(1.0 - x), CLs_exp)),
            }

        _, model, data = self.model(poi_test=1.0, expected=expected)

        def update_bounds(bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            current_bounds = []
            for idx, bound in enumerate(bounds):
                if idx != model.config.poi_index:
                    min_bound = bound[0] * 2.0 if bound[0] < 0.0 else 0.0
                    max_bound = bound[1] * 2.0
                    current_bounds.append((min_bound, max_bound))
                else:
                    min_bound = self.model.minimum_poi_test if allow_negative_signal else 0.0
                    current_bounds.append((min_bound, 2.0 * bound[1]))
            return current_bounds

        # pyhf can raise an error if the poi_test bounds are too stringent
        # they need to be updated dynamically.
        arguments = dict(bounds=update_bounds(model.config.suggested_bounds()))
        poi_test_bounds = model.config.suggested_bounds()[model.config.poi_index]
        poi_update = False
        if not poi_test_bounds[0] <= poi_test <= poi_test_bounds[1]:
            _, model, data = self.model(poi_test=poi_test, expected=expected)
            poi_update = True
        it = 0
        while True:
            CLs = get_CLs(1.0 if poi_update else poi_test, model, data, **arguments)
            if CLs == "update bounds" or np.isnan(CLs["CLs_obs"][0]):
                arguments["bounds"] = update_bounds(arguments["bounds"])
                it += 1
            else:
                break
            # hard limit on iteration required if it exceeds this value it means
            # Nsig >>>>> Nobs
            if it >= iteration_threshold:
                return [1.0] if expected == ExpectationType.observed else [1.0] * 5

        return CLs["CLs_obs" if expected == ExpectationType.observed else "CLs_exp"]

    def poi_upper_limit(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        confidence_level: float = 0.95,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, apriori or aposteriori
        :param confidence_level: confidence level (default 95%)
        :param allow_negative_signal: if true, allow negative mu
        :return: mu
        """
        assert 0.0 <= confidence_level <= 1.0, "Confidence level must be between zero and one."

        def computer(poi_test: float) -> float:
            CLs = self.exclusion_confidence_level(
                expected=expected, poi_test=poi_test, allow_negative_signal=allow_negative_signal
            )
            return CLs[0 if expected == ExpectationType.observed else 2] - confidence_level

        muhat, nllmin = self.maximize_likelihood(
            expected=expected, allow_negative_signal=allow_negative_signal
        )

        low, hig = find_root_limits(
            computer,
            loc=0.0,
            low_ini=muhat + 1.5 if muhat >= 0.0 else 1.0,
            hig_ini=muhat + 2.5 if muhat >= 0.0 else 1.0,
        )

        return scipy.optimize.brentq(computer, low, hig, xtol=low / 100.0)
