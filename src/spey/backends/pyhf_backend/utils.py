import pyhf, logging, warnings, copy
import numpy as np

from pyhf import Workspace

from spey.utils import ExpectationType
from spey.system.exceptions import InvalidInput
from typing import Dict, Union, Optional, Tuple, List, Text, Any

pyhf.pdf.log.setLevel(logging.CRITICAL)
pyhf.workspace.log.setLevel(logging.CRITICAL)
pyhf.set_backend("numpy", precision="64b")

__all__ = ["initialise_workspace", "fixed_poi_fit", "compute_min_negloglikelihood"]


def initialise_workspace(
    signal: Union[List[float], List[Dict]],
    background: Union[Dict, List[float]],
    nb: Optional[List[float]] = None,
    delta_nb: Optional[List[float]] = None,
    expected: ExpectationType = ExpectationType.observed,
    return_full_data: bool = False,
) -> Union[
    tuple[
        Union[list, Any],
        Union[Optional[dict], Any],
        Optional[Any],
        Optional[Any],
        Optional[Workspace],
        Any,
        Any,
        Union[Union[int, float, complex], Any],
    ],
    tuple[Optional[Workspace], Any, Any],
]:
    """
    Construct the statistical model with respect to the given inputs.

    :param signal (`Union[List[float], List[Dict]]`): number of signal events or json patch
    :param background (`Union[Dict, List[float]]`): number of observed events or json dictionary
    :param nb (`Optional[List[float]]`, default `None`): number of expected background events (MC).
    :param delta_nb (`Optional[List[float]]`, default `None`): uncertainty on expected background events.
    :param expected (`ExpectationType`, default `ExpectationType.observed`):
                                                                    observed, apriori or aposteriori.
    :param return_full_data (`bool`, default `False`): if true, returns input values as well.
    :raises `InvalidInput`: if input types are not correctly initialised
    :return `Union[ tuple[ Union[list, Any], Union[Optional[dict], Any],
    Optional[Any], Optional[Any], Optional[Workspace], Any, Any,
    Union[Union[int, float, complex], Any], ],
    tuple[Optional[Workspace], Any, Any], ]`: Workspace(can be none in simple case), model, data

    .. code-block:: python3

        >>> workspace, model, data = initialise_workspace(3., 5., 4., 0.5)

    above example returns a simple model with a single region.
    """
    # Check the origin of signal
    signal_from_patch = False
    if isinstance(signal, (float, np.ndarray)):
        signal = np.array(signal, dtype=np.float32).reshape(-1)
    elif isinstance(signal, list):
        if isinstance(signal[0], dict):
            signal_from_patch = True
        else:
            signal = np.array(signal, dtype=np.float32).reshape(-1)
    else:
        raise InvalidInput(f"An unexpected type of signal has been presented: {signal}")

    # check the origin of background
    bkg_from_json = False
    if isinstance(background, dict):
        bkg_from_json = True
    elif isinstance(background, (float, list, np.ndarray)):
        background = np.array(background, dtype=np.float32).reshape(-1)
    else:
        raise InvalidInput(f"An unexpected type of background has been presented: {background}")

    if (bkg_from_json and not signal_from_patch) or (signal_from_patch and not bkg_from_json):
        raise InvalidInput("Signal and background types does not match.")

    if not bkg_from_json:
        # check if bkg uncertainties are valid
        if isinstance(delta_nb, (float, list, np.ndarray)):
            delta_nb = np.array(delta_nb, dtype=np.float32).reshape(-1)
        else:
            raise InvalidInput(
                f"An unexpected type of background uncertainty has been presented: {delta_nb}"
            )
        # check if MC bkg data is valid
        if isinstance(nb, (float, list, np.ndarray)):
            nb = np.array(nb, dtype=np.float32).reshape(-1)
        else:
            raise InvalidInput(f"An unexpected type of background has been presented: {nb}")
        assert (
            len(signal) == len(background) == len(nb) == len(delta_nb)
        ), "Dimensionality of the data does not match."
    else:
        delta_nb, nb = None, None

    workspace, model, data, minimum_poi = None, None, None, -np.inf

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if not bkg_from_json:
            if expected == ExpectationType.apriori:
                # set data as expected background events
                background = nb
            # Create model from uncorrelated region
            model = pyhf.simplemodels.uncorrelated_background(
                signal.tolist(), nb.tolist(), delta_nb.tolist()
            )
            data = background.tolist() + model.config.auxdata

            if return_full_data:
                if len(signal[signal != 0.0]) == 0:
                    minimum_poi = -np.inf
                else:
                    minimum_poi = -np.min(np.true_divide(nb[signal != 0.0], signal[signal != 0.0]))

        else:
            workspace = pyhf.Workspace(background)
            model = workspace.model(
                patches=[signal],
                modifier_settings={
                    "normsys": {"interpcode": "code4"},
                    "histosys": {"interpcode": "code4p"},
                },
            )

            data = workspace.data(model)

            if expected == ExpectationType.apriori:
                init_param = model.config.suggested_init()
                init_param[model.config.poi_index] = 0.0
                data = model.main_model.expected_data(init_param, False).tolist()

            if return_full_data and None not in [model, workspace, data]:
                min_ratio = []
                for idc, channel in enumerate(background.get("channels", [])):
                    current_signal = []
                    for sigch in signal:
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
                minimum_poi = (
                    -np.min(min_ratio).astype(np.float32) if len(min_ratio) > 0 else -np.inf
                )

    if return_full_data:
        return signal, background, nb, delta_nb, workspace, model, data, minimum_poi

    return workspace, model, data


def fixed_poi_fit(
    mu: float,
    data: np.ndarray,
    model: pyhf.pdf,
    iteration_threshold: int,
    options: Optional[Dict] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute negative log-likelihood for given statistical model at a certain POI

    :param mu: POI (signal strength)
    :param data: dataset retreived from `pyhf.Workspace.data(model)`
    :param model: statistical model
    :param iteration_threshold: number of iterations to be held for convergence of the fit.
                                this should not need to be larger than 3.
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
    :return: (float) negative log-likelihood

    .. code-block:: python3

        workspace, model, data = initialise_workspace(3., 5., 4., 0.5)
        nll, theta = compute_negloglikelihood(1., data, model, True, 3)
    """
    if options is None:
        options = {}

    _options = dict(
        maxiter=options.get("maxiter", 200),
        verbose=options.get("verbose", False),
        method=options.get("method", "SLSQP"),
        tolerance=options.get("tolerance", 1e-6),
        solver_options=options.get("solver_options", {}),
    )

    def compute_nll(
        current_model: pyhf.pdf.Model,
        current_data: np.ndarray,
        new_bounds: List[Tuple[float, float]],
    ) -> Union[Tuple[float, np.ndarray], Tuple[Text, None]]:
        try:
            pars, twice_nllh = pyhf.infer.mle.fixed_poi_fit(
                mu,
                current_data,
                current_model,
                return_fitted_val=True,
                par_bounds=new_bounds,
                **_options,
            )
        except (AssertionError, pyhf.exceptions.FailedMinimization, ValueError) as err:
            warnings.warn(err.args[0], RuntimeWarning)
            return "update bounds", None

        return twice_nllh, pars

    def update_bounds(bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        current_bounds = []
        for idx, bound in enumerate(bounds):
            if idx != model.config.poi_index:
                min_bound = bound[0] * 2.0 if bound[0] < 0.0 else 0.0
                max_bound = bound[1] * 2.0
                current_bounds.append((min_bound, max_bound))
            else:
                current_bounds.append(bound)
        return current_bounds

    bounds = model.config.suggested_bounds()
    if bounds[model.config.poi_index][0] > mu:
        bounds[model.config.poi_index] = (mu, bounds[model.config.poi_index][1])
    if bounds[model.config.poi_index][1] < mu:
        bounds[model.config.poi_index] = (bounds[model.config.poi_index][0], mu)
    it = 0
    while True:
        twice_nllh, pars = compute_nll(model, data, bounds)
        if twice_nllh == "update bounds":
            bounds = update_bounds(bounds)
            it += 1
        else:
            break
        if it >= iteration_threshold:
            warnings.warn(message="pyhf mle.fit failed", category=RuntimeWarning)
            return np.nan, pars

    return twice_nllh / 2.0, pars


def compute_min_negloglikelihood(
    data: np.ndarray,
    model: pyhf.pdf,
    allow_negative_signal: bool,
    iteration_threshold: int,
    minimum_poi_test: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute minimum negative log-likelihood (i.e. maximum likelihood) for given statistical model

    :param data: dataset retreived from `pyhf.Workspace.data(model)`
    :param model: statistical model
    :param allow_negative_signal: if true, POI can get negative values
    :param iteration_threshold: number of iterations to be held for convergence of the fit.
    :param minimum_poi_test: declare minimum safe POI test
    :return: muhat, negative log-likelihood

    .. code-block:: python3

        workspace, model, data = initialise_workspace(3., 5., 4., 0.5)
        muhat, nll = compute_min_negloglikelihood(1., data, model, 3)
    """

    def compute_nll(model, data, bounds):
        try:
            muhat, twice_nllh = pyhf.infer.mle.fit(
                data,
                model,
                return_fitted_val=True,
                maxiter=200,
                par_bounds=bounds,
            )
        except (AssertionError, pyhf.exceptions.FailedMinimization, ValueError) as err:
            warnings.warn(err.args[0], RuntimeWarning)
            return None, "update bounds"

        return muhat, twice_nllh

    def update_bounds(current_bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Update given bounds with respect to min POI criteria"""
        min_bound = (
            current_bounds[model.config.poi_index][0] - 5.0 if allow_negative_signal else 0.0
        )
        if allow_negative_signal and minimum_poi_test is not None:
            min_bound = (
                minimum_poi_test
                if not np.isinf(minimum_poi_test)
                else current_bounds[model.config.poi_index][0] - 5.0
            )
        current_bounds[model.config.poi_index] = (
            min_bound,
            2.0 * current_bounds[model.config.poi_index][1],
        )
        return current_bounds

    bounds = update_bounds(copy.deepcopy(model.config.suggested_bounds()))
    it = 0
    while True:
        muhat, twice_nllh = compute_nll(model, data, bounds)
        if twice_nllh == "update bounds":
            print("updating bounds")
            bounds = update_bounds(bounds)
            it += 1
        else:
            break
        if it >= iteration_threshold and isinstance(twice_nllh, str):
            warnings.warn("pyhf mle.fit failed", RuntimeWarning)
            return np.nan, np.nan

    return muhat, twice_nllh / 2.0
