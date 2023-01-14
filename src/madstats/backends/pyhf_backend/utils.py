import pyhf
import logging
import numpy as np

from numpy import warnings, isnan

from madstats.utils import ExpectationType

from typing import Sequence, Dict, Union, Optional, Tuple, List

pyhf.pdf.log.setLevel(logging.CRITICAL)
pyhf.workspace.log.setLevel(logging.CRITICAL)
pyhf.set_backend("numpy", precision="64b")


def initialise_workspace(
    signal: Union[Sequence, float],
    background: Union[Dict, float],
    nb: Optional[float] = None,
    delta_nb: Optional[float] = None,
    expected: Optional[ExpectationType] = ExpectationType.observed,
) -> Tuple[pyhf.Workspace, pyhf.pdf.Model, np.ndarray]:
    """
    Construct the statistical model with respect to the given inputs.

    :param signal: number of signal events or json patch
    :param background: number of observed events or json dictionary
    :param nb: number of expected background events (MC)
    :param delta_nb: uncertainty on expected background events
    :param expected: if true prepare apriori expected workspace, default False
    :return: Workspace(can be none in simple case), model, data

    .. code-block:: python3

        workspace, model, data = initialise_workspace(3., 5., 4., 0.5)

    above example returns a simple model with a single region.
    """

    workspace, model, data = None, None, None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if isinstance(signal, float) and isinstance(background, (float, int)):
                if expected == ExpectationType.apriori:
                    # set data as expected background events
                    background = nb
                # Create model from uncorrelated region
                model = pyhf.simplemodels.uncorrelated_background(
                    [max(signal, 0.0)], [nb], [delta_nb]
                )
                data = [background] + model.config.auxdata

            elif isinstance(signal, list) and isinstance(background, dict):
                if expected == ExpectationType.apriori:
                    # set data as expected background events
                    obs = []
                    for channel in background.get("channels", []):
                        current = []
                        for ch in channel["samples"]:
                            if len(current) == 0:
                                current = [0.0] * len(ch["data"])
                            current = [cur + dt for cur, dt in zip(current, ch["data"])]
                        obs.append({"name": channel["name"], "data": current})
                    background["observations"] = obs

                workspace = pyhf.Workspace(background)
                model = workspace.model(
                    patches=[signal],
                    modifier_settings={
                        "normsys": {"interpcode": "code4"},
                        "histosys": {"interpcode": "code4p"},
                    },
                )

                data = workspace.data(model)
    except (pyhf.exceptions.InvalidSpecification, KeyError) as err:
        logging.getLogger("MA5").error("Invalid JSON file!! " + str(err))
    except Exception as err:
        logging.getLogger("MA5").debug("Unknown error, check PyhfInterface " + str(err))

    return workspace, model, data


def compute_negloglikelihood(
    mu: float,
    data: np.ndarray,
    model: pyhf.pdf,
    allow_negative_signal: bool,
    iteration_threshold: int,
) -> float:
    """
    Compute negative log-likelihood for given statistical model at a certain POI

    :param mu: POI (signal strength)
    :param data: dataset retreived from `pyhf.Workspace.data(model)`
    :param model: statistical model
    :param allow_negative_signal: if true, POI can get negative values
    :param iteration_threshold: number of iterations to be held for convergence of the fit.
                                this should not need to be larger than 3.
    :return: (float) negative log-likelihood

    .. code-block:: python3

        workspace, model, data = initialise_workspace(3., 5., 4., 0.5)
        nll = compute_negloglikelihood(1., data, model, True, 3)
    """

    def compute_nll(model, data, bounds):
        try:
            _, twice_nllh = pyhf.infer.mle.fixed_poi_fit(
                mu,
                data,
                model,
                return_fitted_val=True,
                maxiter=200,
                par_bounds=bounds,
            )
        except (AssertionError, pyhf.exceptions.FailedMinimization, ValueError) as err:
            logging.getLogger("MA5").debug(str(err))
            return "update bounds"

        return twice_nllh

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Values in x were outside bounds during a minimize step, clipping to bounds",
        )

        bounds = model.config.suggested_bounds()
        it = 0
        while True:
            twice_nllh = compute_nll(model, data, bounds)
            if twice_nllh == "update bounds":
                min_bound = (
                    bounds[model.config.poi_index][0] - 5.0 if allow_negative_signal else 0.0
                )
                bounds[model.config.poi_index] = (
                    min_bound,
                    2.0 * bounds[model.config.poi_index][1],
                )
                it += 1
            else:
                break
            if it >= iteration_threshold:
                logging.getLogger("MA5").debug("pyhf mle.fit failed")
                return float("nan")

        return twice_nllh / 2.0


def compute_min_negloglikelihood(
    data: np.ndarray,
    model: pyhf.pdf,
    allow_negative_signal: bool,
    iteration_threshold: int,
) -> Tuple[float, float]:
    """
    Compute minimum negative log-likelihood (i.e. maximum likelihood) for given statistical model

    :param data: dataset retreived from `pyhf.Workspace.data(model)`
    :param model: statistical model
    :param allow_negative_signal: if true, POI can get negative values
    :param iteration_threshold: number of iterations to be held for convergence of the fit.
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
            logging.getLogger("MA5").debug(str(err))
            return "update bounds", None

        return muhat[model.config.poi_index], twice_nllh

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Values in x were outside bounds during a minimize step, clipping to bounds",
        )

        bounds = model.config.suggested_bounds()
        it = 0
        while True:
            muhat, twice_nllh = compute_nll(model, data, bounds)
            if twice_nllh == "update bounds":
                min_bound = (
                    bounds[model.config.poi_index][0] - 5.0 if allow_negative_signal else 0.0
                )
                bounds[model.config.poi_index] = (
                    min_bound,
                    2.0 * bounds[model.config.poi_index][1],
                )
                it += 1
            else:
                break
            if it >= iteration_threshold:
                logging.getLogger("MA5").debug("pyhf mle.fit failed")
                return float("nan"), float("nan")

        return muhat, twice_nllh / 2.0
