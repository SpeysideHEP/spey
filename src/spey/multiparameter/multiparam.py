import numpy as np
import spey

from spey.multiparameter.templates import MultiParamTemplate


def chi2_test(
    likelihood_config: list[MultiParamTemplate],
    confidence_level: float = 0.95,
    warm_start: int = 100,
):

    npar = likelihood_config[0].number_of_parameters
    assert all(
        npar == llhd.number_of_parameters for llhd in likelihood_config
    ), "Invalid likelihood construction"
    chi2 = []
    for _ in range(warm_start):
        current_parameters = np.random.random(npar)
        likelihoods = [llhd(current_parameters) for llhd in likelihood_config]
        best_llhd = likelihoods[
            np.argmin(
                llhd.poi_upper_limit(expected=spey.ExpectationType.apriori)
                for llhd in likelihoods
            )
        ]
        chi2.append({"parameters": current_parameters.tolist(), "chi2": best_llhd.chi2()})

    return chi2
