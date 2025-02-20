"""Test simplified models at the limit"""

import numpy as np

import spey
from spey.helper_functions import covariance_to_correlation


def test_simplified_llhds_at_limit():
    """Test models at the limit"""

    base_model = spey.get_backend("default.uncorrelated_background")(
        signal_yields=[1, 1, 1],
        background_yields=[2, 1, 3],
        data=[2, 2, 2],
        absolute_uncertainties=[1, 1, 1],
    )
    base_model_cls = base_model.exclusion_confidence_level()[0]

    correlated_model = spey.get_backend("default.correlated_background")(
        signal_yields=[1, 1, 1],
        background_yields=[2, 1, 3],
        data=[2, 2, 2],
        covariance_matrix=np.diag([1, 1, 1]),
    )
    correlated_model_cls = correlated_model.exclusion_confidence_level()[0]

    assert np.isclose(
        correlated_model_cls, base_model_cls
    ), "Correlated model is not same as base model"

    third_moment_model = spey.get_backend("default.third_moment_expansion")(
        signal_yields=[1, 1, 1],
        background_yields=[2, 1, 3],
        data=[2, 2, 2],
        covariance_matrix=np.diag([1, 1, 1]),
        third_moment=[0.0, 0.0, 0.0],
    )
    third_moment_model_cls = third_moment_model.exclusion_confidence_level()[0]

    assert np.isclose(
        third_moment_model_cls, base_model_cls
    ), "third moment model is not same as base model"

    eff_sigma_model = spey.get_backend("default.effective_sigma")(
        signal_yields=[1, 1, 1],
        background_yields=[2, 1, 3],
        data=[2, 2, 2],
        correlation_matrix=covariance_to_correlation(np.diag([1, 1, 1])),
        absolute_uncertainty_envelops=[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
    )
    eff_sigma_model_cls = eff_sigma_model.exclusion_confidence_level()[0]

    assert np.isclose(
        eff_sigma_model_cls, base_model_cls
    ), "effective sigma model is not same as base model"
