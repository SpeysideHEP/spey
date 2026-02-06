import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm, poisson

import spey
from spey.helper_functions import covariance_to_correlation


def test_uncorrelated_background():
    pdf_wrapper = spey.get_backend("default.uncorrelated_background")

    data = np.array([36, 33])
    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    background_unc = np.array([12.0, 16.0])
    scale_unc = np.array([2.0, 3.0])

    pdf_up = np.array([1.0, 5.0])
    pdf_dn = np.array([5.0, 3.0])
    pdf_unc = np.vstack([pdf_up, pdf_dn]).T

    stat_model = pdf_wrapper(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        absolute_uncertainties=background_unc,
        modifiers=[scale_unc, pdf_unc],
    )

    poi = 1.0

    nui = np.array([poi, 1.0, 2.0, 3.0, 4.0])

    model_nll = stat_model.backend.get_logpdf_func()(nui)

    def logprob(param, data):
        return poisson.logpmf(
            data,
            param[0] * signal_yields
            + ((1 + param[3] * scale_unc) * (1 + param[4] * pdf_unc[:, 0]) - 1) * param[0]
            + background_yields
            + background_unc * param[1:-2],
        )

    normal = norm(loc=[0, 0, 0.0, 0], scale=[1, 1, 1, 1])

    exact_nll = logprob(nui, data).sum() + normal.logpdf(nui[1:]).sum()

    assert (
        pytest.approx(model_nll) == exact_nll
    ), "Model llhd does not match with analytic."


def test_correlated_background():
    pdf_wrapper = spey.get_backend("default.correlated_background")

    data = np.array([36, 33])
    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])
    scale_unc = np.array([2.0, 3.0])

    pdf_up = np.array([1.0, 5.0])
    pdf_dn = np.array([5.0, 3.0])
    pdf_unc = np.vstack([pdf_up, pdf_dn]).T

    stat_model = pdf_wrapper(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        covariance_matrix=covariance_matrix,
        modifiers=[scale_unc, pdf_unc],
    )

    for p in [1.0, 2.0, 3.0]:
        nui = np.array([p, 1.0, 2.0, 3.0, 4.0])

        model_nll = stat_model.backend.get_logpdf_func()(nui)

        def logprob(param, data):
            return poisson.logpmf(
                data,
                param[0] * signal_yields
                + ((1 + param[3] * scale_unc) * (1 + param[4] * pdf_unc[:, 0]) - 1)
                * param[0]
                + background_yields
                + np.sqrt(np.diag(covariance_matrix)) * param[1:-2],
            )

        multinorm = multivariate_normal(
            mean=[0.0, 0.0], cov=covariance_to_correlation(covariance_matrix)
        )
        normal = norm(loc=[0.0, 0.0], scale=[1.0, 1.0])

        exact_nll = (
            logprob(nui, data).sum()
            + multinorm.logpdf(nui[1:-2]).sum()
            + normal.logpdf(nui[3:]).sum()
        )

        assert (
            pytest.approx(model_nll, 2e-2) == exact_nll
        ), f"Model llhd does not match with analytic at poi={p}."
