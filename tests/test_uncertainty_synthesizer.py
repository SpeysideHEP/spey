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

    delta_up_scale = 1.0 + scale_unc / signal_yields
    delta_dn_scale = 1.0 + scale_unc / signal_yields
    delta_up_pdf = 1.0 + pdf_up / signal_yields
    delta_dn_pdf = 1.0 + pdf_dn / signal_yields

    def logprob(param, data):
        return poisson.logpmf(
            data,
            param[0]
            * signal_yields
            * np.exp(
                param[3] * np.log(delta_up_scale if param[3] > 0 else delta_dn_scale)
            )
            * np.exp(param[4] * np.log(delta_up_pdf if param[4] > 0 else delta_dn_pdf))
            + background_yields
            + background_unc * param[1:-2],
        )

    normal = norm(loc=[0, 0, 0.0, 0], scale=[1, 1, 1, 1])

    exact_nll = logprob(nui, data).sum() + normal.logpdf(nui[1:]).sum()

    assert (
        pytest.approx(exact_nll) == model_nll
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

    delta_up_scale = 1.0 + scale_unc / signal_yields
    delta_dn_scale = 1.0 + scale_unc / signal_yields
    delta_up_pdf = 1.0 + pdf_up / signal_yields
    delta_dn_pdf = 1.0 + pdf_dn / signal_yields

    for p in [1.0, 2.0, 3.0]:
        nui = np.array([p, 1.0, 2.0, 3.0, 4.0])

        model_nll = stat_model.backend.get_logpdf_func()(nui)

        def logprob(param, data):
            return poisson.logpmf(
                data,
                param[0]
                * signal_yields
                * np.exp(
                    param[3] * np.log(delta_up_scale if param[3] > 0 else delta_dn_scale)
                )
                * np.exp(
                    param[4] * np.log(delta_up_pdf if param[4] > 0 else delta_dn_pdf)
                )
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

        # NOTE: The difference is due to the handling of covariance matrix in scipy
        # see `spey.backends.distributions.MultivariateNormal.log_prob` for details
        assert (
            pytest.approx(exact_nll, 2e-2) == model_nll
        ), f"Model llhd does not match with analytic at poi={p}."
