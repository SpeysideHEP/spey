"""Test default pdf plugin"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm, poisson, chi2

import spey


def test_uncorrelated_background():
    """tester for uncorrelated background model"""

    pdf_wrapper = spey.get_backend("default.uncorrelated_background")

    data = [36, 33]
    signal_yields = [12.0, 15.0]
    background_yields = [50.0, 48.0]
    background_unc = [12.0, 16.0]

    stat_model = pdf_wrapper(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        absolute_uncertainties=background_unc,
        analysis="multi_bin",
        xsection=0.123,
    )

    assert np.isclose(stat_model.poi_upper_limit(), 0.8563345655114185), "POI is wrong."
    assert np.isclose(
        stat_model.exclusion_confidence_level()[0], 0.9701795436411219
    ), "CLs is wrong"


def test_correlated_background():
    """tester for correlated background"""

    pdf_wrapper = spey.get_backend("default.correlated_background")
    statistical_model = pdf_wrapper(
        signal_yields=[12.0, 15.0],
        background_yields=[50.0, 48.0],
        data=[36, 33],
        covariance_matrix=[[144.0, 13.0], [25.0, 256.0]],
        analysis="example",
        xsection=0.123,
    )

    assert np.isclose(statistical_model.poi_upper_limit(), 0.907104376899138), "POI wrong"
    assert np.isclose(
        statistical_model.exclusion_confidence_level()[0], 0.9635100547173434
    ), "CLs is wrong"
    assert np.isclose(
        statistical_model.sigma_mu(1.0), 0.8456469932844632
    ), "Sigma mu is wrong"


def test_third_moment():
    """tester for the third moment"""

    pdf_wrapper = spey.get_backend("default.third_moment_expansion")
    statistical_model = pdf_wrapper(
        signal_yields=[12.0, 15.0],
        background_yields=[50.0, 48.0],
        data=[36, 33],
        covariance_matrix=[[144.0, 13.0], [25.0, 256.0]],
        third_moment=[0.5, 0.8],
        analysis="example",
        xsection=0.123,
    )

    assert np.isclose(
        statistical_model.exclusion_confidence_level()[0], 0.9614329616396733
    ), "CLs is wrong"
    assert np.isclose(
        statistical_model.poi_upper_limit(), 0.9221339770245336
    ), "POI is wrong"
    assert np.isclose(
        statistical_model.sigma_mu(1.0), 0.854551194250324
    ), "Sigma mu is wrong"


def test_effective_sigma():
    """tester for the effective sigma"""

    pdf_wrapper = spey.get_backend("default.effective_sigma")
    statistical_model = pdf_wrapper(
        signal_yields=[12.0, 15.0],
        background_yields=[50.0, 48.0],
        data=[36, 33],
        correlation_matrix=[[1.0, 0.06770833], [0.13020833, 1.0]],
        absolute_uncertainty_envelops=[(10.0, 15.0), (13.0, 18.0)],
        analysis="example",
        xsection=0.123,
    )

    assert np.isclose(
        statistical_model.exclusion_confidence_level()[0], 0.8567802529243093
    ), "CLs is wrong"
    assert np.isclose(
        statistical_model.poi_upper_limit(), 1.5298573610113775
    ), "POI is wrong."
    assert np.isclose(
        statistical_model.sigma_mu(1.0), 1.2152765953701747
    ), "Sigma mu is wrong"


def test_poisson():
    """tester for uncorrelated background model"""

    pdf_wrapper = spey.get_backend("default.poisson")

    data = 55
    signal_yields = 12.0
    background_yields = 50.0

    stat_model = pdf_wrapper(
        signal_yields=[signal_yields], background_yields=[background_yields], data=[data]
    )

    def logprob(mu, data):
        return poisson.logpmf(data, mu * signal_yields + background_yields)

    opt = minimize_scalar(lambda x: -logprob(x, data), bounds=(-0.5, 1))
    muhat, maxnll = stat_model.maximize_likelihood()

    assert np.isclose(
        muhat, opt.x, rtol=1e-3
    ), f"Poisson:: Muhat is wrong {muhat} != {opt.x}"
    assert np.isclose(maxnll, opt.fun), "Poisson:: MLE is wrong"

    ## Test Exclusion limit
    tmu = 2 * (-logprob(1, data) - opt.fun)

    optA = minimize_scalar(lambda x: -logprob(x, background_yields), bounds=(-0.5, 0.5))
    tmuA = 2 * (-logprob(1, background_yields) - optA.fun)

    sqrt_qmuA = np.sqrt(tmuA)
    sqrt_qmu = np.sqrt(tmu)
    delta_teststat = sqrt_qmu - sqrt_qmuA

    CLsb = norm.cdf(-sqrt_qmuA - delta_teststat)
    CLb = norm.cdf(-delta_teststat)
    CLs = CLsb / CLb
    st_cls = stat_model.exclusion_confidence_level()[0]
    assert np.isclose(
        1 - CLs, st_cls
    ), f"Poisson:: Exclusion limit is wrong {1 - CLs} != {st_cls}"

    # compare with analytic results
    s_b = signal_yields + background_yields
    sqrt_qmu = np.sqrt(-2 * (-s_b + data * np.log(s_b) - data * np.log(data) + data))
    sqrt_qmuA = np.sqrt(
        -2
        * (
            -signal_yields
            + background_yields * np.log(1 + signal_yields / background_yields)
        )
    )
    delta_teststat = sqrt_qmu - sqrt_qmuA

    logp_sb = norm.logcdf(-sqrt_qmu)
    logp_b = norm.logcdf(-sqrt_qmu + sqrt_qmuA)
    CLs_analytic = 1 - np.exp(logp_sb - logp_b)
    assert np.isclose(
        CLs_analytic, st_cls
    ), f"Poisson:: Analytic exclusion limit is wrong {CLs_analytic} != {st_cls}"


def test_normal():
    """tester for gaussian model"""

    sig, bkg, obs, unc = 12.0, 50.0, 36.0, 20.0

    dist = norm(loc=obs, scale=unc)
    opt = minimize_scalar(lambda x: -dist.logpdf(x * sig + bkg), bounds=(-2.0, 0.0))

    statistical_model = spey.get_backend("default.normal")(
        signal_yields=[sig],
        background_yields=[bkg],
        data=[obs],
        absolute_uncertainties=[unc],
    )
    muhat, maxnll = statistical_model.maximize_likelihood()

    assert np.isclose(
        statistical_model.chi2(poi_test_denominator=0),
        2.0
        * (
            0.5 * ((12.0 + 50.0 - 36.0) ** 2 / 20.0**2)
            - (0.5 * ((50.0 - 36.0) ** 2 / 20.0**2))
        ),
    ), "Gaussian chi2 is wrong"
    assert np.isclose(muhat, opt.x), "Normal:: Muhat is wrong"
    assert np.isclose(maxnll, opt.fun), "Normal:: MLE is wrong"

    # add chi2-test

    left_lim, right_lim = statistical_model.chi2_test()
    left_chi2 = statistical_model.chi2(poi_test=left_lim, allow_negative_signal=True)
    right_chi2 = statistical_model.chi2(poi_test=right_lim, allow_negative_signal=True)
    chi2_threshold = chi2.isf((1 - 0.95) / 2, df=1)

    assert np.isclose(
        [left_chi2, right_chi2], chi2_threshold
    ).all(), "chi2 test doesnt match chi2 threshold"


def test_multivariate_gauss():
    """tester for multivar gauss"""

    signal = np.array([12.0, 15.0])
    bkg = np.array([50.0, 48.0])
    data = np.array([36, 33])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    statistical_model = spey.get_backend("default.multivariate_normal")(
        signal_yields=signal,
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
    )

    dist = multivariate_normal(mean=data, cov=cov)
    opt = minimize_scalar(lambda x: -dist.logpdf(x * signal + bkg), bounds=(-2, 0))
    muhat, maxnll = statistical_model.maximize_likelihood()

    assert np.isclose(
        statistical_model.chi2(poi_test_denominator=0),
        2.0
        * (
            (0.5 * (signal + bkg - data) @ np.linalg.inv(cov) @ (signal + bkg - data))
            - (0.5 * (bkg - data) @ np.linalg.inv(cov) @ (bkg - data))
        ),
    ), "Multivariate gauss wrong"
    assert np.isclose(muhat, opt.x, rtol=1e-3), "MultivariateNormal:: Muhat is wrong"
    assert np.isclose(maxnll, opt.fun, rtol=1e-3), "MultivariateNormal:: MLE is wrong"
