"""Test default pdf plugin"""

import numpy as np
import spey


def test_uncorrelated_background():
    """tester for uncorrelated background model"""

    pdf_wrapper = spey.get_backend("default_pdf.uncorrelated_background")

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

    pdf_wrapper = spey.get_backend("default_pdf.correlated_background")
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

    pdf_wrapper = spey.get_backend("default_pdf.third_moment_expansion")
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

    pdf_wrapper = spey.get_backend("default_pdf.effective_sigma")
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

    pdf_wrapper = spey.get_backend("default_pdf.poisson")

    data = [36, 33]
    signal_yields = [12.0, 15.0]
    background_yields = [50.0, 48.0]

    stat_model = pdf_wrapper(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        analysis="multi_bin",
        xsection=0.123,
    )

    assert np.isclose(stat_model.poi_upper_limit(), 0.3140867496931846), "POI is wrong."
    assert np.isclose(
        stat_model.exclusion_confidence_level()[0], 0.9999807105228611
    ), "CLs is wrong"
    assert np.isclose(stat_model.sigma_mu(1.0), 0.5573350296644078), "Sigma mu is wrong"


def test_normal():
    """tester for gaussian model"""

    statistical_model = spey.get_backend("default_pdf.normal")(
        signal_yields=[12.0],
        background_yields=[50.0],
        data=[36],
        absolute_uncertainties=[20.0],
    )

    assert np.isclose(
        statistical_model.chi2(),
        2.0
        * (
            0.5 * ((12.0 + 50.0 - 36.0) ** 2 / 20.0**2)
            - (0.5 * ((50.0 - 36.0) ** 2 / 20.0**2))
        ),
    ), "Gaussian chi2 is wrong"


def test_multivariate_gauss():
    """tester for multivar gauss"""

    signal = np.array([12.0, 15.0])
    bkg = np.array([50.0, 48.0])
    data = np.array([36, 33])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    statistical_model = spey.get_backend("default_pdf.multivariate_normal")(
        signal_yields=signal,
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
    )

    assert np.isclose(
        statistical_model.chi2(),
        2.0
        * (
            (0.5 * (signal + bkg - data) @ np.linalg.inv(cov) @ (signal + bkg - data))
            - (0.5 * (bkg - data) @ np.linalg.inv(cov) @ (bkg - data))
        ),
    )
