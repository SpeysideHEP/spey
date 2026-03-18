"""Test default pdf plugin"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, multivariate_normal, norm, poisson

import spey
from spey.helper_functions import covariance_to_correlation, merge_correlated_bins


def test_uncorrelated_background():
    """tester for uncorrelated background model"""

    pdf_wrapper = spey.get_backend("default.uncorrelated_background")

    data = np.array([36, 33])
    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    background_unc = np.array([12.0, 16.0])

    stat_model = pdf_wrapper(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        absolute_uncertainties=background_unc,
    )
    model_nll = stat_model.backend.get_logpdf_func()(np.array([1.0, 1.0, 1.0]))

    def logprob(param, data):
        return poisson.logpmf(
            data,
            param[0] * signal_yields + background_yields + background_unc * param[1:],
        )

    normal = norm(loc=[0, 0], scale=[1, 1])

    exact_nll = (
        logprob(np.array([1.0, 1.0, 1.0]), data).sum() + normal.logpdf([1.0, 1.0]).sum()
    )

    assert np.isclose(
        model_nll, exact_nll, rtol=0.00001
    ), f"Correlated background NLL is wrong. {model_nll=}, {exact_nll=}"
    assert np.isclose(stat_model.poi_upper_limit(), 0.8563345655114185), "POI is wrong."
    assert np.isclose(
        stat_model.exclusion_confidence_level()[0], 0.9701795436411219
    ), "CLs is wrong"


def test_correlated_background():
    """tester for correlated background"""

    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    data = np.array([36, 33])
    covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])

    pdf_wrapper = spey.get_backend("default.correlated_background")
    statistical_model = pdf_wrapper(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        covariance_matrix=covariance_matrix,
        analysis="example",
        xsection=0.123,
    )
    model_nll = statistical_model.backend.get_logpdf_func()(np.array([1.0, 1.0, 1.0]))

    multivar_norm = multivariate_normal(
        mean=[0, 0], cov=covariance_to_correlation(covariance_matrix)
    )

    def logprob(param, data):
        return poisson.logpmf(
            data,
            param[0] * signal_yields
            + background_yields
            + np.sqrt(np.diag(covariance_matrix)) * param[1:],
        )

    exact_nll = sum(logprob(np.array([1.0, 1.0, 1.0]), data)) + multivar_norm.logpdf(
        [1, 1]
    )

    assert np.isclose(
        model_nll, exact_nll, rtol=0.001
    ), f"Correlated background NLL is wrong. {model_nll=}, {exact_nll=}"
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

    CLs = statistical_model.exclusion_confidence_level()[0]
    assert np.isclose(
        CLs, 0.961432961, rtol=1e-4
    ), f"CLs is wrong, expected 0.961432961 got {CLs}"
    poi_ul = statistical_model.poi_upper_limit()
    assert np.isclose(
        poi_ul, 0.9221339, rtol=1e-4
    ), f"POI is wrong, expected 0.9221339 got {poi_ul}"
    sigma_mu = statistical_model.sigma_mu(1.0)
    assert np.isclose(
        sigma_mu, 0.85455, rtol=1e-4
    ), f"Sigma mu is wrong, expected 0.85455 got {sigma_mu}"


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

    assert (
        pytest.approx(0.8567, 1e-4) == statistical_model.exclusion_confidence_level()[0]
    ), "CLs is wrong"
    assert (
        pytest.approx(1.5298, 1e-4) == statistical_model.poi_upper_limit()
    ), "POI is wrong."
    assert pytest.approx(1.2152, 1e-4) == statistical_model.sigma_mu(
        1.0
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

    pull = statistical_model.pull(0.0)
    pull_analytic = (obs - bkg) / unc
    assert np.isclose(pull, pull_analytic), "Pull is wrong."


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

    def logpdf(x, mean, cov):
        diff = x - mean
        return -0.5 * (diff @ np.linalg.inv(cov) @ diff) - 0.5 * (
            len(x) * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1]
        )

    opt = minimize_scalar(
        lambda x: -logpdf(data, mean=x * signal + bkg, cov=cov), bounds=(-2, 0)
    )
    muhat, maxnll = statistical_model.maximize_likelihood()

    assert pytest.approx(statistical_model.chi2(poi_test_denominator=0)) == -2.0 * (
        logpdf(data, mean=signal + bkg, cov=cov) - logpdf(data, mean=bkg, cov=cov)
    ), "Multivariate gauss wrong"
    assert pytest.approx(opt.x) == muhat, "MultivariateNormal:: Muhat is wrong"
    assert pytest.approx(opt.fun) == maxnll, "MultivariateNormal:: MLE is wrong"


def test_multivariate_gauss_array_signal_yields_is_alive():
    """Array signal_yields: is_alive reflects non-zero bins."""
    from spey.backends.default_pdf.simple_pdf import MultivariateNormal

    bkg = np.array([50.0, 48.0])
    data = np.array([36.0, 33.0])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    m_alive = MultivariateNormal(
        signal_yields=np.array([12.0, 15.0]),
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
    )
    assert m_alive.is_alive

    m_dead = MultivariateNormal(
        signal_yields=np.array([0.0, 0.0]),
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
    )
    assert not m_dead.is_alive


def test_multivariate_gauss_callable_signal_yields_is_always_alive():
    """Callable signal_yields: is_alive always returns True."""
    from spey.backends.default_pdf.simple_pdf import MultivariateNormal

    bkg = np.array([50.0, 48.0])
    data = np.array([36.0, 33.0])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    # even a callable that would return zeros must report is_alive=True
    model = MultivariateNormal(
        signal_yields=lambda pars: np.zeros(2),
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
        n_signal_parameters=1,
    )
    assert model.is_alive


def test_multivariate_gauss_callable_signal_yields_model_config():
    """Callable signal_yields: ModelConfig has correct npar and parameter_names."""
    from spey.backends.default_pdf.simple_pdf import MultivariateNormal

    bkg = np.array([50.0, 48.0])
    data = np.array([36.0, 33.0])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    model = MultivariateNormal(
        signal_yields=lambda pars: np.array([12.0, 15.0]),
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
        n_signal_parameters=2,
    )
    cfg = model.config()

    assert cfg.npar == 3, f"Expected npar=3, got {cfg.npar}"
    assert cfg.parameter_names == ["mu", "signal_par_0", "signal_par_1"]
    assert len(cfg.suggested_init) == 3
    assert len(cfg.suggested_bounds) == 3
    # extra-parameter bounds should be (None, None)
    assert cfg.suggested_bounds[1] == (None, None)
    assert cfg.suggested_bounds[2] == (None, None)


def test_multivariate_gauss_n_signal_parameters_zero_no_extra_params():
    """n_signal_parameters=0 (default) keeps the single-parameter config."""
    from spey.backends.default_pdf.simple_pdf import MultivariateNormal

    bkg = np.array([50.0, 48.0])
    data = np.array([36.0, 33.0])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    model = MultivariateNormal(
        signal_yields=np.array([12.0, 15.0]),
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
    )
    cfg = model.config()

    assert cfg.npar == 1
    assert cfg.parameter_names == ["mu"]


def test_multivariate_gauss_callable_signal_yields_logpdf_matches_array():
    """Callable that returns constant signal must give same logpdf as array."""
    from spey.backends.default_pdf.simple_pdf import MultivariateNormal

    signal = np.array([12.0, 15.0])
    bkg = np.array([50.0, 48.0])
    data = np.array([36.0, 33.0])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    model_array = MultivariateNormal(
        signal_yields=signal, background_yields=bkg, data=data, covariance_matrix=cov
    )
    # callable that ignores its argument and returns the same fixed signal
    model_callable = MultivariateNormal(
        signal_yields=lambda pars: signal,
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
        n_signal_parameters=1,
    )

    logpdf_array = model_array.get_logpdf_func()(np.array([1.0]))
    # for callable model pars = [mu=1.0, signal_par_0=0.0]
    logpdf_callable = model_callable.get_logpdf_func()(np.array([1.0, 0.0]))

    assert np.isclose(
        logpdf_array, logpdf_callable
    ), f"logpdf mismatch: array={logpdf_array}, callable={logpdf_callable}"


def test_multivariate_gauss_callable_signal_yields_maximize_likelihood():
    """Callable signal_yields: maximize_likelihood converges and muhat is finite."""
    from spey.backends.default_pdf.simple_pdf import MultivariateNormal

    base_signal = np.array([12.0, 15.0])
    bkg = np.array([50.0, 48.0])
    data = np.array([36.0, 33.0])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    def signal_yields(extra_pars):
        # scale signal by (1 + extra_pars[0])
        return base_signal * (1.0 + extra_pars[0])

    model = MultivariateNormal(
        signal_yields=signal_yields,
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
        n_signal_parameters=1,
    )
    stat_model = spey.StatisticalModel(backend=model, analysis="callable_test")
    muhat, nll = stat_model.maximize_likelihood()

    assert np.isfinite(muhat), f"muhat not finite: {muhat}"
    assert np.isfinite(nll), f"nll not finite: {nll}"


def test_multivariate_gauss_config_preserves_extra_bounds_when_allow_negative_signal_false():
    """config(allow_negative_signal=False) preserves extra signal-parameter bounds."""
    from spey.backends.default_pdf.simple_pdf import MultivariateNormal

    bkg = np.array([50.0, 48.0])
    data = np.array([36.0, 33.0])
    cov = np.array([[144.0, 13.0], [25.0, 256.0]])

    model = MultivariateNormal(
        signal_yields=lambda pars: np.array([12.0, 15.0]) * (1.0 + pars[0]),
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
        n_signal_parameters=1,
    )
    cfg = model.config(allow_negative_signal=False, poi_upper_bound=5.0)

    assert cfg.npar == 2
    assert cfg.suggested_bounds[0] == (0, 5.0)
    assert cfg.suggested_bounds[1] == (None, None)
    assert cfg.parameter_names == ["mu", "signal_par_0"]


# ---------------------------------------------------------------------------
# Tests for callable signal_yields (new feature) — all four default backends
# ---------------------------------------------------------------------------


def _analytic_loglike_uncorr(
    mu, scale, data, signal_yields, background_yields, background_unc
):
    """
    Analytic log-likelihood for UncorrelatedBackground with nuisances profiled out.

    L(mu, theta) = prod_i Poiss(n_obs_i | mu*s_i*scale + b_i + sigma_i*theta_i)
                   * prod_i N(theta_i | 0, 1)

    At fixed mu, the nuisance parameters are at their constraint centres (theta=0)
    for evaluating the likelihood at a specific point.
    """
    lam = mu * signal_yields * scale + background_yields
    log_main = np.sum(poisson.logpmf(data, lam))
    log_constraint = np.sum(norm.logpdf(np.zeros_like(background_unc), 0.0, 1.0))
    return log_main + log_constraint


def test_uncorrelated_background_callable_signal_yields_logpdf():
    """
    UncorrelatedBackground: callable signal_yields with one extra parameter.

    The callable s(p) = base_signal * (1 + p[0]) introduces a linear scale factor.
    At p[0]=0 the result must equal the array-based model at the same mu.
    At a general point (mu, p[0], theta_1, theta_2) the NLL must match the
    analytic expression computed by hand.
    """
    from spey.backends.default_pdf import UncorrelatedBackground

    base_signal = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    data = np.array([36, 33])
    background_unc = np.array([12.0, 16.0])

    # array model (baseline)
    model_array = UncorrelatedBackground(
        signal_yields=base_signal,
        background_yields=background_yields,
        data=data,
        absolute_uncertainties=background_unc,
    )

    # callable model: extra parameter p[0] scales the signal
    def signal_yields_fn(extra):
        return base_signal * (1.0 + extra[0])

    model_callable = UncorrelatedBackground(
        signal_yields=signal_yields_fn,
        background_yields=background_yields,
        data=data,
        absolute_uncertainties=background_unc,
        n_signal_parameters=1,
    )

    # --- parameter-vector layout check ---
    cfg = model_callable.config()
    assert (
        cfg.npar == 4
    ), f"Expected 4 parameters [mu, sig_par_0, theta_0, theta_1], got {cfg.npar}"
    assert cfg.parameter_names == ["mu", "signal_par_0", "theta_bkg_0", "theta_bkg_1"]

    # --- at p[0]=0 logpdf must match the array model ---
    pars_array = np.array([1.0, 0.5, -0.3])  # [mu, theta_0, theta_1]
    pars_callable = np.array([1.0, 0.0, 0.5, -0.3])  # [mu, sig_par_0=0, theta_0, theta_1]

    ll_array = model_array.get_logpdf_func()(pars_array)
    ll_callable = model_callable.get_logpdf_func()(pars_callable)
    assert np.isclose(
        ll_array, ll_callable, rtol=1e-6
    ), f"logpdf mismatch at sig_par=0: array={ll_array}, callable={ll_callable}"

    # --- analytic check at a non-trivial point ---
    # pars = [mu=1.0, sig_par_0=0.5, theta_0=0.0, theta_1=0.0]
    mu_test, scale_test = 1.0, 1.5  # signal_yields = base * (1 + 0.5)
    pars_test = np.array([mu_test, 0.5, 0.0, 0.0])

    ll_model = model_callable.get_logpdf_func()(pars_test)

    # analytic: nuisances at theta=0 → constraint term = sum log N(0|0,1)
    lam = mu_test * base_signal * scale_test + background_yields
    ll_analytic = np.sum(poisson.logpmf(data, lam)) + np.sum(
        norm.logpdf([0.0, 0.0], 0.0, 1.0)
    )
    assert np.isclose(
        ll_model, ll_analytic, rtol=1e-6
    ), f"analytic mismatch: model={ll_model}, analytic={ll_analytic}"

    # --- maximize_likelihood converges ---
    stat_model = spey.StatisticalModel(backend=model_callable, analysis="unc_callable")
    muhat, nll = stat_model.maximize_likelihood()
    assert np.isfinite(muhat), f"muhat not finite: {muhat}"
    assert np.isfinite(nll), f"nll not finite: {nll}"


def test_uncorrelated_background_callable_signal_yields_modifiers_raises():
    """UncorrelatedBackground: callable + modifiers must raise ValueError."""
    from spey.backends.default_pdf import UncorrelatedBackground

    with pytest.raises(ValueError, match="modifiers cannot be combined"):
        UncorrelatedBackground(
            signal_yields=lambda p: np.array([12.0, 15.0]),
            background_yields=[50.0, 48.0],
            data=[36, 33],
            absolute_uncertainties=[12.0, 16.0],
            modifiers=[
                {"type": "normalization", "name": "pdf", "uncertainties": [2.0, 3.0]}
            ],
            n_signal_parameters=1,
        )


def test_correlated_background_callable_signal_yields_logpdf():
    """
    CorrelatedBackground: callable signal_yields with two extra parameters.

    The callable s(p) = [p[0]*12, p[1]*15] lets the optimiser scale each
    bin's signal independently.  At (p[0]=1, p[1]=1, theta=0) the result
    must reproduce the analytic log-likelihood for the standard model.
    """
    from spey.backends.default_pdf import CorrelatedBackground

    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    data = np.array([36, 33])
    covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])

    sigma = np.sqrt(np.diag(covariance_matrix))
    corr = covariance_to_correlation(covariance_matrix)

    # callable: p = [scale_0, scale_1] → signal per bin
    def signal_fn(extra):
        return signal_yields * extra  # element-wise scale

    model_callable = CorrelatedBackground(
        signal_yields=signal_fn,
        background_yields=background_yields,
        data=data,
        covariance_matrix=covariance_matrix,
        n_signal_parameters=2,
    )

    # --- parameter-vector layout ---
    cfg = model_callable.config()
    assert cfg.npar == 5, f"Expected 5 parameters, got {cfg.npar}"
    # pars = [mu, sig_par_0, sig_par_1, theta_0, theta_1]
    assert cfg.parameter_names == [
        "mu",
        "signal_par_0",
        "signal_par_1",
        "theta_bkg_0",
        "theta_bkg_1",
    ]

    # --- analytic check at (mu=1, scale=[1,1], theta=[0,0]) ---
    # This matches the plain array model at (mu=1, theta=[0,0])
    pars_test = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    ll_model = model_callable.get_logpdf_func()(pars_test)

    mv_norm = multivariate_normal(mean=[0, 0], cov=corr)
    lam = (
        1.0 * signal_yields * np.array([1.0, 1.0])
        + background_yields
        + sigma * np.array([0.0, 0.0])
    )
    ll_analytic = np.sum(poisson.logpmf(data, lam)) + mv_norm.logpdf([0.0, 0.0])
    # rtol=0.001 matches the tolerance in test_correlated_background: spey uses a
    # slightly different log-determinant computation than scipy.
    assert np.isclose(
        ll_model, ll_analytic, rtol=0.001
    ), f"analytic mismatch: model={ll_model}, analytic={ll_analytic}"

    # --- at p=(mu=1, scale=[0.5, 2.0], theta=[0.3, -0.2]) ---
    scales = np.array([0.5, 2.0])
    thetas = np.array([0.3, -0.2])
    pars_test2 = np.concatenate([[1.0], scales, thetas])
    ll_model2 = model_callable.get_logpdf_func()(pars_test2)

    lam2 = 1.0 * signal_yields * scales + background_yields + sigma * thetas
    ll_analytic2 = np.sum(poisson.logpmf(data, lam2)) + mv_norm.logpdf(thetas)
    assert np.isclose(
        ll_model2, ll_analytic2, rtol=0.001
    ), f"analytic mismatch (non-trivial point): model={ll_model2}, analytic={ll_analytic2}"

    # --- maximize_likelihood converges ---
    stat_model = spey.StatisticalModel(backend=model_callable, analysis="corr_callable")
    muhat, nll = stat_model.maximize_likelihood()
    assert np.isfinite(muhat), f"muhat not finite: {muhat}"
    assert np.isfinite(nll), f"nll not finite: {nll}"


def test_third_moment_expansion_callable_signal_yields_logpdf():
    """
    ThirdMomentExpansion: callable signal_yields with one extra parameter.

    The callable s(p) = base_signal * p[0] allows a free per-event signal scale.
    At p[0]=1, theta=0 the result must match the analytic NLL for the
    third-moment-expanded background.
    """
    from spey.backends.default_pdf import ThirdMomentExpansion
    from spey.backends.default_pdf.third_moment import third_moment_expansion

    base_signal = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    data = np.array([36, 33])
    covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])
    third_moment = np.array([0.5, 0.8])

    def signal_fn(extra):
        return base_signal * extra[0]

    model_callable = ThirdMomentExpansion(
        signal_yields=signal_fn,
        background_yields=background_yields,
        data=data,
        covariance_matrix=covariance_matrix,
        third_moment=third_moment,
        n_signal_parameters=1,
    )

    cfg = model_callable.config()
    assert cfg.npar == 4, f"Expected 4 parameters, got {cfg.npar}"
    assert cfg.parameter_names == ["mu", "signal_par_0", "theta_bkg_0", "theta_bkg_1"]

    # Compute A, B, C, corr from the third-moment expansion to build the analytic expression
    A, B, C, corr = third_moment_expansion(
        background_yields, covariance_matrix, third_moment, True
    )
    mv_norm = multivariate_normal(mean=[0, 0], cov=corr)

    # at (mu=1, scale=1, theta=[0,0])
    pars_test = np.array([1.0, 1.0, 0.0, 0.0])
    ll_model = model_callable.get_logpdf_func()(pars_test)

    thetas = np.array([0.0, 0.0])
    lam = 1.0 * base_signal * 1.0 + A + B * thetas + C * thetas**2
    ll_analytic = np.sum(poisson.logpmf(data, lam)) + mv_norm.logpdf(thetas)
    assert np.isclose(
        ll_model, ll_analytic, rtol=1e-5
    ), f"analytic mismatch at theta=0: model={ll_model}, analytic={ll_analytic}"

    # at (mu=0.5, scale=2.0, theta=[0.4, -0.3])
    pars_test2 = np.array([0.5, 2.0, 0.4, -0.3])
    ll_model2 = model_callable.get_logpdf_func()(pars_test2)

    thetas2 = np.array([0.4, -0.3])
    lam2 = 0.5 * base_signal * 2.0 + A + B * thetas2 + C * thetas2**2
    ll_analytic2 = np.sum(poisson.logpmf(data, lam2)) + mv_norm.logpdf(thetas2)
    assert np.isclose(
        ll_model2, ll_analytic2, rtol=1e-5
    ), f"analytic mismatch (non-trivial): model={ll_model2}, analytic={ll_analytic2}"

    # maximize_likelihood must converge
    stat_model = spey.StatisticalModel(backend=model_callable, analysis="tme_callable")
    muhat, nll = stat_model.maximize_likelihood()
    assert np.isfinite(muhat), f"muhat not finite: {muhat}"
    assert np.isfinite(nll), f"nll not finite: {nll}"


def test_effective_sigma_callable_signal_yields_logpdf():
    """
    EffectiveSigma: callable signal_yields with one extra parameter.

    The callable s(p) = base_signal * (1 + p[0]).  At p[0]=0, theta=0 the
    log-likelihood must match the analytic expression built from the
    effective-sigma formula (arXiv:physics/0406120, eqs. 18-19).
    """
    from spey.backends.default_pdf import EffectiveSigma

    base_signal = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    data = np.array([36, 33])
    correlation_matrix = np.array([[1.0, 0.06770833], [0.13020833, 1.0]])
    envelops = [(10.0, 15.0), (13.0, 18.0)]  # (sigma_plus, sigma_minus) per bin

    sigma_plus = np.array([10.0, 13.0])
    sigma_minus = np.array([15.0, 18.0])

    def signal_fn(extra):
        return base_signal * (1.0 + extra[0])

    model_callable = EffectiveSigma(
        signal_yields=signal_fn,
        background_yields=background_yields,
        data=data,
        correlation_matrix=correlation_matrix,
        absolute_uncertainty_envelops=envelops,
        n_signal_parameters=1,
    )

    cfg = model_callable.config()
    assert cfg.npar == 4, f"Expected 4 parameters, got {cfg.npar}"
    assert cfg.parameter_names == ["mu", "signal_par_0", "theta_bkg_0", "theta_bkg_1"]

    # analytic effective sigma at theta=0
    def sigma_eff(thetas):
        return np.sqrt(
            np.clip(
                sigma_plus * sigma_minus
                + (sigma_plus - sigma_minus) * (thetas - background_yields),
                1e-10,
                None,
            )
        )

    mv_norm = multivariate_normal(mean=[0, 0], cov=correlation_matrix)

    # at (mu=1, sig_par=0, theta=[0,0])
    thetas = np.array([0.0, 0.0])
    pars_test = np.array([1.0, 0.0, 0.0, 0.0])
    ll_model = model_callable.get_logpdf_func()(pars_test)

    lam = background_yields + sigma_eff(thetas) * thetas + 1.0 * base_signal * 1.0
    ll_analytic = np.sum(poisson.logpmf(data, lam)) + mv_norm.logpdf(thetas)
    # rtol=0.001 to allow for spey's log-det convention vs scipy (same as correlated test)
    assert np.isclose(
        ll_model, ll_analytic, rtol=0.001
    ), f"analytic mismatch at theta=0: model={ll_model}, analytic={ll_analytic}"

    # at (mu=0.8, sig_par=0.5, theta=[0.2, -0.1])
    thetas2 = np.array([0.2, -0.1])
    pars_test2 = np.array([0.8, 0.5, 0.2, -0.1])
    ll_model2 = model_callable.get_logpdf_func()(pars_test2)

    lam2 = background_yields + sigma_eff(thetas2) * thetas2 + 0.8 * base_signal * 1.5
    ll_analytic2 = np.sum(poisson.logpmf(data, lam2)) + mv_norm.logpdf(thetas2)
    assert np.isclose(
        ll_model2, ll_analytic2, rtol=0.001
    ), f"analytic mismatch (non-trivial): model={ll_model2}, analytic={ll_analytic2}"

    # maximize_likelihood must converge
    stat_model = spey.StatisticalModel(backend=model_callable, analysis="es_callable")
    muhat, nll = stat_model.maximize_likelihood()
    assert np.isfinite(muhat), f"muhat not finite: {muhat}"
    assert np.isfinite(nll), f"nll not finite: {nll}"


def test_callable_signal_yields_is_alive_default_backends():
    """is_alive returns True for all backends when signal_yields is callable."""
    from spey.backends.default_pdf import (
        CorrelatedBackground,
        EffectiveSigma,
        ThirdMomentExpansion,
        UncorrelatedBackground,
    )

    sig_fn = lambda p: np.zeros(2)  # noqa: E731
    bkg = [50.0, 48.0]
    data = [36, 33]
    cov = [[144.0, 13.0], [25.0, 256.0]]

    assert UncorrelatedBackground(
        signal_yields=sig_fn,
        background_yields=bkg,
        data=data,
        absolute_uncertainties=[12.0, 16.0],
        n_signal_parameters=1,
    ).is_alive

    assert CorrelatedBackground(
        signal_yields=sig_fn,
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
        n_signal_parameters=1,
    ).is_alive

    assert ThirdMomentExpansion(
        signal_yields=sig_fn,
        background_yields=bkg,
        data=data,
        covariance_matrix=cov,
        third_moment=[0.5, 0.8],
        n_signal_parameters=1,
    ).is_alive

    assert EffectiveSigma(
        signal_yields=sig_fn,
        background_yields=bkg,
        data=data,
        correlation_matrix=[[1.0, 0.07], [0.13, 1.0]],
        absolute_uncertainty_envelops=[(10.0, 15.0), (13.0, 18.0)],
        n_signal_parameters=1,
    ).is_alive


def test_uncertainty_synthesizer_n_signal_parameters_domain_shift():
    """
    signal_uncertainty_synthesizer: n_signal_parameters and domain_start shift
    constraint domains for normalization modifiers.

    With N=2 bins, 1 normalization modifier, and n_signal_parameters=2:
      domain = 1 + 2 + 2 = 5  →  [5]
    With n_signal_parameters=0 (default):
      domain = 1 + 0 + 2 = 3  →  [3]
    With explicit domain_start=7:
      domain = 7  →  [7]
    """
    from spey.backends.default_pdf.uncertainty_synthesizer import (
        signal_uncertainty_synthesizer,
    )

    signal_yields = [10.0, 20.0]
    modifiers = [{"type": "normalization", "name": "pdf", "uncertainties": [1.0, 2.0]}]

    result_default = signal_uncertainty_synthesizer(signal_yields, modifiers)
    result_shifted = signal_uncertainty_synthesizer(
        signal_yields, modifiers, n_signal_parameters=2
    )
    result_explicit = signal_uncertainty_synthesizer(
        signal_yields, modifiers, domain_start=7
    )

    domain_default = result_default["constraint"][0]["kwargs"]["domain"]
    domain_shifted = result_shifted["constraint"][0]["kwargs"]["domain"]
    domain_explicit = result_explicit["constraint"][0]["kwargs"]["domain"]

    assert domain_default[0] == 3, f"Expected domain index 3, got {domain_default[0]}"
    assert domain_shifted[0] == 5, f"Expected domain index 5, got {domain_shifted[0]}"
    assert domain_explicit[0] == 7, f"Expected domain index 7, got {domain_explicit[0]}"

    # normalization: exactly 1 nuisance parameter
    assert result_default["n_parameters"] == 1
    assert result_default["parameter_names"] == ["theta_sig_pdf"]


def test_correlated_background_modifiers_with_n_signal_parameters_zero():
    """
    With n_signal_parameters=0 and modifiers, existing modifier behavior is preserved.
    The signal-uncertainty nuisance parameters are placed after the background
    nuisances (at index N+1 = 3 for 2 bins).
    """
    from spey.backends.default_pdf import CorrelatedBackground

    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    data = np.array([36, 33])
    covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])

    model = CorrelatedBackground(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        covariance_matrix=covariance_matrix,
        modifiers=[{"type": "normalization", "name": "pdf", "uncertainties": [2.0, 3.0]}],
    )

    cfg = model.config()
    # pars = [mu, theta_bkg_0, theta_bkg_1, theta_sig_pdf]
    assert cfg.npar == 4, f"Expected 4 parameters, got {cfg.npar}"
    assert cfg.parameter_names == [
        "mu",
        "theta_bkg_0",
        "theta_bkg_1",
        "theta_sig_pdf",
    ], f"Unexpected parameter names: {cfg.parameter_names}"

    # signal-uncertainty constraint domain must be index 3 (= 1 + 0 + 2)
    sig_unc_constraint = model.signal_uncertainty_configuration["constraint"][0]
    domain = sig_unc_constraint["kwargs"]["domain"]
    assert domain[0] == 3, f"Expected signal-unc domain index 3, got {domain[0]}"

    # logpdf must be finite at the nominal point
    pars = np.array([1.0, 0.0, 0.0, 0.0])
    ll = model.get_logpdf_func()(pars)
    assert np.isfinite(ll), f"logpdf not finite: {ll}"


def test_uncertainty_synthesizer_shape_modifier():
    """
    signal_uncertainty_synthesizer: shape modifier assigns one nuisance per bin.

    With N=2 bins and one shape modifier:
      - n_parameters == 2  (one per bin)
      - constraint domains == [3, 4]  (for n_signal_parameters=0)
      - parameter_names == ["theta_sig_scale_0", "theta_sig_scale_1"]
    """
    from spey.backends.default_pdf.uncertainty_synthesizer import (
        signal_uncertainty_synthesizer,
    )

    signal_yields = [10.0, 20.0]
    modifiers = [{"type": "shape", "name": "scale", "uncertainties": [1.0, 2.0]}]

    result = signal_uncertainty_synthesizer(signal_yields, modifiers)

    assert (
        result["n_parameters"] == 2
    ), f"Expected 2 parameters, got {result['n_parameters']}"
    assert result["parameter_names"] == [
        "theta_sig_scale_0",
        "theta_sig_scale_1",
    ], f"Unexpected names: {result['parameter_names']}"
    assert len(result["constraint"]) == 2, "Expected 2 constraints for shape modifier"
    assert result["constraint"][0]["kwargs"]["domain"][0] == 3
    assert result["constraint"][1]["kwargs"]["domain"][0] == 4

    # lambda must return per-bin array of length 2 at the nominal point (alpha=0)
    pars = np.zeros(5)  # [mu, theta_bkg_0, theta_bkg_1, alpha_0, alpha_1]
    val = result["lambda"](pars)
    assert val.shape == (2,), f"Expected shape (2,), got {val.shape}"
    assert np.allclose(val, 1.0), f"At alpha=0 modifier must be 1, got {val}"


def test_uncertainty_synthesizer_mixed_normalization_shape():
    """
    signal_uncertainty_synthesizer: mixed normalization + shape modifiers.

    With N=2 bins, one normalization (1 par) and one shape (2 pars):
      - n_parameters == 3
      - normalization domain index == 3
      - shape domain indices == [4, 5]
    """
    from spey.backends.default_pdf.uncertainty_synthesizer import (
        signal_uncertainty_synthesizer,
    )

    signal_yields = [10.0, 20.0]
    modifiers = [
        {"type": "normalization", "name": "pdf", "uncertainties": [1.0, 2.0]},
        {"type": "shape", "name": "scale", "uncertainties": [0.5, 1.0]},
    ]

    result = signal_uncertainty_synthesizer(signal_yields, modifiers)

    assert result["n_parameters"] == 3
    assert result["parameter_names"] == [
        "theta_sig_pdf",
        "theta_sig_scale_0",
        "theta_sig_scale_1",
    ]
    assert len(result["constraint"]) == 3
    # normalization at index 3
    assert result["constraint"][0]["kwargs"]["domain"][0] == 3
    # shape bins at indices 4 and 5
    assert result["constraint"][1]["kwargs"]["domain"][0] == 4
    assert result["constraint"][2]["kwargs"]["domain"][0] == 5

    # at alpha=0 for all nuisances the total modifier must be 1
    pars = np.zeros(6)  # [mu, bkg0, bkg1, theta_pdf, alpha0, alpha1]
    val = result["lambda"](pars)
    assert np.allclose(val, 1.0), f"At zero nuisances modifier must be 1, got {val}"


def test_correlated_background_shape_modifier_parameter_count():
    """
    CorrelatedBackground with a shape modifier produces N nuisance parameters
    (one per bin), so ModelConfig must reflect the larger parameter count.

    For N=2 bins and one shape modifier:
      pars = [mu, theta_bkg_0, theta_bkg_1, alpha_scale_0, alpha_scale_1]  (5 total)
    """
    from spey.backends.default_pdf import CorrelatedBackground

    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    data = np.array([36, 33])
    covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])

    model = CorrelatedBackground(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        covariance_matrix=covariance_matrix,
        modifiers=[{"type": "shape", "name": "scale", "uncertainties": [1.0, 2.0]}],
    )

    cfg = model.config()
    assert cfg.npar == 5, f"Expected 5 parameters, got {cfg.npar}"
    assert cfg.parameter_names == [
        "mu",
        "theta_bkg_0",
        "theta_bkg_1",
        "theta_sig_scale_0",
        "theta_sig_scale_1",
    ], f"Unexpected names: {cfg.parameter_names}"

    pars = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    ll = model.get_logpdf_func()(pars)
    assert np.isfinite(ll), f"logpdf not finite at nominal point: {ll}"


def test_uncertainty_synthesizer_invalid_type_raises():
    """signal_uncertainty_synthesizer: unknown modifier type raises InvalidUncertaintyDefinition."""
    from spey.backends.default_pdf.uncertainty_synthesizer import (
        signal_uncertainty_synthesizer,
    )
    from spey.system.exceptions import InvalidUncertaintyDefinition

    with pytest.raises(InvalidUncertaintyDefinition, match="Unknown modifier type"):
        signal_uncertainty_synthesizer(
            signal_yields=[10.0, 20.0],
            modifiers=[{"type": "unknown", "uncertainties": [1.0, 2.0]}],
        )


def test_bin_merge():
    """Test merging of correlated bins in a histogram/cutflow."""
    results = merge_correlated_bins(
        background_yields=np.array([10, 20, 30, 40, 50, 60, 70]),
        data=np.array([12, 22, 32, 42, 52, 62, 72]),
        covariance_matrix=np.array(
            [
                [4, 1, 0.5, 0.2, 0.1, 0.2, 0.3],
                [1, 3, 0.3, 0.1, 0.1, 0.1, 0.2],
                [0.5, 0.3, 5, 0.2, 0.1, 0.0, 0.0],
                [0.2, 0.1, 0.2, 4, 0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1, 0.1, 6, 2, 0.3],
                [0.2, 0.1, 0.0, 0.2, 2, 5, 0.4],
                [0.3, 0.2, 0.0, 0.1, 0.3, 0.4, 7],
            ]
        ),
        merge_groups=[[0, 1], [2, 3]],
        signal_yields=np.array([5, 15, 25, 35, 45, 55, 65]),
    )

    assert np.allclose(
        results["background_yields"], np.array([30.0, 70.0, 50.0, 60.0, 70.0])
    ), "Background yields after merging are incorrect"
    assert np.allclose(
        results["data"], np.array([34.0, 74.0, 52.0, 62.0, 72.0])
    ), "Data after merging is incorrect"
    assert np.allclose(
        results["covariance_matrix"],
        np.array(
            [
                [9.0, 1.1, 0.2, 0.3, 0.5],
                [1.1, 9.4, 0.2, 0.2, 0.1],
                [0.2, 0.2, 6.0, 2.0, 0.3],
                [0.3, 0.2, 2.0, 5.0, 0.4],
                [0.5, 0.1, 0.3, 0.4, 7.0],
            ]
        ),
    ), "Covariance matrix after merging is incorrect"
    if "signal_yields" in results:
        assert np.allclose(
            results["signal_yields"], np.array([20.0, 60.0, 45.0, 55.0, 65.0])
        ), "Signal yields after merging are incorrect"
