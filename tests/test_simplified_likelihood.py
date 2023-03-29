"""Test implementations for simplified likelihoods"""

import numpy as np
import pytest, spey


def test_simplified_likelihood_data():
    """Test single bin statistical model"""
    sl_data, _ = spey.get_backend("simplified_likelihoods")

    data = sl_data(
        observed=np.array([51]),
        signal=np.array([12.0]),
        background=np.array([50.0]),
        covariance=np.array([[9.0]]),
        delta_sys=0.0,
        third_moment=np.array([0.2]),
        name="test_data",
    )

    expansion = data.compute_expansion()
    assert np.isclose(expansion.A[0], 49.99629629), "Expeansion A is incorrect"
    assert np.isclose(expansion.B[0], 2.99999543), "Expeansion B is incorrect"
    assert np.isclose(expansion.C[0], 0.00370371), "Expeansion C is incorrect"
    assert np.isclose(expansion.rho[0, 0], 1.0), "Expeansion rho is incorrect"
    assert np.isclose(expansion.V[0, 0], 8.99997257), "Expeansion V is incorrect"
    assert expansion.logdet_covariance == (
        1.0,
        2.197221529005514,
    ), "Expeansion logdet_covariance is incorrect"
    assert np.isclose(
        expansion.inv_covariance[0, 0], 0.11111145
    ), "Expeansion inv_covariance is incorrect"

    assert np.isclose(data.minimum_poi, -4.16666), "Minimum POI is wrong"
    assert data.diag_cov[0] == 9.0, "Diag cov is wrong"
    assert data.is_single_region, "It is a single bin statistical model"
    assert np.all(data * 3 == 36.0), "Multiplication error"
    assert np.all(3 * data == 36.0), "Multiplication error"
    assert data.var_s[0, 0] == 0.0, "variance of signal is wrong"
    assert data.var_smu(2.0)[0, 0] == 0.0, "variance of signal is wrong"
    assert not data.isLinear, "Data is not linear"
    assert data.isAlive, "Data is alive"
    assert data.config().suggested_init == [1, 1], "suggested init are wrong"
    assert data.config().suggested_bounds[0][0] == -4.166666666666667, "Suggested bounds are wrong"

    with pytest.raises(TypeError):
        data = sl_data(
            observed=[51],
            signal=np.array([12.0]),
            background=np.array([50.0]),
            covariance=np.array([[9.0]]),
            delta_sys=0.0,
            third_moment=np.array([0.2]),
            name="test_data",
        )

    with pytest.raises(AssertionError, match="Covariance input has to be matrix."):
        data = sl_data(
            observed=np.array([51]),
            signal=np.array([12.0]),
            background=np.array([50.0]),
            covariance=np.array([9.0]),
            delta_sys=0.0,
            third_moment=np.array([0.2]),
            name="test_data",
        )

    with pytest.raises(AssertionError, match="Input shapes does not match"):
        data = sl_data(
            observed=np.array([51, 1]),
            signal=np.array([12.0]),
            background=np.array([50.0]),
            covariance=np.array([[9.0]]),
            delta_sys=0.0,
            third_moment=np.array([0.2]),
            name="test_data",
        )


def test_simplified_likelihood():
    """Test statistical model"""
    stat_model_sl = spey.get_multi_region_statistical_model(
        "simple_sl_test",
        signal=[12.0, 11.0],
        observed=[51.0, 48.0],
        covariance=[[3.0, 0.5], [0.6, 7.0]],
        nb=[50.0, 52.0],
        delta_sys=0.0,
        third_moment=[0.2, 0.1],
        xsection=0.5,
    )

    twice_nll, grads = stat_model_sl.backend.get_objective_function(do_grad=True)(
        stat_model_sl.backend.model.config().suggested_init
    )

    assert np.isclose(twice_nll, 25.43995316717471), "twice negloglikelihood is wrong"

    assert np.isclose(grads[0], 10.068618), "first Gradient"
    assert np.isclose(grads[1], 1.0037898), "second Gradient"
    assert np.isclose(grads[2], 0.73284644), "third Gradient"

    assert np.isclose(stat_model_sl.likelihood(), 11.82836290150149), "Observed likelihood"
    assert np.isclose(
        stat_model_sl.likelihood(expected=spey.ExpectationType.apriori), 11.21885221668087
    ), "Apriori likelihood"

    muhat, nll = stat_model_sl.maximize_likelihood()

    assert np.isclose(muhat, -0.1082972254923035), "Muhat"
    assert np.isclose(nll, 9.215870074935179), "nll"

    muhat, nll = stat_model_sl.maximize_likelihood(allow_negative_signal=False)

    assert np.isclose(muhat, 1.903051765299034e-17), "Muhat positive signal"
    assert np.isclose(nll, 9.243972035167737), "nll positive signal"
