"""Test default grad and hessian"""

import numpy as np
from spey.math import value_and_grad, hessian
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
    hess = hessian(stat_model)([1.0, 1.0])
    nll, grad = value_and_grad(stat_model)([1.0, 1.0])
    nll_apri, grad_apri = value_and_grad(
        stat_model, expected=spey.ExpectationType.apriori
    )([1.0, 1.0])
    hess_apri = hessian(stat_model, expected=spey.ExpectationType.apriori)([1.0, 1.0])

    hess_data = hessian(stat_model, expected=spey.ExpectationType.apriori, data=[22, 34])(
        [1.0, 1.0]
    )
    nll_dat, grad_dat = value_and_grad(
        stat_model, expected=spey.ExpectationType.apriori, data=[22, 34]
    )([1.0, 1.0])

    assert np.allclose(
        hess, np.array([[2.13638959, 2.21570381], [2.21570381, 4.30030563]])
    ), "Hessian is wrong"
    assert np.isclose(nll, 37.47391613937222), "NLL is wrong"
    assert np.allclose(grad, np.array([14.89633938, 17.47861786])), "Gradient is wrong"
    assert np.isclose(nll_apri, 20.052816087791097), "NLL apriori is wrong"
    assert np.allclose(
        grad_apri, np.array([9.77796784, 12.1703729])
    ), "apriori Gradient is wrong"
    assert np.allclose(
        hess_apri, np.array([[3.04532025, 3.16068638], [3.16068638, 5.28374358]])
    ), "Hessian apriori is wrong"
    assert np.allclose(
        hess_data, np.array([[1.80428957, 1.88600725], [1.88600725, 3.97317276]])
    ), "Hessian data is wrong"
    assert np.isclose(nll_dat, 49.63922692607177), "NLL data is wrong"
    assert np.allclose(
        grad_dat, np.array([16.97673623, 19.54635648])
    ), "data Gradient is wrong"
