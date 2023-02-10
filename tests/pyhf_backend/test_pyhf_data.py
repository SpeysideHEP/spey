import pytest
from spey.backends.pyhf_backend.pyhfdata import PyhfDataWrapper, PyhfData
from spey.backends.pyhf_backend.interface import PyhfInterface
from spey.interface.statistical_model import StatisticalModel
from spey import AvailableBackends, ExpectationType
from spey.system.exceptions import NegativeExpectedYields
import pyhf
import numpy as np


def test_data_single_bin():
    """Testing single bin data"""

    pyhf_data = PyhfDataWrapper(signal=12.0, background=51.0, nb=50.0, delta_nb=3.0, name="model")

    pyhf_model = pyhf.simplemodels.uncorrelated_background([12.0], [50.0], [3.0])
    pyhfdat = [51.0] + pyhf_model.config.auxdata
    _, model, data = pyhf_data()

    assert isinstance(pyhf_data, PyhfData), "wrapper does not return phyfdata instance"
    assert (
        pyhf_data.npar == pyhf_model.config.npars - 1
    ), "number of parameters should be nuissanse - 1 "
    assert pyhf_data.poi_index == pyhf_model.config.poi_index, "poi index is wrong"
    assert (
        pyhf_data.suggested_init[0] == pyhf_model.config.suggested_init()[1]
    ), "suggested initialisation is wrong."
    assert (
        pyhf_data.suggested_bounds
        == pyhf_model.config.suggested_bounds()[pyhf_model.config.poi_index + 1 :]
    )
    assert (
        pyhf_data.suggested_fixed
        == pyhf_model.config.suggested_fixed()[pyhf_model.config.poi_index + 1 :]
    )

    assert np.isclose(data, [51.0, 277.777]).all(), "Data is wrong"
    assert (
        pyhf_data.suggested_poi_init
        == pyhf_model.config.suggested_init()[pyhf_model.config.poi_index]
    )
    assert np.isclose(model.logpdf([1.0, 1.0], data), -7.6583882), "logpdf is wrong"
    assert (
        pyhf_data.suggested_poi_bounds
        == pyhf_model.config.suggested_bounds()[pyhf_model.config.poi_index]
    )

    with pytest.raises(
        NegativeExpectedYields,
        match="PyhfInterface::Statistical model involves negative "
        "expected bin yields. Bin value: -9.000",
    ):
        res = pyhf_data(-5)

    assert pyhf_data.isAlive, "This region should be alive"
    assert np.isclose(pyhf_data.minimum_poi, -4.1666665), "Minimum POI is not correct"

    interface: StatisticalModel = PyhfInterface(pyhf_data, xsection=0.4, analysis="test")

    assert interface.analysis == "test", "Analysis name does not match"
    assert interface.xsection == 0.4, "Cross section value does not match"
    assert interface.backend.model == pyhf_data, "pyhf data does not match"
    assert interface.backend_type == AvailableBackends.pyhf, "Backend type does not match"

    # Compute likelihood: default

    pars, twice_nllh = pyhf.infer.mle.fixed_poi_fit(
        1.0,
        pyhfdat,
        pyhf_model,
        return_fitted_val=True,
        par_bounds=pyhf_model.config.suggested_bounds(),
    )

    nll = interface.likelihood()
    assert np.isclose(nll, twice_nllh / 2.0), "pyhf result does not match with interface"

    # Compute Maximum likelihood: default

    pars, twice_nllh = pyhf.infer.mle.fit(
        pyhfdat,
        pyhf_model,
        return_fitted_val=True,
        maxiter=200,
        par_bounds=pyhf_model.config.suggested_bounds(),
    )
    muhat, nllmin = interface.maximize_likelihood(allow_negative_signal=False)

    assert np.isclose(
        muhat, pars[pyhf_model.config.poi_index]
    ), "pyhf result does not match with interface"
    assert np.isclose(nllmin, twice_nllh / 2.0), "pyhf result does not match with interface"
    assert np.isclose(
        interface.backend.logpdf(pars[0], pars[1:]), pyhf_model.logpdf(pars, data)[0]
    ), "pyhf result does not match with interface"

    # Test apriori data

    pyhf_model = pyhf.simplemodels.uncorrelated_background([12.0], [50.0], [3.0])
    pyhfdat = [50.0] + pyhf_model.config.auxdata
    _, model, data = pyhf_data(expected=ExpectationType.apriori)

    assert data == pyhfdat, "Invalid data"


def test_data_json_input():
    """Testing single bin data"""

    background = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "background",
                        "data": [50.0, 52.0],
                        "modifiers": [
                            {"name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0]}
                        ],
                    }
                ],
            }
        ],
        "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
        "measurements": [{"name": "Measurement", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }
    signal = [
        {
            "op": "add",
            "path": "/channels/0/samples/1",
            "value": {
                "name": "signal",
                "data": [12.0, 11.0],
                "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
            },
        }
    ]

    pyhf_data = PyhfDataWrapper(signal=signal, background=background, name="model")

    pyhf_workspace = pyhf.Workspace(background)
    pyhf_model = pyhf_workspace.model(
        patches=[signal],
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    pyhfdat = pyhf_workspace.data(pyhf_model)
    ws, model, data = pyhf_data()

    assert isinstance(pyhf_data, PyhfData), "wrapper does not return phyfdata instance"
    assert (
        pyhf_data.npar == pyhf_model.config.npars - 1
    ), "number of parameters should be nuissanse - 1 "
    assert pyhf_data.poi_index == pyhf_model.config.poi_index, "poi index is wrong"

    pars_init = pyhf_model.config.suggested_init()
    assert (
        pyhf_data.suggested_init
        == pars_init[: pyhf_model.config.poi_index] + pars_init[pyhf_model.config.poi_index + 1 :]
    ), "suggested initialisation is wrong."

    pars_bounds = pyhf_model.config.suggested_bounds()
    assert (
        pyhf_data.suggested_bounds
        == pars_bounds[: pyhf_model.config.poi_index]
        + pars_bounds[pyhf_model.config.poi_index + 1 :]
    )

    pars_fixed = pyhf_model.config.suggested_fixed()
    assert (
        pyhf_data.suggested_fixed
        == pars_fixed[: pyhf_model.config.poi_index] + pars_fixed[pyhf_model.config.poi_index + 1 :]
    )

    assert np.isclose(data, pyhfdat).all(), "Data is wrong"

    with pytest.raises(
        NegativeExpectedYields,
        match="PyhfInterface::Statistical model involves negative expected bin "
        "yields in region 'singlechannel'. Bin values: -10.000, -3.000",
    ):
        res = pyhf_data(-5)

    assert pyhf_data.isAlive, "This region should be alive"
    assert np.isclose(pyhf_data.minimum_poi, -4.1666665), "Minimum POI is not correct"

    interface: StatisticalModel = PyhfInterface(pyhf_data, xsection=0.4, analysis="test")

    # Compute likelihood: default

    pars, twice_nllh = pyhf.infer.mle.fixed_poi_fit(
        1.0,
        pyhfdat,
        pyhf_model,
        return_fitted_val=True,
        par_bounds=pyhf_model.config.suggested_bounds(),
    )

    nll = interface.likelihood()
    assert np.isclose(nll, twice_nllh / 2.0), "pyhf result does not match with interface"

    # Compute Maximum likelihood: default

    pars, twice_nllh = pyhf.infer.mle.fit(
        pyhfdat,
        pyhf_model,
        return_fitted_val=True,
        maxiter=200,
        par_bounds=pyhf_model.config.suggested_bounds(),
    )
    muhat, nllmin = interface.maximize_likelihood(allow_negative_signal=False)

    assert np.isclose(
        muhat, pars[pyhf_model.config.poi_index]
    ), "pyhf result does not match with interface"
    assert np.isclose(nllmin, twice_nllh / 2.0), "pyhf result does not match with interface"
    assert np.isclose(
        interface.backend.logpdf(pars[0], pars[1:]), pyhf_model.logpdf(pars, pyhfdat)[0]
    ), "pyhf result does not match with interface"
