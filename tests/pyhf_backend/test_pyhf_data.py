import pytest
from spey.backends.pyhf_backend.pyhfdata import PyhfDataWrapper, PyhfData
from spey.backends.pyhf_backend.interface import PyhfInterface
from spey.interface.statistical_model import StatisticalModel
from spey import AvailableBackends
from spey.system.exceptions import NegativeExpectedYields
import pyhf
import numpy as np


def test_data_single_bin():
    """Testing single bin data"""

    pyhf_data = PyhfDataWrapper(signal=12.0, background=51.0, nb=50.0, delta_nb=3.0, name="model")

    assert isinstance(pyhf_data, PyhfData)
    assert pyhf_data.npar == 1
    assert pyhf_data.poi_index == 0
    assert pyhf_data.suggested_init == [1.0]

    ws, model, data = pyhf_data()

    assert np.isclose(data, [51.0, 277.777]).all()
    assert np.isclose(model.logpdf([1.0, 1.0], data), -7.6583882)

    with pytest.raises(
        NegativeExpectedYields,
        match="PyhfInterface::Statistical model involves negative "
        "expected bin yields. Bin value: -9.000",
    ):
        res = pyhf_data(-5)

    assert pyhf_data.isAlive
    assert np.isclose(pyhf_data.minimum_poi, -4.1666665)

    interface: StatisticalModel = PyhfInterface(pyhf_data, xsection=0.4, analysis="test")

    assert interface.analysis == "test"
    assert interface.backend.model == pyhf_data
    assert interface.backend_type == AvailableBackends.pyhf

    # Compute likelihood: default

    pars, twice_nllh = pyhf.infer.mle.fixed_poi_fit(
        1.0,
        data,
        model,
        return_fitted_val=True,
        par_bounds=model.config.suggested_bounds(),
    )

    nll = interface.likelihood()
    assert np.isclose(nll, twice_nllh / 2.0)

    # Compute Maximum likelihood: default

    pars, twice_nllh = pyhf.infer.mle.fit(
        data,
        model,
        return_fitted_val=True,
        maxiter=200,
        par_bounds=model.config.suggested_bounds(),
    )
    muhat, nllmin = interface.maximize_likelihood(allow_negative_signal=False)

    assert np.isclose(muhat, pars[model.config.poi_index])
    assert np.isclose(nllmin, twice_nllh / 2.0)
    assert np.isclose(interface.backend.logpdf(pars[0], pars[1:]), model.logpdf(pars, data)[0])
