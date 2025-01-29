"""Test uncorrelated statistics combiner"""

import numpy as np

import spey
from spey.combiner.uncorrelated_statistics_combiner import UnCorrStatisticsCombiner


def test_combiner():
    """Test uncorrelated statistics combiner"""

    normal = spey.get_backend("default.normal")(
        signal_yields=[1, 1, 1],
        background_yields=[2, 1, 3],
        data=[2, 2, 2],
        absolute_uncertainties=[1, 1, 1],
    )
    normal_cls = normal.exclusion_confidence_level()[0]
    normal = spey.get_backend("default.multivariate_normal")(
        signal_yields=[1, 1, 1],
        background_yields=[2, 1, 3],
        data=[2, 2, 2],
        covariance_matrix=np.diag([1, 1, 1]),
    )
    multivar_norm_cls = normal.exclusion_confidence_level()[0]

    assert np.isclose(
        normal_cls, multivar_norm_cls
    ), "Normal CLs is not the same as Multivariant normal CLs"

    normal1 = spey.get_backend("default.normal")(
        signal_yields=[1],
        background_yields=[3],
        data=[2],
        absolute_uncertainties=[1],
        analysis="norm1",
    )
    normal2 = spey.get_backend("default.normal")(
        signal_yields=[1],
        background_yields=[1],
        data=[2],
        absolute_uncertainties=[1],
        analysis="norm2",
    )
    normal3 = spey.get_backend("default.normal")(
        signal_yields=[1],
        background_yields=[2],
        data=[2],
        absolute_uncertainties=[1],
        analysis="norm3",
    )
    combined = UnCorrStatisticsCombiner(normal1, normal2, normal3)
    combined_cls = combined.exclusion_confidence_level()[0]

    assert np.isclose(multivar_norm_cls, combined_cls), "Combined CLs is wrong"
