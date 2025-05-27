"""Various helper functions"""

from typing import List, Dict

import numpy as np


def correlation_to_covariance(
    correlation_matrix: np.ndarray, standard_deviations: np.ndarray
) -> np.ndarray:
    """
    Convert correlation matrix into covariance matrix

    Args:
        correlation_matrix (``np.ndarray``): a real NxN matrix
        standard_deviations (``np.ndarray``): a real N-dimensional
          vector representing standard deviations.

    Returns:
        ``np.ndarray``:
        Covariance matrix
    """
    sigma = np.diag(standard_deviations)
    return sigma @ correlation_matrix @ sigma


def covariance_to_correlation(covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix into correlation matrix.

    Args:
        covariance_matrix (``np.ndarray``): a real NxN matrix

    Returns:
        ``np.ndarray``:
        Correlation matrix
    """
    sigma = np.linalg.inv(np.diag(np.sqrt(np.diag(covariance_matrix))))
    return sigma @ covariance_matrix @ sigma


def merge_correlated_bins(
    background_yields: np.ndarray,
    data: np.ndarray,
    covariance_matrix: np.ndarray,
    merge_groups: List[List[int]],
    signal_yields: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Merge correlated bins in a dataset.
    This function takes a set of background yields, data, and a covariance matrix,
    and merges specified groups of bins into single bins. The resulting background yields,
    data, and covariance matrix are returned in a dictionary.
    The merging is done by summing the yields and data for the specified groups,
    and summing the covariance matrix entries for the merged bins.

    .. versionadded:: 0.2.4

    Args:
        background_yields (``np.ndarray``): background yields for each bin.
        data (``np.ndarray``): observed data for each bin.
        covariance_matrix (``np.ndarray``): covariance matrix for the bins.
        merge_groups (``list[list[int]]``): indices of bins to merge.
        signal_yields (``np.ndarray``, default ``None``): signal yields for each bin.
            If provided, these will also be merged according to the specified groups.
    Raises:
        AssertionError:
          * If the lengths of the input arrays do not match or if the covariance matrix is not square.
          * If there are overlapping indices in ``merge_groups``.
          * If the lengths of ``data``, ``background_yields``, and ``signal_yields`` do not match.
          * If the covariance matrix is not square.
          * If the lengths of ``data``, ``background_yields``, and ``signal_yields`` do not match.

    Returns:
        ``dict[str, np.ndarray]``:
        A dictionary containing the merged background yields, data, and covariance matrix
        (and signal if included).
    """

    assert len(data) == len(
        background_yields
    ), "Data and background yields must have the same length."
    assert len(data) == len(
        covariance_matrix
    ), "Data and covariance matrix must have the same length."
    assert (
        len(data) == len(signal_yields) if signal_yields is not None else True
    ), "Data and signal yields must have the same length."
    assert len(covariance_matrix) == len(
        covariance_matrix[0]
    ), "Covariance matrix must be square."

    N = covariance_matrix.shape[0]

    # Flatten merge_groups and find missing indices
    merged_indices = sorted({i for g in merge_groups for i in g})
    assert len(set(merged_indices)) == len(
        merged_indices
    ), "Overlapping indices in merge_groups"

    all_indices = set(range(N))
    unmerged = sorted(all_indices - set(merged_indices))

    # Build full list of groups: merged + singletons
    full_groups = merge_groups + [[i] for i in unmerged]

    M = len(full_groups)
    new_cov = np.zeros((M, M))
    new_background_yields = np.zeros(M)
    new_data = np.zeros(M)
    new_signal_yields = np.zeros(M)

    for i, Gi in enumerate(full_groups):
        new_background_yields[i] = np.sum(background_yields[Gi])
        new_data[i] = np.sum(data[Gi])
        if signal_yields is not None:
            new_signal_yields[i] = np.sum(signal_yields[Gi])
        for j, Gj in enumerate(full_groups):
            new_cov[i, j] = np.sum(covariance_matrix[np.ix_(Gi, Gj)])

    result = {
        "background_yields": new_background_yields,
        "data": new_data,
        "covariance_matrix": new_cov,
    }
    if signal_yields is not None:
        result["signal_yields"] = new_signal_yields

    return result
