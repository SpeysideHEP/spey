"""Various helper functions"""

from typing import Dict, List

import numpy as np


def symmetrise_matrix(matrix: np.ndarray, name: str = "covariance matrix") -> np.ndarray:
    r"""
    Return the symmetric part of a square matrix.

    A covariance or correlation matrix is symmetric by construction. An
    asymmetric input is therefore ill-defined, and silently computing with it
    is worse than it looks: the quadratic form of a multivariate normal only
    sees the symmetric part, but its inverse and its determinant do not, so the
    resulting likelihood is neither the one implied by
    :math:`\Sigma` nor the one implied by :math:`\Sigma^T`.

    This function replaces such an input with

    .. math::

        \Sigma_{\rm sym} = \frac{1}{2}\left(\Sigma + \Sigma^T \right)\ ,

    which is the closest symmetric matrix to :math:`\Sigma` in the Frobenius
    norm, and therefore the least-assumption reading of an input whose two
    off-diagonal entries disagree. A warning is issued whenever the input has
    to be modified.

    .. versionadded:: 0.2.8

    Args:
        matrix (``np.ndarray``): a square matrix.
        name (``str``, default ``"covariance matrix"``): how to refer to the
          matrix in the warning message.

    Raises:
        ``InvalidInput``: if ``matrix`` is not square.

    Returns:
        ``np.ndarray``:
        The symmetric part of ``matrix``, or ``matrix`` itself when it is
        already symmetric.

    Example:

        .. code-block:: python3

            >>> import numpy as np
            >>> from spey.helper_functions import symmetrise_matrix
            >>> symmetrise_matrix(np.array([[144.0, 13.0], [25.0, 256.0]]))
            array([[144.,  19.],
                   [ 19., 256.]])

    .. note::

        Symmetry is necessary but not sufficient: the function does not check
        that the result is positive semi-definite.
    """
    from spey.system.exceptions import InvalidInput  # avoids a circular import

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InvalidInput(
            f"The {name} has to be a square matrix, got shape {matrix.shape}."
        )
    if np.allclose(matrix, matrix.T):
        return matrix

    from spey import log_once  # avoids a circular import

    symmetric = 0.5 * (matrix + matrix.T)
    log_once(
        f"The {name} is not symmetric; using its symmetric part "
        "(cov + cov.T) / 2 instead. Please check the input.",
        log_type="warning",
    )
    return symmetric


def ensure_positive_definite(
    matrix: np.ndarray, name: str = "covariance matrix", tolerance: float = 1e-8
) -> np.ndarray:
    r"""
    Validate that a covariance or correlation matrix defines a normal distribution.

    The matrix is first replaced by its symmetric part, see
    :func:`~spey.helper_functions.symmetrise_matrix`, and is then required to be
    **positive definite**. Positive *semi*-definiteness alone is not enough: a
    singular matrix has a zero eigenvalue, so
    :math:`\det\Sigma = 0` and :math:`\Sigma^{-1}` does not exist, and the
    multivariate normal

    .. math::

        \mathcal{N}(x\vert\mu,\Sigma)\propto
        \frac{1}{\sqrt{\det\Sigma}}
        \exp\left[-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right]

    has no density. A negative eigenvalue is worse still: the exponent is then
    unbounded above, so the "likelihood" can be made arbitrarily large.

    The test is whether a Cholesky decomposition exists. That is the operative
    criterion rather than a convenience: it holds exactly when the inverse and
    the determinant the likelihood needs can be computed, and it costs about a
    third of an eigendecomposition. Eigenvalues are only computed when the
    decomposition fails, to decide whether the matrix is merely badly
    conditioned and to report what is wrong.

    .. versionadded:: 0.2.8

    Args:
        matrix (``np.ndarray``): a square covariance or correlation matrix.
        name (``str``, default ``"covariance matrix"``): how to refer to the
          matrix in the error message.
        tolerance (``float``, default ``1e-8``): only consulted when the Cholesky
          decomposition fails. The matrix is still accepted if every eigenvalue
          exceeds ``tolerance * max(1, largest eigenvalue)``, which rescues a
          matrix that is positive definite but too badly conditioned for the
          decomposition to go through.

    Raises:
        ``InvalidInput``: if ``matrix`` is not square, or is not positive
          definite.

    Returns:
        ``np.ndarray``:
        The symmetric part of ``matrix``, validated.

    Example:

        .. code-block:: python3

            >>> import numpy as np
            >>> from spey.helper_functions import ensure_positive_definite
            >>> ensure_positive_definite(np.array([[4.0, 2.4], [2.4, 16.0]]))
            array([[ 4. ,  2.4],
                   [ 2.4, 16. ]])

    .. note::

        A ``callable`` covariance matrix cannot be validated ahead of time and
        is left to the user; it has to return a positive definite matrix for
        every parameter point the optimiser visits.

    .. note::

        The condition number is not checked. A matrix can be positive definite
        and still be ill-conditioned enough to make the likelihood numerically
        unreliable.
    """
    from spey.system.exceptions import InvalidInput  # avoids a circular import

    matrix = symmetrise_matrix(matrix, name)
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        pass

    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest = float(np.min(eigenvalues))
    threshold = tolerance * max(1.0, float(np.max(eigenvalues)))
    if smallest > threshold:
        # Cholesky can fail on a badly conditioned but still positive definite
        # matrix; the eigenvalues are the authority.
        return matrix

    if smallest < -threshold:
        detail = (
            f"it has a negative eigenvalue ({smallest:.6g}), so the exponent of the "
            "normal distribution is unbounded above"
        )
    else:
        detail = (
            f"it is singular (smallest eigenvalue {smallest:.6g}), so it has no "
            "inverse and the normal distribution has no density"
        )
    raise InvalidInput(
        f"The {name} is not positive definite: {detail}. Eigenvalues: "
        f"{np.array2string(eigenvalues, precision=4)}. A rank-deficient matrix "
        "usually means two or more bins are perfectly correlated; they can be "
        "merged with spey.helper_functions.merge_correlated_bins."
    )


def correlation_to_covariance(
    correlation_matrix: np.ndarray, standard_deviations: np.ndarray
) -> np.ndarray:
    r"""
    Convert correlation matrix into covariance matrix.

    Computes :math:`\Sigma_{ij} = \sigma_i \rho_{ij} \sigma_j` where
    :math:`\sigma_i` are the per-bin standard deviations and :math:`\rho_{ij}`
    is the input correlation matrix.

    Args:
        correlation_matrix (``np.ndarray``): A real NxN correlation matrix
          :math:`\rho` (diagonal entries equal to one, off-diagonals in
          ``[-1, 1]``).
        standard_deviations (``np.ndarray``): A real N-dimensional vector
          of per-bin standard deviations :math:`\sigma_i`.

    Returns:
        ``np.ndarray``:
        Covariance matrix :math:`\Sigma` of shape ``(N, N)``.

    Example:

        .. code-block:: python3

            >>> import numpy as np
            >>> from spey.helper_functions import correlation_to_covariance
            >>> rho = np.array([[1.0, 0.3], [0.3, 1.0]])
            >>> sigma = np.array([2.0, 4.0])
            >>> correlation_to_covariance(rho, sigma)
            array([[ 4. ,  2.4],
                   [ 2.4, 16. ]])
    """
    correlation_matrix = symmetrise_matrix(correlation_matrix, "correlation matrix")
    sigma = np.diag(standard_deviations)
    return sigma @ correlation_matrix @ sigma


def covariance_to_correlation(covariance_matrix: np.ndarray) -> np.ndarray:
    r"""
    Convert covariance matrix into correlation matrix.

    Computes :math:`\rho_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)` where
    :math:`\sigma_i = \sqrt{\Sigma_{ii}}` are the per-bin standard deviations.

    Args:
        covariance_matrix (``np.ndarray``): A real NxN covariance matrix
          :math:`\Sigma` with strictly positive diagonal entries.

    Returns:
        ``np.ndarray``:
        Correlation matrix :math:`\rho` of shape ``(N, N)``.

    Example:

        .. code-block:: python3

            >>> import numpy as np
            >>> from spey.helper_functions import covariance_to_correlation
            >>> cov = np.array([[4.0, 2.4], [2.4, 16.0]])
            >>> covariance_to_correlation(cov)
            array([[1. , 0.3],
                   [0.3, 1. ]])
    """
    covariance_matrix = symmetrise_matrix(covariance_matrix)
    sigma_inv = np.diag(1.0 / np.sqrt(np.diag(covariance_matrix)))
    return sigma_inv @ covariance_matrix @ sigma_inv


def merge_correlated_bins(
    background_yields: np.ndarray,
    data: np.ndarray,
    covariance_matrix: np.ndarray,
    merge_groups: List[List[int]],
    signal_yields: np.ndarray = None,
    return_group_indices: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Merge correlated bins in a histogram/cutflow.

    This function takes a set of background yields, data, and a covariance matrix,
    and merges specified groups of bins into single bins. The resulting background yields,
    data, and covariance matrix are returned in a dictionary. The merging is done by summing
    the yields and data for the specified groups, and summing the covariance matrix entries
    for the merged bins.

    .. versionadded:: 0.2.4

    **Example:**

    .. code-block:: python3

        >>> from spey.helper_functions import merge_correlated_bins
        >>> import numpy as np
        >>> background_yields = np.array([10, 20, 30, 40])
        >>> data = np.array([12, 22, 32, 42])
        >>> covariance_matrix = np.array(
        ...   [[4, 1, 0.5, 0.2],
        ...    [1, 3, 0.3, 0.1],
        ...    [0.5, 0.3, 5, 0.2],
        ...    [0.2, 0.1, 0.2, 4]]
        >>> )
        >>> merge_groups = [[0, 1], [2, 3]]
        >>> result = merge_correlated_bins(
        ...     background_yields=background_yields,
        ...     data=data,
        ...     covariance_matrix=covariance_matrix,
        ...     merge_groups=merge_groups
        ... )
        >>> print(result)
        >>> # {
        ... #    'background_yields': array([30., 70.]),
        ... #    'data': array([34., 74.]),
        ... #    'covariance_matrix': array([[ 9. ,  1.1],
        ... #                                [ 1.1,  9.4]])
        ... # }


    The resulting ``result`` dictionary will contain:
        - ``background_yields``: Merged background yields.
        - ``data``: Merged data.
        - ``covariance_matrix``: Merged covariance matrix.

    .. note::

        The function assumes that the input arrays are 1-dimensional and that the covariance
        matrix is square. It also checks for overlapping indices in ``merge_groups`` and raises
        an assertion error if any are found.

    .. warning::

        The function does not check for the validity of the covariance matrix (e.g., positive
        definiteness). It is assumed that the input covariance matrix is valid for the given
        background yields and data.


    Args:
        background_yields (``np.ndarray``): background yields for each bin.
        data (``np.ndarray``): observed data for each bin.
        covariance_matrix (``np.ndarray``): covariance matrix for the bins.
        merge_groups (``list[list[int]]``): indices of bins to merge.
        signal_yields (``np.ndarray``, default ``None``): signal yields for each bin.
            If provided, these will also be merged according to the specified groups.
        return_group_indices (``bool``, default ``False``): if ``True``, the function will
          return the indices of the merged groups in the output dictionary. This is to help
          user to keep track of which bins were merged together and how the bins are reordered.
          New signal yields can be formed by running the following code:

          .. code-block:: python

                >>> new_signal_yields = [sum(np.array(signal_yields)[Gi]) for Gi in output["group_indices"]]

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
    new_signal_yields = np.zeros(M) if signal_yields is not None else None

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
    if return_group_indices:
        result["group_indices"] = full_groups

    return result
