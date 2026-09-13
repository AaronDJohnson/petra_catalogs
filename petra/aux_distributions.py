"""
Evaluate the fitted auxiliary distributions of a relabeling pass.

Second of the four stages (:mod:`~petra.parametric_fits` ->
:mod:`~petra.aux_distributions` -> :mod:`~petra.cost_matrix` ->
:mod:`~petra.relabel`).  Each function here takes the ``(locations, scales)``
produced by the matching fit in :mod:`petra.parametric_fits` and returns
log-densities, which :mod:`petra.cost_matrix` turns into the reward matrix the
Hungarian solver maximizes.

Conventions
-----------
* `sample` is a **single** posterior sample of shape
  ``(num_sources, num_params_per_source)``: row *j* is the parameter vector
  currently sitting in slot *j*.
* `source_index` selects which fitted distributions (i.e. which *labels*) to
  evaluate.  The returned array is indexed ``[label, slot]`` -- labels down the
  rows, slots across the columns -- which is the orientation
  :mod:`petra.cost_matrix` and :mod:`petra.relabel` assume.  A scalar
  `source_index` collapses the leading axis, giving shape ``(num_sources,)``.
* Values are natural-log probability *densities*, so they may be positive; only
  differences between them are meaningful.
* ``NaN`` in `sample` means the source is absent from that sample.  It is
  deliberately **not** handled here: it propagates into the returned array, and
  :mod:`petra.cost_matrix` replaces every NaN entry with the log-probability
  that the corresponding label is out of the model.

Degenerate fits fail loudly rather than returning NaN.  A non-positive-definite
covariance and a non-positive standard deviation both raise, because a NaN row
reaching :mod:`petra.cost_matrix` becomes a *constant* row, and a constant row is
invariant under ``scipy.optimize.linear_sum_assignment`` -- that label would then
be assigned arbitrarily, with nothing in the output to show it happened.
"""

import numpy as np
from typing import List, Sequence, Tuple, Union

#: Anything accepted as `source_index`: one label, or several.
SourceIndex = Union[int, Sequence[int], np.ndarray]


def _is_positive_definite(matrix: np.ndarray) -> bool:
    """Return True if `matrix` admits a Cholesky factorization."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def mv_normal_aux_distribution(sample: np.ndarray,
                               aux_parameters: Tuple[List[np.ndarray], List[np.ndarray]],
                               source_index: SourceIndex) -> np.ndarray:
    """
    Compute the log-pdf of multivariate normal distributions for each source.

    Parameters
    ----------
    sample : ndarray, shape (num_sources, num_params_per_source)
        Array of parameter values for each source.
    aux_parameters : tuple of lists
        Tuple ``(means, cov_matrices)`` where
        - means : list of ndarray, each of shape (num_params_per_source,)
        - cov_matrices : list of ndarray, each of shape (num_params_per_source, num_params_per_source)
    source_index : int or array-like
        Index or indices of which fitted distributions to evaluate.

    Returns
    -------
    logpdf : ndarray
        If `source_index` is a scalar, returns shape `(num_sources,)`.
        Otherwise returns shape `(len(source_index), num_sources)`.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import mv_normal_aux_distribution
    >>> # 5 samples, 3 parameters per source, 2 fitted distributions
    >>> sample = np.random.randn(5, 3)
    >>> means = [np.zeros(3), np.ones(3)]
    >>> covs = [np.eye(3), 2*np.eye(3)]
    >>> # evaluate both distributions
    >>> logpdf = mv_normal_aux_distribution(sample, (means, covs), [0, 1])
    >>> logpdf.shape
    (2, 5)
    >>> # single distribution
    >>> logpdf0 = mv_normal_aux_distribution(sample, (means, covs), 0)
    >>> logpdf0.shape
    (5,)
    """
    means, cov_matrices = aux_parameters
    # Ensure source_index is array-like.
    source_indices = np.atleast_1d(source_index)
    means = np.array(means)[source_indices]         # shape: (n, d)
    cov_matrices = np.array(cov_matrices)[source_indices]  # shape: (n, d, d)

    num_sources, d = sample.shape
    # Broadcast sample to match the number of distributions.
    diff = sample[np.newaxis, :, :] - means[:, np.newaxis, :]  # shape: (n, num_sources, d)

    # Cholesky rather than an explicit inverse. `np.linalg.inv` does NOT raise on a merely
    # ill-conditioned matrix: for a rank-deficient covariance it returns entries of order
    # 1/eps, and `slogdet` returns a finite but meaningless value, so the intended
    # LinAlgError fallback never fired and the bad numbers flowed straight into the cost
    # matrix. Cholesky fails loudly on exactly the matrices we cannot use.
    try:
        chol = np.linalg.cholesky(cov_matrices)  # shape: (n, d, d), lower triangular
    except np.linalg.LinAlgError:
        bad = [int(source_indices[i]) for i in range(cov_matrices.shape[0])
               if not _is_positive_definite(cov_matrices[i])]
        raise np.linalg.LinAlgError(
            f"Covariance matrices for source indices {bad} are not positive definite, so the "
            f"multivariate normal log-pdf is undefined. This usually means the source was fit "
            f"from too few retained samples, or a parameter is constant across them. Check the "
            f"fit that produced these covariances (see petra.parametric_fits.mv_normal_fit, "
            f"which regularizes with COVARIANCE_RIDGE)."
        )

    # log(det(cov)) = 2 * sum(log(diag(L)))
    logdet = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)  # shape: (n,)

    # Mahalanobis term: diff^T inv(cov) diff == || solve(L, diff) ||^2
    # solve expects the d axis second-to-last, so put sources on the trailing axis.
    solved = np.linalg.solve(chol, np.swapaxes(diff, 1, 2))  # shape: (n, d, num_sources)
    mahal = np.sum(solved * solved, axis=1)  # shape: (n, num_sources)

    logpdf = -0.5 * (d * np.log(2 * np.pi) + logdet[:, None] + mahal)  # shape: (n, num_sources)

    # Return result: squeeze out axis if only one distribution was requested.
    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf


def uni_normal_aux_distribution_single_parameter(sample: np.ndarray,
                                                 aux_parameters: Tuple[List[np.ndarray], List[np.ndarray]],
                                                 source_index: SourceIndex,
                                                 single_parameter: int) -> np.ndarray:
    """
    Compute the log-pdf of univariate normal distributions on one parameter.

    Parameters
    ----------
    sample : ndarray, shape (num_sources, num_params_per_source)
        Array of parameter values for each source.
    aux_parameters : tuple of lists
        Tuple ``(means, stds)`` where each is a list of length n_distributions.
    source_index : int or array-like
        Index or indices of which fitted distributions to evaluate.
    single_parameter : int
        Index of the parameter dimension to evaluate.

    Returns
    -------
    logpdf : ndarray
        If `source_index` is a scalar, returns shape `(num_sources,)`.
        Otherwise returns shape `(len(source_index), num_sources)`.

    Raises
    ------
    ValueError
        If any requested standard deviation is not strictly positive and finite.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import uni_normal_aux_distribution_single_parameter
    >>> # 4 samples, 5 parameters per source, 3 fitted normals
    >>> sample = np.random.randn(4, 5)
    >>> means = [0.0, 1.0, -1.0]
    >>> stds = [1.0, 0.5, 2.0]
    >>> # evaluate distributions 0 and 2 on parameter index 3
    >>> logpdf = uni_normal_aux_distribution_single_parameter(
    ...     sample, (means, stds), [0, 2], single_parameter=3)
    >>> logpdf.shape
    (2, 4)
    >>> # single distribution
    >>> logpdf1 = uni_normal_aux_distribution_single_parameter(
    ...     sample, (means, stds), 1, single_parameter=3)
    >>> logpdf1.shape
    (4,)

    A degenerate scale is rejected instead of silently producing a NaN row:

    >>> uni_normal_aux_distribution_single_parameter(  # doctest: +ELLIPSIS
    ...     np.zeros((2, 1)), ([0.0, 1.0], [1.0, 0.0]), [0, 1], single_parameter=0)
    Traceback (most recent call last):
        ...
    ValueError: Standard deviations for source indices [1] are not strictly positive...
    """
    values = sample[:, single_parameter]  # shape: (num_sources,)
    means, stds = aux_parameters
    source_indices = np.atleast_1d(source_index)
    means = np.array(means)[source_indices]  # shape: (n,)
    stds = np.array(stds)[source_indices]      # shape: (n,)

    # Fail loudly, mirroring the Cholesky branch of `mv_normal_aux_distribution`. A
    # non-positive std divides by zero, giving a NaN row that `petra.cost_matrix`
    # replaces with a single constant; a row-constant cost row is invariant under
    # `linear_sum_assignment`, so the label would be assigned arbitrarily and nothing
    # downstream would record that it happened.
    bad_scale = ~(stds > 0.0) | ~np.isfinite(stds)
    if np.any(bad_scale):
        bad = [int(i) for i in np.atleast_1d(source_indices)[bad_scale]]
        raise ValueError(
            f"Standard deviations for source indices {bad} are not strictly positive "
            f"and finite, so the univariate normal log-pdf is undefined. This usually "
            f"means the parameter is constant across the samples retained for that "
            f"source. Check the fit that produced these scales (see "
            f"petra.parametric_fits.uni_normal_fit_single_parameter, which floors them "
            f"with COVARIANCE_RIDGE)."
        )

    # Compute logpdf in a vectorized way. Broadcasting: (n,1) vs (num_sources,)
    logpdf = (
        -0.5 * np.log(2 * np.pi) - np.log(stds)[:, None] - 0.5 * (((values - means[:, None]) / stds[:, None]) ** 2)
    )  # shape: (n, num_sources)

    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf
