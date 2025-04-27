import numpy as np
from typing import List, Tuple


def mv_normal_aux_distribution(sample: np.ndarray,
                               aux_parameters: Tuple[List[np.ndarray]],
                               source_index) -> np.ndarray:
    """
    Compute the log‑pdf of multivariate normal distributions for each source.

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
    inv_cov = np.linalg.inv(cov_matrices)  # shape: (n, d, d)
    # Compute log determinant for each covariance matrix.
    _, logdet = np.linalg.slogdet(cov_matrices)  # shape: (n,)
    # Compute the Mahalanobis term over distributions and sources.
    mahal = np.einsum('nsi, nij, nsj -> ns', diff, inv_cov, diff)  # shape: (n, num_sources)
    logpdf = -0.5 * (d * np.log(2 * np.pi) + logdet[:, None] + mahal)  # shape: (n, num_sources)

    # Return result: squeeze out axis if only one distribution was requested.
    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf


def uni_normal_aux_distribution_single_parameter(sample: np.ndarray,
                                                 aux_parameters: Tuple[List[np.ndarray]],
                                                 source_index,
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
    """
    values = sample[:, single_parameter]  # shape: (num_sources,)
    means, stds = aux_parameters
    source_indices = np.atleast_1d(source_index)
    means = np.array(means)[source_indices]  # shape: (n,)
    stds = np.array(stds)[source_indices]      # shape: (n,)

    # Compute logpdf in a vectorized way. Broadcasting: (n,1) vs (num_sources,)
    logpdf = (
        -0.5 * np.log(2 * np.pi) - np.log(stds)[:, None] - 0.5 * (((values - means[:, None]) / stds[:, None]) ** 2)
    )  # shape: (n, num_sources)

    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf
