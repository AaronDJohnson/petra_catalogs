import numpy as np
from typing import List, Tuple


def mv_normal_aux_distribution(sample: np.ndarray,
                               aux_parameters: Tuple[List[np.ndarray]],
                               source_index) -> np.ndarray:
    """
    Vectorized: Compute the logpdf of a multivariate normal distribution evaluated at each source in sample.
    Accepts source_index as an int or array-like.

    Parameters:
        sample: numpy array of shape (num_sources, num_params_per_source)
        aux_parameters: Tuple of lists (means, cov_matrices) where each is of length num_distributions.
        source_index: int or array-like indices specifying which distributions to evaluate.

    Returns:
        If source_index is a scalar, returns an array of shape (num_sources,).
        Otherwise, returns an array of shape (len(source_index), num_sources).
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
    Vectorized: Compute the logpdf of a univariate normal distribution on a single parameter for each source.
    Accepts source_index as an int or array-like.
    
    Parameters:
        sample: numpy array of shape (num_sources, num_params_per_source)
        aux_parameters: Tuple of lists (means, stds) for the univariate case.
        source_index: int or array-like indices specifying which distributions to evaluate.
        single_parameter: the parameter index to process.
    
    Returns:
        If source_index is a scalar, returns an array of shape (num_sources,).
        Otherwise, returns an array of shape (len(source_index), num_sources).
    """
    values = sample[:, single_parameter]  # shape: (num_sources,)
    means, stds = aux_parameters
    source_indices = np.atleast_1d(source_index)
    means = np.array(means)[source_indices]  # shape: (n,)
    stds = np.array(stds)[source_indices]      # shape: (n,)
    
    # Compute logpdf in a vectorized way. Broadcasting: (n,1) vs (num_sources,)
    logpdf = (
        -0.5 * np.log(2 * np.pi)
        - np.log(stds)[:, None]
        - 0.5 * (((values - means[:, None]) / stds[:, None]) ** 2)
    )  # shape: (n, num_sources)
    
    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf
