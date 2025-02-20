import pandas as pd
import numpy as np
from functools import partial
from typing import Callable
from petra.utils import find_prob_in_model


def create_parametric_fit(fit_function, single_parameter=None):
    """
    Create a function to fit a parametric distribution to the chain of samples. Extensible to any parametric distribution.

    Parameters
    ----------
    fit_function: Callable function with the signature (chain, max_num_sources, single_parameter (optional))
        The function to fit the parametric distribution to the chain of samples

    Return
    ------
    parametric_fit: Callable function with the signature (chain, max_num_sources)
        The function to fit the parametric distribution to the chain of samples
    """

    if single_parameter is not None:
        fit_function = partial(fit_function, fit_parameter=single_parameter)

    def parametric_fit(chain, max_num_sources):
        """
        Fit a parametric distribution to the chain of samples .

        Parameters
        ----------
        chain: A numpy array of shape (num_samples, num_entries, num_params_per_source)
        max_num_sources: The maximum number of sources to consider in the catalog

        Return
        ------
        fit: The fit to the chain of samples
        """
        return fit_function(chain, max_num_sources)

    return parametric_fit


def mv_normal_fit(chain, max_num_sources):
    """
    Get a MV Gaussian fit (mu, cov) to use in the cost matrix. This function takes in a chain of samples and returns the
    mean and covariance matrix. NaN values are removed by pandas before the calculation.

    Parameters
    ----------
    chain: A numpy array of shape (num_samples, num_entries, num_params_per_source)
    max_num_sources: The maximum number of sources to consider in the catalog

    Return
    -------
    means: A list of length max_num_sources of numpy arrays of shape (num_params_per_source,)
    cov_matrices: A list of length max_num_sources of numpy arrays of shape (num_params_per_source, num_params_per_source)
    """

    means = []
    cov_matrices = []
    for source in range(max_num_sources):
        sample_i = chain[:, source, :]  # shape: (num_samples, num_params)
        valid = ~np.isnan(sample_i).any(axis=1)
        valid_samples = sample_i[valid]
        if valid_samples.shape[0] < 8:
            print('Fewer than 8 values in source index {}. Appending a standard normal distribution.'.format(source))
            means.append(np.zeros(chain.shape[2]))
            cov_matrices.append(np.identity(chain.shape[2]))
            continue
        df = pd.DataFrame(chain[:, source, :])
        means.append(np.array(df.dropna().mean()))
        cov_matrices.append(np.array(df.dropna().cov()))
    return np.array(means), np.array(cov_matrices)


def uni_normal_fit_single_parameter(chain, max_num_sources, fit_parameter):
    """
    Get a univariate Gaussian fit (mu, std) to use in the cost matrix. This function takes in a chain of samples and returns the
    mean and standard deviation. NaN values are removed by pandas before the calculation.

    Parameters
    ----------
    chain: A numpy array of shape (num_samples, num_entries, num_params_per_source)
    max_num_sources: The maximum number of sources to consider in the catalog
    fit_parameter: The index of the parameter to fit

    Return
    ------
    means: A list of length max_num_sources of numpy arrays of shape (num_params_per_source,)
    stds: A list of length max_num_sources of numpy arrays of shape (num_params_per_source,)
    """
    means = []
    stds = []
    for source in range(max_num_sources):
        sample_i = chain[:, source, fit_parameter]  # shape: (num_samples, num_params)
        valid = np.where(~np.isnan(sample_i))
        valid_samples = sample_i[valid]
        if valid_samples.shape[0] < 8:
            print('Fewer than 8 values in source index {}. Appending a standard normal distribution.'.format(source))
            means.append(0)
            stds.append(1)
            continue
        else:
            df = pd.DataFrame(chain[:, source, fit_parameter])
            mean = np.nanmean(df)
            std = np.nanstd(df)
            means.append(mean)
            stds.append(std)
    return np.array(means), np.array(stds)


def update_parametric_fit_and_prob_in_model(posterior_chain, max_num_sources, parametric_fit_function: Callable, eps=1e-2):
    """
    Update the parametric fit and the probability of each source being in the model.

    Parameters
    ----------
    chain: A numpy array of shape (num_samples, num_entries, num_params_per_source)
    max_num_sources: The maximum number of sources to consider in the catalog
    fit_function: A function that fits a parametric distribution to the chain of samples
    prob_in_model: A numpy array of shape (max_num_sources,) that contains the probability of each source being in the model

    Return
    ------
    parametric_fit: The fit to the chain of samples
    prob_in_model: A numpy array of shape (max_num_sources,)
    """
    aux_params = parametric_fit_function(posterior_chain.get_chain(), max_num_sources)
    prob_in_model = find_prob_in_model(posterior_chain.get_chain(), max_num_sources, eps=eps)
    return aux_params, prob_in_model
