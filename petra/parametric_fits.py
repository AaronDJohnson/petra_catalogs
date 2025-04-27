import pandas as pd
import numpy as np
from functools import partial
from typing import Callable
from petra.utils import find_prob_in_model


def create_parametric_fit(fit_function, single_parameter=None):
    """
    Wrap a fitting function to unify its interface.

    Parameters
    ----------
    fit_function : callable
        A function with signature
        `(chain, max_num_sources[, fit_parameter])` that returns fit parameters.
    single_parameter : int, optional
        If provided, fixes the parameter index for single‐parameter fit functions.

    Returns
    -------
    parametric_fit : callable
        A function with signature `(chain, max_num_sources)`
        that calls `fit_function` and returns its output.

    Examples
    --------
    >>> from petra.parametric_fits import create_parametric_fit, uni_normal_fit_single_parameter
    >>> fit = create_parametric_fit(uni_normal_fit_single_parameter, single_parameter=2)
    >>> chain = np.random.randn(100, 3, 5)
    >>> means, stds = fit(chain, max_num_sources=3)
    >>> means.shape, stds.shape
    ((3,), (3,))
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
    Fit a multivariate normal distribution to each source.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Posterior samples.
    max_num_sources : int
        Number of sources to fit.

    Returns
    -------
    means : ndarray, shape (max_num_sources, n_params)
        Mean vectors for each source.
    cov_matrices : ndarray, shape (max_num_sources, n_params, n_params)
        Covariance matrices for each source.

    Examples
    --------
    >>> from petra.parametric_fits import mv_normal_fit
    >>> chain = np.random.randn(500, 4, 2)
    >>> means, covs = mv_normal_fit(chain, max_num_sources=4)
    >>> means.shape, covs.shape
    ((4, 2), (4, 2, 2))
    """

    means = []
    cov_matrices = []
    for source in range(max_num_sources):
        sample_i = chain[:, source, :]  # shape: (num_samples, num_params)
        valid = ~np.isnan(sample_i).any(axis=1)
        valid_samples = sample_i[valid]
        if valid_samples.shape[0] < 8:
            print('Fewer than 8 values in source index {}. Appending normal distribution fit to all entries.'.format(source))
            df_all = pd.DataFrame(chain.reshape(-1, chain.shape[2]))
            means.append(np.array(df_all.dropna().mean()))
            cov_matrices.append(np.array(df_all.dropna().cov()))
            continue
        df = pd.DataFrame(chain[:, source, :])
        means.append(np.array(df.dropna().mean()))
        cov_matrices.append(np.array(df.dropna().cov()))
    return np.array(means), np.array(cov_matrices)


def uni_normal_fit_single_parameter(chain, max_num_sources, fit_parameter):
    """
    Fit a univariate normal distribution to one parameter for each source.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Posterior samples.
    max_num_sources : int
        Number of sources to fit.
    fit_parameter : int
        Index of the parameter to fit.

    Returns
    -------
    means : ndarray, shape (max_num_sources,)
        Means for each source.
    stds : ndarray, shape (max_num_sources,)
        Standard deviations for each source.

    Examples
    --------
    >>> from petra.parametric_fits import uni_normal_fit_single_parameter
    >>> chain = np.random.randn(200, 3, 5)
    >>> means, stds = uni_normal_fit_single_parameter(chain, max_num_sources=3, fit_parameter=1)
    >>> means.shape, stds.shape
    ((3,), (3,))
    """
    means = []
    stds = []
    for source in range(max_num_sources):
        sample_i = chain[:, source, fit_parameter]  # shape: (num_samples, num_params)
        valid = np.where(~np.isnan(sample_i))
        valid_samples = sample_i[valid]
        if valid_samples.shape[0] < 8:
            print(f'Fewer than 8 values in source index {source}. Appending normal distribution fit to all entries.')
            df_all = pd.DataFrame(chain.reshape(-1))
            means.append(df_all.dropna().mean().to_numpy(dtype=np.float64)[0])
            stds.append(df_all.dropna().std().to_numpy(dtype=np.float64)[0])
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
    Compute parametric fit and inclusion probabilities.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Object with `.get_chain()` method returning an ndarray.
    max_num_sources : int
        Number of sources to consider.
    parametric_fit_function : callable
        Function `(chain, max_num_sources) -> aux_params`.
    eps : float, optional
        Threshold for inclusion probability (default 1e-2).

    Returns
    -------
    aux_params : tuple
        Output of `parametric_fit_function`, e.g. (means, covs) or (means, stds).
    prob_in_model : ndarray, shape (max_num_sources,)
        Inclusion probabilities.

    Examples
    --------
    >>> from petra.parametric_fits import update_parametric_fit_and_prob_in_model, mv_normal_fit
    >>> from petra.posterior_chain import PosteriorChain
    >>> pc = PosteriorChain(np.random.randn(150, 4, 2), 4, 2, True, None, {})
    >>> aux, probs = update_parametric_fit_and_prob_in_model(pc, 4, mv_normal_fit)
    >>> aux[0].shape, probs.shape
    ((4, 2), (4,))
    """
    aux_params = parametric_fit_function(posterior_chain.get_chain(), max_num_sources)
    prob_in_model = find_prob_in_model(posterior_chain.get_chain(), max_num_sources, eps=eps)
    return aux_params, prob_in_model
