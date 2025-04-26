import pandas as pd
import numpy as np
from functools import partial
from typing import Callable
from petra.utils import find_prob_in_model


def create_parametric_fit(fit_function, single_parameter=None):
    """
    Wrap a fit function to unify its interface.

    Parameters
    ----------
    fit_function : callable
        Function with signature
        `(chain, max_num_sources[, single_parameter])` that computes a parametric fit.
    single_parameter : int, optional
        If provided, fixes the index of the parameter for single‐parameter fits.

    Returns
    -------
    parametric_fit : callable
        A function with signature `(chain, max_num_sources)` that returns the fit.

    Examples
    --------
    >>> from petra.parametric_fits import create_parametric_fit, uni_normal_fit_single_parameter
    >>> fit = create_parametric_fit(uni_normal_fit_single_parameter, single_parameter=0)
    >>> chain = np.random.randn(100, 2, 3)
    >>> means, stds = fit(chain, max_num_sources=2)
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
    Fit a multivariate normal distribution to each source across samples.

    Parameters
    ----------
    chain : ndarray, shape (num_samples, num_entries, num_params_per_source)
        Posterior samples array.
    max_num_sources : int
        Number of sources to fit.

    Returns
    -------
    means : ndarray, shape (max_num_sources, num_params_per_source)
        Mean vector for each source.
    cov_matrices : ndarray, shape (max_num_sources, num_params_per_source, num_params_per_source)
        Covariance matrix for each source.

    Examples
    --------
    >>> from petra.parametric_fits import mv_normal_fit
    >>> chain = np.random.randn(500, 3, 2)
    >>> means, covs = mv_normal_fit(chain, max_num_sources=3)
    >>> means.shape
    (3, 2)
    >>> covs.shape
    (3, 2, 2)
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
    chain : ndarray, shape (num_samples, num_entries, num_params_per_source)
        Posterior samples array.
    max_num_sources : int
        Number of sources to fit.
    fit_parameter : int
        Index of the parameter to fit.

    Returns
    -------
    means : ndarray, shape (max_num_sources,)
        Mean of the fitted normal for each source.
    stds : ndarray, shape (max_num_sources,)
        Standard deviation of the fitted normal for each source.

    Examples
    --------
    >>> from petra.parametric_fits import uni_normal_fit_single_parameter
    >>> chain = np.random.randn(200, 2, 5)
    >>> means, stds = uni_normal_fit_single_parameter(chain, max_num_sources=2, fit_parameter=3)
    >>> len(means), len(stds)
    (2, 2)
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
    Compute both the parametric fit and each source's inclusion probability.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Object containing the chain and metadata.
    max_num_sources : int
        Number of sources to include.
    parametric_fit_function : callable
        Function `(chain, max_num_sources) -> fit_params`.
    eps : float, optional
        Tolerance for inclusion probability (default is 1e-2).

    Returns
    -------
    aux_params : tuple
        Output of `parametric_fit_function`, e.g. (means, covs) or (means, stds).
    prob_in_model : ndarray, shape (max_num_sources,)
        Probability that each source is in the model.

    Examples
    --------
    >>> from petra.parametric_fits import update_parametric_fit_and_prob_in_model, mv_normal_fit
    >>> from petra.posterior_chain import PosteriorChain
    >>> chain = np.random.randn(100, 4, 2)
    >>> pc = PosteriorChain(chain, 4, 2, True, None, {})
    >>> aux, probs = update_parametric_fit_and_prob_in_model(
    ...     pc, max_num_sources=4, parametric_fit_function=lambda c, m: mv_normal_fit(c, m))
    >>> len(probs)
    4
    """
    aux_params = parametric_fit_function(posterior_chain.get_chain(), max_num_sources)
    prob_in_model = find_prob_in_model(posterior_chain.get_chain(), max_num_sources, eps=eps)
    return aux_params, prob_in_model
