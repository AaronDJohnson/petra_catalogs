from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import uni_normal_fit_single_parameter
from petra.aux_distributions import uni_normal_aux_distribution_single_parameter
from petra.relabel import create_relabel_samples

import numpy as np
from copy import deepcopy

def get_max_count_bin_center(chain: np.ndarray, delta_freq: float, relabeling_parameter: int) -> float:
    """Snap points to frequency grid and find the bin with the maximum count."""
    x = chain[:, :, relabeling_parameter].reshape(-1)
    x = x[~np.isnan(x)]

    # snap to grid
    k = np.rint(x / delta_freq).astype(np.int64)  # indices on grid
    k_mode = np.bincount(k).argmax()

    max_count_bin_center = k_mode * delta_freq
    return max_count_bin_center

def tetris_rise_nd(a, *, axis=1, empty=np.nan):
    """
    Apply 'anti-gravity' along `axis` for an N-D array:
    filled values rise to the start of `axis`, independently
    for every combination of the other axes.
    """
    a = np.asarray(a)

    # Build filled mask
    if isinstance(empty, float) and np.isnan(empty):
        filled = ~np.isnan(a)
    else:
        filled = (a != empty)

    # Stable sort: filled first, empties last
    order = np.argsort(~filled, axis=axis, kind="stable")

    return np.take_along_axis(a, order, axis=axis)

def count_nonnan_samples(chain: np.ndarray, source_index: int, relabeling_parameter: int) -> int:
    """Count non-NaN samples along axis 0."""
    chain = chain[:, source_index, relabeling_parameter]
    return np.sum(~np.isnan(chain))

def sort_array_by_sample_count(sources_array: np.ndarray, relabeling_parameter: int) -> np.ndarray:
    """Sort sources_array by number of non-NaN samples in relabeling_parameter."""
    source_sample_counts = []
    for source_index in range(sources_array.shape[1]):
        source_sample_counts.append(count_nonnan_samples(sources_array, source_index, relabeling_parameter))
    sorted_source_indices = np.argsort(source_sample_counts)[::-1]  # descending order

    sources_array_sorted = np.zeros_like(sources_array)
    for new_index, old_index in enumerate(sorted_source_indices):
        sources_array_sorted[:, new_index, :] = sources_array[:, old_index, :]
    return sources_array_sorted

def label_chain_by_histogram(chain: PosteriorChain, obs_time_yrs: float=1, num_surrounding_bins: int=1, relabeling_parameter: int=0, num_extra_entries: int=1000, low_num_samples: int=100) -> PosteriorChain:
    delta_freq = 1.0 / (525600 * 60 * obs_time_yrs)

    chain = deepcopy(chain)
    num_surrounding_bins = 1  # above and below
    relabeling_parameter = 0
    sources_array = np.zeros((chain.shape[0], chain.shape[1] + num_extra_entries, chain.shape[2]))
    sources_array[:] = np.nan

    low_sample_count_source_indices = []

    source_index = 0
    while np.any(~np.isnan(chain.chain)):
        # create histogram of all samples in relabeling_parameter
        max_count_bin_center = get_max_count_bin_center(chain.chain, delta_freq, relabeling_parameter)

        # take samples within the surrounding bins of max_count_bin_center
        # source: for each sample, compute the distance to max_count_bin_center
        # take the nearest value (minimum) in each sample and if that value is within the surrounding bins, keep it, else continue!
        for i in range(chain.shape[0]):
            distances = np.abs(chain[i, :, relabeling_parameter] - max_count_bin_center)
            # check if the distance array is all nans:
            if np.all(np.isnan(distances)):
                sources_array[i, source_index, :] = np.nan
                continue
            elif np.nanmin(distances) <= num_surrounding_bins * delta_freq:
                min_index = np.nanargmin(distances)
            else:
                sources_array[i, source_index, :] = np.nan
                continue
            sources_array[i, source_index, :] = chain[i, min_index, :]
            # turn used sample into nan to avoid reusing it
            chain[i, min_index, :] = np.nan

        num_valid_samples = count_nonnan_samples(sources_array, source_index, relabeling_parameter)
        if (num_valid_samples == 0) or (num_valid_samples < low_num_samples):
            low_sample_count_source_indices.append(source_index)

        source_index += 1

    # sort the source indices by sample count
    sources_array = sort_array_by_sample_count(sources_array, relabeling_parameter)

    # get the low sample count indices again
    low_sample_count_source_indices = []
    for source_index in range(sources_array.shape[1]):
        num_valid_samples = count_nonnan_samples(sources_array, source_index, relabeling_parameter)
        if num_valid_samples < low_num_samples:
            low_sample_count_source_indices.append(source_index)

    # apply tetris rise to low sample count sources
    low_sample_count_sources = sources_array[:, low_sample_count_source_indices, :]
    sources_array[:, low_sample_count_source_indices, :] = tetris_rise_nd(low_sample_count_sources)

    # remove empty sources
    empty_source_indices = []
    for source_index in range(sources_array.shape[1]):
        num_valid_samples = count_nonnan_samples(sources_array, source_index, relabeling_parameter)
        if num_valid_samples == 0:
            empty_source_indices.append(source_index)

    non_empty_source_indices = [i for i in range(sources_array.shape[1]) if i not in empty_source_indices]
    non_empty_sources = sources_array[:, non_empty_source_indices, :]

    # reorder the arrays based on valid sample counts
    non_empty_sources = sort_array_by_sample_count(non_empty_sources, relabeling_parameter)

    # for source_index in range(non_empty_sources.shape[1]):
    #     num_valid_samples = count_nonnan_samples(non_empty_sources, source_index, relabeling_parameter)
    #     print(f'Source {source_index}: {num_valid_samples} samples')

    return PosteriorChain(chain=non_empty_sources, num_sources=non_empty_sources.shape[1], num_params_per_source=non_empty_sources.shape[2], trans_dimensional=chain.trans_dimensional)


def relabel_univariate_normal(posterior_chain: PosteriorChain,
                              max_num_sources: int|None = None,
                              num_iterations: int = 20,
                              init_parameter_index: int = 0,
                              eps=1e-6):
    """
    Iteratively relabels a posterior chain using univariate normal fits.

    At each iteration, fits a normal distribution to a single parameter of each source,
    computes a cost matrix based on the fit, and applies the Hungarian algorithm
    to align labels across samples until convergence.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        The chain of posterior samples to relabel.
    max_num_sources : int, optional
        Maximum number of sources to include; if None, uses the chain's num_sources.
    num_iterations : int, default 20
        Number of relabeling iterations to perform.
    init_parameter_index : int, default 0
        Index of the parameter used to initialize the relabeling.
    eps : float, default 1e-2
        Convergence tolerance on the change in relabeling cost.

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain instance with relabeled samples.
    cost_trace : list of float
        Cost values at each iteration, showing convergence behavior.

    Examples
    --------
    >>> from petra.initialization import relabel_univariate_normal
    >>> from petra.posterior_chain import PosteriorChain
    >>> # assume `pc` is a PosteriorChain with samples
    >>> relabeled_pc, trace = relabel_univariate_normal(
    ...     pc, max_num_sources=3, num_iterations=50, init_parameter_index=2)
    >>> isinstance(relabeled_pc, PosteriorChain)
    True
    >>> len(trace)
    50
    """

    # create single parameter function to relabel samples
    relabel_samples = create_relabel_samples(uni_normal_fit_single_parameter,
                                             uni_normal_aux_distribution_single_parameter,
                                             single_parameter=init_parameter_index,
                                             eps=eps)

    return relabel_samples(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations
    )
