import numpy as np


def find_prob_in_model(chain, max_num_sources, eps=1e-2):
    """
    Compute the probability that each source is present in the model.

    For each source index i, counts the fraction of samples where
    the parameter at index i is not NaN, then clips to [eps, 1-eps]
    to avoid log-domain issues.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples, possibly containing NaNs for missing sources.
    max_num_sources : int
        Maximum number of sources to consider.
    eps : float, optional
        Small value to clip probabilities away from 0 or 1 (default 1e-2).

    Returns
    -------
    prob_in_model : ndarray, shape (max_num_sources,)
        Probability each source index is active in the samples.

    Examples
    --------
    >>> chain = np.array([
    ...     [[1.0], [np.nan], [2.0]],
    ...     [[0.5], [ 2.1], [np.nan]],
    ...     [[np.nan], [1.8], [2.3]],
    ... ])
    >>> # chain.shape = (3 samples, 3 sources, 1 param)
    >>> find_prob_in_model(chain, max_num_sources=3)
    array([0.33..., 0.66..., 0.66...])
    """
    num_samples = chain.shape[0]
    prob_in_model = np.zeros(max_num_sources)
    for i in range(max_num_sources):
        prob_in_model[i] = np.sum(~np.isnan(chain[:, i, 0])) / num_samples
    # clip the probabilities to avoid division by zero in np.log
    prob_in_model = np.clip(prob_in_model, eps, 1 - eps)
    return prob_in_model


def fill_missing_indices(total_sources, given_indices):
    """
    Fill in missing indices by appending those not in `given_indices`.

    Takes the list of selected indices and appends all other indices
    from 0 to total_sources-1 in ascending order.

    Parameters
    ----------
    total_sources : int
        The total number of source indices desired.
    given_indices : array-like of int
        Indices that are already assigned or filled.

    Returns
    -------
    filled_indices : ndarray, shape (total_sources,)
        Array starting with `given_indices`, then the missing indices.

    Examples
    --------
    >>> fill_missing_indices(5, [2, 4])
    array([2, 4, 0, 1, 3])
    >>> fill_missing_indices(3, [])
    array([0, 1, 2])
    """
    # Step 1: Generate a list of all indices from 0 to total_sources - 1
    all_indices = np.arange(total_sources)
    # Step 2: Convert given_indices to a set for faster operations
    given_indices_set = set(given_indices)
    # Step 3: Filter out the given indices from all_indices to get missing indices
    missing_indices = [index for index in all_indices if index not in given_indices_set]
    # Step 4: Combine the given indices with the missing indices
    filled_indices = list(given_indices) + missing_indices

    return np.array(filled_indices)


def count_swaps(arr):
    """
    Count the minimum number of swaps needed to sort an array.

    Compares the array to its sorted version and counts mismatches,
    dividing by two since each swap corrects two positions.

    Parameters
    ----------
    arr : ndarray of shape (n,)
        Input array of comparable elements.

    Returns
    -------
    swaps : int
        Minimum number of pairwise swaps to sort `arr`.

    Examples
    --------
    >>> count_swaps(np.array([2, 1, 3]))
    1
    >>> count_swaps(np.array([3, 1, 2]))
    2
    """
    sorted_arr = np.sort(arr)
    swaps = np.sum(arr != sorted_arr)
    return swaps // 2


def sort_by_number(filenames):
    """
    Sort filenames by the integer suffix after the final dot.

    Assumes each filename ends with ".<number>".

    Parameters
    ----------
    filenames : list of str
        Filenames to sort.

    Returns
    -------
    sorted_list : list of str
        Filenames sorted in ascending order of their numeric suffix.

    Examples
    --------
    >>> sort_by_number(['file.10', 'file.2', 'file.1'])
    ['file.1', 'file.2', 'file.10']
    """
    # Extract the number after the dot and convert it to an integer
    def extract_number(filename):
        return int(filename.split(".")[-1])

    # Sort the filenames using the extracted number
    return sorted(filenames, key=extract_number)
