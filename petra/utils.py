import numpy as np
import jax.numpy as jnp
from scipy.stats import uniform


def find_prob_in_model(chain, max_num_sources, eps=1e-6):
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


def make_uniform_prior(chain: np.ndarray):
    """
    Create a uniform prior distribution over each parameter based on the full range observed in the chain.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples array; NaNs indicate missing entries.

    Returns
    -------
    uniform_dist : scipy.stats._continuous_distns.uniform_gen
        A SciPy `uniform` distribution with `loc=mins` and `scale=maxs-mins`,
        where `mins` and `maxs` are the per-parameter minima and maxima
        over all samples and sources.
    """
    all_entries = np.reshape(chain, (-1, chain.shape[2]))
    mins = np.nanmin(all_entries, axis=0)
    maxs = np.nanmax(all_entries, axis=0)
    uniform_dist = uniform(loc=mins, scale=maxs - mins)
    value = np.sum(uniform_dist.logpdf(all_entries[0]))  # Use the first set of parameters for logpdf

    def uniform_prior():
        """
        Compute the PDF of the uniform prior for a given sample.

        Parameters
        ----------
        sample : ndarray, shape (n_samples, n_params_per_source)
            Sample of parameter values to evaluate.

        Returns
        -------
        pdf_values : ndarray, shape (n_samples,)
            PDF values for each sample point.
        """
        pass

    def log_prob(sample):
        # Fix for stacking error: return array matching sample's source dimension
        sample = jnp.asarray(sample)
        if sample.ndim == 1:
            # Single source: return scalar
            return value
        elif sample.ndim == 2:
            # Multiple sources: return array with same log-prob for each source
            return jnp.full(sample.shape[0], value)
        else:
            raise ValueError("Sample must be 1D or 2D array.")

    uniform_prior.log_prob = log_prob

    return uniform_prior


def process_array(arr):
    """
    Checks for duplicate values in a sorted numpy array.
    If duplicates exist, perturbs the values slightly to make them unique
    and resorts the array if necessary after perturbation.
    
    Parameters:
    arr (numpy.ndarray): A sorted numpy array.
    
    Returns:
    numpy.ndarray: The processed array with no duplicates.
    """
    # Convert to float for perturbation
    arr = np.sort(arr)
    arr = np.asarray(arr, dtype=float)

    # Check for duplicates
    unique, counts = np.unique(arr, return_counts=True)
    if np.all(counts <= 1):
        return arr

    # Determine epsilon for perturbation
    max_count = np.max(counts)
    diffs = np.diff(arr)
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) > 0:
        min_pos_diff = np.min(pos_diffs)
        epsilon = min_pos_diff / (max_count * 2)
    else:
        epsilon = 1e-10

    # Copy array for modification
    new_arr = arr.copy()
    
    # Perturb duplicates
    i = 0
    while i < len(new_arr):
        val = new_arr[i]
        j = i
        while j < len(new_arr) and new_arr[j] == val:
            j += 1
        group_size = j - i
        if group_size > 1:
            for k in range(group_size):
                new_arr[i + k] += k * epsilon
        i = j

    # Check if still sorted
    is_sorted = np.all(np.diff(new_arr) >= 0)
    if not is_sorted:
        new_arr.sort()

    return new_arr

def find_uniform_bounds(chain: np.ndarray):
    all_entries = chain.reshape(-1, chain.shape[2])
    lower_bound = np.nanmin(all_entries, axis=0)
    upper_bound = np.nanmax(all_entries, axis=0)
    return lower_bound, upper_bound
