import numpy as np


def find_prob_in_model(chain, max_num_sources, eps=1e-2):
    """
    Find the number of samples for each entry in the chain and divide by the total number of samples.

    Parameters
    ----------
    chains: A numpy array of shape (num_samples, num_entries, num_params_per_source)
    max_num_sources: The maximum number of sources to consider in the catalog
    eps: A small number to avoid division by zero

    Return
    ------
    prob_in_model: A numpy array of shape (max_num_sources,)
    """
    num_samples = chain.shape[0]
    prob_in_model = np.zeros(max_num_sources)
    for i in range(max_num_sources):
        prob_in_model[i] = np.sum(~np.isnan(chain[:, i, 0])) / num_samples
    # clip the probabilities to avoid division by zero in np.log
    prob_in_model = np.clip(prob_in_model, eps, 1 - eps)
    return prob_in_model


def fill_missing_indices(total_sources, given_indices):
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
    Counts the number of swaps needed to sort an array.

    Args:
        arr (np.array): The input array.

    Returns:
        int: The number of swaps.
    """
    sorted_arr = np.sort(arr)
    swaps = np.sum(arr != sorted_arr)
    return swaps // 2


def sort_by_number(filenames):
    # Extract the number after the dot and convert it to an integer
    def extract_number(filename):
        return int(filename.split(".")[-1])

    # Sort the filenames using the extracted number
    return sorted(filenames, key=extract_number)
