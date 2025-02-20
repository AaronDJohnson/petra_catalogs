import numpy as np
from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import uni_normal_fit_single_parameter
from petra.aux_distributions import uni_normal_aux_distribution_single_parameter
from petra.relabel import create_relabel_samples


def relabel_by_sorting(posterior_chain: PosteriorChain, init_parameter_index: int = 0):
    """
    Relabel sources by simply sorting each row by the chosen parameter.
    
    NaNs are replaced by infinity so that they come last, then after sorting
    we revert infinity back to NaN.
    """
    chain = posterior_chain.get_chain().copy()  # shape: (n_samples, n_sources, n_params)
    n_samples, n_sources, _ = chain.shape

    # Work on the parameter we want to sort on.
    param = chain[:, :, init_parameter_index].copy()
    # Replace NaNs with infinity so they appear at the end when sorting.
    param[np.isnan(param)] = np.inf

    # For each row, get the permutation that sorts the parameter values.
    ordering = np.argsort(param, axis=1)
    
    # Reorder the entire chain.
    reordered_chain = np.take_along_axis(chain, ordering[:, :, np.newaxis], axis=1)

    # Convert any inf back to NaN in the parameter.
    sorted_param = reordered_chain[:, :, init_parameter_index]
    sorted_param[np.isinf(sorted_param)] = np.nan
    reordered_chain[:, :, init_parameter_index] = sorted_param

    return PosteriorChain(
        reordered_chain,
        posterior_chain.num_sources,
        posterior_chain.num_params_per_source,
        posterior_chain.trans_dimensional,
        posterior_chain.prob_in_model,
        posterior_chain.cost_dict,
    )


def relabel_by_histogram(posterior_chain: PosteriorChain, init_parameter_index: int = 0):
    """
    Assign labels to sources based on samples for one parameter,
    ensuring that each original column (source) is used exactly once per row.
    
    The algorithm:
      1. Extract the parameter values (for ordering) and replace NaNs with ∞ so they’re
         chosen only after all finite values.
      2. For each ordering column (i.e. for each source position):
         a. Compute a histogram from all remaining finite values (across all rows).
         b. Find the histogram’s mode.
         c. For each row, choose the index (from the not-yet-assigned values) whose value is 
            closest to the mode.
         d. Mark that chosen value as “used” by setting it to NaN.
      3. Reorder the full chain according to the computed ordering.
      4. Convert any ∞ values in the sorted parameter back to NaN.
    
    Parameters
    ----------
    posterior_chain : PosteriorChain
        Contains a 3D array of samples with shape (n_samples, n_sources, n_params).
    init_parameter_index : int, optional
        The parameter index used for sorting (default 0).
    
    Returns
    -------
    PosteriorChain
        A new PosteriorChain with sources reordered so that each original column is used once per row.
    """
    # Get a copy of the full chain (shape: [n_samples, n_sources, n_params])
    chain = posterior_chain.get_chain().copy()
    n_samples, n_sources, _ = chain.shape

    # Extract the parameter values we are sorting on.
    # Work on a separate copy so that we can mark chosen entries.
    param_values = chain[:, :, init_parameter_index].copy()
    # Replace NaNs with infinity so they are assigned last.
    param_values[np.isnan(param_values)] = np.inf

    # Prepare an array to store the new ordering (each row should be a permutation of 0..n_sources-1).
    reordered_indices = np.empty((n_samples, n_sources), dtype=int)

    # Iterate for each new column in the ordering.
    for new_col in range(n_sources):
        # Compute the global histogram mode from all remaining finite values.
        remaining = param_values[np.isfinite(param_values)]
        if remaining.size > 0:
            hist, edges = np.histogram(remaining, bins="auto")
            centers = (edges[:-1] + edges[1:]) / 2
            mode = centers[np.argmax(hist)]
        else:
            # Fallback if no finite values remain (should not occur given our assumptions).
            mode = 0.0

        # Now, for each row, pick the index (from values not yet chosen) that is closest to the mode.
        for i in range(n_samples):
            row = param_values[i, :]
            # Find the positions that have not been chosen yet (i.e. are not NaN).
            available = np.where(~np.isnan(row))[0]
            if available.size > 0:
                # Compute differences only over available values.
                diffs = np.abs(row[available] - mode)
                best_idx = available[np.argmin(diffs)]
            else:
                # Should not happen; every row originally has at least one finite value.
                best_idx = 0
            # Record the chosen index for this new column.
            reordered_indices[i, new_col] = best_idx
            # Mark the chosen value as used by setting it to NaN.
            param_values[i, best_idx] = np.nan

    # Reorder the entire chain using the computed ordering.
    # This reorders along the source dimension.
    reordered_chain = np.take_along_axis(chain, reordered_indices[:, :, np.newaxis], axis=1)

    # In the sorted parameter, convert any ∞ values back to NaN.
    sorted_param = reordered_chain[:, :, init_parameter_index]
    sorted_param[np.isinf(sorted_param)] = np.nan
    reordered_chain[:, :, init_parameter_index] = sorted_param

    return PosteriorChain(
        reordered_chain,
        posterior_chain.num_sources,
        posterior_chain.num_params_per_source,
        posterior_chain.trans_dimensional,
        posterior_chain.prob_in_model,
        posterior_chain.cost_dict,
    )


def relabel_uni_normal_one_parameter(posterior_chain: PosteriorChain,
                                     max_num_sources: int = None,
                                     num_iterations: int = 20,
                                     init_parameter_index: int = 0,
                                     eps=1e-2):
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
