import numpy as np
from typing import Callable, List
from functools import partial


def create_compute_cost_matrix(aux_distribution: Callable, single_parameter: int = None) -> Callable:
    """
    Creates the function to compute the cost matrix for a given auxiliary parametric distribution.

    Parameters
    ----------
    aux_distribution: The function to compute the cost matrix elements for the auxiliary distribution

    Return
    ------
    compute_cost_matrix: The function to compute the cost matrix
    """
    if single_parameter is not None:
        aux_distribution = partial(aux_distribution, single_parameter=single_parameter)

    def compute_cost_matrix(sample: np.ndarray, aux_parameters: List[np.ndarray],
                            prob_in_model: np.ndarray, num_distributions: int) -> np.ndarray:
        """
        Takes in a single sample and a set of parameters for all fitting distributions,
        and returns the cost matrix.

        Parameters
        ----------
        sample: A numpy array of shape (num_sources, num_params_per_source)
        aux_parameters: A tuple of lists containing the auxiliary distribution parameters
        prob_in_model: A numpy array of shape (num_distributions,) that contains the probability of each source being in the catalog
        num_distributions: The number of distributions to use in the cost matrix

        Return
        ------
        cost_matrix: A numpy array of shape (num_sources, num_distributions)
        """
        # Precompute logarithms.
        with np.errstate(divide='ignore'):  # avoid divide by zero warnings, they are expected when we don't clip prob_in_model
            log_prob = np.log(prob_in_model)  # shape: (num_distributions,)
            log_prob_not = np.log1p(-prob_in_model)[np.newaxis, :]  # shape: (1, num_distributions)

        # Vectorize over distribution indices.
        distribution_indices = np.arange(num_distributions)
        # Assume that aux_distribution is vectorized so that it returns an array of shape (num_distributions, num_sources)
        cost_matrix = aux_distribution(sample, aux_parameters, distribution_indices)
        cost_matrix = log_prob[:, np.newaxis] + cost_matrix

        # Replace any NaN values with log(1 - prob_in_model)
        cost_matrix = np.where(np.isnan(cost_matrix), log_prob_not, cost_matrix)

        return cost_matrix

    return compute_cost_matrix
