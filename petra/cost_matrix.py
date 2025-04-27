import numpy as np
from typing import Callable, List
from functools import partial


def create_compute_cost_matrix(aux_distribution: Callable, single_parameter: int = None) -> Callable:
    """
    Create a cost-matrix computation function for a given auxiliary distribution.

    Parameters
    ----------
    aux_distribution : callable
        Function to compute log-pdf values for the auxiliary distribution.
    single_parameter : int, optional
        Fix an index if `aux_distribution` is single-parameter.

    Returns
    -------
    compute_cost_matrix : callable
        A function with signature
            (sample, aux_parameters, prob_in_model, num_distributions) -> cost_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import mv_normal_aux_distribution
    >>> from petra.cost_matrix import create_compute_cost_matrix
    >>> compute_cost = create_compute_cost_matrix(mv_normal_aux_distribution)
    >>> # sample: 4 sources x 3 params each
    >>> sample = np.random.randn(4, 3)
    >>> means = [np.zeros(3), np.ones(3)]
    >>> covs = [np.eye(3), 2*np.eye(3)]
    >>> prob = np.array([0.6, 0.4])
    >>> cost = compute_cost(sample, (means, covs), prob, num_distributions=2)
    >>> cost.shape
    (2, 4)
    """
    if single_parameter is not None:
        aux_distribution = partial(aux_distribution, single_parameter=single_parameter)

    def compute_cost_matrix(sample: np.ndarray, aux_parameters: List[np.ndarray],
                            prob_in_model: np.ndarray, num_distributions: int) -> np.ndarray:
        """
        Compute the cost matrix for one sample and a set of distributions.

        Parameters
        ----------
        sample : ndarray, shape (n_sources, n_params_per_source)
            A single sample of parameter values for each source.
        aux_parameters : tuple of lists of ndarray
            Auxiliary distribution parameters,
            e.g., (means, cov_matrices) for multivariate normals.
        prob_in_model : ndarray, shape (num_distributions,)
            Probability that each distribution is active.
        num_distributions : int
            Number of distributions to include.

        Returns
        -------
        cost_matrix : ndarray, shape (num_distributions, n_sources)
            Cost where entry (i, j) = log(prob_in_model[i]) + logpdf_i(sample[j]).

        Examples
        --------
        >>> from petra.aux_distributions import mv_normal_aux_distribution
        >>> compute_cost = create_compute_cost_matrix(mv_normal_aux_distribution)
        >>> sample = np.random.randn(5, 4)
        >>> means = [np.zeros(4), np.ones(4)]
        >>> covs = [np.eye(4), 1.5*np.eye(4)]
        >>> prob = np.array([0.7, 0.3])
        >>> cost = compute_cost(sample, (means, covs), prob, num_distributions=2)
        >>> cost.shape
        (2, 5)
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
