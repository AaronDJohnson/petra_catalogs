import numpy as np
from scipy import optimize
from typing import Callable

from petra.utils import fill_missing_indices
from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import update_parametric_fit_and_prob_in_model, create_parametric_fit
from petra.cost_matrix import create_compute_cost_matrix


def relabel_samples_one_iteration(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix: Callable):
    """
    Perform one iteration of label assignment using the Hungarian algorithm.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples to relabel.
    aux_parameters : tuple
        Parameters (e.g., means, covariances) returned by a parametric fit.
    prob_in_model : ndarray, shape (max_num_sources,)
        Probability each source is present in the model.
    max_num_sources : int
        Maximum number of sources to consider in assignment.
    compute_cost_matrix : callable
        Function with signature
        `(sample, aux_parameters, prob_in_model, max_num_sources) -> cost_matrix`.

    Returns
    -------
    relabeled_chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Samples reordered according to optimal assignments.
    cost : float
        Average assignment cost over all samples.

    Examples
    --------
    >>> from petra.relabel import relabel_samples_one_iteration
    >>> # chain: np.ndarray shape (100, 3, 5)
    >>> # aux_params, prob: obtained from a fit
    >>> relabeled, cost = relabel_samples_one_iteration(
    ...     chain, aux_params, prob, 3, compute_cost_matrix)
    >>> relabeled.shape
    (100, 3, 5)
    """
    # TODO(Aaron): Parallelize this for loop over samples
    relabeled_list = []
    total_cost = 0
    for sample in chain:
        cost_matrix = compute_cost_matrix(sample, aux_parameters, prob_in_model, max_num_sources)
        total_entries = max(len(sample), max_num_sources)  # pick the larger of number of aux distributions or sample entries
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)  # solve the linear sum assignment problem
        total_cost -= cost_matrix[row_ind, col_ind].sum()
        # TODO(Aaron): Is the following line still necessary? It shouldn't change the result either way.
        relabeled_list.append(fill_missing_indices(total_entries, col_ind))  # sum assignment only outputs 1 number for each labeling distribution input
    relabeled_array = np.array(relabeled_list)

    relabeled_samples = np.array([row[m] for row, m in zip(chain, relabeled_array)])  # reorder the original samples
    total_cost /= len(chain)
    return relabeled_samples, total_cost


def relabel_posterior_chain_one_iteration(posterior_chain: PosteriorChain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix: Callable):
    """
    Perform one relabeling step on a PosteriorChain instance.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Object containing chain, metadata and previous cost dictionary.
    aux_parameters : tuple
        Parameters from a parametric fit (e.g., means, covariances).
    prob_in_model : ndarray, shape (max_num_sources,)
        Probability each source is present in the model.
    max_num_sources : int
        Maximum number of sources for this iteration.
    compute_cost_matrix : callable
        Cost matrix builder function for each sample.

    Returns
    -------
    new_chain : PosteriorChain
        PosteriorChain with updated `.chain` and `.cost_dict`.
    cost : float
        Assignment cost recorded under `new_chain.cost_dict[max_num_sources]`.

    Examples
    --------
    >>> from petra.posterior_chain import PosteriorChain
    >>> from petra.relabel import relabel_posterior_chain_one_iteration
    >>> new_pc, cost = relabel_posterior_chain_one_iteration(
    ...     pc, aux_params, prob, 3, compute_cost_matrix)
    >>> isinstance(new_pc, PosteriorChain)
    True
    >>> cost == new_pc.cost_dict[3]
    True
    """
    cost_dict = posterior_chain.cost_dict
    relabeled_chain, total_cost = relabel_samples_one_iteration(posterior_chain.get_chain(), aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix)
    if cost_dict is None:
        cost_dict = {}
    cost_dict[max_num_sources] = total_cost
    return PosteriorChain(relabeled_chain, posterior_chain.num_sources, posterior_chain.num_params_per_source, prob_in_model=prob_in_model, cost_dict=cost_dict)


def create_relabel_samples(parametric_fit_function: Callable,
                           aux_distribution: Callable,
                           single_parameter: int = None,
                           eps: float = 1e-2):
    """
    Build a relabeling procedure combining fitting, cost computation, and assignment.

    Parameters
    ----------
    parametric_fit_function : callable
        Fit function `(chain, max_num_sources) -> aux_parameters`.
    aux_distribution : callable
        PDF function for cost matrix `(sample, aux_parameters, prob_in_model, idx)`.
    single_parameter : int, optional
        Fixes the parameter index for single-parameter fit/distribution.
    eps : float, optional
        Convergence tolerance for inclusion probabilities.

    Returns
    -------
    relabel_samples : callable
        Function with signature
        `(posterior_chain, max_num_sources, num_iterations) -> PosteriorChain`.

    Examples
    --------
    >>> from petra.relabel import create_relabel_samples
    >>> relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    >>> new_pc = relabel(pc, max_num_sources=3, num_iterations=10)
    >>> isinstance(new_pc, PosteriorChain)
    True
    """

    # make all the necessary pieces
    param_fit = create_parametric_fit(parametric_fit_function, single_parameter=single_parameter)
    compute_cost_matrix = create_compute_cost_matrix(aux_distribution, single_parameter=single_parameter)

    def relabel_samples(posterior_chain: PosteriorChain,
                        max_num_sources: int = None,
                        num_iterations: int = 200):
        """
        Iteratively relabel a PosteriorChain with a chosen aux distribution.

        Parameters
        ----------
        posterior_chain : PosteriorChain
            Chain to process; may be expanded or trimmed.
        max_num_sources : int, optional
            Target number of sources (defaults to chain.num_sources).
        num_iterations : int, default 200
            Maximum relabeling iterations before stopping.

        Returns
        -------
        PosteriorChain
            Relabeled chain with updated cost history.

        Examples
        --------
        >>> from petra.relabel import create_relabel_samples
        >>> relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
        >>> result_pc = relabel(pc, max_num_sources=4, num_iterations=50)
        >>> result_pc.cost_dict.keys()
        dict_keys([4])
        """

        # if shuffle_entries:
        #     posterior_chain.randomize_entries(shuffle_seed)
        if max_num_sources is None:
            max_num_sources = posterior_chain.num_sources
        if max_num_sources > posterior_chain.num_sources:
            # increase the width of the posterior chain to the maximum number of sources
            posterior_chain = posterior_chain.expand_chain(max_num_sources)
        if max_num_sources < posterior_chain.num_sources:
            raise ValueError("The maximum number of sources cannot be less than the number of entries in the chain.")

        print()
        print(f"Sorting the posterior chain:\n\tMaximum number of iterations: {num_iterations}\n\tMaximum number of source labels:{max_num_sources}\n")

        # set up the for loop
        old_posterior_chain = posterior_chain
        old_parametric_fit, old_prob_in_model = update_parametric_fit_and_prob_in_model(posterior_chain, max_num_sources, param_fit, eps=eps)
        old_cost_of_assignment = 0

        for iteration in range(num_iterations):
            # get the new values
            new_posterior_chain = relabel_posterior_chain_one_iteration(old_posterior_chain, old_parametric_fit, old_prob_in_model, max_num_sources, compute_cost_matrix)
            new_parametric_fit, new_prob_in_model = update_parametric_fit_and_prob_in_model(new_posterior_chain, max_num_sources, param_fit, eps=eps)
            new_cost_of_assignment = new_posterior_chain.cost_dict[max_num_sources]

            # print out the results
            delta_cost_of_assignment = new_cost_of_assignment - old_cost_of_assignment
            print(f"Iteration {iteration + 1}: Difference in cost of assignment is {delta_cost_of_assignment} with total cost of {new_cost_of_assignment}.")
            print(f"\tProbabilities in model: {new_prob_in_model}")

            # break if converged
            if (delta_cost_of_assignment == 0):
                print(f"Stopped after {iteration + 1} iterations because the cost didn't change from the previous iteration.")
                new_parametric_fit = old_parametric_fit
                new_prob_in_model = old_prob_in_model
                new_posterior_chain = old_posterior_chain
                break

            # update the old values to the new values
            old_posterior_chain = new_posterior_chain
            old_parametric_fit = new_parametric_fit
            old_prob_in_model = new_prob_in_model
            old_cost_of_assignment = new_cost_of_assignment

        if iteration == num_iterations - 1:
            print(f"Final cost of assignment: {new_cost_of_assignment} after the maximum number ({num_iterations}) of iterations.")

        return new_posterior_chain

    return relabel_samples
