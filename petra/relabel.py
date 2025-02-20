import numpy as np
from scipy import optimize
from typing import Callable

from petra.utils import fill_missing_indices
from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import update_parametric_fit_and_prob_in_model, create_parametric_fit
from petra.cost_matrix import create_compute_cost_matrix


def relabel_samples_one_iteration(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix: Callable):
    """
    Takes in a chain of samples and a MV Gaussian fit and returns the sorted chain

    Parameters
    ----------
    chain: A numpy array of shape (num_chains, num_sources, num_params_per_source)
    parameters: A set of parameters that fit the aux distribution to the chain
    prob_in_model: A numpy array of shape (max_num_sources,) that contains the probability of each source being in the model
    max_num_sources: The maximum number of sources to consider in the catalog (can be less than, equal to, or greater than the number of entries in the chain)

    Return
    ------
    relabeled_chain: A numpy array of shape (num_chains, num_sources, num_params_per_source)
    cost: Cost of the optimal assignment
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
    Takes in a chain of samples and a MV Gaussian fit and returns the sorted chain

    Parameters
    ----------
    chain: A PosteriorChain object
    parameters: A set of parameters that fit the aux distribution to the chain
    prob_in_model: A numpy array of shape (max_num_sources,) that contains the probability of each source being in the model
    max_num_sources: The maximum number of sources to consider in the catalog (can be less than, equal to, or greater than the number of entries in the chain)

    Return
    ------
    relabeled_chain: A PosteriorChain object
    cost: Cost of the optimal assignment
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

    # make all the necessary pieces
    param_fit = create_parametric_fit(parametric_fit_function, single_parameter=single_parameter)
    compute_cost_matrix = create_compute_cost_matrix(aux_distribution, single_parameter=single_parameter)

    def relabel_samples(posterior_chain: PosteriorChain,
                        max_num_sources: int = None,
                        num_iterations: int = 200):
        """
        Takes in a posterior_chain and relabels the samples using a multivariate normal auxiliary fitting distribution

        Parameters
        ----------
        posterior_chain: A PosteriorChain object
        max_num_sources: The maximum number of sources to consider in the catalog (can be less than, equal to, or greater than the number of entries in the chain)
        num_iterations: How many tries to use before stopping the algorithm
        init_parameter_index: Which parameter index to use in the initialization procedures
        shuffle_entries: Whether to shuffle the entries in the chain at the start
        shuffle_seed: Seed for the random number generator

        Return
        ------
        relabeled_chain: A PosteriorChain object
        cost: Cost of the optimal assignment
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
