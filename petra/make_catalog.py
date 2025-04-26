from petra.posterior_chain import PosteriorChain
from petra.relabel import create_relabel_samples
from petra.aux_distributions import mv_normal_aux_distribution
from petra.parametric_fits import mv_normal_fit
from petra.initialization import relabel_uni_normal_one_parameter


def relabel_mv_normal(posterior_chain: PosteriorChain,
                      max_num_sources: int = None,
                      num_iterations: int = 20,
                      eps=1e-2):

    relabel_samples = create_relabel_samples(mv_normal_fit,
                                             mv_normal_aux_distribution,
                                             eps=eps)

    return relabel_samples(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations
    )


def make_catalog_mv_normal(posterior_chain: PosteriorChain,
                           max_num_sources: int,
                           num_iterations: int = 200,
                           init_num_iterations: int = 200,
                           initialization_param_index: int = None,
                           shuffle_seed: int = None):
    # shuffle the entries
    posterior_chain = posterior_chain.randomize_entries(shuffle_seed)  # works with a copy of the chain

    if posterior_chain.num_sources > max_num_sources:
        raise ValueError("max_num_sources must be greater than the number of entries in the chain.")

    # make sure that posterior_chain has the right shape
    if posterior_chain.num_sources < max_num_sources:
        print("Expanding posterior chain to max_num_sources.")
        posterior_chain.expand_chain(max_num_sources)

    # initialize here:
    if initialization_param_index is not None:
        print("Initializing with univariate normal distribution.")
        initial_posterior_chain = relabel_uni_normal_one_parameter(posterior_chain,
                                                                   max_num_sources=max_num_sources,
                                                                   num_iterations=init_num_iterations,
                                                                   init_parameter_index=initialization_param_index)

    else:
        initial_posterior_chain = posterior_chain

    # relabel using mv normal
    print("Relabeling with multivariate normal distribution.")
    relabeled_chain = relabel_mv_normal(initial_posterior_chain,
                                        max_num_sources=max_num_sources,
                                        num_iterations=num_iterations)
    
    return relabeled_chain
