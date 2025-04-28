from petra.posterior_chain import PosteriorChain
from petra.relabel import create_relabel_samples
from petra.aux_distributions import mv_normal_aux_distribution
from petra.parametric_fits import mv_normal_fit
from petra.initialization import relabel_univariate_normal


def relabel_mv_normal(posterior_chain: PosteriorChain,
                      max_num_sources: int = None,
                      num_iterations: int = 20,
                      eps=1e-2):
    """
    Iteratively relabels a posterior chain using multivariate normal fits.

    At each iteration, fits a multivariate normal distribution to each source,
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
    >>> from petra.make_catalog import relabel_mv_normal
    >>> from petra.posterior_chain import PosteriorChain
    >>> pc = PosteriorChain(chain_array, num_sources, num_params, True, None, {})
    >>> relabeled_pc, trace = relabel_mv_normal(pc, max_num_sources=3)
    >>> isinstance(relabeled_pc, PosteriorChain)
    True
    >>> len(trace)
    20
    """

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
    """
    Build a catalog by relabeling samples using multivariate normal fits,
    with optional univariate initialization and shuffling.

    The chain is first shuffled (if a seed is provided), then expanded
    or trimmed to `max_num_sources`. If `initialization_param_index` is set,
    a univariate normal relabeling is run for `init_num_iterations` to
    initialize labels. Finally, multivariate relabeling is run for
    `num_iterations`.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        The chain of posterior samples to process.
    max_num_sources : int
        Target number of sources in the catalog.
    num_iterations : int, default 200
        Number of iterations for the final multivariate relabeling.
    init_num_iterations : int, default 200
        Number of iterations for the optional univariate initialization.
    initialization_param_index : int, optional
        Index of the parameter for the univariate initialization step.
    shuffle_seed : int, optional
        Seed for random shuffling of chain entries (for reproducibility).

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain instance with relabeled samples.

    Examples
    --------
    >>> from petra.make_catalog import make_catalog_mv_normal
    >>> from petra.posterior_chain import PosteriorChain
    >>> pc = PosteriorChain(chain_array, num_sources, num_params, True, None, {})
    >>> catalog = make_catalog_mv_normal(
    ...     pc,
    ...     max_num_sources=4,
    ...     num_iterations=100,
    ...     init_num_iterations=50,
    ...     initialization_param_index=2,
    ...     shuffle_seed=42
    ... )
    >>> isinstance(catalog, PosteriorChain)
    True
    """

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
        initial_posterior_chain = relabel_univariate_normal(posterior_chain,
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
