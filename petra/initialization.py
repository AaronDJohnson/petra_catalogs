from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import uni_normal_fit_single_parameter
from petra.aux_distributions import uni_normal_aux_distribution_single_parameter
from petra.relabel import create_relabel_samples


def relabel_univariate_normal(posterior_chain: PosteriorChain,
                              max_num_sources: int = None,
                              num_iterations: int = 20,
                              init_parameter_index: int = 0,
                              eps=1e-2):
    """
    Iteratively relabels a posterior chain using univariate normal fits.

    At each iteration, fits a normal distribution to a single parameter of each source,
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
    init_parameter_index : int, default 0
        Index of the parameter used to initialize the relabeling.
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
    >>> from petra.initialization import relabel_univariate_normal
    >>> from petra.posterior_chain import PosteriorChain
    >>> # assume `pc` is a PosteriorChain with samples
    >>> relabeled_pc, trace = relabel_univariate_normal(
    ...     pc, max_num_sources=3, num_iterations=50, init_parameter_index=2)
    >>> isinstance(relabeled_pc, PosteriorChain)
    True
    >>> len(trace)
    50
    """

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
