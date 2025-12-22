from typing import Callable, List, Optional
import numpy as np
from petra.parametric_fits import update_parametric_fit_and_prob_in_model
from petra.cost_matrix import create_compute_cost_matrix
from petra.relabel import relabel_posterior_chain_one_iteration
import jax
import jax.numpy as jnp
import jax.random as jr
from petra.posterior_chain import PosteriorChain
from petra.utils import make_uniform_prior
from petra.relabel import create_relabel_samples
from equinox import filter_jit

from corner import corner

from coppuccino.copula_flows import normalizing_flows_fit

jax.config.update('jax_enable_x64', True)


def copula_flows_aux_distribution(sample: np.ndarray,
                                       aux_parameters: List,
                                       source_index) -> np.ndarray:

    # Ensure source_index is array-like.
    source_indices = jnp.atleast_1d(source_index)  # shape: (n,)

    # Get log probabilities from all flows using safe evaluation
    log_probs = [filter_jit(flow.log_prob)(sample) for flow in aux_parameters]

    # Ensure all log_probs have consistent shapes for stacking
    log_probs_arrays = [jnp.asarray(lp) for lp in log_probs]
    shapes = [lp.shape for lp in log_probs_arrays]

    if len(set(shapes)) > 1:
        # Handle shape mismatch: determine target shape
        array_shapes = [s for s in shapes if s != ()]
        if array_shapes:
            # Some flows return arrays - use that shape as target
            target_shape = array_shapes[0]
            log_probs_arrays = [
                lp if lp.shape == target_shape
                else jnp.full(target_shape, lp) if lp.shape == ()
                else lp  # Keep as-is if already correct array shape
                for lp in log_probs_arrays
            ]
        # If all are scalars, keep as scalars (handled by original code)

    all_lp = jnp.stack(log_probs_arrays, axis=0)
    all_lp = jnp.where(jnp.isfinite(all_lp), all_lp, -1e10)  # Replace -inf with a very small value

    return all_lp[source_indices]


def make_copula_flows_fit(chain:np.ndarray, rng_seed: int = 999, threshold_samples: int = 50,
                          knots:int = 16, patience: int = 20, learning_rate: float = 1e-3, max_epochs: int = 800,
                          flow_layers: int = 8) -> Callable:
    """
    Fit a flow to each entry in the chain.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples, possibly containing NaNs for missing sources.
    rng_seed : int, optional
        Random seed for reproducibility (default is 999).

    Returns
    -------
    list of Transformed
        List of fitted flows for each chain entry.
    """
    uniform_prior = make_uniform_prior(chain)

    def copula_flows_fit(chain: np.ndarray, max_num_sources: int) -> List:
        fitted_distributions = [None] * max_num_sources
        rng_key = jr.key(rng_seed + 1)
        fit_seeds = jax.random.randint(rng_key, shape=(max_num_sources,), minval=0, maxval=999999)

        for i in range(max_num_sources):
            chain_entry = chain[:, i, :]  # Get the i-th source across all samples
            chain_entry = chain_entry[~np.isnan(chain_entry).any(axis=(1))]  # Remove samples with NaNs in this source

            if chain_entry.shape[0] <= threshold_samples:  # If there are not enough samples, use uniform prior
                fitted_distributions[i] = uniform_prior

            else:  # Everything else gets an NF fit
                fitted_distributions[i] = normalizing_flows_fit(chain_entry, rng_seed=int(fit_seeds[i]), knots=knots,
                                                                patience=patience, learning_rate=learning_rate,
                                                                max_epochs=max_epochs, flow_layers=flow_layers)
        return fitted_distributions

    return copula_flows_fit


def relabel_copula_flows(posterior_chain: PosteriorChain,
                         max_num_sources: int|None = None,
                         num_iterations: int = 20,
                         checkpoint_dir: Optional[str] = None,
                         eps=1e-2):

    copula_flows_fit = make_copula_flows_fit(posterior_chain.chain, rng_seed = 999, threshold_samples = 50)

    relabel_samples = create_relabel_samples(copula_flows_fit,
                                             copula_flows_aux_distribution,
                                             eps=eps)

    return relabel_samples(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        checkpoint_dir=checkpoint_dir
    )


def make_catalog_copula_flows(posterior_chain: PosteriorChain,
                              max_num_sources: int,
                              num_iterations: int = 50,
                              eps=1e-6,
                              checkpoint_dir: Optional[str] = None):
    """
    Create a catalog using Gaussian copula marginal transforms + normalizing flows.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        The chain of posterior samples to process.
    max_num_sources : int
        Target number of sources in the catalog.
    num_iterations : int, default 50
        Number of relabeling iterations.
    plot_flows : bool, default False
        Whether to plot flow quality at each iteration.
    tail_decay : float, default 4.0
        Controls extrapolation decay rate in tails. Higher values = slower decay (more conservative).
        Lower values = faster decay (more aggressive penalty for outliers).
    eps : float, default 1e-6
        Small value for numerical stability in probability computations.
    checkpoint_dir : str, optional
        Directory to save iteration checkpoints. If None, no checkpointing is performed.

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain with relabeled samples using the hybrid approach.

    Examples
    --------
    >>> from petra.flows import make_catalog_copula_flows
    >>> relabeled_chain = make_catalog_copula_flows(
    ...     posterior_chain, max_num_sources=3, plot_flows=True, tail_decay=2.0)
    >>> # With checkpointing enabled
    >>> relabeled_chain = make_catalog_copula_flows(
    ...     posterior_chain, max_num_sources=3, checkpoint_dir="./my_checkpoints")
    """

    if posterior_chain.num_sources > max_num_sources:
        raise ValueError("max_num_sources must be greater than the number of entries in the chain.")

    # make sure that posterior_chain has the right shape
    if posterior_chain.num_sources < max_num_sources:
        print("Expanding posterior chain to max_num_sources.")
        posterior_chain.expand_chain(max_num_sources)
    else:
        initial_posterior_chain = posterior_chain

    # relabel using copula-flow hybrid approach with smooth extrapolation
    print("Relabeling with Gaussian copula marginals + normalizing flows.")
    relabeled_chain = relabel_copula_flows(initial_posterior_chain,
                                           max_num_sources=max_num_sources,
                                           num_iterations=num_iterations,
                                           eps=eps,
                                           checkpoint_dir=checkpoint_dir)

    return relabeled_chain
