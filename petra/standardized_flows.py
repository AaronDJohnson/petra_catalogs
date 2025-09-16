"""
Standardized normalizing flows for petra catalogs.

This module provides normalizing flows with simple standardization (mean/std normalization)
instead of copula transforms. This allows label swapping to occur naturally while still
benefiting from data standardization for improved numerical stability.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import List, Callable, ClassVar
from equinox import filter_jit

from flowjax.bijections import AbstractBijection, Stack
from flowjax.train import fit_to_data
from flowjax.flows import masked_autoregressive_flow
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed

from paramax import non_trainable
from petra.posterior_chain import PosteriorChain
from petra.utils import make_uniform_prior
from petra.relabel import create_relabel_samples


class StandardizationBijection(AbstractBijection):
    """
    Simple standardization transform: z = (x - mean) / std

    This is a basic linear transform that centers and scales the data
    without imposing copula constraints, allowing natural label swapping.
    """

    mean: float
    std: float
    cond_shape: ClassVar[None] = None
    shape: ClassVar[tuple[int, ...]] = ()

    def __init__(self, mean: float, std: float):
        """
        Initialize standardization transform.

        Parameters
        ----------
        mean : float
            Mean to subtract
        std : float
            Standard deviation to divide by
        """
        self.mean = mean
        self.std = max(std, 1e-6)  # Avoid division by zero

    def inverse_and_log_det(self, x, condition=None):
        """Transform from original space to standardized space."""
        # Ensure x is treated as scalar for this univariate transform
        x = jnp.asarray(x)
        if x.ndim == 0:
            # Scalar input
            z = (x - self.mean) / self.std
            log_det = -jnp.log(self.std)
        else:
            # Array input - apply element-wise
            z = (x - self.mean) / self.std
            log_det = -jnp.log(self.std) * x.size
        return z, log_det

    def transform_and_log_det(self, z, condition=None):
        """Transform from standardized space to original space."""
        # Ensure z is treated correctly
        z = jnp.asarray(z)
        if z.ndim == 0:
            # Scalar input
            x = self.std * z + self.mean
            log_det = jnp.log(self.std)
        else:
            # Array input - apply element-wise
            x = self.std * z + self.mean
            log_det = jnp.log(self.std) * z.size
        return x, log_det


def find_uniform_bounds(chain: np.ndarray):
    """Find uniform bounds across all chain entries."""
    all_entries = chain.reshape(-1, chain.shape[2])
    lower_bound = np.nanmin(all_entries, axis=0)
    upper_bound = np.nanmax(all_entries, axis=0)
    return lower_bound, upper_bound


def make_standardized_flows_fit(chain: np.ndarray,
                               max_num_sources: int,
                               rng_seed: int = 999,
                               threshold_samples: int = 50,
                               plot_flows: bool = False,
                               knots: int = 16,
                               interval: float = 4.0) -> Callable:
    """
    Create standardized normalizing flows that preserve label swapping.

    Uses simple mean/std standardization instead of copula transforms,
    allowing natural label permutations while maintaining numerical stability.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples, possibly containing NaNs for missing sources
    max_num_sources : int
        Maximum number of sources to fit flows for
    rng_seed : int, default 999
        Random seed for reproducibility
    threshold_samples : int, default 50
        Minimum samples needed to fit a flow (else use uniform)
    plot_flows : bool, default False
        Whether to plot flow comparisons during training
    knots : int, default 16
        Number of knots for rational quadratic splines
    interval : float, default 4.0
        Interval size for spline transforms

    Returns
    -------
    standardized_flows_fit : callable
        Function that fits standardized flows to chain data
    """
    flows = []
    transforms = []
    inverse_log_dets = []
    uniform_prior = make_uniform_prior(chain)

    key = jr.key(rng_seed)
    keys = jr.split(key, max_num_sources + 1)

    for i in range(max_num_sources):
        subkey_i = keys[i]
        chain_entry = chain[:, i, :]
        x = chain_entry[~np.isnan(chain_entry).any(axis=1)]

        # Create simple standardization transforms for each parameter
        standardization_transforms = []
        for j in range(x.shape[1]):
            param_samples = x[:, j]
            mean = np.mean(param_samples)
            std = np.std(param_samples)
            standardization_transforms.append(
                non_trainable(StandardizationBijection(mean, std))
            )

        transform = Stack(standardization_transforms)
        transforms.append(transform)
        inverse_log_dets.append(jax.jit(jax.vmap(transform.inverse_and_log_det)))

        if len(x) <= threshold_samples:
            flows.append(uniform_prior)
        else:
            flow = masked_autoregressive_flow(
                subkey_i,
                base_dist=Normal(jnp.zeros(x.shape[1])),
                transformer=RationalQuadraticSpline(knots=knots, interval=interval),
                invert=True,
                nn_depth=2,
            )
            flows.append(flow)

    # Keep track of training history
    training_history = {}
    current_iteration = [0]

    def clear_training_cache():
        """Clear the training cache to force retraining."""
        nonlocal training_history
        training_history = {}
        current_iteration[0] = 0

    def set_iteration(iteration_num):
        """Set current iteration for tracking."""
        current_iteration[0] = iteration_num

    def fit_chain_entry_standardized(input_flow, transform, inverse_log_det,
                                   chain_entry: np.ndarray, rng_seed: int = 999):
        """Fit a single chain entry with standardized inputs."""
        key = jr.key(rng_seed)
        key, subkey = jr.split(key)

        # Transform to standardized space
        x_train, _ = inverse_log_det(chain_entry)

        # Train the flow
        updated_flow, losses = fit_to_data(
            subkey, input_flow, x_train,
            max_epochs=100, max_patience=5, learning_rate=1e-3
        )

        # Wrap with transform to map back to original space
        final_flow = Transformed(updated_flow, transform)
        return final_flow

    def standardized_flows_fit(chain: np.ndarray, max_num_sources: int, **kwargs):
        """Fit standardized flows to the chain data."""
        rng_key = jr.key(rng_seed + 1)
        fit_keys = jr.split(rng_key, max_num_sources)

        for i in range(max_num_sources):
            flow = flows[i]
            chain_entry = chain[:, i, :]
            chain_entry = chain_entry[~jnp.isnan(chain_entry).any(axis=1)]

            # Create hash for change detection
            chain_hash = hash(chain_entry.tobytes()) if len(chain_entry) > 0 else 0

            # Always retrain each iteration since relabeling changes the sample assignments
            # Even if data points are the same, their assignment to sources changes
            need_retrain = True

            if len(chain_entry) <= threshold_samples:
                flows[i] = uniform_prior
                training_history[i] = {
                    'hash': chain_hash,
                    'n_samples': len(chain_entry),
                    'flow_type': 'uniform'
                }
            elif (flow is uniform_prior) and (len(chain_entry) > threshold_samples):
                # Transitioning from uniform to flow
                # Recalculate standardization transforms for the current chain_entry
                updated_transforms = []
                for j in range(chain_entry.shape[1]):
                    param_samples = chain_entry[:, j]
                    mean = np.mean(param_samples)
                    std = np.std(param_samples)
                    updated_transforms.append(
                        non_trainable(StandardizationBijection(mean, std))
                    )

                updated_transform = Stack(updated_transforms)
                updated_inverse_log_det = jax.jit(jax.vmap(updated_transform.inverse_and_log_det))

                flow = masked_autoregressive_flow(
                    fit_keys[i],
                    base_dist=Normal(jnp.zeros(chain_entry.shape[1])),
                    transformer=RationalQuadraticSpline(knots=knots, interval=interval),
                    invert=True,
                    nn_depth=2,
                )
                flow = fit_chain_entry_standardized(
                    flow, updated_transform, updated_inverse_log_det,
                    chain_entry, rng_seed=rng_seed+i
                )
                flows[i] = flow
                # Update the transforms and inverse_log_dets for this source
                transforms[i] = updated_transform
                inverse_log_dets[i] = updated_inverse_log_det
                training_history[i] = {
                    'hash': chain_hash,
                    'n_samples': len(chain_entry),
                    'flow_type': 'trained'
                }
            elif need_retrain:
                # Need to retrain existing flow
                # Recalculate standardization transforms for the current chain_entry
                updated_transforms = []
                for j in range(chain_entry.shape[1]):
                    param_samples = chain_entry[:, j]
                    mean = np.mean(param_samples)
                    std = np.std(param_samples)
                    updated_transforms.append(
                        non_trainable(StandardizationBijection(mean, std))
                    )

                updated_transform = Stack(updated_transforms)
                updated_inverse_log_det = jax.jit(jax.vmap(updated_transform.inverse_and_log_det))

                flow = masked_autoregressive_flow(
                    fit_keys[i],
                    base_dist=Normal(jnp.zeros(chain_entry.shape[1])),
                    transformer=RationalQuadraticSpline(knots=knots, interval=interval),
                    invert=True,
                    nn_depth=2,
                )
                flow = fit_chain_entry_standardized(
                    flow, updated_transform, updated_inverse_log_det,
                    chain_entry, rng_seed=rng_seed+i
                )
                flows[i] = flow
                # Update the transforms and inverse_log_dets for this source
                transforms[i] = updated_transform
                inverse_log_dets[i] = updated_inverse_log_det
                training_history[i] = {
                    'hash': chain_hash,
                    'n_samples': len(chain_entry),
                    'flow_type': 'trained'
                }

        if plot_flows:
            display_standardized_flows(chain, flows, transforms, current_iteration[0])

        return flows

    # Attach utility functions
    standardized_flows_fit.clear_cache = clear_training_cache
    standardized_flows_fit.set_iteration = set_iteration

    return standardized_flows_fit


def display_standardized_flows(chain: np.ndarray, flows, transforms, iteration: int = None):
    """Display comparison plots for standardized flows."""
    from matplotlib import pyplot as plt
    uniform_prior = make_uniform_prior(chain)

    key = jr.key(999)

    # Count flows to plot
    n_sources_to_plot = sum(1 for flow in flows if flow is not uniform_prior)
    if n_sources_to_plot == 0:
        print("No trained flows to plot (all sources using uniform prior)")
        return

    n_params = chain.shape[2]
    fig, axes = plt.subplots(n_sources_to_plot, n_params, figsize=(4*n_params, 3*n_sources_to_plot))
    if n_sources_to_plot == 1:
        axes = axes.reshape(1, -1)
    if n_params == 1:
        axes = axes.reshape(-1, 1)

    plot_row = 0
    for i, (flow, transform) in enumerate(zip(flows, transforms)):
        if flow is uniform_prior:
            continue

        # Get clean samples for this source
        x = chain[:, i, :]
        x = x[~np.isnan(x).any(axis=1)]
        n_samples = len(x)

        # Sample from flow
        key, subkey = jr.split(key)
        try:
            x_samples = flow.sample(subkey, (n_samples,))
        except Exception as e:
            print(f"Error sampling from flow for source {i}: {e}")
            continue

        # Plot marginals
        for j in range(x.shape[1]):
            ax = axes[plot_row, j] if n_sources_to_plot > 1 else axes[j]

            # Original samples
            counts, bins, _ = ax.hist(
                x[:, j], bins=40, density=True, alpha=0.7,
                color='blue', label="Original samples"
            )

            # Flow samples
            ax.hist(
                x_samples[:, j], bins=bins, density=True, alpha=0.7,
                color='red', label="Flow samples"
            )

            # Statistics
            orig_mean, orig_std = np.mean(x[:, j]), np.std(x[:, j])
            flow_mean, flow_std = np.mean(x_samples[:, j]), np.std(x_samples[:, j])

            title = f"Source {i} • Param {j}"
            if iteration is not None:
                title += f" (Iter {iteration})"
            title += f"\nOrig: μ={orig_mean:.3f}, σ={orig_std:.3f}"
            title += f"\nFlow: μ={flow_mean:.3f}, σ={flow_std:.3f}"

            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plot_row += 1

    # Overall title
    main_title = "Standardized Flow Comparison"
    if iteration is not None:
        main_title += f" - Iteration {iteration}"
    fig.suptitle(main_title, fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def _safe_flow_log_prob_standardized(flow, sample):
    """Safely evaluate flow log_prob for standardized flows."""
    if not hasattr(flow, 'bijection'):
        # Uniform prior
        return flow.log_prob(sample)

    try:
        log_prob = flow.log_prob(sample)
        if jnp.all(jnp.isfinite(log_prob)):
            return log_prob
        else:
            return jnp.array(-50.0)
    except Exception:
        return jnp.array(-50.0)


def standardized_flows_aux_distribution(sample: np.ndarray,
                                       aux_parameters: List,
                                       source_index) -> np.ndarray:
    """Compute auxiliary distribution for standardized flows."""
    source_indices = jnp.atleast_1d(source_index)

    # Get log probabilities: each flow evaluates each source
    # sample has shape (n_sources, n_params)
    n_sources = sample.shape[0]
    n_flows = len(aux_parameters)

    log_probs_matrix = []

    for flow_idx, flow in enumerate(aux_parameters):
        source_log_probs = []
        for source_idx in range(n_sources):
            source_data = sample[source_idx]  # Get data for this source
            log_prob = _safe_flow_log_prob_standardized(flow, source_data)
            source_log_probs.append(log_prob)
        log_probs_matrix.append(source_log_probs)

    # Convert to array: shape (n_flows, n_sources)
    all_lp = jnp.array(log_probs_matrix)
    all_lp = jnp.where(jnp.isfinite(all_lp), all_lp, -50.0)

    return all_lp[source_indices]


def relabel_standardized_flows(posterior_chain: PosteriorChain,
                              max_num_sources: int = None,
                              num_iterations: int = 20,
                              eps: float = 1e-2,
                              plot_flows: bool = False,
                              knots: int = 16,
                              interval: float = 4.0):
    """
    Relabel samples using standardized normalizing flows.

    This approach uses simple standardization instead of copula transforms,
    allowing natural label swapping while maintaining numerical stability.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain to relabel
    max_num_sources : int, optional
        Maximum number of sources (defaults to chain.num_sources)
    num_iterations : int, default 20
        Number of relabeling iterations
    eps : float, default 1e-2
        Convergence tolerance
    plot_flows : bool, default False
        Whether to plot flow quality
    knots : int, default 16
        Number of spline knots
    interval : float, default 4.0
        Spline interval size

    Returns
    -------
    relabeled_chain : PosteriorChain
        Relabeled chain with standardized flows
    """
    # Create standardized flow fitter
    standardized_flows_fit = make_standardized_flows_fit(
        posterior_chain.chain,
        max_num_sources or posterior_chain.num_sources,
        rng_seed=999,
        threshold_samples=50,
        plot_flows=plot_flows,
        knots=knots,
        interval=interval
    )

    # Create relabeling function
    # Note: Removed filter_jit wrapper as it causes JAX compilation issues with flows
    relabel_samples = create_relabel_samples(
        standardized_flows_fit,
        standardized_flows_aux_distribution,
        eps=eps
    )

    # Perform relabeling
    return relabel_samples(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations
    )


def make_catalog_standardized_flows(posterior_chain: PosteriorChain,
                                   max_num_sources: int,
                                   num_iterations: int = 50,
                                   plot_flows: bool = False,
                                   knots: int = 16,
                                   interval: float = 4.0):
    """
    Create catalog using standardized normalizing flows.

    Uses simple mean/std standardization instead of copula transforms,
    preserving natural label swapping behavior while benefiting from
    improved numerical stability through standardization.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain to process
    max_num_sources : int
        Target number of sources
    num_iterations : int, default 50
        Number of relabeling iterations
    plot_flows : bool, default False
        Whether to plot flow quality
    knots : int, default 16
        Number of spline knots
    interval : float, default 4.0
        Spline interval size

    Returns
    -------
    relabeled_chain : PosteriorChain
        Relabeled chain using standardized flows
    """
    if posterior_chain.num_sources > max_num_sources:
        raise ValueError("max_num_sources must be >= chain.num_sources")

    # Expand chain if needed
    if posterior_chain.num_sources < max_num_sources:
        print("Expanding posterior chain to max_num_sources.")
        posterior_chain = posterior_chain.expand_chain(max_num_sources)

    # Relabel using standardized flows
    print("Relabeling with standardized normalizing flows.")
    relabeled_chain = relabel_standardized_flows(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        plot_flows=plot_flows,
        knots=knots,
        interval=interval
    )

    return relabeled_chain