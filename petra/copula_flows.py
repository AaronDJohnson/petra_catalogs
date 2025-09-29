from typing import Callable, List, Optional
import numpy as np
from petra.parametric_fits import update_parametric_fit_and_prob_in_model
from petra.cost_matrix import create_compute_cost_matrix
from petra.relabel import relabel_posterior_chain_one_iteration
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Stack
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed
from paramax import non_trainable
from petra.posterior_chain import PosteriorChain
from petra.utils import make_uniform_prior
from petra.relabel import create_relabel_samples
from petra.bijections import EmpiricalMarginalToGaussian, NormalToUniformInverseStandardize
from petra.utils import find_uniform_bounds
from matplotlib import pyplot as plt
from equinox import filter_jit


jax.config.update('jax_enable_x64', False)


def _create_empirical_transforms_for_source(chain_entry: np.ndarray, uniform_lower_bound: np.ndarray, uniform_upper_bound: np.ndarray, tail_decay: float = 4.0):
    """
    Create empirical marginal transforms for a single source's data.

    Parameters
    ----------
    chain_entry : ndarray
        Data for one source, shape (n_samples, n_params)
    uniform_lower_bound : ndarray
        Global lower bounds for fallback transforms
    uniform_upper_bound : ndarray
        Global upper bounds for fallback transforms
    tail_decay : float
        Controls extrapolation decay rate in tails

    Returns
    -------
    transform : Stack
        Stacked empirical transforms for this source
    inverse_log_det : callable
        JIT-compiled inverse log determinant function
    """
    x = chain_entry[~np.isnan(chain_entry).any(axis=1)]
    empirical_transforms = []

    for j in range(x.shape[1]):
        param_samples = x[:, j]
        if len(param_samples) > 20:  # Need sufficient samples for empirical CDF
            empirical_transforms.append(non_trainable(EmpiricalMarginalToGaussian(param_samples, tail_decay=tail_decay)))
        else:
            # Fallback to simple transform if insufficient samples
            empirical_transforms.append(non_trainable(NormalToUniformInverseStandardize(
                uniform_lower_bound[j], uniform_upper_bound[j], np.mean(param_samples), np.std(param_samples)
            )))

    transform = Stack(empirical_transforms)
    inverse_log_det = jax.jit(jax.vmap(transform.inverse_and_log_det))

    return transform, inverse_log_det


def display_flows(chain: np.ndarray, flows, transforms, iteration: int = None):
    uniform_prior = make_uniform_prior(chain)

    key = jr.key(999)  # Use a fixed key for reproducibility

    # Create a figure with subplots for all sources and parameters
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

        # pull out all non-NaN samples for source i
        x = chain[:, i, :]
        x = x[~np.isnan(x).any(axis=1)]
        n_samples, n_params_actual = x.shape

        # sample from the flow
        key, subkey = jr.split(key)
        try:
            x_samples = filter_jit(flow.sample)(subkey, (n_samples,))
        except Exception as e:
            print(f"Error sampling from flow for source {i}: {e}")
            continue

        # plot each marginal
        for j in range(n_params_actual):
            ax = axes[plot_row, j] if n_sources_to_plot > 1 else axes[j]

            # Original samples histogram
            counts, bins, _ = ax.hist(
                x[:, j],
                bins=40,
                density=True,
                alpha=0.7,
                color='blue',
                label="Original samples",
            )

            # Flow samples histogram
            ax.hist(
                x_samples[:, j],
                bins=bins,
                density=True,
                alpha=0.7,
                color='red',
                label="Flow samples",
            )

            # Statistics comparison
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
    main_title = "Normalizing Flow Comparison"
    if iteration is not None:
        main_title += f" - Iteration {iteration}"
    fig.suptitle(main_title, fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # Print summary statistics
    print(f"\n{'='*50}")
    if iteration is not None:
        print(f"Flow Quality Summary - Iteration {iteration}")
    else:
        print("Flow Quality Summary")
    print(f"{'='*50}")

    for i, flow in enumerate(flows):
        if flow is uniform_prior:
            print(f"Source {i}: Using uniform prior (insufficient samples)")
        else:
            x = chain[:, i, :]
            x = x[~np.isnan(x).any(axis=1)]
            print(f"Source {i}: {len(x)} samples, trained flow")
    print()


def _compute_adaptive_flow_params(chain_entry: np.ndarray, tail_decay: float = 2.0, base_knots: int = 16, base_interval: float = 4.0):
    """
    Compute adaptive flow architecture parameters based on data characteristics.
    Fast approximation using data range instead of full marginal transforms.

    Parameters
    ----------
    chain_entry : ndarray
        Training data for one source
    tail_decay : float
        Tail decay parameter for marginal transforms (used for scaling)
    base_knots : int
        Base number of knots
    base_interval : float
        Base interval size

    Returns
    -------
    adaptive_knots : int
        Number of knots to use
    adaptive_interval : float
        Interval size to use
    """
    if len(chain_entry) == 0:
        return base_knots, base_interval

    # Fast approximation: use data range to estimate z-range
    # This avoids expensive marginal transform computations
    marginal_ranges = []
    for j in range(chain_entry.shape[1]):
        param_samples = chain_entry[:, j]
        if len(np.unique(param_samples)) > 1:
            # Estimate z-range from data range using empirical scaling
            data_range = np.max(param_samples) - np.min(param_samples)
            data_std = np.std(param_samples)

            # Conservative estimate: larger ranges need more support
            # This approximates what EmpiricalMarginalToGaussian would produce
            if data_std > 0:
                # Rough heuristic: z-range scales with data range/std, clamped by tail_decay
                estimated_z_range = min(data_range / (2 * data_std), 6.0 - tail_decay/2)
                marginal_ranges.append(max(estimated_z_range, 2.5))  # Minimum reasonable range
            else:
                marginal_ranges.append(base_interval / 1.5)
        else:
            marginal_ranges.append(base_interval / 1.5)

    # Use maximum range across all parameters
    max_z_range = max(marginal_ranges) if marginal_ranges else base_interval / 1.5

    # Adaptive interval: handle 150% of estimated range, with conservative bounds
    adaptive_interval = float(np.clip(max_z_range * 1.2, base_interval, 10.0))  # Reduced max from 12 to 10

    # Scale knots more conservatively to avoid excessive complexity
    adaptive_knots = min(base_knots + int((adaptive_interval - base_interval)), 32)  # Reduced max from 48 to 32
    
    return adaptive_knots, adaptive_interval

# @partial(jax.jit, static_argnames=['z_clamp_limit'])
# def _safe_flow_log_prob(flow, sample, z_clamp_limit=6.0):
#     """
#     Safely evaluate flow log_prob with z-value clamping to prevent out-of-support issues.

#     Parameters
#     ----------
#     flow : Flow object
#         The normalizing flow to evaluate
#     sample : ndarray
#         Sample to evaluate (will be transformed to Gaussian space first)
#     z_clamp_limit : float
#         Maximum |z| value allowed before clamping

#     Returns
#     -------
#     log_prob : float or ndarray
#         Log probability, with safety clamping applied
#     """

#     # Standard evaluation - let's try it first
#     log_prob = flow.log_prob(sample)

#     # Check if result is reasonable
#     return log_prob

#     # except Exception:
#     #     # Fallback: manually transform with clamping
#     #     # Transform sample through the bijection with clamping
#     #     z, log_det = flow.bijection.inverse_and_log_det(sample)

#     #     # Clamp z values to safe range
#     #     z_clamped = jnp.clip(z, -z_clamp_limit, z_clamp_limit)

#     #     # Evaluate base distribution (typically Normal)
#     #     base_log_prob = flow.base_dist.log_prob(z_clamped)

#     #     # Add jacobian (note: this is approximate due to clamping)
#     #     return base_log_prob + log_det


def fit_chain_entry(input_flow, transform, inverse_log_det, chain_entry: np.ndarray, rng_seed: int = 999, max_epochs: int = 100):
    """Enhanced fit_chain_entry with adaptive training strategies and optional compactness penalty."""
    key = jr.key(rng_seed)
    key, subkey_2 = jr.split(key)
    x_train, __ = inverse_log_det(chain_entry)

    # Adaptive training parameters based on iteration
    learning_rate = 1e-3
    patience = 5
    epochs = max_epochs

    # Train with adaptive parameters and optional compactness penalty
    kwargs = {
        'max_epochs': epochs,
        'max_patience': patience,
        'learning_rate': learning_rate
    }

    # Standard training
    updated_flow, losses = fit_to_data(subkey_2, input_flow, x_train, **kwargs)

    final_flow = Transformed(updated_flow, transform)  # unstandardize and back to uniform distribution
    return final_flow


def normalizing_flows_aux_distribution(sample: np.ndarray,
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


def relabel_normalizing_flows_with_plots(posterior_chain: PosteriorChain, normalizing_flows_fit, max_num_sources, num_iterations, eps, checkpoint_dir: Optional[str] = None):
    """Custom relabeling process that updates iteration numbers for plotting."""
    

    if max_num_sources is None:
        max_num_sources = posterior_chain.num_sources
    if max_num_sources > posterior_chain.num_sources:
        posterior_chain = posterior_chain.expand_chain(max_num_sources)
    if max_num_sources < posterior_chain.num_sources:
        raise ValueError("The maximum number of sources cannot be less than the number of entries in the chain.")

    print()
    print(f"Sorting the posterior chain with flow plotting enabled:\n\tMaximum number of iterations: {num_iterations}\n\tMaximum number of source labels: {max_num_sources}\n")

    # Set up the compute cost matrix function
    compute_cost_matrix = create_compute_cost_matrix(normalizing_flows_aux_distribution)

    # Set up the for loop
    old_posterior_chain = posterior_chain
    old_parametric_fit, old_prob_in_model = update_parametric_fit_and_prob_in_model(posterior_chain, max_num_sources, normalizing_flows_fit, eps=eps)
    old_cost_of_assignment = 0

    for iteration in range(num_iterations):

        # get the new values
        print("relabeling samples...")
        new_posterior_chain = relabel_posterior_chain_one_iteration(old_posterior_chain, old_parametric_fit, old_prob_in_model, max_num_sources, compute_cost_matrix)
        print('updating parametric fit and probabilities in model...')
        new_parametric_fit, new_prob_in_model = update_parametric_fit_and_prob_in_model(new_posterior_chain, max_num_sources, normalizing_flows_fit, eps=eps)
        print('updating cost of assignment...')
        new_cost_of_assignment = new_posterior_chain.cost_dict[max_num_sources]

        # print out the results
        delta_cost_of_assignment = new_cost_of_assignment - old_cost_of_assignment
        print(f"Iteration {iteration + 1}: Difference in cost of assignment is {delta_cost_of_assignment} with total cost of {new_cost_of_assignment}.")
        print(f"\tProbabilities in model: {new_prob_in_model}")

        # Save checkpoint if requested
        if checkpoint_dir is not None:
            from petra.relabel import _checkpoint_posterior_chain
            _checkpoint_posterior_chain(new_posterior_chain, checkpoint_dir, iteration + 1)

        # break if converged
        if (delta_cost_of_assignment == 0):
            print(f"Stopped after {iteration + 1} iterations because the cost didn't change from the previous iteration.")
            new_parametric_fit, new_prob_in_model = update_parametric_fit_and_prob_in_model(old_posterior_chain, max_num_sources, normalizing_flows_fit, eps=0)
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

def make_normalizing_flows_fit(chain:np.ndarray, max_num_sources: int, rng_seed: int = 999, threshold_samples: int = 50, tail_decay: float = 4.0) -> Callable:
    """
    Fit a flow to each entry in the chain with smooth extrapolation support.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples, possibly containing NaNs for missing sources.
    rng_seed : int, optional
        Random seed for reproducibility (default is 999).
    tail_decay : float, optional
        Controls extrapolation decay rate in tails (default is 4.0).
        Higher values = slower decay (more conservative).
        Lower values = faster decay (more aggressive penalty for outliers).
    compactness_weight : float, optional
        Weight for compactness penalty in flow training (default is 0.0).
        Higher values encourage blob-like, compact distributions.
        Typical range: 0.0 (no penalty) to 1.0 (strong penalty).

    Returns
    -------
    list of Transformed
        List of fitted flows for each chain entry.
    """
    flows = []
    transforms = []
    inverse_log_dets = []
    uniform_lower_bound, uniform_upper_bound = find_uniform_bounds(chain)
    uniform_prior = make_uniform_prior(chain)
    key = jr.key(rng_seed)
    keys = jr.split(key, max_num_sources + 1)  # +1 if you want to keep master key
    for i in range(max_num_sources):
        subkey_i = keys[i]
        chain_entry = chain[:, i, :]  # Get the i-th entry across all samples
        # Create initial transforms using the helper function
        transform, inverse_log_det = _create_empirical_transforms_for_source(
            chain_entry, uniform_lower_bound, uniform_upper_bound, tail_decay
        )
        transforms.append(transform)
        inverse_log_dets.append(inverse_log_det)

        # Clean data for flow creation
        x = chain_entry[~np.isnan(chain_entry).any(axis=1)]
        if x.shape[0] <= threshold_samples:
            flows.append(uniform_prior)
        else:
            # Compute adaptive flow parameters based on data characteristics
            adaptive_knots, adaptive_interval = _compute_adaptive_flow_params(x, tail_decay)
            
            flow = masked_autoregressive_flow(
                subkey_i,
                base_dist=Normal(jnp.zeros(x.shape[1])),
                transformer=RationalQuadraticSpline(knots=adaptive_knots, interval=adaptive_interval),
                invert=True,
                nn_depth=2,
            )
            flows.append(flow)

    def normalizing_flows_fit(chain: np.ndarray, max_num_sources: int, flow_knots:int = 16, flow_interval:int = 4, plot_flows=True):
        rng_key = jr.key(rng_seed + 1)
        fit_keys = jr.split(rng_key, max_num_sources)

        for i in range(max_num_sources):
            flow = flows[i]
            chain_entry = chain[:, i, :]  # Get the i-th source across all samples
            chain_entry = chain_entry[~jnp.isnan(chain_entry).any(axis=(1))]  # Remove samples with NaNs in this source

            if chain_entry.shape[0] <= threshold_samples:
                flows[i] = uniform_prior

            elif (flow is uniform_prior) and (chain_entry.shape[0] > threshold_samples):
                # Transitioning from uniform to flow - update transforms with current data
                transforms[i], inverse_log_dets[i] = _create_empirical_transforms_for_source(
                    np.array(chain_entry), uniform_lower_bound, uniform_upper_bound, tail_decay
                )

                flow = masked_autoregressive_flow(
                    fit_keys[i],
                    base_dist=Normal(jnp.zeros(chain_entry.shape[1])),
                    transformer=RationalQuadraticSpline(knots=flow_knots, interval=flow_interval),
                    invert=True,
                )
                flow = fit_chain_entry(flow, transforms[i], inverse_log_dets[i], chain_entry, rng_seed=rng_seed+i)
                flows[i] = flow

            else:
                # Retraining existing flow - update transforms with current iteration's data
                transforms[i], inverse_log_dets[i] = _create_empirical_transforms_for_source(
                    np.array(chain_entry), uniform_lower_bound, uniform_upper_bound, tail_decay
                )

                flow = masked_autoregressive_flow(
                    fit_keys[i],
                    base_dist=Normal(jnp.zeros(chain_entry.shape[1])),
                    transformer=RationalQuadraticSpline(knots=flow_knots, interval=flow_interval),
                    invert=True,
                )
                flow = fit_chain_entry(flow, transforms[i], inverse_log_dets[i], chain_entry, rng_seed=rng_seed+i)
                flows[i] = flow
        if plot_flows:
            display_flows(chain, flows, transforms, iteration=None)
        return flows

    # def normalizing_flows_fit(chain: np.ndarray, max_num_sources: int):
    #     # worker that fits one source-index
    #     def _fit_one(i):
    #         sub = chain[:, i, :]
    #         sub = sub[~jnp.isnan(sub).any(axis=1)]
    #         if sub.shape[0] <= threshold_samples:
    #             return uniform_prior
    #         return fit_chain_entry(
    #             initial_flows[i],
    #             transforms[i],
    #             inverse_log_dets[i],
    #             sub,
    #             rng_seed + i  # vary seed per worker if you like
    #         )

    #     # parallel map over all source‐indices
    #     with ProcessPoolExecutor() as exe:
    #         updated_flows = list(exe.map(_fit_one, range(max_num_sources)))
    #     return updated_flows

    # flows = inner_fit(chain, uniform_prior, max_num_sources, rng_seed = rng_seed, threshold_samples = threshold_samples)

    return normalizing_flows_fit


def relabel_normalizing_flows(posterior_chain: PosteriorChain,
                              max_num_sources: int|None = None,
                              num_iterations: int = 20,
                              eps=1e-6,
                              plot_flows: bool = False,
                              tail_decay: float = 4.0,
                              checkpoint_dir: Optional[str] = None):
    normalizing_flows_fit = make_normalizing_flows_fit(posterior_chain.chain, max_num_sources, rng_seed = 999, threshold_samples = 100, tail_decay=tail_decay)

    if plot_flows:
        # Use a custom relabeling process that tracks iterations for plotting
        return relabel_normalizing_flows_with_plots(posterior_chain, normalizing_flows_fit, max_num_sources, num_iterations, eps, checkpoint_dir)
    else:
        # Use the standard relabeling process
        # Note: Removed filter_jit wrapper as it causes JAX compilation issues with flows
        relabel_samples = create_relabel_samples(normalizing_flows_fit,
                                                 normalizing_flows_aux_distribution,
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
                              plot_flows: bool = False,
                              tail_decay: float = 4.0,
                              eps=1e-6,
                              checkpoint_dir: Optional[str] = None):
    """
    Create a catalog using Gaussian copula marginal transforms + normalizing flows with smooth extrapolation.

    This hybrid approach:
    1. Uses empirical CDFs with smooth extrapolation to transform each parameter's marginal to Gaussian
    2. Applies normalizing flows to model the dependence structure in Gaussian space
    3. Should handle non-uniform marginals better than standard flows
    4. Uses smooth exponential extrapolation to avoid artificial boundary concentration

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
    print("Relabeling with Gaussian copula marginals + normalizing flows (smooth extrapolation).")
    relabeled_chain = relabel_normalizing_flows(initial_posterior_chain,
                                        max_num_sources=max_num_sources,
                                        num_iterations=num_iterations,
                                        plot_flows=plot_flows,
                                        tail_decay=tail_decay,
                                        eps=eps,
                                        checkpoint_dir=checkpoint_dir)

    return relabeled_chain
