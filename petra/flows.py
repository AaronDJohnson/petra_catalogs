from typing import ClassVar, List, Callable
import numpy as np
from equinox import filter_jit

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import erf, erfinv

from flowjax.bijections import AbstractBijection, Stack
from flowjax.train import fit_to_data
from flowjax.flows import masked_autoregressive_flow
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed

from paramax import non_trainable

from petra.posterior_chain import PosteriorChain
from petra.utils import make_uniform_prior
from petra.relabel import create_relabel_samples
from scipy.interpolate import PchipInterpolator

jax.config.update('jax_enable_x64', False)


def make_empirical_cdf_spline(x_grid, samples, tail_scale=None, min_eps=1e-7):
    """
    Create empirical CDF and PDF splines with smooth extrapolation.
    
    IMPROVED VERSION: Uses smooth exponential extrapolation instead of hard boundaries
    to preserve natural likelihood ordering for out-of-support samples.
    
    Parameters
    ----------
    x_grid : array_like
        Grid points for CDF evaluation
    samples : array_like
        Training data samples
    tail_scale : float, optional
        Controls extrapolation decay rate. If None, computed adaptively.
    min_eps : float
        Minimum probability to prevent numerical issues
    """
    from petra.utils import process_array

    # 1. Sort the grid
    x_grid = np.asarray(x_grid)
    sort_idx = np.argsort(x_grid)
    xg = x_grid[sort_idx]

    # 2. Empirical CDF
    sorted_chain = process_array(samples)
    counts = np.searchsorted(sorted_chain, xg, side='right')
    cdf_vals = counts / len(sorted_chain)

    # 3. Data bounds and adaptive tail scale
    data_min, data_max = np.min(samples), np.max(samples)
    
    # if tail_scale is None:
    #     # Adaptive tail scale based on data characteristics
    #     q25, q75 = np.percentile(samples, [25, 75])
    #     iqr = q75 - q25
    #     data_range = data_max - data_min
    #     tail_scale = max(iqr / 4, data_range / 10, 0.1)  # Ensure minimum scale

    # 4. Create monotonic spline for interior region
    try:
        cs = PchipInterpolator(xg, cdf_vals, extrapolate=False)
    except ValueError as e:
        # Fallback to simple linear interpolation
        from scipy.interpolate import interp1d
        cs = interp1d(xg, cdf_vals, kind='linear', bounds_error=False, fill_value=(0, 1))

    # 5. Smooth CDF function with exponential extrapolation
    def cdf_fn(u):
        u = np.asarray(u)
        y = np.zeros_like(u, dtype=float)
        
        # Masks for different regions
        below_mask = u < data_min
        above_mask = u > data_max
        inside_mask = ~(below_mask | above_mask)
        
        # Smooth extrapolation below minimum
        if np.any(below_mask):
            distances = data_min - u[below_mask]
            y[below_mask] = min_eps * np.exp(-distances / tail_scale)
            
        # Smooth extrapolation above maximum
        if np.any(above_mask):
            distances = u[above_mask] - data_max
            y[above_mask] = 1.0 - min_eps * np.exp(-distances / tail_scale)
            
        # Interior: use spline
        if np.any(inside_mask):
            y[inside_mask] = cs(u[inside_mask])
            # Handle any NaNs from spline
            nan_mask = np.isnan(y[inside_mask])
            if np.any(nan_mask):
                # Fallback to linear interpolation
                y[inside_mask][nan_mask] = np.interp(
                    u[inside_mask][nan_mask], xg, cdf_vals
                )
        
        # Final bounds check
        y = np.clip(y, min_eps, 1.0 - min_eps)
        return y

    # 6. Smooth PDF function  
    def pdf_fn(u):
        u = np.asarray(u)
        y = np.zeros_like(u, dtype=float)
        
        # Masks for different regions
        below_mask = u < data_min
        above_mask = u > data_max
        inside_mask = ~(below_mask | above_mask)
        
        # Analytical derivatives in tails (from exponential extrapolation)
        if np.any(below_mask):
            distances = data_min - u[below_mask]
            y[below_mask] = (min_eps / tail_scale) * np.exp(-distances / tail_scale)
            
        if np.any(above_mask):
            distances = u[above_mask] - data_max
            y[above_mask] = (min_eps / tail_scale) * np.exp(-distances / tail_scale)
            
        # Interior: finite difference approximation
        if np.any(inside_mask):
            eps_fd = min(1e-6, tail_scale / 1000)  # Adaptive step size
            cdf_plus = cdf_fn(u[inside_mask] + eps_fd)
            cdf_minus = cdf_fn(u[inside_mask] - eps_fd)
            y[inside_mask] = (cdf_plus - cdf_minus) / (2 * eps_fd)
            
        # Ensure positive and bounded
        y = np.maximum(y, min_eps)
        return y

    return cdf_vals[np.argsort(sort_idx)], cdf_fn, pdf_fn


class EmpiricalMarginalToGaussian(AbstractBijection):
    """
    Transform from empirical marginal to standard Gaussian with smooth extrapolation.
    
    IMPROVED VERSION: Uses smooth exponential extrapolation beyond training data bounds
    to preserve natural likelihood ordering for out-of-support samples.
    """

    x_grid: jnp.ndarray
    raw_cdf: jnp.ndarray
    data_min: float
    data_max: float
    tail_scale: float
    min_eps: float = 1e-7
    cond_shape: ClassVar[None] = None
    shape: ClassVar[tuple[int, ...]] = ()

    def __init__(self, samples: np.ndarray, num_points: int = 200, tail_decay: float = 2.0):
        """
        Initialize empirical marginal transformation with smooth extrapolation.
        
        Parameters
        ----------
        samples : ndarray
            Clean samples (no NaNs) for this parameter
        num_points : int
            Number of quantile points for CDF approximation
        tail_decay : float
            Controls how quickly probability decays in tails (higher = slower decay)
        """
        # Compute adaptive tail scale based on data characteristics
        q25, q75 = np.percentile(samples, [25, 75])
        iqr = q75 - q25
        data_range = np.max(samples) - np.min(samples)
        tail_scale = max(iqr / tail_decay, data_range / 10, 0.1)
        
        # Create empirical CDF with smooth extrapolation
        x_grid = np.quantile(samples, np.linspace(0, 1, num_points))
        raw_cdf, _, _ = make_empirical_cdf_spline(x_grid, samples, tail_scale=tail_scale)

        # Store data as JAX arrays for Equinox compatibility
        self.x_grid = jnp.array(x_grid)
        self.raw_cdf = jnp.array(raw_cdf)
        self.data_min = float(np.min(samples))
        self.data_max = float(np.max(samples))
        self.tail_scale = tail_scale
        self.min_eps = 1e-7

    def _smooth_interp_cdf(self, x):
        """
        Interpolate CDF using smooth extrapolation.

        IMPROVED VERSION: Uses exponential decay in tails instead of hard clamping
        to preserve natural likelihood ordering for outliers.
        """
        # Use JIT-compiled version for better performance
        return _jit_smooth_interp_cdf(x, self.x_grid, self.raw_cdf,
                                    self.data_min, self.data_max,
                                    self.tail_scale, self.min_eps)

    def _smooth_finite_diff_pdf(self, x):
        """
        Estimate PDF using smooth extrapolation.

        Uses finite differences with the smooth CDF function.
        """
        # Use JIT-compiled version for better performance
        return _jit_smooth_finite_diff_pdf(x, self.x_grid, self.raw_cdf,
                                         self.data_min, self.data_max,
                                         self.tail_scale, self.min_eps)

    def inverse_and_log_det(self, x, condition=None):
        """Transform from original space to standard Gaussian with smooth extrapolation."""
        # Get CDF value using smooth interpolation
        u = self._smooth_interp_cdf(x)

        # Transform to standard Gaussian
        from jax.scipy.special import ndtri  # Inverse normal CDF
        z = ndtri(u)  # JAX equivalent of norm.ppf

        # Jacobian: |dz/dx| = |dz/du| * |du/dx| = (1/φ(z)) * pdf(x)
        pdf_x = self._smooth_finite_diff_pdf(x)

        # JAX normal PDF
        pdf_z = jnp.exp(-0.5 * z**2) / jnp.sqrt(2 * jnp.pi)
        log_det = jnp.log(pdf_x) - jnp.log(pdf_z)

        return z, jnp.sum(log_det)

    def transform_and_log_det(self, z, condition=None):
        """Transform from standard Gaussian to original space with smooth extrapolation."""
        # Transform to uniform using JAX normal CDF
        from jax.scipy.special import ndtr  # Normal CDF
        u = ndtr(z)  # JAX equivalent of norm.cdf
        u = jnp.clip(u, self.min_eps, 1.0 - self.min_eps)

        # Inverse CDF with smooth extrapolation handling
        # For very low u (left tail)
        x_left = jnp.where(
            u <= self.min_eps * 2,  # Safety factor
            self.data_min + self.tail_scale * jnp.log(u / self.min_eps),
            0.0
        )
        
        # For very high u (right tail) 
        x_right = jnp.where(
            u >= 1.0 - self.min_eps * 2,  # Safety factor
            self.data_max - self.tail_scale * jnp.log((1.0 - u) / self.min_eps),
            0.0
        )
        
        # For interior (use interpolation)
        x_interior = jnp.where(
            (u > self.min_eps * 2) & (u < 1.0 - self.min_eps * 2),
            jnp.interp(u, self.raw_cdf, self.x_grid),
            0.0
        )
        
        # Combine (only one will be non-zero for each point)
        x = x_left + x_right + x_interior

        # Jacobian: |dx/dz| = |dx/du| * |du/dz| = (1/pdf(x)) * φ(z)  
        pdf_x = self._smooth_finite_diff_pdf(x)

        # JAX normal PDF
        pdf_z = jnp.exp(-0.5 * z**2) / jnp.sqrt(2 * jnp.pi)
        log_det = jnp.log(1.0 / pdf_x) + jnp.log(pdf_z)

        return x, jnp.sum(log_det)


# JIT-compiled helper functions for improved performance
def _smooth_interp_cdf_impl(x, x_grid, raw_cdf, data_min, data_max, tail_scale, min_eps):
    """Implementation of smooth CDF interpolation."""
    # Smooth extrapolation below minimum
    below_contrib = jnp.where(
        x < data_min,
        min_eps * jnp.exp(-(data_min - x) / tail_scale),
        0.0
    )

    # Smooth extrapolation above maximum
    above_contrib = jnp.where(
        x > data_max,
        1.0 - min_eps * jnp.exp(-(x - data_max) / tail_scale),
        0.0
    )

    # Interior: JAX-compatible interpolation
    interior_contrib = jnp.where(
        (x >= data_min) & (x <= data_max),
        jnp.interp(x, x_grid, raw_cdf),
        0.0
    )

    # Combine contributions
    u = below_contrib + above_contrib + interior_contrib

    # Ensure numerical bounds
    return jnp.clip(u, min_eps, 1.0 - min_eps)


def _smooth_finite_diff_pdf_impl(x, x_grid, raw_cdf, data_min, data_max, tail_scale, min_eps):
    """Implementation of smooth PDF estimation."""
    # Adaptive step size for finite differences
    eps_fd = tail_scale / 1000

    cdf_plus = _smooth_interp_cdf_impl(x + eps_fd, x_grid, raw_cdf, data_min, data_max, tail_scale, min_eps)
    cdf_minus = _smooth_interp_cdf_impl(x - eps_fd, x_grid, raw_cdf, data_min, data_max, tail_scale, min_eps)
    pdf = (cdf_plus - cdf_minus) / (2 * eps_fd)

    return jnp.maximum(pdf, min_eps)


# Create JIT-compiled versions at module level
_jit_smooth_interp_cdf = jax.jit(_smooth_interp_cdf_impl)
_jit_smooth_finite_diff_pdf = jax.jit(_smooth_finite_diff_pdf_impl)


class NormalToUniform(AbstractBijection):
    r"""Bijection mapping x ∈ [a, b] → z ∈ ℝ via a uniform→normal CDF transform.

    The forward transform is

        u = clip((x - a) / (b - a), eps, 1 - eps)
        z = sqrt(2) * erfinv(2 u - 1)

    and the inverse is

        u = 0.5 * (1 + erf(z / sqrt(2)))
        x = a + u (b - a)

    Args
    ----
    a : array_like or float
        Lower bound(s) of the uniform support.
    b : array_like or float
        Upper bound(s) of the uniform support.
    eps : float, default=1e-6
        Clamping parameter to avoid CDF values exactly 0 or 1.
    """

    a: float
    b: float
    eps: float = 1e-6
    cond_shape: ClassVar[None] = None

    shape: ClassVar[tuple[int, ...]] = ()

    def inverse_and_log_det(self, x, condition=None):
        # map into (0,1)
        u = (x - self.a) / (self.b - self.a)
        u = jnp.clip(u, self.eps, 1.0 - self.eps)
        # uniform→normal
        y = jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)
        # log|dz/du|
        log_dz_du = 0.5 * jnp.log(2.0 * jnp.pi) + erfinv(2.0 * u - 1.0) ** 2
        # log|du/dx|
        log_du_dx = -jnp.log(self.b - self.a)
        log_det = jnp.sum(log_dz_du + log_du_dx)
        return y, log_det

    def transform_and_log_det(self, y, condition=None):
        # normal→uniform
        u = 0.5 * (1.0 + erf(y / jnp.sqrt(2.0)))
        # uniform→original
        x = self.a + u * (self.b - self.a)
        # log|dx/dy|
        log_du_dy = -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * y ** 2
        log_dx_du = jnp.log(self.b - self.a)
        log_det = jnp.sum(log_dx_du + log_du_dy)
        return x, log_det


class InverseStandardize(AbstractBijection):

    mean: float
    std: float
    cond_shape: ClassVar[None] = None

    shape: ClassVar[tuple[int, ...]] = ()

    def inverse_and_log_det(self, x, condition=None):
        u = (x - self.mean) / self.std
        log_du_dx = -jnp.log(self.std)
        log_det = jnp.sum(log_du_dx)
        return u, log_det

    def transform_and_log_det(self, y, condition=None):
        u = self.std * y + self.mean
        log_du_dy = jnp.log(self.std)
        log_det = jnp.sum(log_du_dy)
        return u, log_det


class NormalToUniformInverseStandardize(AbstractBijection):
    a: float
    b: float
    mean: float
    std: float
    eps: float = 1e-6
    cond_shape: ClassVar[None] = None

    shape: ClassVar[tuple[int, ...]] = ()

    def inverse_and_log_det(self, x, condition=None):
        # map into (0,1)
        u = (x - self.a) / (self.b - self.a)
        u = jnp.clip(u, self.eps, 1.0 - self.eps)
        # uniform→normal
        y = self.mean + self.std * jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)
        # standardize
        z = (y - self.mean) / self.std
        # log|dz/dy|
        log_dz_dy = -jnp.log(self.std)
        # log|dy/du|
        log_dy_du = jnp.log(self.std) + 0.5 * jnp.log(2.0 * jnp.pi) + erfinv(2.0 * u - 1.0) ** 2
        # log|du/dx|
        log_du_dx = -jnp.log(self.b - self.a)
        log_det = jnp.sum(log_dy_du + log_du_dx + log_dz_dy)
        return z, log_det

    def transform_and_log_det(self, z, condition=None):
        # unstandardize
        y = self.std * z + self.mean
        # normal→uniform
        u = 0.5 * (1.0 + erf((y - self.mean) / (self.std * jnp.sqrt(2.0))))
        # uniform→original
        x = self.a + u * (self.b - self.a)
        # jacobians:
        log_dz_dy = jnp.log(self.std)
        log_du_dy = -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * ((y - self.mean)/self.std)**2 - jnp.log(self.std)
        log_dx_du = jnp.log(self.b - self.a)
        log_det = jnp.sum(log_dz_dy + log_dx_du + log_du_dy)
        return x, log_det


# def find_uniform_parameters(chain_entry: np.ndarray, threshold: float = 3e-7):
#     p_values = []
#     for dim in range(chain_entry.shape[1]):
#         data = chain_entry[:, dim]
#         lower = data.min()
#         width = data.max() - lower
#         __, p_value = kstest(data, 'uniform', args=(lower, width))
#         # print(f"Axis {dim}: KS statistic = {stat:.4f}, p-value = {p_value:.4f}")
#         p_values.append(p_value)
#     # print(f"p-values: {p_values}")
#     # import matplotlib.pyplot as plt
#     # for i in range(len(p_values)):
#     #     plt.hist(chain_entry[:, i], bins=50, alpha=0.5, label=f"Param {i}")
#     #     plt.legend()
#     #     plt.show()
#     uniform_idxs = np.where(np.array(p_values) > threshold)[0]
#     return uniform_idxs


def find_uniform_bounds(chain: np.ndarray):
    all_entries = chain.reshape(-1, chain.shape[2])
    lower_bound = np.nanmin(all_entries, axis=0)
    upper_bound = np.nanmax(all_entries, axis=0)
    return lower_bound, upper_bound


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

# def fit_chain_entry_incremental(existing_flow, transform, inverse_log_det, chain_entry: np.ndarray, rng_seed: int = 999):
#     """Incrementally update an existing flow with fewer epochs."""
#     if existing_flow is None:
#         # No existing flow, do full training
#         return fit_chain_entry(None, transform, inverse_log_det, chain_entry, rng_seed, max_epochs=100)

#     # For incremental training, use fewer epochs and start from existing flow
#     key = jr.key(rng_seed)
#     key, subkey_2 = jr.split(key)
#     x_train, __ = inverse_log_det(chain_entry)

#     # Extract the base flow from the Transformed wrapper
#     if isinstance(existing_flow, Transformed):
#         base_flow = existing_flow.base_dist
#     else:
#         base_flow = existing_flow

#     # Incremental training with fewer epochs
#     updated_flow, losses = fit_to_data(subkey_2, base_flow, x_train, max_epochs=20, max_patience=3, learning_rate=5e-4)
#     final_flow = Transformed(updated_flow, transform)
#     return final_flow


def display_flows(chain: np.ndarray, flows, transforms, iteration: int = None):
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches
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
            x_samples = flow.sample(subkey, (n_samples,))
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
        x = chain_entry[~np.isnan(chain_entry).any(axis=1)]  # NaNs should have been removed already, but just in case!
        # Use empirical marginal transforms for better marginal modeling
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
        transforms.append(transform)
        inverse_log_dets.append(jax.jit(jax.vmap(transform.inverse_and_log_det)))
        if chain_entry.shape[0] <= threshold_samples:
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

    def normalizing_flows_fit(chain: np.ndarray, max_num_sources: int, flow_knots:int = 16, flow_interval:int = 4):
        rng_key = jr.key(rng_seed + 1)
        fit_keys = jr.split(rng_key, max_num_sources)

        for i in range(max_num_sources):
            flow = flows[i]
            chain_entry = chain[:, i, :]  # Get the i-th source across all samples
            chain_entry = chain_entry[~jnp.isnan(chain_entry).any(axis=(1))]  # Remove samples with NaNs in this source

            if chain_entry.shape[0] <= threshold_samples:
                flows[i] = uniform_prior

            elif (flow is uniform_prior) and (chain_entry.shape[0] > threshold_samples):
                # Transitioning from uniform to flow
                # Compute adaptive parameters for this specific chain_entry
                adaptive_knots, adaptive_interval = _compute_adaptive_flow_params(chain_entry, tail_decay)

                flow = masked_autoregressive_flow(
                    fit_keys[i],
                    base_dist=Normal(jnp.zeros(chain_entry.shape[1])),
                    transformer=RationalQuadraticSpline(knots=adaptive_knots, interval=adaptive_interval),
                    invert=True,
                )
                flow = fit_chain_entry(flow, transforms[i], inverse_log_dets[i], chain_entry, rng_seed=rng_seed+i, iteration=current_iteration[0])
                flows[i] = flow

            elif need_retrain:
                # Data has changed, decide between full retraining vs incremental
                existing_flow = flows[i] if flows[i] is not uniform_prior else None

                if existing_flow is None or force_retrain:
                    # No existing trained flow OR force_retrain=True: do full training with new flow
                    # Compute adaptive parameters for current chain_entry
                    adaptive_knots, adaptive_interval = _compute_adaptive_flow_params(chain_entry, tail_decay)

                    # Use more sophisticated architecture for later iterations
                    if current_iteration[0] > 3:
                        adaptive_knots = min(adaptive_knots + 8, 48)
                        adaptive_interval = min(adaptive_interval + 1, 12.0)

                    flow = masked_autoregressive_flow(
                        fit_keys[i],
                        base_dist=Normal(jnp.zeros(chain_entry.shape[1])),
                        transformer=RationalQuadraticSpline(knots=adaptive_knots, interval=adaptive_interval),
                        invert=True,
                    )
                    flow = fit_chain_entry(flow, transforms[i], inverse_log_dets[i], chain_entry, rng_seed=rng_seed+i, iteration=current_iteration[0])
                else:
                    # Use incremental training (only when force_retrain=False)
                    flow = fit_chain_entry_incremental(existing_flow, transforms[i], inverse_log_dets[i], chain_entry, rng_seed=rng_seed+i)

                flows[i] = flow

            # If no retraining needed, flows[i] remains unchanged (reused!)

        if plot_flows:
            display_flows(chain, flows, transforms, iteration=current_iteration[0] if current_iteration[0] > 0 else None)
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

    # Attach utility functions as attributes
    normalizing_flows_fit.clear_cache = clear_training_cache
    normalizing_flows_fit.set_iteration = set_iteration

    return normalizing_flows_fit


# def make_normalizing_flows_fit(chain:np.ndarray, rng_seed: int = 999, threshold_samples: int = 100, plot_flows: bool = False) -> Callable:
#     """
#     Fit a flow to each entry in the chain.

#     Parameters
#     ----------
#     chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
#         Posterior samples, possibly containing NaNs for missing sources.
#     rng_seed : int, optional
#         Random seed for reproducibility (default is 999).

#     Returns
#     -------
#     list of Transformed
#         List of fitted flows for each chain entry.
#     """
#     initial_flows = []
#     transforms = []
#     inverse_log_dets = []
#     uniform_prior = make_uniform_prior(chain)
#     for i in range(chain.shape[1]):
#         chain_entry = chain[:, i, :]  # Get the i-th entry across all samples
#         x = chain_entry[~np.isnan(chain_entry).any(axis=1)]  # NaNs should have been removed already, but just in case!
#         uniform_idxs = find_uniform_parameters(x)
#         print(uniform_idxs)
#         transform = Stack([non_trainable(InverseStandardize(np.mean(x[:, i]), np.std(x[:, i]))) if i not in uniform_idxs else non_trainable(NormalToUniform(np.min(x[:, i]), np.max(x[:, i]))) for i in range(x.shape[1])])
#         transforms.append(transform)
#         inverse_log_dets.append(jax.jit(jax.vmap(transform.inverse_and_log_det)))
#         key, subkey_1 = jr.split(jr.key(rng_seed))
#         flow = masked_autoregressive_flow(
#             subkey_1,
#             base_dist=Normal(jnp.zeros(x.shape[1])),
#             transformer=RationalQuadraticSpline(knots=32, interval=8),
#             invert=True,
#         )
#         initial_flows.append(flow)

#     # if plot_flows:
#     #     display_flows(chain, initial_flows, transforms)

#     def normalizing_flows_fit(chain: np.ndarray, max_num_sources: int):
#         updated_flows = []
#         for i in range(max_num_sources):
#             chain_entry = chain[:, i, :]  # Get the i-th source across all samples
#             chain_entry = chain_entry[~jnp.isnan(chain_entry).any(axis=(1))]  # Remove samples with NaNs in this source
#             # check for empty subchain
#             if chain_entry.shape[0] <= threshold_samples:  # How many samples are needed to fit a flow?
#                 updated_flows.append(uniform_prior)
#             else:
#                 flow = fit_chain_entry(initial_flows[i], transforms[i], inverse_log_dets[i], chain_entry)
#                 updated_flows.append(flow)
#         if plot_flows:
#             display_flows(chain, updated_flows, transforms)
#         return updated_flows

#     # def normalizing_flows_fit(chain: np.ndarray, max_num_sources: int):
#     #     # worker that fits one source-index
#     #     def _fit_one(i):
#     #         sub = chain[:, i, :]
#     #         sub = sub[~jnp.isnan(sub).any(axis=1)]
#     #         if sub.shape[0] <= threshold_samples:
#     #             return uniform_prior
#     #         return fit_chain_entry(
#     #             initial_flows[i],
#     #             transforms[i],
#     #             inverse_log_dets[i],
#     #             sub,
#     #             rng_seed + i  # vary seed per worker if you like
#     #         )

#     #     # parallel map over all source‐indices
#     #     with ProcessPoolExecutor() as exe:
#     #         updated_flows = list(exe.map(_fit_one, range(max_num_sources)))
#     #     return updated_flows

#     # flows = inner_fit(chain, uniform_prior, max_num_sources, rng_seed = rng_seed, threshold_samples = threshold_samples)

#     return normalizing_flows_fit

def _safe_flow_log_prob(flow, sample, z_clamp_limit=6.0):
    """
    Safely evaluate flow log_prob with z-value clamping to prevent out-of-support issues.

    Parameters
    ----------
    flow : Flow object
        The normalizing flow to evaluate
    sample : ndarray
        Sample to evaluate (will be transformed to Gaussian space first)
    z_clamp_limit : float
        Maximum |z| value allowed before clamping

    Returns
    -------
    log_prob : float or ndarray
        Log probability, with safety clamping applied
    """
    # Check if this is a uniform prior (has simple log_prob)
    if not hasattr(flow, 'bijection'):
        # This is likely uniform_prior - call directly
        return flow.log_prob(sample)

    # For trained flows, we need to be more careful
    try:
        # Standard evaluation - let's try it first
        log_prob = flow.log_prob(sample)

        # Check if result is reasonable
        if jnp.all(jnp.isfinite(log_prob)):
            return log_prob
        else:
            # If we get -inf or nan, fall back to clamped version
            raise ValueError("Non-finite log_prob detected")

    except Exception:
        # Fallback: manually transform with clamping
        # Transform sample through the bijection with clamping
        z, log_det = flow.bijection.inverse_and_log_det(sample)

        # Clamp z values to safe range
        z_clamped = jnp.clip(z, -z_clamp_limit, z_clamp_limit)

        # Evaluate base distribution (typically Normal)
        base_log_prob = flow.base_dist.log_prob(z_clamped)

        # Add jacobian (note: this is approximate due to clamping)
        return base_log_prob + log_det


# JIT-compiled version for single flow evaluation
@jax.jit
def _jit_safe_flow_log_prob_single(flow, sample, z_clamp_limit=6.0):
    """
    JIT-compiled version of safe flow log_prob evaluation for a single flow.
    This version assumes the flow has bijection (is a trained flow, not uniform prior).
    """
    # Direct evaluation first
    try:
        log_prob = flow.log_prob(sample)
        return jnp.where(jnp.isfinite(log_prob), log_prob, -1e10)
    except:
        # Fallback: manual evaluation with clamping
        z, log_det = flow.bijection.inverse_and_log_det(sample)
        z_clamped = jnp.clip(z, -z_clamp_limit, z_clamp_limit)
        base_log_prob = flow.base_dist.log_prob(z_clamped)
        return base_log_prob + log_det


def normalizing_flows_aux_distribution(sample: np.ndarray,
                                       aux_parameters: List,
                                       source_index) -> np.ndarray:

    # Ensure source_index is array-like.
    source_indices = jnp.atleast_1d(source_index)  # shape: (n,)

    # Get log probabilities from all flows using safe evaluation
    log_probs = [_safe_flow_log_prob(f, sample) for f in aux_parameters]

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


# JIT-compiled version for better performance during relabeling
def _jit_normalizing_flows_aux_distribution(flows, sample, source_index):
    """
    JIT-compiled version of normalizing_flows_aux_distribution for better performance.

    This function separates uniform priors from trained flows to enable JIT compilation.
    """
    source_indices = jnp.atleast_1d(source_index)

    # Separate flows into uniform priors and trained flows for different handling
    log_probs = []
    uniform_prior = make_uniform_prior(sample.reshape(1, -1))  # Temporary uniform prior

    for flow in flows:
        if not hasattr(flow, 'bijection') or flow is uniform_prior:
            # Uniform prior - use simple computation
            log_probs.append(flow.log_prob(sample))
        else:
            # Trained flow - use JIT-compiled safe evaluation
            log_probs.append(_jit_safe_flow_log_prob_single(flow, sample))

    all_lp = jnp.stack(log_probs, axis=0)
    all_lp = jnp.where(jnp.isfinite(all_lp), all_lp, -1e10)

    return all_lp[source_indices]


# Create specialized JIT functions for uniform and trained flows separately
@jax.jit
def _jit_uniform_flow_log_prob(sample, a, b):
    """JIT-compiled uniform prior log probability."""
    # Uniform distribution: log_prob = -log(volume) if inside bounds, -inf otherwise
    volume = jnp.prod(b - a)
    inside_bounds = jnp.all((sample >= a) & (sample <= b))
    return jnp.where(inside_bounds, -jnp.log(volume), -jnp.inf)

@jax.jit
def _jit_trained_flow_log_prob(flow, sample):
    """JIT-compiled trained flow log probability with safety checks."""
    try:
        log_prob = flow.log_prob(sample)
        return jnp.where(jnp.isfinite(log_prob), log_prob, -1e10)
    except:
        # Fallback: manual evaluation with clamping
        z, log_det = flow.bijection.inverse_and_log_det(sample)
        z_clamped = jnp.clip(z, -6.0, 6.0)
        base_log_prob = flow.base_dist.log_prob(z_clamped)
        return base_log_prob + log_det


def _compute_cost_matrix_batch_optimized(samples, flows, uniform_bounds=None):
    """
    Optimized batch computation of cost matrices.

    This function separates uniform flows from trained flows to enable better JIT optimization.
    """
    batch_size, n_sources, n_params = samples.shape
    n_flows = len(flows)

    # Pre-allocate result array
    all_log_probs = jnp.zeros((batch_size, n_flows))

    # Process each flow
    for flow_idx, flow in enumerate(flows):
        if not hasattr(flow, 'bijection'):
            # Uniform prior - use bounds-based computation
            if uniform_bounds is not None:
                a, b = uniform_bounds
                # Compute for all samples at once
                flow_log_probs = jax.vmap(_jit_uniform_flow_log_prob, in_axes=(0, None, None))(
                    samples.reshape(batch_size, -1), a, b
                )
            else:
                # Fallback to direct evaluation
                flow_log_probs = jax.vmap(flow.log_prob)(samples.reshape(batch_size, -1))
        else:
            # Trained flow - use specialized JIT function
            flow_log_probs = jax.vmap(_jit_trained_flow_log_prob, in_axes=(None, 0))(
                flow, samples.reshape(batch_size, -1)
            )

        all_log_probs = all_log_probs.at[:, flow_idx].set(flow_log_probs)

    # Ensure finite values
    all_log_probs = jnp.where(jnp.isfinite(all_log_probs), all_log_probs, -1e10)

    return all_log_probs


def relabel_normalizing_flows_with_plots(posterior_chain: PosteriorChain, normalizing_flows_fit, max_num_sources, num_iterations, eps):
    """Custom relabeling process that updates iteration numbers for plotting."""
    from petra.parametric_fits import update_parametric_fit_and_prob_in_model
    from petra.cost_matrix import create_compute_cost_matrix
    from petra.relabel import relabel_posterior_chain_one_iteration

    if max_num_sources is None:
        max_num_sources = posterior_chain.num_sources
    if max_num_sources > posterior_chain.num_sources:
        posterior_chain = posterior_chain.expand_chain(max_num_sources)
    if max_num_sources < posterior_chain.num_sources:
        raise ValueError("The maximum number of sources cannot be less than the number of entries in the chain.")

    print()
    print(f"Sorting the posterior chain with flow plotting enabled:\n\tMaximum number of iterations: {num_iterations}\n\tMaximum number of source labels: {max_num_sources}\n")

    # Set up the compute cost matrix function
    # Note: Removed filter_jit wrapper as it causes JAX compilation issues with flows
    compute_cost_matrix = create_compute_cost_matrix(normalizing_flows_aux_distribution)

    # Set up the for loop
    old_posterior_chain = posterior_chain
    normalizing_flows_fit.set_iteration(0)  # Initial state
    old_parametric_fit, old_prob_in_model = update_parametric_fit_and_prob_in_model(posterior_chain, max_num_sources, normalizing_flows_fit, eps=eps)
    old_cost_of_assignment = 0

    for iteration in range(num_iterations):
        # Update iteration number for plotting
        normalizing_flows_fit.set_iteration(iteration + 1)

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


def relabel_normalizing_flows(posterior_chain: PosteriorChain,
                              max_num_sources: int|None = None,
                              num_iterations: int = 20,
                              eps=1e-2,
                              plot_flows: bool = False,
                              tail_decay: float = 4.0,
                              compactness_weight: float = 0.0):
    normalizing_flows_fit = make_normalizing_flows_fit(posterior_chain.chain, max_num_sources, rng_seed = 999, threshold_samples = 100, plot_flows=plot_flows, tail_decay=tail_decay, compactness_weight=compactness_weight)

    if plot_flows:
        # Use a custom relabeling process that tracks iterations for plotting
        return relabel_normalizing_flows_with_plots(posterior_chain, normalizing_flows_fit, max_num_sources, num_iterations, eps)
    else:
        # Use the standard relabeling process
        # Note: Removed filter_jit wrapper as it causes JAX compilation issues with flows
        relabel_samples = create_relabel_samples(normalizing_flows_fit,
                                                 normalizing_flows_aux_distribution,
                                                 eps=eps)
        return relabel_samples(
            posterior_chain,
            max_num_sources=max_num_sources,
            num_iterations=num_iterations
        )


# def make_catalog_normalizing_flows(posterior_chain: PosteriorChain,
#                                    max_num_sources: int,
#                                    num_iterations: int = 50,
#                                    plot_flows: bool = False):

#     if posterior_chain.num_sources > max_num_sources:
#         raise ValueError("max_num_sources must be greater than the number of entries in the chain.")

#     # make sure that posterior_chain has the right shape
#     if posterior_chain.num_sources < max_num_sources:
#         print("Expanding posterior chain to max_num_sources.")
#         posterior_chain.expand_chain(max_num_sources)

#     else:
#         initial_posterior_chain = posterior_chain

#     # relabel using normalizing flows
#     print("Relabeling with normalizing flows.")
#     relabeled_chain = relabel_normalizing_flows(initial_posterior_chain,
#                                         max_num_sources=max_num_sources,
#                                         num_iterations=num_iterations,
#                                         plot_flows=plot_flows)

#     return relabeled_chain


def make_catalog_copula_flows(posterior_chain: PosteriorChain,
                              max_num_sources: int,
                              num_iterations: int = 50,
                              plot_flows: bool = False,
                              tail_decay: float = 4.0,
                              compactness_weight: float = 0.0):
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
    compactness_weight : float, default 0.0
        Weight for compactness penalty in flow training. Higher values encourage blob-like distributions.
        Typical range: 0.0 (no penalty) to 1.0 (strong compactness preference).

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain with relabeled samples using the hybrid approach.

    Examples
    --------
    >>> from petra.flows import make_catalog_copula_flows
    >>> relabeled_chain = make_catalog_copula_flows(
    ...     posterior_chain, max_num_sources=3, plot_flows=True, tail_decay=2.0)
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
                                        compactness_weight=compactness_weight)

    return relabeled_chain
