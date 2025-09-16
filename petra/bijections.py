from typing import ClassVar
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection
from petra.utils import process_array
from matplotlib import pyplot as plt
from interpax._ppoly import PchipInterpolator
from jax.scipy.special import erf, erfinv
import jax


def make_empirical_cdf_spline(x_grid, samples):
    # 1. Sort the grid
    x_grid = np.asarray(x_grid)
    sort_idx = np.argsort(x_grid)
    xg       = x_grid[sort_idx]

    # 2. Empirical CDF/
    sorted_chain = process_array(samples)
    counts       = np.searchsorted(sorted_chain, xg, side='right')
    cdf_vals     = counts / len(sorted_chain)

    # 3. Monotonic spline, no extrapolation
    try:
        # check for duplicate values and perturb them if there are any:

        cs = PchipInterpolator(xg, cdf_vals, extrapolate=False, check=False)
    except ValueError:
        plt.plot(xg, cdf_vals)
        plt.show()

    # 4. Safe wrapper for true CDF
    def cdf_fn(u):
        y = cs(u)
        y = np.where(u < xg[0], 0.0, y)
        y = np.where(u > xg[-1], 1.0, y)
        return y

    # 5. PDF spline
    pdf_spline = cs.derivative()

    def pdf_fn(u):
        y = pdf_spline(u)
        y = np.where(u < xg[0], 1e-20, y)
        y = np.where(u > xg[-1], 1e-20, y)
        return y

    # Return raw CDF values (in the same order as x_grid),
    # the safe CDF function, and the PDF spline
    return (
        cdf_vals[np.argsort(sort_idx)],  # back in original x_grid order
        cdf_fn,
        pdf_fn
    )

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
        raw_cdf, _, _ = make_empirical_cdf_spline(x_grid, samples)

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
