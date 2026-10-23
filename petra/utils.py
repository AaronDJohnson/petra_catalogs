"""Dependency-light helpers shared by every petra module: argument checks, source presence and the uniform prior.

Only numpy is imported at module scope, so ``import petra`` stays free of jax. :data:`logger` is the package
logger ``petra``, with a NullHandler: petra prints nothing unless the application configures logging.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("petra")
logger.addHandler(logging.NullHandler())


def check_int(name: str, value: Any, minimum: int = 1) -> int:
    """Return `value` as a builtin int, refusing a bool, a non-integer or a value below `minimum`.

    Parameters:
        name: argument name, quoted in the error.
        value: candidate; an int or NumPy integer.
        minimum: smallest accepted value.

    Returns:
        `value` as an int.

    Raises:
        ValueError: `value` is a bool, not an integer, or below `minimum`.
    """
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def check_eps(eps: Any) -> None:
    """Require a finite probability clip bound with ``0 <= eps <= 0.5``; eps=0 means unclipped.

    Parameters:
        eps: candidate clip bound.

    Raises:
        ValueError: eps is not finite or lies outside [0, 0.5].
    """
    if not np.isfinite(eps) or not 0 <= eps <= 0.5:
        raise ValueError(f"eps must be finite and between 0 and 0.5, got {eps!r}.")


def check_shuffle_seed(shuffle_seed: Any) -> None:
    """Refuse a `shuffle_seed` that is not None or a non-negative integer.

    Parameters:
        shuffle_seed: the `shuffle_seed` argument of an entry point.

    Raises:
        TypeError: it is a bool, a numpy.random.Generator or not an integer.
        ValueError: it is a negative integer.
    """
    if shuffle_seed is None:
        return
    if isinstance(shuffle_seed, np.random.Generator):
        raise TypeError(
            "shuffle_seed must be None or a non-negative integer, got a numpy.random.Generator. "
            "Pass an integer seed instead: every shuffle advances a Generator, so two calls given "
            "the same one would shuffle differently and return different catalogs, and the run's "
            "checkpoints could not record a seed that reproduces either."
        )
    if not isinstance(shuffle_seed, (int, np.integer)) or isinstance(shuffle_seed, bool):
        raise TypeError(
            f"shuffle_seed must be None or a non-negative integer, got {shuffle_seed!r} "
            f"({type(shuffle_seed).__name__}). Pass shuffle_seed=None not to shuffle, or an "
            f"integer seed to shuffle reproducibly."
        )
    if shuffle_seed < 0:
        raise ValueError(f"shuffle_seed must be None or a non-negative integer, got {shuffle_seed!r}.")


def source_present(source_data: npt.ArrayLike) -> np.ndarray:
    """Return which source rows are present: a row is present only when none of its values is NaN.

    Parameters:
        source_data: array whose last axis is the parameter axis; leading axes are kept.

    Returns:
        bool array with the last axis reduced away; a row mixing NaN and numbers is absent.
    """
    return ~np.isnan(np.asarray(source_data)).any(axis=-1)


def find_prob_in_model(chain: np.ndarray, max_num_sources: int, eps: float = 1e-6) -> np.ndarray:
    """Return each leading slot's share of samples with a present row, clipped to [eps, 1 - eps].

    Parameters:
        chain: (N, S, P) samples; an all-NaN row is absent.
        max_num_sources: number of leading slots to report, at most S.
        eps: clip bound in [0, 0.5].

    Returns:
        (max_num_sources,) float64 array.

    Raises:
        ValueError: eps out of range, or max_num_sources above S.
    """
    check_eps(eps)
    chain = np.asarray(chain)
    if max_num_sources > chain.shape[1]:
        raise ValueError(
            f"max_num_sources ({max_num_sources}) exceeds the {chain.shape[1]} source "
            f"slots of a chain of shape {chain.shape}."
        )
    prob_in_model = source_present(chain[:, :max_num_sources, :]).mean(axis=0)
    prob_in_model = np.clip(prob_in_model, eps, 1 - eps)
    return prob_in_model


class UniformPrior:
    """Uniform density over a hyper-rectangle; zero-width dimensions are dropped with a warning.

    Parameters:
        mins: (n_params,) lower bounds.
        maxs: (n_params,) upper bounds.

    Raises:
        ValueError: the bounds are not finite 1-D arrays of one length, maxs < mins somewhere, or every width is 0.
    """

    def __init__(self, mins: npt.ArrayLike, maxs: npt.ArrayLike) -> None:
        """Validate the bounds and precompute the constant log density."""
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)
        if mins.ndim != 1 or maxs.ndim != 1 or mins.shape != maxs.shape:
            raise ValueError(
                f"mins and maxs must be 1-D arrays of the same length, got shapes "
                f"{mins.shape} and {maxs.shape}."
            )
        if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs))):
            raise ValueError("UniformPrior bounds must be finite; got NaN or inf.")
        if np.any(maxs < mins):
            raise ValueError("UniformPrior requires maxs >= mins in every dimension.")

        widths = maxs - mins
        self._active = widths > 0
        if not np.all(self._active):
            logger.warning(
                "UniformPrior: %d of %d dimensions have zero width and are "
                "ignored by the prior.",
                int(np.sum(~self._active)),
                widths.size,
            )
        if not np.any(self._active):
            raise ValueError("UniformPrior requires at least one dimension of non-zero width.")

        self.mins = mins
        self.maxs = maxs
        self.log_density = float(-np.sum(np.log(widths[self._active])))

    @property
    def n_params(self) -> int:
        """Number of dimensions the prior is defined over."""
        return self.mins.shape[0]

    def __repr__(self) -> str:
        """Return a representation naming the dimension and the log density."""
        return f"UniformPrior(n_params={self.n_params}, log_density={self.log_density:.6g})"

    def log_prob(self, sample: npt.ArrayLike) -> Any:
        """Return the log density at one point or a batch: `log_density` inside the support, -inf outside or at NaN.

        Parameters:
            sample: (n_params,) point or (n_points, n_params) batch.

        Returns:
            jax array: a scalar for a 1-D sample, shape (n_points,) for a 2-D one.

        Raises:
            ValueError: sample is not 1-D or 2-D, or its last axis is not n_params long.
        """
        # Imported here so that ``import petra`` stays jax-free.
        xnp: Any
        try:
            import jax.numpy as jax_numpy
            xnp = jax_numpy
        except ImportError:  # pragma: no cover - jax is a heavy optional dep
            xnp = np

        x = xnp.asarray(sample)
        if x.ndim not in (1, 2):
            raise ValueError("Sample must be 1D or 2D array.")
        if x.shape[-1] != self.n_params:
            raise ValueError(
                f"Sample has {x.shape[-1]} parameters but the prior is defined over "
                f"{self.n_params}."
            )

        active = xnp.asarray(self._active)
        inside_dim = (x >= xnp.asarray(self.mins)) & (x <= xnp.asarray(self.maxs))
        inside = xnp.all(inside_dim | ~active, axis=-1)
        return xnp.where(inside, self.log_density, -xnp.inf)


def make_uniform_prior(chain: np.ndarray) -> UniformPrior:
    """Build a :class:`UniformPrior` over the per-parameter range of a chain, ignoring NaN.

    Parameters:
        chain: (N, S, P) samples.

    Returns:
        the prior over [min, max] of each parameter across every sample and source.

    Raises:
        ValueError: a parameter is NaN in every entry, so it has no bounds.
    """
    all_entries = chain.reshape(-1, chain.shape[2])
    mins = np.nanmin(all_entries, axis=0)
    maxs = np.nanmax(all_entries, axis=0)
    if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs))):
        raise ValueError(
            "Cannot build a uniform prior: at least one parameter is NaN in every "
            "entry of the chain."
        )
    return UniformPrior(mins, maxs)
