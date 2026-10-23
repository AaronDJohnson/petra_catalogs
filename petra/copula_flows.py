"""Copula-flow relabeling: each slot is fit with a Gaussian-copula marginal transform plus a normalizing flow.

The flows come from ``coppuccino.copula_flows.normalizing_flows_fit``. Importing this module imports jax but leaves its
64-bit mode alone; set ``JAX_ENABLE_X64=1`` before any JAX array exists if the fits need double precision.
"""

import contextlib
import logging
import math
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array

from coppuccino.copula_flows import normalizing_flows_fit

from petra.fits import compute_cost_matrix
from petra.posterior_chain import PosteriorChain
from petra.relabel import prepare_chain, run_relabeling_loop
from petra.utils import UniformPrior, check_eps, check_int, make_uniform_prior

logger = logging.getLogger(__name__)

# Low enough that the assignment refuses the pairing, high enough not to swamp the finite costs as -1e10 did.
DEFAULT_LOG_PROB_FLOOR = -50.0
DEFAULT_MAX_PATIENCE = 20


@contextlib.contextmanager
def quiet_external_fit(quiet: bool) -> Iterator[None]:
    """Discard stderr inside the block, silencing a progress bar the fitter offers no switch for.

    Parameters:
        quiet: discard stderr, warnings included, when True; do nothing when False.

    Yields:
        None.
    """
    if not quiet:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        yield


def safe_flow_log_prob(flow: Optional[Any], x: npt.ArrayLike,
                       floor: float = DEFAULT_LOG_PROB_FLOOR) -> Array:
    """Return ``flow.log_prob(x)`` with every non-finite value, such as an absent NaN row, replaced by `floor`.

    Parameters:
        flow: anything with ``log_prob``, a flowjax distribution or a UniformPrior; None marks an unfitted slot.
        x: (P,) point or (n, P) points.
        floor: value substituted for a non-finite log-density.

    Returns:
        a finite scalar for a 1-D `x`, else (n,), a scalar result broadcast; floor throughout for None or an empty x.

    Raises:
        ValueError: `x` is not 1-D or 2-D.
    """
    x = jnp.asarray(x)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}.")
    out_shape = x.shape[:-1]

    if flow is None or x.size == 0:
        return jnp.full(out_shape, floor)

    if isinstance(flow, UniformPrior):
        log_prob = jnp.asarray(flow.log_prob(x))
    else:
        # Only jax_debug_nans' FloatingPointError: any other exception is a mis-wired flow and must surface.
        try:
            log_prob = jnp.asarray(flow.log_prob(x))
        except FloatingPointError:
            return jnp.full(out_shape, floor)

    log_prob = jnp.where(jnp.isfinite(log_prob), log_prob, floor)
    if log_prob.shape != out_shape:
        log_prob = jnp.broadcast_to(log_prob, out_shape)
    return log_prob


def stack_flow_log_probs(flows: Sequence[Any], sample: np.ndarray,
                         floor: float = DEFAULT_LOG_PROB_FLOOR) -> Array:
    """Return the floored log-density every flow gives every row of one sample.

    Parameters:
        flows: one fitted distribution per label; an entry may be None or a UniformPrior.
        sample: (S, P) rows; a NaN row is an absent source.
        floor: value substituted for a non-finite log-density.

    Returns:
        finite (len(flows), S) array.

    Raises:
        ValueError: `sample` is not 2-D.
    """
    points = jnp.asarray(sample)
    if points.ndim != 2:
        raise ValueError(f"sample must be 2-D of shape (n_sources, n_params), got {points.shape}.")
    rows = [safe_flow_log_prob(flow, points, floor=floor) for flow in flows]
    return jnp.stack(rows, axis=0)


def _check_flow_settings(knots: int, flow_layers: int, max_epochs: int, max_patience: int, learning_rate: float,
                         log_prob_floor: float = DEFAULT_LOG_PROB_FLOOR) -> None:
    """Refuse a count that is not an integer >= 1, a learning rate not finite and > 0, or a non-finite floor."""
    check_int("knots", knots)
    check_int("flow_layers", flow_layers)
    check_int("max_epochs", max_epochs)
    check_int("max_patience", max_patience)
    if not learning_rate > 0:
        raise ValueError(f"learning_rate must be strictly positive, got {learning_rate}.")
    if not math.isfinite(learning_rate):
        raise ValueError(f"learning_rate must be finite, got {learning_rate}.")
    if not math.isfinite(log_prob_floor):
        raise ValueError(f"log_prob_floor must be finite, got {log_prob_floor}.")


def _resolve_max_patience(func_name: str, max_patience: int, patience: int | None) -> int:
    """Map the deprecated `patience` onto `max_patience` with a DeprecationWarning; a TypeError if both are set."""
    if patience is None:
        return max_patience
    warnings.warn(
        f"{func_name}(): keyword 'patience' is deprecated, use 'max_patience' instead.",
        DeprecationWarning, stacklevel=3,
    )
    if max_patience != DEFAULT_MAX_PATIENCE:
        raise TypeError("Pass either 'patience' or 'max_patience', not both.")
    return patience


def copula_flows_aux_distribution(sample: np.ndarray,
                                  aux_parameters: List,
                                  source_index: int | Sequence[int] | np.ndarray,
                                  floor: float = DEFAULT_LOG_PROB_FLOOR) -> Array:
    """Return the floored log-density each requested flow gives every row of one sample.

    Parameters:
        sample: (S, P) rows; a NaN row is an absent source.
        aux_parameters: fitted flows, one per label.
        source_index: label or labels to evaluate.
        floor: value substituted for a non-finite log-density; the same in every entry point, so costs compare.

    Returns:
        (L, S) array; row i belongs to label ``source_index[i]``.
    """
    source_indices = jnp.atleast_1d(np.asarray(source_index))
    all_lp = stack_flow_log_probs(aux_parameters, sample, floor=floor)
    return all_lp[source_indices]


def make_copula_flows_fit(chain: np.ndarray,
                          rng_seed: int = 999,
                          threshold_samples: int = 50,
                          knots: int = 16,
                          max_patience: int = DEFAULT_MAX_PATIENCE,
                          learning_rate: float = 1e-3,
                          max_epochs: int = 800,
                          flow_layers: int = 8,
                          progress: bool = True,
                          patience: int | None = None) -> Callable:
    """Return a fit that trains one copula flow per slot and gives a sparse slot the chain-wide uniform prior.

    Parameters:
        chain: (N, S, P) samples; only the uniform prior's bounds are taken from it.
        rng_seed: seed for the flow initialization and training.
        threshold_samples: a slot with at most this many present rows gets the uniform prior instead of a flow.
        knots: spline knots per flow layer.
        max_patience: early-stopping patience, in epochs without validation improvement.
        learning_rate: Adam learning rate.
        max_epochs: maximum training epochs per flow.
        flow_layers: number of flow layers.
        progress: show each flow's training progress bar.
        patience: deprecated spelling of max_patience.

    Returns:
        ``copula_flows_fit(chain, max_num_sources)``, giving one fitted distribution per slot.

    Raises:
        ValueError: a flow setting out of range.
        TypeError: patience together with a max_patience other than the default.

    Warns:
        DeprecationWarning: patience was given.
    """
    max_patience = _resolve_max_patience("make_copula_flows_fit", max_patience, patience)
    # Direct factory callers get the same validation as the catalog entry point.
    _check_flow_settings(knots, flow_layers, max_epochs, max_patience, learning_rate)
    uniform_prior = make_uniform_prior(chain)

    def copula_flows_fit(chain: np.ndarray, max_num_sources: int) -> List:
        """Fit a copula flow, or the uniform prior, to each of the first `max_num_sources` slots of `chain`."""
        fitted_distributions: List = [None] * max_num_sources
        rng_key = jr.key(rng_seed + 1)
        fit_seeds = jr.randint(rng_key, shape=(max_num_sources,), minval=0, maxval=999999)

        for i in range(max_num_sources):
            chain_entry = chain[:, i, :]
            chain_entry = chain_entry[~np.isnan(chain_entry).any(axis=1)]

            if chain_entry.shape[0] <= threshold_samples:
                logger.info("Source %d has %d usable samples (<= threshold_samples=%d); "
                            "falling back to the uniform prior.",
                            i, chain_entry.shape[0], threshold_samples)
                fitted_distributions[i] = uniform_prior

            else:
                logger.debug("Fitting a copula flow for source %d on %d samples.", i, chain_entry.shape[0])
                # normalizing_flows_fit has no progress switch, so progress=False silences its bar instead.
                with quiet_external_fit(not progress):
                    fitted_distributions[i] = normalizing_flows_fit(chain_entry, rng_seed=int(fit_seeds[i]),
                                                                    knots=knots, patience=max_patience,
                                                                    learning_rate=learning_rate,
                                                                    max_epochs=max_epochs, flow_layers=flow_layers)
        return fitted_distributions

    return copula_flows_fit


def relabel_copula_flows(posterior_chain: PosteriorChain,
                         max_num_sources: int | None = None,
                         num_iterations: int = 20,
                         *,
                         eps: float = 1e-6,
                         rng_seed: int = 999,
                         threshold_samples: int = 50,
                         knots: int = 16,
                         max_patience: int = DEFAULT_MAX_PATIENCE,
                         learning_rate: float = 1e-3,
                         max_epochs: int = 800,
                         flow_layers: int = 8,
                         log_prob_floor: float = DEFAULT_LOG_PROB_FLOOR,
                         checkpoint_dir: str | Path | None = None,
                         resume_from: str | Path | None = None,
                         progress: bool = True,
                         patience: int | None = None) -> PosteriorChain:
    """Relabel with copula-flow fits until an iteration fails to lower the cost.

    Parameters:
        posterior_chain: chain to relabel; widened to max_num_sources if needed.
        max_num_sources: number of slots; None means the chain's num_sources.
        num_iterations: iteration ceiling, at least 1.
        eps: clip bound on prob_in_model inside the cost matrix only.
        rng_seed: seed for the flow initialization and training.
        threshold_samples: a slot with at most this many present rows gets the uniform prior instead of a flow.
        knots: spline knots per flow layer.
        max_patience: early-stopping patience, in epochs without validation improvement.
        learning_rate: Adam learning rate.
        max_epochs: maximum training epochs per flow.
        flow_layers: number of flow layers.
        log_prob_floor: value substituted for a non-finite flow log-density.
        checkpoint_dir: directory for a checkpoint of the best labeling after each iteration; None for none.
        resume_from: checkpoint file, or directory of them, of a run with the same chain and settings.
        progress: show progress bars over the samples and over flow training.
        patience: deprecated spelling of max_patience.

    Returns:
        the cheapest labeling seen, its cost in ``cost_dict[max_num_sources]`` and its prob_in_model unclipped.

    Raises:
        ValueError: a setting out of range, or a checkpoint written by another run or with other flow settings.
        TypeError: patience together with a max_patience other than the default.

    Warns:
        DeprecationWarning: patience was given.
    """
    max_patience = _resolve_max_patience("relabel_copula_flows", max_patience, patience)
    copula_flows_fit = make_copula_flows_fit(posterior_chain.chain,
                                             rng_seed=rng_seed,
                                             threshold_samples=threshold_samples,
                                             knots=knots,
                                             max_patience=max_patience,
                                             learning_rate=learning_rate,
                                             max_epochs=max_epochs,
                                             flow_layers=flow_layers,
                                             progress=progress)
    flow_log_prob: Callable[..., Any] = partial(copula_flows_aux_distribution, floor=log_prob_floor)

    return run_relabeling_loop(
        posterior_chain,
        copula_flows_fit,
        partial(compute_cost_matrix, log_prob=flow_log_prob),
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        eps=eps,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress=progress,
        method="copula_flows",
        settings={"rng_seed": rng_seed, "threshold_samples": threshold_samples, "knots": knots,
                  "max_patience": max_patience, "learning_rate": learning_rate, "max_epochs": max_epochs,
                  "flow_layers": flow_layers, "log_prob_floor": log_prob_floor},
    )


def make_catalog_copula_flows(posterior_chain: PosteriorChain,
                              max_num_sources: int,
                              *,
                              num_iterations: int = 50,
                              initialization_param_index: int | None = 0,
                              shuffle_seed: int | None = None,
                              rng_seed: int = 999,
                              checkpoint_dir: str | Path | None = None,
                              resume_from: str | Path | None = None,
                              progress: bool = True,
                              eps: float = 1e-6,
                              threshold_samples: int = 50,
                              init_with_mv_normal: bool = True,
                              init_num_iterations: int = 200,
                              knots: int = 16,
                              flow_layers: int = 8,
                              max_epochs: int = 800,
                              max_patience: int = DEFAULT_MAX_PATIENCE,
                              learning_rate: float = 1e-3,
                              log_prob_floor: float = DEFAULT_LOG_PROB_FLOOR,
                              patience: int | None = None) -> PosteriorChain:
    """Build a catalog with copula-flow relabeling after the shared initialization passes.

    Parameters:
        posterior_chain: input chain.
        max_num_sources: number of slots in the catalog, at least the chain's num_sources.
        num_iterations: ceiling on the flow iterations; each one retrains every populated slot's flow.
        initialization_param_index: parameter for the univariate initialization pass; None skips that pass.
        shuffle_seed: seed for shuffling the entries of each sample; None keeps their order.
        rng_seed: seed for the flow initialization and training.
        checkpoint_dir: directory for checkpoints of the flow relabeling; None for none.
        resume_from: checkpoint of the flow relabeling to resume; the initialization passes are then skipped.
        progress: show progress bars over the samples and over flow training.
        eps: clip bound on prob_in_model inside the cost matrix only.
        threshold_samples: a slot with at most this many present rows gets the uniform prior instead of a flow.
        init_with_mv_normal: run the multivariate normal initialization pass.
        init_num_iterations: iteration ceiling of each initialization pass.
        knots: spline knots per flow layer.
        flow_layers: number of flow layers.
        max_epochs: maximum training epochs per flow.
        max_patience: early-stopping patience, in epochs without validation improvement.
        learning_rate: Adam learning rate.
        log_prob_floor: value substituted for a non-finite flow log-density.
        patience: deprecated spelling of max_patience.

    Returns:
        the relabeled chain, with its cost in cost_dict and its prob_in_model unclipped.

    Raises:
        ValueError: a setting out of range, or initialization_param_index not None or a column of the chain.
        TypeError: patience together with a max_patience other than the default.

    Warns:
        DeprecationWarning: patience was given.
    """
    max_patience = _resolve_max_patience("make_catalog_copula_flows", max_patience, patience)
    check_int("init_num_iterations", init_num_iterations)
    _check_flow_settings(knots, flow_layers, max_epochs, max_patience, learning_rate, log_prob_floor)
    check_int("num_iterations", num_iterations)
    check_eps(eps)

    from petra.make_catalog import relabel_mv_normal
    from petra.initialization import relabel_univariate_normal, run_initialization_passes

    posterior_chain = prepare_chain(posterior_chain, max_num_sources, shuffle_seed=shuffle_seed)

    posterior_chain = run_initialization_passes(
        posterior_chain,
        max_num_sources,
        univariate_relabeler=relabel_univariate_normal,
        mv_normal_relabeler=relabel_mv_normal,
        with_mv_normal=init_with_mv_normal,
        param_index=initialization_param_index,
        num_iterations=init_num_iterations,
        resume_from=resume_from,
        progress=progress,
    )

    logger.info("Relabeling with Gaussian copula marginals + normalizing flows.")
    return relabel_copula_flows(posterior_chain,
                                max_num_sources=max_num_sources,
                                num_iterations=num_iterations,
                                eps=eps,
                                rng_seed=rng_seed,
                                threshold_samples=threshold_samples,
                                knots=knots,
                                max_patience=max_patience,
                                learning_rate=learning_rate,
                                max_epochs=max_epochs,
                                flow_layers=flow_layers,
                                log_prob_floor=log_prob_floor,
                                checkpoint_dir=checkpoint_dir,
                                resume_from=resume_from,
                                progress=progress)
