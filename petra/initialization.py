"""Warm starts for the relabelers: the frequency-histogram labeling and the univariate-normal relabeling.

`run_initialization_passes` is the shared preamble of the Bayesian and copula entry points; it takes both relabelers
as arguments, so each entry point's own namespace is where a pass can be substituted.
"""

import logging
import os
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from petra.fits import compute_cost_matrix, uni_normal_fit, uni_normal_log_prob
from petra.posterior_chain import PosteriorChain
from petra.relabel import run_relabeling_loop
from petra.utils import source_present

logger = logging.getLogger(__name__)


def label_chain_by_histogram(chain: PosteriorChain, obs_time_yrs: float = 1, num_surrounding_bins: int = 1,
                             relabeling_parameter: int = 0, num_extra_entries: int = 1000,
                             low_num_samples: int = 100) -> PosteriorChain:
    """Label a chain by greedily peeling sources off a histogram of one parameter, fullest bin first.

    Parameters:
        chain: chain to label; never modified.
        obs_time_yrs: observation time in years; the bin width is ``1 / (obs_time_yrs * 31_536_000)`` Hz.
        num_surrounding_bins: bins on either side of the fullest bin that still count as the same source.
        relabeling_parameter: parameter to histogram, normally the frequency.
        num_extra_entries: labels beyond ``chain.num_sources`` the labeling may create before it gives up.
        low_num_samples: labels holding fewer samples than this are compacted towards the front.

    Returns:
        New chain of the non-empty labels, sorted by decreasing sample count; equal counts keep reverse slot order.
        ``trans_dimensional`` and ``validate_nan_convention`` are inherited.

    Raises:
        ValueError: obs_time_yrs is not positive, or more than ``num_sources + num_extra_entries`` labels are needed.

    Warns:
        UserWarning: some source rows mix NaN with finite values; they are absent and left out of the result.
    """
    if not obs_time_yrs > 0:
        raise ValueError(f"obs_time_yrs must be positive, got {obs_time_yrs}")
    delta_freq = 1.0 / (obs_time_yrs * 31_536_000)
    window = (num_surrounding_bins + 0.5) * delta_freq
    max_labels = chain.num_sources + num_extra_entries

    present = source_present(chain.chain)
    num_partial = int(np.count_nonzero(~present & ~np.isnan(chain.chain).all(axis=-1)))
    if num_partial:
        warnings.warn(
            f"label_chain_by_histogram: {num_partial} source "
            f"{'row mixes' if num_partial == 1 else 'rows mix'} NaN with finite values. Such a row is "
            f"absent under petra's all-or-nothing NaN convention, as find_prob_in_model, "
            f"the cost matrix and the multivariate fits read it, so it is not labeled and "
            f"is left out of the returned chain.",
            UserWarning, stacklevel=2,
        )
    # Blank partial rows like absent ones: the compaction below moves single values and would splice them.
    remaining = np.where(present[..., None], chain.chain, np.nan)
    num_samples, _, num_params = remaining.shape
    num_unclaimed = int(np.count_nonzero(present))
    samples = np.arange(num_samples)

    labels: list[np.ndarray] = []
    while num_unclaimed:
        if len(labels) >= max_labels:
            raise ValueError(
                f"Ran out of source labels after {len(labels)}; increase "
                f"num_extra_entries (currently {num_extra_entries})."
            )
        x = remaining[:, :, relabeling_parameter].reshape(-1)
        x = x[~np.isnan(x)]
        center = float(np.bincount(np.rint(x / delta_freq).astype(np.int64)).argmax() * delta_freq)

        # An absent entry is never nearest; a sample with nothing left has a NaN distance, which no window admits.
        distances = np.abs(remaining[:, :, relabeling_parameter] - center)
        nearest = np.argmin(np.where(np.isnan(distances), np.inf, distances), axis=1)
        claimed = samples[distances[samples, nearest] <= window]

        label = np.full((num_samples, num_params), np.nan)
        label[claimed] = remaining[claimed, nearest[claimed]]
        remaining[claimed, nearest[claimed]] = np.nan
        num_unclaimed -= claimed.size
        labels.append(label)

    sources_array = np.stack(labels, axis=1) if labels else np.full((num_samples, 0, num_params), np.nan)

    counts = np.count_nonzero(~np.isnan(sources_array[:, :, relabeling_parameter]), axis=0)
    sources_array = sources_array[:, np.argsort(counts, kind="stable")[::-1], :]

    counts = np.count_nonzero(~np.isnan(sources_array[:, :, relabeling_parameter]), axis=0)
    low = sources_array[:, counts < low_num_samples, :]
    sources_array[:, counts < low_num_samples, :] = np.take_along_axis(
        low, np.argsort(np.isnan(low), axis=1, kind="stable"), axis=1)

    counts = np.count_nonzero(~np.isnan(sources_array[:, :, relabeling_parameter]), axis=0)
    non_empty_sources = sources_array[:, counts > 0, :][:, np.argsort(counts[counts > 0], kind="stable")[::-1], :]

    logger.info("Histogram labeling found %d non-empty source labels.",
                non_empty_sources.shape[1])

    return PosteriorChain(chain=non_empty_sources, num_sources=non_empty_sources.shape[1],
                          num_params_per_source=non_empty_sources.shape[2],
                          trans_dimensional=chain.trans_dimensional,
                          validate_nan_convention=chain.validate_nan_convention)


def relabel_univariate_normal(posterior_chain: PosteriorChain, max_num_sources: int | None = None,
                              num_iterations: int = 20, init_parameter_index: int = 0, eps: float = 1e-06,
                              checkpoint_dir: str | os.PathLike[str] | None = None,
                              resume_from: str | os.PathLike[str] | None = None,
                              progress: bool = True) -> PosteriorChain:
    """Relabel a chain with univariate normal fits to one parameter, without split moves.

    Parameters:
        posterior_chain: chain to relabel; not modified.
        max_num_sources: number of source slots; None uses the chain's num_sources.
        num_iterations: ceiling on the iterations, counting those a resumed checkpoint holds.
        init_parameter_index: parameter to fit.
        eps: clip bound on prob_in_model inside the cost matrix.
        checkpoint_dir: directory to save the best labeling to each time it improves.
        resume_from: checkpoint file or directory to resume from; it must come from this method and parameter.
        progress: show a progress bar over the samples of each iteration.

    Returns:
        New chain with the cheapest labeling found, unclipped prob_in_model and ``cost_dict[max_num_sources]``.

    Raises:
        ValueError: an argument is invalid, or the checkpoint belongs to another run.
    """
    return run_relabeling_loop(
        posterior_chain, partial(uni_normal_fit, param_index=init_parameter_index),
        partial(compute_cost_matrix, log_prob=partial(uni_normal_log_prob, param_index=init_parameter_index)),
        max_num_sources, num_iterations, eps, checkpoint_dir, resume_from, progress,
        method="univariate_normal", settings={"init_parameter_index": init_parameter_index})


def check_param_index(num_params: int, initialization_param_index: Any) -> None:
    """Require `initialization_param_index` to be None or an integer column of the chain, not a bool.

    Parameters:
        num_params: parameters per source in the chain.
        initialization_param_index: the entry point's `initialization_param_index`.

    Raises:
        ValueError: it is neither None nor an integer in ``[0, num_params)``.
    """
    if initialization_param_index is None:
        return
    # numpy would wrap a negative index onto another column and read a bool as a mask; neither raises.
    if (not isinstance(initialization_param_index, (int, np.integer))
            or isinstance(initialization_param_index, (bool, np.bool_))
            or not 0 <= initialization_param_index < num_params):
        raise ValueError(
            f"initialization_param_index must be None or in [0, {num_params}), got "
            f"{initialization_param_index}; the chain has {num_params} parameters per source."
        )


def run_initialization_passes(posterior_chain: PosteriorChain,
                              max_num_sources: int,
                              *,
                              univariate_relabeler: Callable[..., PosteriorChain],
                              mv_normal_relabeler: Callable[..., PosteriorChain],
                              with_mv_normal: bool,
                              param_index: int | None,
                              num_iterations: int,
                              resume_from: str | os.PathLike[str] | None = None,
                              progress: bool = True) -> PosteriorChain:
    """Run the univariate pass if `param_index` is set, then the multivariate-normal pass if `with_mv_normal` is.

    Parameters:
        posterior_chain: chain already prepared by `petra.relabel.prepare_chain`; not modified.
        max_num_sources: number of source slots, passed to both passes.
        univariate_relabeler: ``f(chain, max_num_sources=, num_iterations=, init_parameter_index=, progress=)``.
        mv_normal_relabeler: ``f(chain, max_num_sources=, num_iterations=, progress=, split_moves=False)``.
        with_mv_normal: run the multivariate-normal pass.
        param_index: parameter of the univariate pass, or None to skip that pass.
        num_iterations: ceiling on the iterations of each pass.
        resume_from: the checkpoint the caller resumes from; anything but None skips both passes.
        progress: show a progress bar over the samples of each pass.

    Returns:
        The chain after the passes that ran; `posterior_chain` itself if none did.

    Raises:
        ValueError: param_index is neither None nor a column of the chain, checked on a resume too.
    """
    check_param_index(posterior_chain.shape[2], param_index)

    if resume_from is not None:
        logger.info("Resuming from %s; skipping both initialization passes.", resume_from)
        return posterior_chain

    if param_index is not None:
        logger.info("Initializing with univariate normal distribution.")
        posterior_chain = univariate_relabeler(
            posterior_chain,
            max_num_sources=max_num_sources,
            num_iterations=num_iterations,
            init_parameter_index=param_index,
            progress=progress,
        )
    else:
        logger.info("initialization_param_index is None; skipping the univariate initialization.")

    if with_mv_normal:
        logger.info("Initializing with multivariate normal distribution.")
        posterior_chain = mv_normal_relabeler(
            posterior_chain,
            max_num_sources=max_num_sources,
            num_iterations=num_iterations,
            progress=progress,
            split_moves=False,
        )
    else:
        logger.info("init_with_mv_normal is False; skipping the multivariate-normal initialization.")

    return posterior_chain
