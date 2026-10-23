"""The multivariate-normal catalog method: `relabel_mv_normal` and the entry point `make_catalog_mv_normal`.

Every ``make_catalog_*`` entry point takes ``(posterior_chain, max_num_sources, *, ...)`` with the shared keywords
``num_iterations``, ``initialization_param_index``, ``shuffle_seed``, ``rng_seed``, ``checkpoint_dir``, ``resume_from``,
``progress`` and ``eps``, checks its arguments before any work, never modifies its input and returns a new chain.
"""

import logging
import os
import warnings
from functools import partial
from typing import Any, Mapping

import numpy as np

from petra.checkpoint import prepare_checkpoint_dir
from petra.fits import compute_cost_matrix, mv_normal_fit, mv_normal_log_prob
from petra.initialization import check_param_index, relabel_univariate_normal
from petra.posterior_chain import PosteriorChain
from petra.relabel import check_chain, prepare_chain, run_relabeling_loop
from petra.utils import check_eps, check_int, check_shuffle_seed, find_prob_in_model

logger = logging.getLogger(__name__)


def relabel_mv_normal(posterior_chain: PosteriorChain, max_num_sources: int | None = None, num_iterations: int = 20,
                      eps: float = 1e-06, checkpoint_dir: str | os.PathLike[str] | None = None,
                      resume_from: str | os.PathLike[str] | None = None, progress: bool = True, *,
                      run_info: Mapping[str, Any] | None = None, keep_checkpoints: int | None = None,
                      split_moves: bool = True, split_exclusive: bool = False) -> PosteriorChain:
    """Relabel a chain with a multivariate normal fit per label, with split moves where the search would stop.

    Parameters:
        posterior_chain: chain to relabel; not modified. On a resume, the chain the run started from.
        max_num_sources: number of source slots; None uses the chain's num_sources.
        num_iterations: ceiling on the iterations, counting split-move evaluations and resumed ones.
        eps: clip bound on prob_in_model inside the cost matrix; the returned prob_in_model is unclipped.
        checkpoint_dir: directory to save the best labeling to each time it improves; one directory per run.
        resume_from: checkpoint file or directory to resume from; it must come from the same chain, width and eps.
        progress: show a progress bar over the samples of each iteration.
        run_info: informational settings recorded in every checkpoint and not compared on resume.
        keep_checkpoints: keep only this many of the newest checkpoints; None keeps them all.
        split_moves: where the search would stop, try splitting a label that holds two clusters of rows.
        split_exclusive: with split_moves, also split a label whose two clusters rarely appear in the same sample.

    Returns:
        New chain with the cheapest labeling found; prob_in_model equals ``find_prob_in_model(chain, K, eps=0)``.

    Raises:
        TypeError: posterior_chain is not a PosteriorChain.
        ValueError: an argument is invalid, checkpoint_dir holds another run, or the checkpoint belongs to another run.

    Warns:
        UserWarning: with split_moves, the budget ran out before the search stopped; on a resume with nothing to run.
    """
    return run_relabeling_loop(posterior_chain, mv_normal_fit,
                               partial(compute_cost_matrix, log_prob=mv_normal_log_prob),
                               max_num_sources, num_iterations, eps, checkpoint_dir, resume_from, progress,
                               method="mv_normal", run_info=run_info, keep_checkpoints=keep_checkpoints,
                               split_moves=split_moves, split_exclusive=split_exclusive)


def make_catalog_mv_normal(posterior_chain: PosteriorChain,
                           max_num_sources: int,
                           *,
                           num_iterations: int = 200,
                           init_num_iterations: int = 200,
                           initialization_param_index: int | None = 0,
                           shuffle_seed: int | None = None,
                           rng_seed: int = 999,
                           checkpoint_dir: str | os.PathLike[str] | None = None,
                           resume_from: str | os.PathLike[str] | None = None,
                           progress: bool = True,
                           eps: float = 1e-06,
                           keep_checkpoints: int | None = None,
                           split_moves: bool = True,
                           split_exclusive: bool = False,
                           num_starts: int = 1) -> PosteriorChain:
    """Build a catalog: widen, shuffle, warm-start with a univariate pass, then relabel with multivariate normals.

    Parameters:
        posterior_chain: chain to catalog, holding at least one sample; never modified.
        max_num_sources: number of labels in the catalog, at least the chain's num_sources.
        num_iterations: ceiling on the multivariate iterations, counting split-move evaluations and resumed ones.
        init_num_iterations: ceiling on the univariate warm start's iterations; checked even when it is skipped.
        initialization_param_index: parameter of the univariate warm start, or None to skip it.
        shuffle_seed: None for no shuffle, else a non-negative integer; the chain is widened before it is shuffled.
        rng_seed: accepted for symmetry with the other entry points and unused.
        checkpoint_dir: directory to save the best labeling to each time it improves; created and checked first.
        resume_from: checkpoint file or directory to resume from, without shuffling, widening or initializing again.
        progress: show a progress bar over the samples of each iteration.
        eps: clip bound on prob_in_model inside the cost matrix; costs under different values are not comparable.
        keep_checkpoints: keep only this many of the newest checkpoints; None keeps them all.
        split_moves: where the search would stop, try splitting a label that holds two clusters of rows.
        split_exclusive: with split_moves, also split a label whose two clusters rarely appear in the same sample.
        num_starts: independent starts, start k shuffled with ``shuffle_seed + k`` (or k, start 0 unshuffled).

    Returns:
        The strictly cheapest start's catalog of max_num_sources labels, with unclipped prob_in_model;
        ``cost_dict[max_num_sources]`` is its labeling's cost under the fit before it.

    Raises:
        TypeError: posterior_chain is not a PosteriorChain, or shuffle_seed is not None or an integer.
        ValueError: a count, eps, max_num_sources, shuffle_seed, initialization_param_index or the checkpoint is
            invalid, or num_starts > 1 is combined with checkpoint_dir or resume_from.
        OSError: checkpoint_dir cannot be created or written to.

    Warns:
        UserWarning: a label that widening added ends empty; with split_moves, the budget ran out before the search
            stopped; a resume has nothing left to run.
    """
    # Checked even when the warm start is skipped; when it runs, the loop would name it num_iterations.
    check_int("init_num_iterations", init_num_iterations)
    check_int("num_iterations", num_iterations)
    check_eps(eps)
    if keep_checkpoints is not None:
        check_int("keep_checkpoints", keep_checkpoints)
    check_int("num_starts", num_starts)
    if num_starts > 1 and (checkpoint_dir is not None or resume_from is not None):
        raise ValueError(
            f"num_starts={num_starts} cannot be combined with checkpoint_dir or resume_from: a "
            f"checkpoint directory holds the checkpoints of one run, and a resume could not tell "
            f"which start a checkpoint belongs to. Run each start as its own call, with its own "
            f"shuffle_seed and checkpoint_dir, and keep the catalog of lowest "
            f"cost_dict[max_num_sources]."
        )
    max_num_sources = check_chain(posterior_chain, max_num_sources, required=True)
    check_shuffle_seed(shuffle_seed)
    check_param_index(posterior_chain.shape[2], initialization_param_index)

    if resume_from is not None:
        # The checkpoint replaces the chain and already reflects the shuffle, widening and warm start.
        logger.info("Resuming from %s; the input chain is not shuffled, widened or initialized.",
                    resume_from)
    elif checkpoint_dir is not None:
        # Before the warm start, so a bad or reused directory is reported before any work is done.
        prepare_checkpoint_dir(checkpoint_dir)

    best: PosteriorChain | None = None
    for start in range(num_starts):
        # Start 0 is the run the other arguments describe; the others reshuffle. A resume has one start.
        seed = shuffle_seed + start if shuffle_seed is not None else start or None
        # Widen first, so that the shuffle spreads the padding over every slot.
        prepared = (posterior_chain if resume_from is not None
                    else prepare_chain(posterior_chain, max_num_sources, shuffle_seed=seed, widen_first=True))
        if initialization_param_index is not None and resume_from is None:
            logger.info("Initializing with univariate normal distribution.")
            prepared = relabel_univariate_normal(
                prepared,
                max_num_sources=max_num_sources,
                num_iterations=init_num_iterations,
                init_parameter_index=initialization_param_index,
                eps=eps,
                progress=progress,
            )

        logger.info("Relabeling with multivariate normal distribution.")
        candidate = relabel_mv_normal(prepared,
                                      max_num_sources=max_num_sources,
                                      num_iterations=num_iterations,
                                      eps=eps,
                                      checkpoint_dir=checkpoint_dir,
                                      resume_from=resume_from,
                                      progress=progress,
                                      run_info={"entry_point": "make_catalog_mv_normal",
                                                "shuffle_seed": seed},
                                      keep_checkpoints=keep_checkpoints,
                                      split_moves=split_moves,
                                      split_exclusive=split_exclusive)
        if num_starts > 1:
            logger.info("Start %d of %d (shuffle_seed=%s): cost %s.", start + 1, num_starts, seed,
                        candidate.cost_dict[max_num_sources])
        # Strict, so the earliest of equally cheap catalogs is kept.
        if best is None or candidate.cost_dict[max_num_sources] < best.cost_dict[max_num_sources]:
            best = candidate
    assert best is not None     # num_starts is at least 1

    # A warning, not a log: the petra logger has no handler, and split moves fill an added slot whenever that pays.
    if max_num_sources > posterior_chain.num_sources:
        empty = int(np.count_nonzero(find_prob_in_model(best.get_chain(), max_num_sources, eps=0) == 0))
        if empty:
            exclusive_hint = (", or split_exclusive=True if sources that are never in the same sample "
                              "should each have a label")
            search = " and split_moves=True" if not split_moves else exclusive_hint if not split_exclusive else ""
            starts = ("num_starts > 1" if num_starts == 1
                      else f"more starts than the num_starts={num_starts} this call ran")
            warnings.warn(
                f"max_num_sources={max_num_sources} widened the chain from {posterior_chain.num_sources} "
                f"source slots, but {empty} of the catalog's {max_num_sources} labels "
                f"{'is' if empty == 1 else 'are'} empty (prob_in_model == 0): the search found no "
                f"cheaper labeling that fills {'it' if empty == 1 else 'them'}. The data may hold "
                f"fewer sources than max_num_sources; if not, try {starts}{search}, and check "
                f"the chain.",
                UserWarning, stacklevel=2,
            )

    return best
