"""The shared relabeling loop: fit, assign every sample with the Hungarian algorithm, keep the cheapest labeling.

A method is a fit and a cost matrix; cost is negated log-likelihood, so lower is better. With ``split_moves`` the loop
first tries, wherever it would stop, to split a label holding two clusters into a sparse one. Inputs are never modified.
"""

import logging
import math
import os
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize
from tqdm import tqdm

from petra.checkpoint import Checkpoint, load_checkpoint, prepare_checkpoint_dir, run_fingerprint, write_checkpoint
from petra.posterior_chain import PosteriorChain
from petra.utils import check_eps, check_int, check_shuffle_seed, find_prob_in_model, source_present

logger = logging.getLogger(__name__)

FitFunction = Callable[[np.ndarray, int], Any]

_STOP_REASONS = {"converged": "its cost did not change",
                 "cost increased": "its cost increased"}
_SPLIT_CANDIDATES = 3
_SPLIT_EVALUATION_ITERATIONS = 10
_TWO_MEANS_ITERATIONS = 20
# One normal cloud of 200 or more rows measures 3.3-4.4, two normal clusters 4.3 standard deviations apart 4.5.
_SPLIT_MIN_SEPARATION = 4.5
# A tenth of the samples two independent sources would share; fewer is one source with two modes.
_SPLIT_MIN_TOGETHER = 0.1
_BLOCK_ROWS = 1 << 18


def check_chain(posterior_chain: Any, max_num_sources: Any = None, *, required: bool = False) -> int:
    """Check a relabeler's chain and target width before any work, and return the width.

    Parameters:
        posterior_chain: the chain argument; a PosteriorChain holding at least one sample.
        max_num_sources: number of source slots, at least the chain's; None means the chain's own width.
        required: refuse None for max_num_sources.

    Returns:
        the width as an int.

    Raises:
        TypeError: posterior_chain is not a PosteriorChain.
        ValueError: the chain holds no samples, or max_num_sources is not an integer >= 1 or is below the chain's width.
    """
    if not isinstance(posterior_chain, PosteriorChain):
        kind = type(posterior_chain)
        raise TypeError(
            f"posterior_chain must be a petra.PosteriorChain, got {kind.__module__}.{kind.__qualname__}. "
            f"Wrap an array as PosteriorChain(chain, num_sources, num_params_per_source), or load "
            f"one with petra.load_samples."
        )
    if posterior_chain.shape[0] == 0:
        raise ValueError(
            f"posterior_chain holds no samples (shape {posterior_chain.shape}), so there is nothing "
            f"to relabel. If it was loaded or sliced with a burn or thin, they removed every sample."
        )
    if max_num_sources is None and not required:
        return int(posterior_chain.num_sources)
    max_num_sources = check_int("max_num_sources", max_num_sources)
    if max_num_sources < posterior_chain.num_sources:
        raise ValueError(
            f"max_num_sources ({max_num_sources}) cannot be less than the number of "
            f"entries in the chain ({posterior_chain.num_sources})."
        )
    return max_num_sources


def prepare_chain(posterior_chain: PosteriorChain,
                  max_num_sources: int | None = None,
                  shuffle_seed: int | None = None,
                  *,
                  widen_first: bool = False) -> PosteriorChain:
    """Check, widen and optionally shuffle a chain before relabeling.

    Parameters:
        posterior_chain: the chain to prepare; never modified.
        max_num_sources: number of source slots, at least the chain's; None means the chain's own width.
        shuffle_seed: None not to shuffle, or a non-negative integer seed for a per-sample shuffle of the slots.
        widen_first: widen before shuffling, spreading the padding over every slot; otherwise added slots start empty.

    Returns:
        the input itself when there is nothing to do; otherwise a new chain with prob_in_model re-derived from it,
        an empty cost_dict and the input's validate_nan_convention.

    Raises:
        TypeError: posterior_chain is not a PosteriorChain, or shuffle_seed is not None or an integer.
        ValueError: the chain holds no samples, max_num_sources is invalid, or shuffle_seed is negative.
    """
    max_num_sources = check_chain(posterior_chain, max_num_sources)
    check_shuffle_seed(shuffle_seed)

    def shuffle(chain: PosteriorChain) -> PosteriorChain:
        if shuffle_seed is None:
            return chain
        logger.info("Shuffling chain entries with seed %s.", shuffle_seed)
        return chain.randomize_entries(shuffle_seed)

    if not widen_first:
        posterior_chain = shuffle(posterior_chain)

    if max_num_sources > posterior_chain.num_sources:
        logger.info("Expanding posterior chain from %d to %d source slots.",
                    posterior_chain.num_sources, max_num_sources)
        posterior_chain = posterior_chain.expand_chain(max_num_sources)

    if widen_first:
        posterior_chain = shuffle(posterior_chain)

    return posterior_chain


def relabel_samples_one_iteration(chain: np.ndarray,
                                  aux_parameters: Any,
                                  prob_in_model: np.ndarray,
                                  max_num_sources: int,
                                  compute_cost_matrix: Callable[..., np.ndarray],
                                  progress: bool = True) -> tuple[np.ndarray, float]:
    """Assign each sample's rows to labels with the Hungarian algorithm under one fit.

    Parameters:
        chain: (N, S, P) samples.
        aux_parameters: the fit, passed to compute_cost_matrix unchanged.
        prob_in_model: (K,) inclusion probabilities fed to the cost matrix.
        max_num_sources: number of labels K, at most S; slots no label is assigned follow in their own order.
        compute_cost_matrix: ``(sample, aux_parameters, prob_in_model, K)`` giving the (K, S) rewards to maximize.
        progress: show a progress bar over the samples.

    Returns:
        the (N, S, P) relabeled samples, each a permutation of its own rows, and the cost: minus the mean maximized
        reward sum, so lower is better.
    """
    relabeled_list = []
    total_cost = 0.0
    for sample in tqdm(chain, disable=not progress):
        cost_matrix = compute_cost_matrix(sample, aux_parameters, prob_in_model, max_num_sources)
        total_entries = max(len(sample), max_num_sources)
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)
        total_cost -= cost_matrix[row_ind, col_ind].sum()
        # `col_ind` has one entry per label, not per slot: a wider chain's unassigned slots are appended, or they
        # would be dropped from the reordering and the chain would lose sources.
        unassigned = np.ones(total_entries, dtype=bool)
        unassigned[col_ind] = False
        relabeled_list.append(np.concatenate([col_ind, np.flatnonzero(unassigned)]))
    relabeled_array = np.array(relabeled_list)

    relabeled_samples = np.array([row[m] for row, m in zip(chain, relabeled_array)])
    total_cost /= len(chain)
    return relabeled_samples, total_cost


def _with_unclipped_prob_in_model(posterior_chain: PosteriorChain,
                                  max_num_sources: int) -> PosteriorChain:
    """Return the chain with prob_in_model re-derived from its samples, unclipped; no copy or rescan, same settings."""
    rebuilt = PosteriorChain(posterior_chain.chain,
                             posterior_chain.num_sources,
                             posterior_chain.num_params_per_source,
                             trans_dimensional=posterior_chain.trans_dimensional,
                             prob_in_model=find_prob_in_model(posterior_chain.get_chain(),
                                                              max_num_sources, eps=0),
                             cost_dict=dict(posterior_chain.cost_dict),
                             copy=False,
                             validate_nan_convention=False)
    rebuilt.copy = posterior_chain.copy
    rebuilt.validate_nan_convention = posterior_chain.validate_nan_convention
    return rebuilt


@dataclass(frozen=True)
class _Labeling:
    """A labeling the loop can continue from: the chain (unclipped prob_in_model), its fit, clipped probs and cost."""

    chain: PosteriorChain
    aux_parameters: Any
    prob_in_model: np.ndarray
    cost: float


def _relabel_and_refit(posterior_chain: PosteriorChain,
                       aux_parameters: Any,
                       prob_in_model: np.ndarray,
                       max_num_sources: int,
                       param_fit: FitFunction,
                       compute_cost_matrix: Callable[..., np.ndarray],
                       eps: float,
                       progress: bool) -> _Labeling:
    """Run one iteration of the loop: assign under the fit of `posterior_chain`, then fit the new labeling."""
    relabeled, cost = relabel_samples_one_iteration(posterior_chain.get_chain(), aux_parameters, prob_in_model,
                                                    max_num_sources, compute_cost_matrix, progress=progress)
    new_chain = PosteriorChain(relabeled, posterior_chain.num_sources, posterior_chain.num_params_per_source,
                               trans_dimensional=posterior_chain.trans_dimensional,
                               prob_in_model=find_prob_in_model(relabeled, max_num_sources, eps=0),
                               cost_dict={**posterior_chain.cost_dict, max_num_sources: cost},
                               validate_nan_convention=False)
    new_chain.validate_nan_convention = posterior_chain.validate_nan_convention
    return _Labeling(new_chain, param_fit(relabeled, max_num_sources),
                     find_prob_in_model(relabeled, max_num_sources, eps=eps), cost)


def _stall_reason(old_cost: float | None, new_cost: float) -> str | None:
    """Return "converged" if the cost did not change, "cost increased" if it rose, and None otherwise or first."""
    if old_cost is None:
        return None
    if new_cost - old_cost == 0:
        return "converged"
    if new_cost > old_cost:
        return "cost increased"
    return None


def _lloyd_two_means(points: np.ndarray, start: np.ndarray) -> tuple[float, np.ndarray]:
    """Run Lloyd's 2-means from ``start`` and ``-start``; return the within sum of squares, inf if a cluster empties."""
    centroids = np.stack([start, -start])
    in_second = np.zeros(points.shape[0], dtype=bool)
    for _ in range(_TWO_MEANS_ITERATIONS):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignment = distances[:, 1] < distances[:, 0]
        if not assignment.any() or assignment.all():
            return math.inf, assignment
        converged = np.array_equal(assignment, in_second)
        in_second = assignment
        centroids = np.stack([points[~in_second].mean(axis=0), points[in_second].mean(axis=0)])
        if converged:
            break
    return float(np.sum((points - centroids[in_second.astype(int)]) ** 2)), in_second


def _one_dimensional_two_means(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split sorted values at the cut minimizing the within sum of squares; all-equal values give both halves whole."""
    ordered = np.sort(values)
    if ordered[0] == ordered[-1]:
        return ordered, ordered
    sums, squares = np.cumsum(ordered), np.cumsum(ordered ** 2)
    left = np.arange(1, ordered.size)
    within = (squares[:-1] - sums[:-1] ** 2 / left
              + (squares[-1] - squares[:-1]) - (sums[-1] - sums[:-1]) ** 2 / (ordered.size - left))
    cut = int(np.argmin(within)) + 1
    return ordered[:cut], ordered[cut:]


def _ashman_d(first: np.ndarray, second: np.ndarray) -> float:
    """Return Ashman's D, ``sqrt(2) |m1 - m2| / sqrt(v1 + v2)``, infinite when neither set varies."""
    spread = float(first.var() + second.var())
    if not spread > 0:
        return math.inf
    return math.sqrt(2.0) * abs(float(first.mean() - second.mean())) / math.sqrt(spread)


def _two_means_split(rows: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Split a label's standardized rows by deterministic 2-means; return the score, separation and smaller cluster."""
    n_rows = rows.shape[0]
    no_split = 0.0, 0.0, np.zeros(n_rows, dtype=bool)
    # A constant parameter is held at exactly 0: its computed standard deviation need not be 0 (the mean of 400
    # copies of 0.3 is not 0.3), and dividing by it would blow rounding up to unit variance.
    varies = np.ptp(rows, axis=0) > 0
    standardized = np.zeros(rows.shape)
    standardized[:, varies] = (rows[:, varies] - rows[:, varies].mean(axis=0)) / rows[:, varies].std(axis=0)
    total = float(np.sum(standardized ** 2))
    if not total > 0.0:
        return no_split
    _, singular_values, components = np.linalg.svd(standardized, full_matrices=False)
    starts = [singular_values[0] / math.sqrt(n_rows) * components[0]]
    starts += [np.eye(rows.shape[1])[column] for column in np.flatnonzero(varies)]
    within, in_second = min((_lloyd_two_means(standardized, start) for start in starts),
                            key=lambda split: split[0])
    if not math.isfinite(within):
        return no_split
    first, second = standardized[~in_second], standardized[in_second]
    direction = first.mean(axis=0) - second.mean(axis=0)
    separation = _ashman_d(_one_dimensional_two_means(first @ direction)[0],
                           _one_dimensional_two_means(second @ direction)[1])
    smaller = in_second if 2 * np.count_nonzero(in_second) <= n_rows else ~in_second
    return 1.0 - within / total, separation, smaller


def _clusters_appear_together(chain: np.ndarray, label: int, samples: np.ndarray, smaller: np.ndarray) -> bool:
    """Say whether enough samples hold a row of the label in one cluster and another label's row like the other."""
    rows = chain[samples, label]
    varies = np.ptp(rows, axis=0) > 0
    centre = rows.mean(axis=0)
    scale = np.where(varies, rows.std(axis=0), 1.0)

    def standardize(values: np.ndarray) -> np.ndarray:
        # Another label's row can lie far out in this label's units; an overflow to infinity still reads as far.
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = np.where(varies, (values - centre) / scale, 0.0)
        return np.where(varies | (values == rows[0]), scaled, np.inf)

    points = standardize(rows)
    clusters = (points[~smaller], points[smaller])
    centres = [cluster.mean(axis=0) for cluster in clusters]
    radii = [float(np.sqrt(((cluster - centre_of) ** 2).sum(axis=1).max()))
             for cluster, centre_of in zip(clusters, centres)]
    n_smaller = int(np.count_nonzero(smaller))
    needed = max(1, math.ceil(_SPLIT_MIN_TOGETHER * 2 * (samples.size - n_smaller) * n_smaller / chain.shape[0]))
    others = [slot for slot in range(chain.shape[1]) if slot != label]
    block = max(1, _BLOCK_ROWS // chain.shape[1])
    together = 0
    for start in range(0, samples.size, block):
        part = chain[samples[start:start + block]][:, others]
        standardized = standardize(part)
        with np.errstate(over="ignore", invalid="ignore"):
            distances = [np.sqrt(((standardized - centre_of) ** 2).sum(axis=-1)) for centre_of in centres]
        present = source_present(part)
        like_larger = present & (distances[0] <= radii[0]) & (distances[0] < distances[1])
        like_smaller = present & (distances[1] <= radii[1]) & (distances[1] < distances[0])
        in_smaller = smaller[start:start + block]
        together += int(np.count_nonzero(np.where(in_smaller, like_larger.any(axis=1), like_smaller.any(axis=1))))
        if together >= needed:
            return True
    return False


def _split_candidates(chain: np.ndarray, exclusive: bool = False) -> list[tuple[int, int, np.ndarray, float]]:
    """List ``(donor, receiver, samples, score)`` split proposals, best donor first, sparsest receiver first."""
    n_slots, n_params = chain.shape[1], chain.shape[2]
    present = source_present(chain)
    minimum = n_params + 2
    donors = []
    for label in range(n_slots):
        samples = np.flatnonzero(present[:, label])
        if samples.size < 2 * minimum:
            continue
        score, separation, smaller = _two_means_split(chain[samples, label])
        if np.count_nonzero(smaller) < minimum or separation < _SPLIT_MIN_SEPARATION:
            continue
        if not exclusive and not _clusters_appear_together(chain, label, samples, smaller):
            logger.info("Split moves: label %d holds two clusters that too few samples hold together, so it "
                        "is kept whole, as one source with two modes; split_exclusive=True would propose it.",
                        label)
            continue
        donors.append((score, label, samples[smaller]))
    donors.sort(key=lambda donor: (-donor[0], donor[1]))
    count = min(_SPLIT_CANDIDATES, n_slots)
    by_probability = np.argsort(present.mean(axis=0), kind="stable")
    proposals: list[tuple[int, int, np.ndarray, float]] = []
    for score, donor, samples in donors[:count]:
        receivers = [int(label) for label in by_probability if label != donor][:count]
        proposals.extend((donor, receiver, samples, score) for receiver in receivers)
    return proposals


def _propose_split(posterior_chain: PosteriorChain, donor: int, receiver: int,
                   samples: np.ndarray) -> PosteriorChain:
    """Swap two labels' whole rows in some samples; the new chain has an unclipped prob_in_model and no cost at K."""
    chain = posterior_chain.get_chain().copy()
    chain[samples, donor], chain[samples, receiver] = chain[samples, receiver], chain[samples, donor]
    width = posterior_chain.num_sources
    proposal = PosteriorChain(chain, width, posterior_chain.num_params_per_source,
                              trans_dimensional=posterior_chain.trans_dimensional,
                              prob_in_model=find_prob_in_model(chain, width, eps=0),
                              cost_dict={key: value for key, value in posterior_chain.cost_dict.items()
                                         if key != width},
                              validate_nan_convention=False)
    proposal.validate_nan_convention = posterior_chain.validate_nan_convention
    return proposal


@dataclass(frozen=True)
class _Evaluation:
    """What the iterations from one split proposal found: count, cheapest labeling, where, whether it paid, stall."""

    iterations: int
    best: _Labeling
    found_at: int
    paid: bool
    stall: tuple[str, _Labeling] | None


def _evaluate_split(proposal: PosteriorChain,
                    max_num_sources: int,
                    param_fit: FitFunction,
                    compute_cost_matrix: Callable[..., np.ndarray],
                    eps: float,
                    max_iterations: int,
                    pays: Callable[[_Labeling], bool]) -> _Evaluation:
    """Run the loop's own iterations from a proposed labeling, and return the cheapest one found."""
    aux_parameters, prob_in_model = (param_fit(proposal.get_chain(), max_num_sources),
                                     find_prob_in_model(proposal.get_chain(), max_num_sources, eps=eps))
    current = proposal
    old_cost: float | None = None
    best: _Labeling | None = None
    found_at = iterations = 0
    paid = False
    stall: tuple[str, _Labeling] | None = None
    while iterations < max_iterations:
        new = _relabel_and_refit(current, aux_parameters, prob_in_model, max_num_sources,
                                 param_fit, compute_cost_matrix, eps, progress=False)
        iterations += 1
        logger.debug("  Split proposal, iteration %d: cost of assignment %s.", iterations, new.cost)
        reason = _stall_reason(old_cost, new.cost)
        if reason is not None:
            stall = (reason, new)
            break
        best, found_at = new, iterations
        paid = paid or pays(new)
        current, aux_parameters, prob_in_model, old_cost = new.chain, new.aux_parameters, new.prob_in_model, new.cost
    assert best is not None, "max_iterations must be at least 1"
    return _Evaluation(iterations, best, found_at, paid, stall)


def _opened_label_cost(n_samples: int, n_params: int) -> float:
    """Return the per-sample BIC price of one more normal label, ``k log(N) / (2 N)`` with ``k = P + P(P+1)/2 + 1``."""
    num_parameters = n_params + n_params * (n_params + 1) / 2 + 1
    return 0.5 * num_parameters * math.log(n_samples) / n_samples


@dataclass(frozen=True)
class _SplitOutcome:
    """What one split pass did: iterations completed, the labeling accepted, where, its stall, whether it finished."""

    iteration: int
    accepted: _Labeling | None = None
    found_at: int = 0
    stall: tuple[str, _Labeling] | None = None
    complete: bool = True


def _split_pass(best_chain: PosteriorChain,
                best_cost: float,
                max_num_sources: int,
                param_fit: FitFunction,
                compute_cost_matrix: Callable[..., np.ndarray],
                eps: float,
                iteration: int,
                num_iterations: int,
                exclusive: bool = False) -> _SplitOutcome:
    """Evaluate the best labeling's split proposals in turn, and accept the first that beats it net of its price."""
    chain = best_chain.get_chain()
    proposals = _split_candidates(chain, exclusive)
    labels_in_use = int(np.count_nonzero(source_present(chain).any(axis=0)))
    price = _opened_label_cost(chain.shape[0], chain.shape[2])

    def opened(labeling: _Labeling) -> int:
        """Count the labels by which `labeling` raises the number in use: the net increase, never below 0."""
        return max(0, int(np.count_nonzero(source_present(labeling.chain.get_chain()).any(axis=0))) - labels_in_use)

    def pays(labeling: _Labeling) -> bool:
        """Whether `labeling` beats the best by the price of the labels it adds."""
        return labeling.cost < best_cost - opened(labeling) * price

    if not proposals:
        logger.info("Split moves: no label splits into two clusters large enough to fit%s, so there is "
                    "nothing to propose.", "" if exclusive else " and held together often enough")
    complete = True
    for donor, receiver, samples, score in proposals:
        if iteration >= num_iterations:
            logger.info("Split moves: the budget of %d iterations is spent.", num_iterations)
            complete = False
            break
        logger.info("Split move: label %d (bimodality %.3f) gives the %d rows of its smaller cluster to "
                    "label %d.", donor, score, samples.size, receiver)
        evaluation = _evaluate_split(
            _propose_split(best_chain, donor, receiver, samples), max_num_sources, param_fit,
            compute_cost_matrix, eps, min(_SPLIT_EVALUATION_ITERATIONS, num_iterations - iteration), pays)
        if evaluation.paid:
            logger.info("Split move accepted after %d iterations: cost %s, down from %s.",
                        evaluation.iterations, evaluation.best.cost, best_cost)
            return _SplitOutcome(iteration + evaluation.iterations, evaluation.best,
                                 iteration + evaluation.found_at, evaluation.stall)
        candidate = evaluation.best
        logger.info("Split move rejected after %d iterations: its best cost %s does not beat %s "
                    "(%d label(s) opened).", evaluation.iterations, candidate.cost,
                    best_cost - opened(candidate) * price, opened(candidate))
        iteration += evaluation.iterations
        # Incomplete when the budget, not the stopping rule or the cap, ended this evaluation.
        complete = complete and (evaluation.stall is not None
                                 or evaluation.iterations == _SPLIT_EVALUATION_ITERATIONS)
    return _SplitOutcome(iteration, complete=complete)


def run_relabeling_loop(posterior_chain: PosteriorChain,
                        param_fit: FitFunction,
                        compute_cost_matrix: Callable[..., np.ndarray],
                        max_num_sources: int | None = None,
                        num_iterations: int = 200,
                        eps: float = 1e-6,
                        checkpoint_dir: str | os.PathLike[str] | None = None,
                        resume_from: str | os.PathLike[str] | None = None,
                        progress: bool = True,
                        *,
                        method: str | None = None,
                        settings: Mapping[str, Any] | None = None,
                        run_info: Mapping[str, Any] | None = None,
                        keep_checkpoints: int | None = None,
                        split_moves: bool = False,
                        split_exclusive: bool = False) -> PosteriorChain:
    """Relabel a chain by alternating fit and assignment, keep the cheapest labeling, and stop when the cost stalls.

    Parameters:
        posterior_chain: the chain to relabel, never modified; when resuming, the chain the run was started with.
        param_fit: ``(chain, K) -> aux_parameters``, refitted to each new labeling.
        compute_cost_matrix: ``(sample, aux_parameters, prob_in_model, K)`` giving the (K, S) rewards to maximize.
        max_num_sources: number of source slots K, at least the chain's; None means the chain's own width.
        num_iterations: iteration budget, counting those a resumed checkpoint holds and those split passes spend.
        eps: clip bound of the probabilities fed to the cost matrix; the returned prob_in_model is unclipped.
        checkpoint_dir: directory the best labeling is written to whenever it or its cost improves, and on stopping.
        resume_from: checkpoint file, or directory of them, of an earlier run of the same method on the same chain.
        progress: show a progress bar over the samples of each iteration.
        method: name of the method; required with checkpoint_dir or resume_from, compared on resume.
        settings: values that change what a cost means; recorded and compared on resume.
        run_info: settings of the entry point; recorded and never compared.
        keep_checkpoints: keep only this many newest checkpoints; None keeps them all.
        split_moves: where the loop would stop, first try to split a label holding two clusters into a sparse label.
        split_exclusive: with split_moves, also split a label whose two clusters the chain rarely shows together.

    Returns:
        the cheapest labeling accepted, with prob_in_model recomputed from it without clipping.
        `cost_dict[K]` is the returned labeling's cost under the fit before it.

    Raises:
        TypeError: posterior_chain is not a PosteriorChain.
        ValueError: an invalid setting, width or empty chain; checkpointing without method; a checkpoint_dir holding
            other checkpoints; or a resumed checkpoint of another run.
        FileNotFoundError: resume_from is neither a checkpoint file nor a directory holding one.
        PermissionError: checkpoint_dir is not writable.

    Warns:
        UserWarning: the resumed run had already stopped or used the budget; with split_moves, the budget ended the
            search before it stopped.
    """
    check_int("num_iterations", num_iterations)
    check_eps(eps)
    if keep_checkpoints is not None:
        check_int("keep_checkpoints", keep_checkpoints)
    width = check_chain(posterior_chain, max_num_sources)
    # Taken before widening, which the digest ignores; hashing is a pass over the chain, so only when checkpointing.
    fingerprint: dict[str, Any] | None = None
    if checkpoint_dir is not None or resume_from is not None:
        if method is None:
            raise ValueError("run_relabeling_loop needs method= to write or resume checkpoints: a checkpoint records "
                             "the method that wrote it, so a resume can refuse another method's costs.")
        fingerprint = run_fingerprint(posterior_chain, method=method, settings=settings, max_num_sources=width,
                                      eps=eps, run_info=run_info)

    start_iteration = best_iteration = 0
    resumed: Checkpoint | None = None
    if resume_from is not None:
        resumed = load_checkpoint(resume_from, fingerprint=fingerprint)
        if checkpoint_dir is not None:
            prepare_checkpoint_dir(checkpoint_dir, resumed)
        logger.warning("Resuming from checkpoint %s after %d completed iterations.",
                       resumed.path, resumed.completed_iterations)
        posterior_chain, start_iteration, best_iteration = (resumed.posterior_chain, resumed.completed_iterations,
                                                            resumed.file_iteration)
    else:
        posterior_chain = prepare_chain(posterior_chain, width)
        if checkpoint_dir is not None:
            prepare_checkpoint_dir(checkpoint_dir)
    max_num_sources = posterior_chain.num_sources

    if resumed is not None and start_iteration >= num_iterations:
        logger.warning("Checkpoint already holds %d of %d iterations; nothing to do.",
                       start_iteration, num_iterations)
        warnings.warn(
            f"Checkpoint {resumed.path!r} already holds {start_iteration} completed iterations, and "
            f"num_iterations={num_iterations} counts those too, so there is nothing to do: it is "
            f"returned unchanged. Pass num_iterations={start_iteration + 1} or more to continue "
            f"the run.",
            UserWarning, stacklevel=2,
        )
        return _with_unclipped_prob_in_model(posterior_chain, max_num_sources)

    if resumed is not None and resumed.state["stopped"]:
        reason = _STOP_REASONS.get(resumed.state["stop_reason"], resumed.state["stop_reason"])
        logger.warning("The run that wrote %s already stopped after %d iterations because %s; "
                       "nothing to do.", resumed.path, start_iteration, reason)
        warnings.warn(
            f"The run that wrote checkpoint {resumed.path!r} already stopped after "
            f"{start_iteration} iterations because {reason}, so there is nothing to do: its "
            f"labeling is final and is returned unchanged. To relabel it further, start a fresh "
            f"run in an empty checkpoint directory.",
            UserWarning, stacklevel=2,
        )
        return _with_unclipped_prob_in_model(posterior_chain, max_num_sources)

    logger.info("Sorting the posterior chain: maximum number of iterations %d, "
                "maximum number of source labels %d.", num_iterations, max_num_sources)

    old_posterior_chain = posterior_chain
    old_parametric_fit, old_prob_in_model = (param_fit(posterior_chain.get_chain(), max_num_sources),
                                             find_prob_in_model(posterior_chain.get_chain(), max_num_sources, eps=eps))
    # None, not 0.0: the first iteration has no previous cost, and 0.0 is a reachable cost.
    old_cost_of_assignment: float | None = None
    best_cost_of_assignment: float | None = None
    best_posterior_chain = _with_unclipped_prob_in_model(posterior_chain, max_num_sources)
    if resumed is not None:
        # A resume never comes back worse than its checkpoint, the run's latest labeling, so its cost is both the one
        # to beat and the previous cost; a fresh run's input cost came from another fit and is not comparable.
        best_cost_of_assignment = old_cost_of_assignment = posterior_chain.cost_dict.get(max_num_sources)
        logger.info("The resumed labeling cost %s; that is what this run has to beat.",
                    best_cost_of_assignment)

    def keep_if_cheaper(candidate: _Labeling, found_at: int) -> bool:
        """Make `candidate` the best labeling if it costs strictly less; say whether it did."""
        nonlocal best_cost_of_assignment, best_posterior_chain, best_iteration
        if best_cost_of_assignment is not None and not candidate.cost < best_cost_of_assignment:
            return False
        # A refit can lower the cost of an unchanged assignment: that labeling keeps the iteration that found it, so
        # its file is rewritten in place rather than copied.
        if best_cost_of_assignment is None or not np.array_equal(
                candidate.chain.get_chain(), best_posterior_chain.get_chain(), equal_nan=True):
            best_iteration = found_at
        best_cost_of_assignment = candidate.cost
        best_posterior_chain = candidate.chain
        return True

    def save(completed: int, stop_reason: str | None) -> None:
        """Write the best labeling, with the run's progress, if the run checkpoints at all."""
        if checkpoint_dir is None:
            return
        assert fingerprint is not None
        write_checkpoint(best_posterior_chain, checkpoint_dir, best_iteration, {
            **fingerprint,
            "iteration": completed,
            "best_iteration": best_iteration,
            "best_cost": best_cost_of_assignment,
            "stopped": stop_reason is not None,
            "stop_reason": stop_reason,
        }, keep=keep_checkpoints)

    iteration = start_iteration
    while iteration < num_iterations:
        new = _relabel_and_refit(old_posterior_chain, old_parametric_fit, old_prob_in_model,
                                 max_num_sources, param_fit, compute_cost_matrix, eps, progress)
        iteration += 1

        if old_cost_of_assignment is None:
            logger.info("Iteration %d: cost of assignment is %s; there is no previous cost to "
                        "compare it with.", iteration, new.cost)
        else:
            logger.info("Iteration %d: Difference in cost of assignment is %s with total cost of %s.",
                        iteration, new.cost - old_cost_of_assignment, new.cost)
        logger.info("\tProbabilities in model: %s", new.prob_in_model)

        # A checkpoint holds the best labeling, not the latest: written when it or its cost improves, and once more on
        # stopping, so a resume knows the run is over.
        if keep_if_cheaper(new, iteration):
            save(iteration, None)
        stop_reason = _stall_reason(old_cost_of_assignment, new.cost)

        # An accepted split move is checkpointed where it was found and the loop carries on from it as a resume
        # would; a stall its evaluation already ran into is taken over, not recomputed.
        while stop_reason is not None and split_moves:
            assert best_cost_of_assignment is not None
            logger.info("Iteration %d: the search stalled (%s) at cost %s; trying split moves.",
                        iteration, stop_reason, best_cost_of_assignment)
            # Even with no budget left: a pass with nothing to propose ends the search, not the budget.
            outcome = _split_pass(best_posterior_chain, best_cost_of_assignment, max_num_sources, param_fit,
                                  compute_cost_matrix, eps, iteration, num_iterations, split_exclusive)
            iteration = outcome.iteration
            if outcome.accepted is None:
                if outcome.complete:
                    logger.info("No split move lowered the cost below %s.", best_cost_of_assignment)
                else:
                    # The budget, not the search, ended here: not a stop, and nothing is written, so a resume with a
                    # larger budget repeats the same iterations and then runs this pass in full.
                    logger.info("The budget of %d iterations ran out before the split pass finished; "
                                "a resume with a larger num_iterations can finish it.", num_iterations)
                    stop_reason = None
                break
            keep_if_cheaper(outcome.accepted, outcome.found_at)
            save(outcome.found_at, None)
            if outcome.stall is None:
                new, stop_reason = outcome.accepted, None
            else:
                old_cost_of_assignment = outcome.accepted.cost
                stop_reason, new = outcome.stall

        if stop_reason is not None:
            save(iteration, stop_reason)
            if stop_reason == "converged":
                logger.info("Stopped after %d iterations because the cost didn't change from the previous "
                            "iteration.", iteration)
            else:
                logger.info("Stopped after %d iterations because the cost increased from %s to %s; "
                            "returning the best labeling accepted, of cost %s.",
                            iteration, old_cost_of_assignment, new.cost, best_cost_of_assignment)
            break

        old_posterior_chain, old_parametric_fit, old_prob_in_model, old_cost_of_assignment = (
            new.chain, new.aux_parameters, new.prob_in_model, new.cost)

    else:
        logger.info("Final cost of assignment: %s after the maximum number (%d) of iterations.",
                    best_cost_of_assignment, num_iterations)
        # With split moves the search is over only when a pass accepts nothing, so a budget that ends it warns.
        if split_moves:
            warnings.warn(
                f"The budget of num_iterations={num_iterations} ran out before the search with split "
                f"moves stopped, so the search did not finish, and the labeling returned is the best "
                f"found so far. Each split proposal can take up to {_SPLIT_EVALUATION_ITERATIONS} "
                f"iterations. Pass a larger num_iterations to finish the search -- a run with a "
                f"checkpoint_dir can be resumed with one -- or split_moves=False for the plain search.",
                UserWarning, stacklevel=2,
            )

    return best_posterior_chain
