"""
The shared relabeling loop: fit, cost matrix, Hungarian assignment, repeat.

This module is the engine every relabeling method in :mod:`petra` is built on.
A method is defined by two callables -- a *parametric fit*
``(chain, max_num_sources) -> aux_parameters`` and an *auxiliary distribution*
``(sample, aux_parameters, indices) -> log_pdf`` -- and
:func:`create_relabel_samples` turns that pair into a full relabeler.  Nothing
here knows what the auxiliary distributions are; adding a method means adding a
fit and a distribution, not another loop.

Conventions
-----------
Chains are ``(n_samples, n_sources, n_params_per_source)`` and ``NaN`` means
"this source is absent from this sample"; see :mod:`petra.posterior_chain`.

*Cost is negated log-likelihood, so lower is better.*
:func:`relabel_samples_one_iteration` maximizes the log-density sum with
:func:`scipy.optimize.linear_sum_assignment` and returns minus that sum averaged
over samples.  Every ``cost_dict`` entry written here follows that sign, and the
loop in :func:`create_relabel_samples` keeps the *cheapest* labeling it has seen
and reverts to it as soon as the cost goes back up, so a relabeler never returns
a labeling worse than its best.

Relabeling is always non-destructive: no function in this module modifies the
chain, the ``cost_dict`` or any other attribute of the
:class:`~petra.posterior_chain.PosteriorChain` it is handed.

*A returned chain's ``prob_in_model`` always describes that chain.*
:func:`run_relabeling_loop` recomputes it from the labeling it is about to hand
back, with no ``eps`` clipping, so that
``find_prob_in_model(result.get_chain(), n, eps=0)`` reproduces
``result.prob_in_model`` exactly.  ``eps`` clips only the copy fed to the cost
matrix during iteration, where a probability of exactly 0 or 1 would make a
``log`` term infinite; see the Notes of :func:`run_relabeling_loop`.

Progress bars are controlled by the shared ``progress`` keyword (default
``True``) and all diagnostics go to ``logging.getLogger("petra.relabel")``;
nothing here prints.
"""

import os
import re
from dataclasses import replace
from typing import Any, Callable, Optional

import numpy as np
from scipy import optimize
from tqdm import tqdm

from petra.utils import _validate_eps, fill_missing_indices, find_prob_in_model, get_logger
from petra.options import _check_at_least_one
from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import (FitFunction, update_parametric_fit_and_prob_in_model,
                                   create_parametric_fit)
from petra.cost_matrix import create_compute_cost_matrix

logger = get_logger(__name__)

#: Filename pattern used for the per-iteration checkpoints written by
#: ``relabel_samples(..., checkpoint_dir=...)`` and understood by
#: :func:`find_latest_checkpoint`.
CHECKPOINT_TEMPLATE = "posterior_chain_iteration_{iteration:03d}.feather"
_CHECKPOINT_RE = re.compile(r"^posterior_chain_iteration_(\d+)\.feather$")


def _validate_relabel_settings(num_iterations: int, eps: float) -> None:
    """Check iteration and clipping settings before fitting or initialization."""
    _check_at_least_one("num_iterations", num_iterations)
    _validate_eps(eps)


def prepare_chain(posterior_chain: PosteriorChain,
                  max_num_sources: int | None = None,
                  shuffle_seed: int | None = None) -> PosteriorChain:
    """
    Validate, optionally shuffle, and widen a chain before relabeling.

    Every ``make_catalog_*`` entry point needs the same preamble, and
    :meth:`PosteriorChain.expand_chain` is *not* in-place -- it returns a new
    object -- so calling it for effect silently does nothing.  This helper is
    the single implementation.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain to prepare.  Never modified in place.
    max_num_sources : int, optional
        Target number of source slots.  Defaults to
        ``posterior_chain.num_sources``.
    shuffle_seed : int, optional
        If given, entries are shuffled per sample with this seed *before* the
        chain is widened, so that the NaN padding is not shuffled in.

    Returns
    -------
    PosteriorChain
        Chain with exactly `max_num_sources` source slots.  This may be the
        input object itself when no shuffling or widening was required.  When
        either step did run, the returned chain's ``prob_in_model`` is
        re-derived from its own samples and its ``cost_dict`` is empty: both
        describe a labeling that shuffling replaced and widening changed the
        width of.  ``validate_nan_convention`` is inherited, so a chain the
        caller opted out for is not re-checked here.

    Raises
    ------
    ValueError
        If `max_num_sources` is smaller than ``posterior_chain.num_sources``;
        the chain would have to lose sources, which is never intended.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.posterior_chain import PosteriorChain
    >>> pc = PosteriorChain(np.random.randn(10, 2, 3), 2, 3)
    >>> prepare_chain(pc, 4).chain.shape
    (10, 4, 3)
    >>> prepare_chain(pc, 2) is pc
    True
    >>> prepare_chain(pc, 1)
    Traceback (most recent call last):
        ...
    ValueError: max_num_sources (1) cannot be less than the number of entries in the chain (2).
    """
    if max_num_sources is None:
        max_num_sources = posterior_chain.num_sources
    if max_num_sources < posterior_chain.num_sources:
        raise ValueError(
            f"max_num_sources ({max_num_sources}) cannot be less than the number of "
            f"entries in the chain ({posterior_chain.num_sources})."
        )

    if shuffle_seed is not None:
        logger.info("Shuffling chain entries with seed %s.", shuffle_seed)
        posterior_chain = posterior_chain.randomize_entries(shuffle_seed)

    if max_num_sources > posterior_chain.num_sources:
        logger.info("Expanding posterior chain from %d to %d source slots.",
                    posterior_chain.num_sources, max_num_sources)
        posterior_chain = posterior_chain.expand_chain(max_num_sources)

    return posterior_chain


def _checkpoint_posterior_chain(posterior_chain: PosteriorChain, checkpoint_dir: str, iteration: int) -> str:
    """
    Save a PosteriorChain to disk as a checkpoint.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        The chain to save, with the ``prob_in_model`` and ``cost_dict`` a
        resumed run will need; :meth:`PosteriorChain.to_feather` persists both.
    checkpoint_dir : str
        Directory to save checkpoints in
    iteration : int
        Number of completed iterations (used in the filename)

    Returns
    -------
    filepath : str
        Path the checkpoint was written to.

    """
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create filename with iteration number
    filename = CHECKPOINT_TEMPLATE.format(iteration=iteration)
    filepath = os.path.join(checkpoint_dir, filename)

    # Save the chain
    posterior_chain.to_feather(filepath)
    logger.info("  Checkpoint saved: %s", filepath)
    return filepath


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[str, int] | None:
    """
    Locate the most advanced checkpoint written into a directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory previously passed as ``checkpoint_dir`` to a relabeler.

    Returns
    -------
    latest : tuple of (str, int) or None
        Path of the highest-numbered checkpoint and the number of completed
        iterations it represents, or ``None`` if the directory does not exist
        or holds no checkpoint.

    Examples
    --------
    >>> find_latest_checkpoint('/no/such/directory') is None
    True
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    found = []
    for filename in os.listdir(checkpoint_dir):
        match = _CHECKPOINT_RE.match(filename)
        if match is not None:
            found.append((int(match.group(1)), filename))
    if not found:
        return None
    iteration, filename = max(found)
    return os.path.join(checkpoint_dir, filename), iteration


def load_checkpoint(resume_from: str) -> tuple[PosteriorChain, int]:
    """
    Load a relabeling checkpoint written by ``relabel_samples``.

    Parameters
    ----------
    resume_from : str
        Either a checkpoint file written by
        :func:`_checkpoint_posterior_chain`, or a directory holding such files,
        in which case the highest-numbered one is used.

    Returns
    -------
    posterior_chain : PosteriorChain
        The checkpointed chain.
    completed_iterations : int
        Number of relabeling iterations already completed, parsed from the
        filename; ``0`` if the name does not follow `CHECKPOINT_TEMPLATE`.

    Raises
    ------
    FileNotFoundError
        If `resume_from` is neither a checkpoint file nor a directory holding
        one.

    Notes
    -----
    The chain, its ``prob_in_model`` and its ``cost_dict`` all survive the round
    trip: :meth:`PosteriorChain.to_feather` writes all three into the Feather
    schema metadata.  A resumed run therefore knows what the labeling it is
    resuming from cost, and :func:`run_relabeling_loop` takes that cost as the
    one to beat.  Only files in the legacy column layout, written before that
    metadata existed, come back with an empty ``cost_dict``; a run resumed from
    one of those has nothing to compare against and behaves like a fresh run.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from petra.posterior_chain import PosteriorChain
    >>> pc = PosteriorChain(np.zeros((4, 2, 1)), 2, 1,
    ...                     prob_in_model=np.array([1.0, 0.0]), cost_dict={2: -3.5})
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     _ = _checkpoint_posterior_chain(pc, tmpdir, 7)
    ...     restored, completed = load_checkpoint(tmpdir)
    >>> completed, restored.cost_dict, restored.prob_in_model
    (7, {2: -3.5}, array([1., 0.]))
    """
    if os.path.isdir(resume_from):
        latest = find_latest_checkpoint(resume_from)
        if latest is None:
            raise FileNotFoundError(f"No relabeling checkpoint found in {resume_from!r}.")
        filepath, completed_iterations = latest
    elif os.path.isfile(resume_from):
        filepath = resume_from
        match = _CHECKPOINT_RE.match(os.path.basename(resume_from))
        completed_iterations = int(match.group(1)) if match is not None else 0
    else:
        raise FileNotFoundError(f"No such checkpoint file or directory: {resume_from!r}.")

    return PosteriorChain.read_feather(filepath), completed_iterations


def relabel_samples_one_iteration(chain: np.ndarray,
                                  aux_parameters: Any,
                                  prob_in_model: np.ndarray,
                                  max_num_sources: int,
                                  compute_cost_matrix: Callable,
                                  progress: bool = True) -> tuple[np.ndarray, float]:
    """
    Perform one iteration of label assignment using the Hungarian algorithm.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples to relabel.
    aux_parameters : object
        Parameters (e.g. means and covariances) returned by a parametric fit,
        passed straight through to `compute_cost_matrix`.
    prob_in_model : ndarray, shape (max_num_sources,)
        Probability each source is present in the model.
    max_num_sources : int
        Maximum number of sources to consider in assignment.  May be smaller
        than ``chain.shape[1]``, in which case the surplus source slots keep
        their own labels; see Notes.
    compute_cost_matrix : callable
        Function with signature
        `(sample, aux_parameters, prob_in_model, max_num_sources) -> cost_matrix`,
        returning an array of shape ``(max_num_sources, n_sources)``.
    progress : bool, default True
        Show a tqdm progress bar over the samples.

    Returns
    -------
    relabeled_chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Samples reordered according to optimal assignments.
    cost : float
        Average assignment cost over all samples.  This is *minus* the average
        of the maximized log-density sum, so lower is better.

    Notes
    -----
    :func:`scipy.optimize.linear_sum_assignment` returns one column per *row*
    of the cost matrix, i.e. ``max_num_sources`` entries, whereas reordering a
    sample needs a permutation of all ``chain.shape[1]`` slots.  When the two
    agree -- the case for every relabeler built by
    :func:`create_relabel_samples`, since
    :func:`prepare_chain` widens the chain to ``max_num_sources`` first --
    ``col_ind`` is already a full permutation and
    :func:`petra.utils.fill_missing_indices` returns it unchanged.  When
    `max_num_sources` is smaller, it is load-bearing: without it the
    unassigned slots would be dropped and the returned chain would silently
    lose sources.

    That narrower case is supported *here*, on a bare array, and not by
    :func:`relabel_posterior_chain_one_iteration` one layer up.  The difference
    is not an oversight: this function returns samples and a number, whereas the
    wrapper returns a :class:`~petra.posterior_chain.PosteriorChain`, which
    carries a ``prob_in_model`` of one entry per source slot and a ``cost_dict``
    keyed by source count.  With fewer auxiliary distributions than slots there
    is no value either field could take for the slots the assignment never
    looked at, so the wrapper requires the two to agree and says so.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import mv_normal_aux_distribution
    >>> from petra.cost_matrix import create_compute_cost_matrix
    >>> from petra.parametric_fits import create_parametric_fit, mv_normal_fit
    >>> from petra.utils import find_prob_in_model
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(20, 2, 1)) + np.array([[0.0], [5.0]])
    >>> aux_params = create_parametric_fit(mv_normal_fit)(chain, max_num_sources=2)
    >>> prob = find_prob_in_model(chain, max_num_sources=2)
    >>> compute_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)
    >>> relabeled, cost = relabel_samples_one_iteration(
    ...     chain, aux_params, prob, 2, compute_cost_matrix, progress=False)
    >>> relabeled.shape
    (20, 2, 1)
    """
    relabeled_list = []
    total_cost = 0.0
    for sample in tqdm(chain, disable=not progress):
        cost_matrix = compute_cost_matrix(sample, aux_parameters, prob_in_model, max_num_sources)
        total_entries = max(len(sample), max_num_sources)  # pick the larger of number of aux distributions or sample entries
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)  # solve the linear sum assignment problem
        total_cost -= cost_matrix[row_ind, col_ind].sum()
        # `col_ind` has one entry per aux distribution, not one per source slot.  When the
        # chain is exactly `max_num_sources` wide -- the shared code path -- it is already a
        # full permutation and this call is the identity.  When the chain is wider, the slots
        # the assignment never looked at have to be appended, or they would be dropped from
        # the reordering below and the chain would silently lose sources.  See the Notes.
        relabeled_list.append(fill_missing_indices(total_entries, col_ind))
    relabeled_array = np.array(relabeled_list)

    relabeled_samples = np.array([row[m] for row, m in zip(chain, relabeled_array)])  # reorder the original samples
    total_cost /= len(chain)
    return relabeled_samples, total_cost


def relabel_posterior_chain_one_iteration(posterior_chain: PosteriorChain,
                                          aux_parameters: Any,
                                          prob_in_model: np.ndarray,
                                          max_num_sources: int,
                                          compute_cost_matrix: Callable,
                                          progress: bool = True) -> PosteriorChain:
    """
    Perform one relabeling step on a PosteriorChain instance.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Object containing chain, metadata and previous cost dictionary.
    aux_parameters : object
        Parameters from a parametric fit (e.g. means and covariances).
    prob_in_model : ndarray, shape (max_num_sources,)
        Probability each source is present in the model.
    max_num_sources : int
        Number of sources for this iteration.  Must equal
        ``posterior_chain.num_sources``; see Notes.
    compute_cost_matrix : callable
        Cost matrix builder function for each sample.
    progress : bool, default True
        Show a tqdm progress bar over the samples.

    Returns
    -------
    new_chain : PosteriorChain
        New PosteriorChain with the relabeled samples, the supplied
        `prob_in_model`, and a *copy* of the input's ``cost_dict`` with this
        iteration's cost recorded under ``cost_dict[max_num_sources]``.  The
        caller's chain -- including its ``cost_dict`` -- is left untouched.
        ``trans_dimensional`` and ``validate_nan_convention`` are inherited from
        the input.

    Raises
    ------
    ValueError
        If `max_num_sources` is not ``posterior_chain.num_sources``.

    Notes
    -----
    The ``prob_in_model`` on the returned chain is the array the cost matrix was
    built from -- computed from the *pre*-relabel labeling and clipped -- not one
    describing the samples that come back.  A caller handing this chain onward
    owes it the correction :func:`run_relabeling_loop` applies; that is what
    ``_with_unclipped_prob_in_model`` is for.

    ``max_num_sources`` has to match the width of the chain, which is *narrower*
    than what :func:`relabel_samples_one_iteration` accepts underneath.  That
    function returns an array; this one returns a chain, and a chain carries a
    ``prob_in_model`` of one entry per source slot.  Handing it the array the
    cost matrix used -- ``max_num_sources`` entries -- attached to a chain that
    is wider produces exactly the self-contradicting object
    :class:`~petra.posterior_chain.PosteriorChain` exists to refuse, and there
    is no honest value for the slots the assignment never scored: the caller
    supplied none, and 0 would claim they are empty when they are not.  Relabel
    a wider chain against fewer distributions through
    :func:`relabel_samples_one_iteration` directly, which is what the narrower
    contract is for.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import mv_normal_aux_distribution
    >>> from petra.cost_matrix import create_compute_cost_matrix
    >>> from petra.parametric_fits import create_parametric_fit, mv_normal_fit
    >>> from petra.posterior_chain import PosteriorChain
    >>> from petra.utils import find_prob_in_model
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(20, 2, 1)) + np.array([[0.0], [5.0]])
    >>> pc = PosteriorChain(chain, 2, 1, trans_dimensional=True)
    >>> aux_params = create_parametric_fit(mv_normal_fit)(chain, max_num_sources=2)
    >>> prob = find_prob_in_model(chain, max_num_sources=2)
    >>> compute_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)
    >>> new_pc = relabel_posterior_chain_one_iteration(
    ...     pc, aux_params, prob, 2, compute_cost_matrix, progress=False)
    >>> isinstance(new_pc, PosteriorChain)
    True
    >>> sorted(new_pc.cost_dict), pc.cost_dict   # the caller's dict is untouched
    ([2], {})
    >>> new_pc.trans_dimensional                 # and its metadata is preserved
    True
    >>> relabel_posterior_chain_one_iteration(pc, aux_params, prob[:1], 1,
    ...                                       compute_cost_matrix, progress=False)
    Traceback (most recent call last):
        ...
    ValueError: max_num_sources (1) must equal the chain's num_sources (2): the returned PosteriorChain carries one inclusion probability per source slot, and there is none to give the 1 slot the assignment never scored. Call relabel_samples_one_iteration on posterior_chain.get_chain() to relabel a wider chain against fewer distributions.
    """
    # Named here rather than left to PosteriorChain's own check further down, which
    # reports a prob_in_model of the wrong shape without saying which argument chose
    # that shape or that the layer beneath does support the mismatch.  See the Notes.
    if max_num_sources != posterior_chain.num_sources:
        raise ValueError(
            f"max_num_sources ({max_num_sources}) must equal the chain's num_sources "
            f"({posterior_chain.num_sources}): the returned PosteriorChain carries one "
            f"inclusion probability per source slot, and there is none to give the "
            f"{abs(posterior_chain.num_sources - max_num_sources)} slot"
            f"{'' if abs(posterior_chain.num_sources - max_num_sources) == 1 else 's'} "
            f"the assignment never scored. Call relabel_samples_one_iteration on "
            f"posterior_chain.get_chain() to relabel a wider chain against fewer "
            f"distributions."
        )
    # Copy: the caller's cost_dict must not be mutated by a relabeling run.
    cost_dict = dict(posterior_chain.cost_dict)
    relabeled_chain, total_cost = relabel_samples_one_iteration(
        posterior_chain.get_chain(), aux_parameters, prob_in_model, max_num_sources,
        compute_cost_matrix, progress=progress)
    cost_dict[max_num_sources] = total_cost
    # `validate_nan_convention` rides along: this rebuild happens once per iteration, so
    # a chain the caller opted out for would otherwise be rejected on the first one, by
    # the very message that told them to pass the flag.
    return PosteriorChain(relabeled_chain,
                          posterior_chain.num_sources,
                          posterior_chain.num_params_per_source,
                          trans_dimensional=posterior_chain.trans_dimensional,
                          prob_in_model=prob_in_model,
                          cost_dict=cost_dict,
                          validate_nan_convention=posterior_chain.validate_nan_convention)


def _with_unclipped_prob_in_model(posterior_chain: PosteriorChain,
                                  max_num_sources: int) -> PosteriorChain:
    """
    Re-derive a chain's ``prob_in_model`` from its own samples, without clipping.

    A chain handed to a caller has to describe itself.  The array
    :func:`relabel_posterior_chain_one_iteration` stores is the one the *cost
    matrix* was built from: computed before the relabeling and clipped into
    ``[eps, 1 - eps]``, so it is stale by one iteration and can never report a
    source that is always -- or never -- present.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain whose ``prob_in_model`` is to be replaced.  Not modified: a new
        instance is returned and every other field is carried over unchanged.
    max_num_sources : int
        Number of source slots to report a probability for.

    Returns
    -------
    PosteriorChain
        Copy of `posterior_chain` whose ``prob_in_model`` is
        ``find_prob_in_model(posterior_chain.get_chain(), max_num_sources, eps=0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.posterior_chain import PosteriorChain
    >>> chain = np.array([[[1.0], [np.nan]], [[2.0], [3.0]]])
    >>> pc = PosteriorChain(chain, 2, 1, trans_dimensional=True,
    ...                     prob_in_model=np.array([0.99, 0.01]))
    >>> _with_unclipped_prob_in_model(pc, 2).prob_in_model
    array([1. , 0.5])
    >>> pc.prob_in_model                      # the caller's chain is untouched
    array([0.99, 0.01])
    """
    return replace(posterior_chain,
                   prob_in_model=find_prob_in_model(posterior_chain.get_chain(),
                                                    max_num_sources, eps=0))


def run_relabeling_loop(posterior_chain: PosteriorChain,
                        param_fit: FitFunction,
                        compute_cost_matrix: Callable,
                        max_num_sources: int | None = None,
                        num_iterations: int = 200,
                        eps: float = 1e-6,
                        checkpoint_dir: Optional[str] = None,
                        resume_from: Optional[str] = None,
                        progress: bool = True) -> PosteriorChain:
    """
    Run the shared relabeling loop: fit, assign, keep the best, stop when it stalls.

    Every relabeling method in petra is this loop plus a `param_fit` and a
    `compute_cost_matrix`.  :func:`create_relabel_samples` builds that pair from
    a ``(fit, aux_distribution)`` pair; :func:`petra.bayesian_gaussian.bayesian_relabel_loop`
    supplies its own.  Nothing else about the methods differs, so nothing else
    is parameterised here.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain to relabel; widened to `max_num_sources` if necessary.  Never
        modified in place.
    param_fit : callable
        Fit function ``(chain, max_num_sources) -> aux_parameters``, as returned
        by :func:`petra.parametric_fits.create_parametric_fit`.  It is re-run on
        the current labeling at the start of every iteration.
    compute_cost_matrix : callable
        Cost matrix builder
        ``(sample, aux_parameters, prob_in_model, max_num_sources) -> cost_matrix``,
        as returned by :func:`petra.cost_matrix.create_compute_cost_matrix`.
    max_num_sources : int, optional
        Target number of source slots (defaults to ``posterior_chain.num_sources``).
    num_iterations : int, default 200
        Maximum relabeling iterations before stopping.  Must be at least 1.
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities *fed to the cost matrix*,
        keeping them inside ``[eps, 1 - eps]`` so the logarithms there stay
        finite.  It does not reach the returned array; see Notes.
    checkpoint_dir : str, optional
        Directory to write the best chain available after every iteration.
        ``None`` disables checkpointing.
    resume_from : str, optional
        Checkpoint file, or directory of checkpoints, written by an earlier run
        with the same `checkpoint_dir`.  The chain stored there replaces
        `posterior_chain`, the iteration counter picks up where that run
        stopped, and the cost stored with the chain becomes the cost to beat, so
        a job killed at the walltime limit can be restarted without losing its
        progress and without coming back worse than the checkpoint.  See
        :func:`load_checkpoint` for what survives the round-trip.
    progress : bool, default True
        Show a tqdm progress bar over the samples of each iteration.

    Returns
    -------
    PosteriorChain
        The cheapest labeling seen during the run, with that cost recorded in
        ``cost_dict[max_num_sources]`` and with ``prob_in_model`` recomputed
        from the returned samples *without* clipping, so that
        ``find_prob_in_model(result.get_chain(), max_num_sources, eps=0)``
        reproduces it exactly.

    Raises
    ------
    ValueError
        If `num_iterations` is less than 1, or if `max_num_sources` is smaller
        than the number of entries in the chain.

    Notes
    -----
    Cost is negated log-likelihood, so lower is better.  The loop keeps the
    cheapest labeling it has produced and stops as soon as an iteration comes
    back more expensive than the one before it, returning that best labeling
    rather than the last one.  Iterating a relabeler is not a descent method --
    refitting the auxiliary distributions to a new labeling can raise the cost
    -- so without this the returned catalog could be worse than one already
    computed on the way to it.

    Resuming extends that guarantee across a restart.  The checkpoint's own cost
    seeds the best-so-far, so a run picked up after a walltime kill cannot come
    back with a labeling more expensive than the one already on disk.  The
    *stopping* rule is not seeded the same way: ``old_cost_of_assignment`` starts
    at ``None`` again, so a resumed run always completes one iteration before it
    has a difference to test, exactly as a fresh one does.

    `eps` governs only the clipping applied *inside* the loop, to the copy of
    the inclusion probabilities that the cost matrix takes logarithms of: a
    source present in every sample would otherwise contribute ``log(1 - 1)``
    and a source present in none ``log(0)``.  The array that comes back is
    unclipped, and is recomputed from the labeling being returned rather than
    from the one the last cost matrix was built on.  A chain that reports
    ``prob_in_model = [0.99, 0.01]`` for sources that are in fact always and
    never present is describing the clip bound, not the posterior.

    The two loops this replaces looked like they differed in how they refreshed
    the fit and the inclusion probabilities each iteration -- one called
    :func:`petra.parametric_fits.update_parametric_fit_and_prob_in_model`, the
    other called its fit and :func:`petra.utils.find_prob_in_model` by hand.
    That helper *is* exactly those two calls, so there is nothing to
    parameterise there: `param_fit` and `eps` cover both spellings.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import mv_normal_aux_distribution
    >>> from petra.cost_matrix import create_compute_cost_matrix
    >>> from petra.parametric_fits import create_parametric_fit, mv_normal_fit
    >>> from petra.posterior_chain import PosteriorChain
    >>> from petra.utils import find_prob_in_model
    >>> rng = np.random.default_rng(0)
    >>> pc = PosteriorChain(rng.normal(size=(20, 2, 1)) + np.array([[0.0], [5.0]]), 2, 1)
    >>> result = run_relabeling_loop(pc,
    ...                              create_parametric_fit(mv_normal_fit),
    ...                              create_compute_cost_matrix(mv_normal_aux_distribution),
    ...                              max_num_sources=2, num_iterations=3, progress=False)
    >>> sorted(result.cost_dict)
    [2]

    Both sources are in every sample, and the returned probabilities say so
    exactly rather than reporting ``1 - eps``:

    >>> result.prob_in_model
    array([1., 1.])
    >>> bool(np.array_equal(result.prob_in_model,
    ...                     find_prob_in_model(result.get_chain(), 2, eps=0)))
    True
    """
    _validate_relabel_settings(num_iterations, eps)

    start_iteration = 0
    if resume_from is not None:
        posterior_chain, start_iteration = load_checkpoint(resume_from)
        logger.warning("Resuming from checkpoint %s after %d completed iterations.",
                       resume_from, start_iteration)

    posterior_chain = prepare_chain(posterior_chain, max_num_sources)
    max_num_sources = posterior_chain.num_sources

    if start_iteration >= num_iterations:
        logger.warning("Checkpoint already holds %d of %d iterations; nothing to do.",
                       start_iteration, num_iterations)
        # Even the do-nothing path owes the caller a self-describing chain: the
        # checkpoint on disk carries the clipped, pre-relabel array of the run
        # that wrote it.
        return _with_unclipped_prob_in_model(posterior_chain, max_num_sources)

    logger.info("Sorting the posterior chain: maximum number of iterations %d, "
                "maximum number of source labels %d.", num_iterations, max_num_sources)

    # set up the for loop
    old_posterior_chain = posterior_chain
    old_parametric_fit, old_prob_in_model = update_parametric_fit_and_prob_in_model(
        posterior_chain, max_num_sources, param_fit, eps=eps)
    # `None` rather than 0.0: on the first iteration there is no previous cost to
    # compare against, and 0.0 is a perfectly reachable cost, not a sentinel.
    old_cost_of_assignment: float | None = None
    best_cost_of_assignment: float | None = None
    best_posterior_chain = _with_unclipped_prob_in_model(posterior_chain, max_num_sources)
    if resume_from is not None:
        # A resumed run must never come back worse than the checkpoint it started
        # from -- that is the one thing resume has to guarantee -- so the cost the
        # checkpoint carries is the cost to beat.  Without this the first
        # post-resume iteration became "best" unconditionally, however expensive.
        # Only on the resume path: an input chain handed to a fresh run may carry a
        # `cost_dict` entry under this same key from an initialization step, and
        # that number came from a different fit, so it is not comparable with
        # anything this loop computes.
        best_cost_of_assignment = posterior_chain.cost_dict.get(max_num_sources)
        logger.info("The resumed labeling cost %s; that is what this run has to beat.",
                    best_cost_of_assignment)

    for iteration in range(start_iteration, num_iterations):
        # get the new values
        new_posterior_chain = relabel_posterior_chain_one_iteration(old_posterior_chain, old_parametric_fit, old_prob_in_model, max_num_sources, compute_cost_matrix, progress=progress)
        # Overwrite the clipped, pre-relabel array that step stored with one derived
        # from the labeling it just produced, before that chain can be checkpointed,
        # kept as the best, or returned.
        new_posterior_chain = _with_unclipped_prob_in_model(new_posterior_chain, max_num_sources)
        new_parametric_fit, new_prob_in_model = update_parametric_fit_and_prob_in_model(
            new_posterior_chain, max_num_sources, param_fit, eps=eps)
        # `relabel_posterior_chain_one_iteration` always records this iteration's cost,
        # so this key exists.
        new_cost_of_assignment = new_posterior_chain.cost_dict[max_num_sources]

        # log the results
        if old_cost_of_assignment is None:
            delta_cost_of_assignment = new_cost_of_assignment
        else:
            delta_cost_of_assignment = new_cost_of_assignment - old_cost_of_assignment
        logger.info("Iteration %d: Difference in cost of assignment is %s with total cost of %s.",
                    iteration + 1, delta_cost_of_assignment, new_cost_of_assignment)
        logger.info("\tProbabilities in model: %s", new_prob_in_model)

        # Remember the cheapest labeling seen so far.  The comparison is strict, so a
        # tie keeps the earlier chain and convergence returns exactly what the previous
        # iteration produced.
        if best_cost_of_assignment is None or new_cost_of_assignment < best_cost_of_assignment:
            best_cost_of_assignment = new_cost_of_assignment
            best_posterior_chain = new_posterior_chain

        # A checkpoint is a resumable snapshot of what this run would return at
        # this point.  Persisting the latest candidate instead loses the best
        # labeling whenever an iteration raises the cost and then stops the loop.
        if checkpoint_dir is not None:
            _checkpoint_posterior_chain(best_posterior_chain, checkpoint_dir, iteration + 1)

        # break if converged
        if old_cost_of_assignment is not None and delta_cost_of_assignment == 0:
            logger.info("Stopped after %d iterations because the cost didn't change from the previous iteration.",
                        iteration + 1)
            break

        # break if the cost went back up: the labeling is no longer improving, and the
        # best one has already been kept.
        if old_cost_of_assignment is not None and new_cost_of_assignment > old_cost_of_assignment:
            logger.info("Stopped after %d iterations because the cost increased from %s to %s; "
                        "returning the best labeling seen, of cost %s.",
                        iteration + 1, old_cost_of_assignment, new_cost_of_assignment,
                        best_cost_of_assignment)
            break

        # update the old values to the new values
        old_posterior_chain = new_posterior_chain
        old_parametric_fit = new_parametric_fit
        old_prob_in_model = new_prob_in_model
        old_cost_of_assignment = new_cost_of_assignment

    else:
        logger.info("Final cost of assignment: %s after the maximum number (%d) of iterations.",
                    best_cost_of_assignment, num_iterations)

    return best_posterior_chain


def create_relabel_samples(parametric_fit_function: Callable,
                           aux_distribution: Callable,
                           single_parameter: int | None = None,
                           eps: float = 1e-2) -> Callable:
    """
    Build a relabeling procedure combining fitting, cost computation, and assignment.

    Parameters
    ----------
    parametric_fit_function : callable
        Fit function `(chain, max_num_sources) -> aux_parameters`.
    aux_distribution : callable
        PDF function for cost matrix `(sample, aux_parameters, prob_in_model, idx)`.
    single_parameter : int, optional
        Fixes the parameter index for single-parameter fit/distribution.
    eps : float, optional
        Clip bound on the inclusion probabilities used *inside* the cost matrix,
        keeping them inside ``[eps, 1 - eps]`` so the logarithms there stay
        finite.  The ``prob_in_model`` the relabeler returns is unclipped; see
        Notes.

    Returns
    -------
    relabel_samples : callable
        Function with signature
        `(posterior_chain, max_num_sources, num_iterations, checkpoint_dir,
        resume_from, progress) -> PosteriorChain`.

    Notes
    -----
    The returned relabeler is :func:`run_relabeling_loop` with `param_fit` and
    `compute_cost_matrix` already bound.  It keeps the cheapest labeling it has
    seen and returns that one, never the merely most recent one, and its
    ``prob_in_model`` is recomputed from the labeling it returns with no
    clipping -- see the Notes of :func:`run_relabeling_loop`.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import mv_normal_aux_distribution
    >>> from petra.parametric_fits import mv_normal_fit
    >>> from petra.posterior_chain import PosteriorChain
    >>> relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    >>> rng = np.random.default_rng(0)
    >>> pc = PosteriorChain(rng.normal(size=(20, 2, 1)) + np.array([[0.0], [5.0]]), 2, 1)
    >>> new_pc = relabel(pc, max_num_sources=2, num_iterations=3, progress=False)
    >>> isinstance(new_pc, PosteriorChain)
    True
    """

    # make all the necessary pieces
    param_fit = create_parametric_fit(parametric_fit_function, single_parameter=single_parameter)
    compute_cost_matrix = create_compute_cost_matrix(aux_distribution, single_parameter=single_parameter)

    def relabel_samples(posterior_chain: PosteriorChain,
                        max_num_sources: int | None = None,
                        num_iterations: int = 200,
                        checkpoint_dir: Optional[str] = None,
                        resume_from: Optional[str] = None,
                        progress: bool = True) -> PosteriorChain:
        """
        Iteratively relabel a PosteriorChain with a chosen aux distribution.

        Parameters
        ----------
        posterior_chain : PosteriorChain
            Chain to process; widened to `max_num_sources` if necessary.
        max_num_sources : int, optional
            Target number of sources (defaults to chain.num_sources).
        num_iterations : int, default 200
            Maximum relabeling iterations before stopping.  Must be at least 1.
        checkpoint_dir : str, optional
            Directory to save the best state available after each iteration.
            If None, no checkpointing.
        resume_from : str, optional
            Checkpoint file, or directory of checkpoints, written by an earlier
            run with the same `checkpoint_dir`.  The chain stored there
            replaces `posterior_chain`, the iteration counter picks up where
            that run stopped, and the cost stored with the chain becomes the
            cost to beat, so a job killed at the walltime limit can be restarted
            without losing its progress and without coming back worse than the
            checkpoint.  See :func:`load_checkpoint` for what survives the
            round-trip.
        progress : bool, default True
            Show a tqdm progress bar over the samples of each iteration.

        Returns
        -------
        PosteriorChain
            The cheapest labeling seen during the run, with its cost recorded
            in ``cost_dict[max_num_sources]`` and with ``prob_in_model``
            recomputed from the returned samples *without* clipping, so that
            ``find_prob_in_model(result.get_chain(), max_num_sources, eps=0)``
            reproduces it exactly.

        Raises
        ------
        ValueError
            If `num_iterations` is less than 1, or if `max_num_sources` is
            smaller than the number of entries in the chain.

        Notes
        -----
        Cost is negated log-likelihood, so lower is better.  The loop keeps the
        cheapest labeling it has produced and stops as soon as an iteration
        comes back more expensive than the one before it, returning that best
        labeling rather than the last one.  Iterating a relabeler is not a
        descent method -- refitting the auxiliary distributions to a new
        labeling can raise the cost -- so without this the returned catalog
        could be worse than one already computed on the way to it.

        The `eps` given to :func:`create_relabel_samples` still governs the
        clipping used *inside* the cost matrix during iteration, where a
        probability of exactly 0 or 1 would make a ``log`` term infinite.  Only
        the returned array is unclipped.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.aux_distributions import mv_normal_aux_distribution
        >>> from petra.parametric_fits import mv_normal_fit
        >>> from petra.posterior_chain import PosteriorChain
        >>> relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
        >>> rng = np.random.default_rng(0)
        >>> pc = PosteriorChain(rng.normal(size=(20, 2, 1)) + np.array([[0.0], [5.0]]), 2, 1)
        >>> result_pc = relabel(pc, max_num_sources=2, num_iterations=3, progress=False)
        >>> sorted(result_pc.cost_dict)
        [2]

        Both sources are in every sample, and the returned probabilities say so
        exactly rather than reporting the clip bound ``1 - eps``:

        >>> result_pc.prob_in_model
        array([1., 1.])
        """
        return run_relabeling_loop(posterior_chain,
                                   param_fit,
                                   compute_cost_matrix,
                                   max_num_sources=max_num_sources,
                                   num_iterations=num_iterations,
                                   eps=eps,
                                   checkpoint_dir=checkpoint_dir,
                                   resume_from=resume_from,
                                   progress=progress)

    return relabel_samples
