"""
Cheap first labelings, used to give the iterative relabelers a warm start.

The iterative relabelers in :mod:`petra.relabel` converge much faster --- and to
a better labeling --- when they do not start from an arbitrary ordering.  This
module holds the two initializations petra offers:

:func:`label_chain_by_histogram`
    Greedily peels sources off a histogram of the frequency parameter.  Fast,
    needs no fitting, and is *not* a posterior-preserving relabeling: it exists
    only to get similar samples into the same slot.
:func:`relabel_univariate_normal`
    The full iterative relabeler restricted to a single parameter, which is
    what all three ``make_catalog_*`` entry points can run before their own
    method.
:func:`run_initialization_passes`
    The shared preamble of the Bayesian and copula entry points: the optional univariate
    pass followed by the optional multivariate-normal one.  It lives here so
    that the two gates are decided in one place; see its Notes for which
    keyword controls which pass.

Conventions
-----------
Chains are ``(n_samples, n_sources, n_params_per_source)`` and ``NaN`` means
"this source is absent from this sample"; see :mod:`petra.posterior_chain`.

`relabeling_parameter` and `init_parameter_index` are both indices into the
parameter axis, and both default to ``0`` on the assumption that the *frequency*
of a source is its first parameter --- the histogram labeling is only meaningful
for a parameter whose posterior is narrow compared with the spacing between
sources.  Frequency bin widths come from
:func:`petra.utils.frequency_bin_width`, so `obs_time_yrs` is in years and
frequencies are in Hz.

Nothing here modifies its input chain, diagnostics go to
``logging.getLogger("petra.initialization")``, and progress bars are controlled
by the shared ``progress`` keyword.
"""

from pathlib import Path
from typing import Callable, Optional

from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import uni_normal_fit_single_parameter
from petra.aux_distributions import uni_normal_aux_distribution_single_parameter
from petra.relabel import create_relabel_samples
from petra.utils import frequency_bin_width, get_logger

import numpy as np
from copy import deepcopy

logger = get_logger(__name__)


def get_max_count_bin_center(chain: np.ndarray, delta_freq: float, relabeling_parameter: int) -> float:
    """
    Find the centre of the most populated frequency bin of a chain.

    Every non-NaN value of parameter `relabeling_parameter` is snapped to the
    grid of multiples of `delta_freq` and the fullest grid point is returned.
    This is the "peel the tallest peak next" step of
    :func:`label_chain_by_histogram`.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Chain to histogram.  Not modified.
    delta_freq : float
        Grid spacing, in the units of the parameter -- the width of one Fourier
        frequency bin, from :func:`petra.utils.frequency_bin_width`.
    relabeling_parameter : int
        Index of the parameter to histogram, along the parameter axis.

    Returns
    -------
    float
        Centre of the fullest bin, i.e. an exact multiple of `delta_freq`.

    Raises
    ------
    ValueError
        If every value of the parameter is NaN, since there is then no bin to
        return.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.initialization import get_max_count_bin_center
    >>> chain = np.array([[[1.0], [5.0]], [[1.1], [np.nan]], [[0.9], [9.0]]])
    >>> get_max_count_bin_center(chain, delta_freq=1.0, relabeling_parameter=0)
    1.0
    """
    x = chain[:, :, relabeling_parameter].reshape(-1)
    x = x[~np.isnan(x)]

    # snap to grid
    k = np.rint(x / delta_freq).astype(np.int64)  # indices on grid
    k_mode = np.bincount(k).argmax()

    return float(k_mode * delta_freq)


def tetris_rise_nd(a: np.ndarray, *, axis: int = 1, empty: float = np.nan) -> np.ndarray:
    """
    Apply 'anti-gravity' along `axis` for an N-D array:
    filled values rise to the start of `axis`, independently
    for every combination of the other axes.

    Parameters
    ----------
    a : ndarray
        Array to compact.
    axis : int, default 1
        Axis along which filled values rise.
    empty : float, default nan
        Value marking an empty slot.

    Returns
    -------
    ndarray
        Copy of `a` with the filled values moved to the start of `axis`.

    Examples
    --------
    >>> tetris_rise_nd(np.array([[np.nan, 1.0, np.nan, 2.0]]))
    array([[ 1.,  2., nan, nan]])
    """
    a = np.asarray(a)

    # Build filled mask
    if isinstance(empty, float) and np.isnan(empty):
        filled = ~np.isnan(a)
    else:
        filled = (a != empty)

    # Stable sort: filled first, empties last
    order = np.argsort(~filled, axis=axis, kind="stable")

    return np.take_along_axis(a, order, axis=axis)


def count_nonnan_samples(chain: np.ndarray, source_index: int, relabeling_parameter: int) -> int:
    """
    Count the samples in which one source slot is present.

    Presence is read off a single parameter: ``NaN`` means the source is absent
    from that sample.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Chain to count in.  Not modified.
    source_index : int
        Index of the source slot, along the source axis.
    relabeling_parameter : int
        Index of the parameter used to test presence.

    Returns
    -------
    int
        Number of samples in which the slot is not NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.initialization import count_nonnan_samples
    >>> chain = np.array([[[1.0], [np.nan]], [[2.0], [3.0]]])
    >>> count_nonnan_samples(chain, source_index=1, relabeling_parameter=0)
    1
    """
    chain = chain[:, source_index, relabeling_parameter]
    return int(np.count_nonzero(~np.isnan(chain)))


def sort_array_by_sample_count(sources_array: np.ndarray, relabeling_parameter: int) -> np.ndarray:
    """
    Reorder source slots so the best populated ones come first.

    Parameters
    ----------
    sources_array : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Chain whose source slots are to be reordered.  Not modified.
    relabeling_parameter : int
        Index of the parameter used to count how populated a slot is, as in
        :func:`count_nonnan_samples`.

    Returns
    -------
    ndarray
        Copy of `sources_array` whose slots are ordered by decreasing number of
        non-NaN samples.  Equally populated slots come back in reverse index
        order, since the ascending stable sort is reversed.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.initialization import sort_array_by_sample_count
    >>> chain = np.array([[[np.nan], [1.0]], [[np.nan], [2.0]]])
    >>> sort_array_by_sample_count(chain, relabeling_parameter=0)[:, 0, 0]
    array([1., 2.])
    """
    source_sample_counts = []
    for source_index in range(sources_array.shape[1]):
        source_sample_counts.append(count_nonnan_samples(sources_array, source_index, relabeling_parameter))
    sorted_source_indices = np.argsort(source_sample_counts, kind="stable")[::-1]  # descending order

    sources_array_sorted = np.zeros_like(sources_array)
    for new_index, old_index in enumerate(sorted_source_indices):
        sources_array_sorted[:, new_index, :] = sources_array[:, old_index, :]
    return sources_array_sorted


def label_chain_by_histogram(chain: PosteriorChain, obs_time_yrs: float = 1, num_surrounding_bins: int = 1, relabeling_parameter: int = 0, num_extra_entries: int = 1000, low_num_samples: int = 100) -> PosteriorChain:
    """
    Assign labels by greedily peeling sources off a frequency histogram.

    Each pass snaps every remaining (non-NaN) entry onto a grid of Fourier
    frequency bins, finds the most populated bin, and claims for one new source
    label the entry of each sample that lies closest to that bin -- provided it
    lies in the center bin or one of `num_surrounding_bins` bins on either side.
    Claimed entries are removed from the working copy of the chain and the pass
    repeats until nothing is left.  The resulting labels are then sorted by how
    many samples they contain, sparsely populated labels are compacted towards
    the front (:func:`tetris_rise_nd`), and labels that ended up empty are
    dropped.

    This is a cheap initialization for the iterative relabelers: it is not
    itself a posterior-preserving relabeling, it just gets similar samples into
    the same slot before :func:`relabel_univariate_normal` or the flow-based
    relabelers refine the assignment.

    Parameters
    ----------
    chain : PosteriorChain
        Chain to label.  Copied, never modified in place.
    obs_time_yrs : float, default 1
        Observation time in years, which sets the frequency bin width
        (:func:`petra.utils.frequency_bin_width`).
    num_surrounding_bins : int, default 1
        How many bins on either side of the most populated bin still count as
        the same source.
    relabeling_parameter : int, default 0
        Index of the frequency parameter within each source.
    num_extra_entries : int, default 1000
        Head-room for the intermediate array: the labeling may need many more
        slots than the chain has sources before empty ones are dropped.
    low_num_samples : int, default 100
        Labels holding fewer than this many samples are treated as sparse and
        compacted together.

    Returns
    -------
    PosteriorChain
        New chain whose ``num_sources`` is the number of non-empty labels
        found, sorted by decreasing number of samples.

    Raises
    ------
    ValueError
        If more than ``chain.num_sources + num_extra_entries`` labels are
        needed; raise `num_extra_entries` and try again.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.posterior_chain import PosteriorChain
    >>> from petra.initialization import label_chain_by_histogram
    >>> # two well-separated frequencies, in the wrong slot half the time
    >>> arr = np.tile(np.array([[1e-3], [2e-3]]), (10, 1, 1))
    >>> arr[::2] = arr[::2, ::-1]
    >>> pc = PosteriorChain(arr, 2, 1, trans_dimensional=True)
    >>> labeled = label_chain_by_histogram(pc, low_num_samples=1, num_extra_entries=2)
    >>> labeled.num_sources
    2
    >>> np.unique(labeled.chain[:, 0, 0])
    array([0.001])
    """
    delta_freq = frequency_bin_width(obs_time_yrs)

    chain = deepcopy(chain)
    sources_array = np.zeros((chain.shape[0], chain.shape[1] + num_extra_entries, chain.shape[2]))
    sources_array[:] = np.nan

    low_sample_count_source_indices = []

    source_index = 0
    while np.any(~np.isnan(chain.chain)):
        if source_index >= sources_array.shape[1]:
            raise ValueError(
                f"Ran out of source labels after {source_index}; increase "
                f"num_extra_entries (currently {num_extra_entries})."
            )
        # create histogram of all samples in relabeling_parameter
        max_count_bin_center = get_max_count_bin_center(chain.chain, delta_freq, relabeling_parameter)

        # take samples within the surrounding bins of max_count_bin_center
        # source: for each sample, compute the distance to max_count_bin_center
        # take the nearest value (minimum) in each sample and if that value is within the surrounding bins, keep it, else continue!
        for i in range(chain.shape[0]):
            distances = np.abs(chain[i, :, relabeling_parameter] - max_count_bin_center)
            # check if the distance array is all nans:
            if np.all(np.isnan(distances)):
                sources_array[i, source_index, :] = np.nan
                continue
            elif np.nanmin(distances) <= (num_surrounding_bins + 0.5) * delta_freq:
                min_index = np.nanargmin(distances)
            else:
                sources_array[i, source_index, :] = np.nan
                continue
            sources_array[i, source_index, :] = chain[i, min_index, :]
            # turn used sample into nan to avoid reusing it
            chain[i, min_index, :] = np.nan

        num_valid_samples = count_nonnan_samples(sources_array, source_index, relabeling_parameter)
        if (num_valid_samples == 0) or (num_valid_samples < low_num_samples):
            low_sample_count_source_indices.append(source_index)

        source_index += 1

    # sort the source indices by sample count
    sources_array = sort_array_by_sample_count(sources_array, relabeling_parameter)

    # get the low sample count indices again
    low_sample_count_source_indices = []
    for source_index in range(sources_array.shape[1]):
        num_valid_samples = count_nonnan_samples(sources_array, source_index, relabeling_parameter)
        if num_valid_samples < low_num_samples:
            low_sample_count_source_indices.append(source_index)

    # apply tetris rise to low sample count sources
    low_sample_count_sources = sources_array[:, low_sample_count_source_indices, :]
    sources_array[:, low_sample_count_source_indices, :] = tetris_rise_nd(low_sample_count_sources)

    # remove empty sources
    empty_source_indices = []
    for source_index in range(sources_array.shape[1]):
        num_valid_samples = count_nonnan_samples(sources_array, source_index, relabeling_parameter)
        if num_valid_samples == 0:
            empty_source_indices.append(source_index)

    non_empty_source_indices = [i for i in range(sources_array.shape[1]) if i not in empty_source_indices]
    non_empty_sources = sources_array[:, non_empty_source_indices, :]

    # reorder the arrays based on valid sample counts
    non_empty_sources = sort_array_by_sample_count(non_empty_sources, relabeling_parameter)

    logger.info("Histogram labeling found %d non-empty source labels.",
                non_empty_sources.shape[1])

    return PosteriorChain(chain=non_empty_sources, num_sources=non_empty_sources.shape[1], num_params_per_source=non_empty_sources.shape[2], trans_dimensional=chain.trans_dimensional)


def relabel_univariate_normal(posterior_chain: PosteriorChain,
                              max_num_sources: int | None = None,
                              num_iterations: int = 20,
                              init_parameter_index: int = 0,
                              eps: float = 1e-6,
                              checkpoint_dir: Optional[str] = None,
                              resume_from: Optional[str] = None,
                              progress: bool = True) -> PosteriorChain:
    """
    Iteratively relabels a posterior chain using univariate normal fits.

    At each iteration, fits a normal distribution to a single parameter of each source,
    computes a cost matrix based on the fit, and applies the Hungarian algorithm
    to align labels across samples until convergence.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        The chain of posterior samples to relabel.
    max_num_sources : int, optional
        Maximum number of sources to include; if None, uses the chain's num_sources.
    num_iterations : int, default 20
        Maximum number of relabeling iterations to perform.
    init_parameter_index : int, default 0
        Index of the parameter used to initialize the relabeling.
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities, keeping them inside
        ``[eps, 1 - eps]`` so the logarithms in the cost matrix stay finite.
    checkpoint_dir : str, optional
        Directory to save the best chain available after every iteration.
    resume_from : str, optional
        Checkpoint file, or directory of checkpoints, to resume an interrupted
        run from.  See :func:`petra.relabel.load_checkpoint`.
    progress : bool, default True
        Show a tqdm progress bar over the samples of each iteration.

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain instance with relabeled samples.  The cost of the
        final assignment is recorded in ``relabeled_chain.cost_dict``.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.initialization import relabel_univariate_normal
    >>> from petra.posterior_chain import PosteriorChain
    >>> rng = np.random.default_rng(0)
    >>> pc = PosteriorChain(rng.normal(size=(20, 2, 1)) + np.array([[0.0], [5.0]]), 2, 1)
    >>> relabeled_pc = relabel_univariate_normal(
    ...     pc, max_num_sources=2, num_iterations=5, init_parameter_index=0,
    ...     progress=False)
    >>> isinstance(relabeled_pc, PosteriorChain)
    True
    >>> sorted(relabeled_pc.cost_dict)
    [2]
    """

    # create single parameter function to relabel samples
    relabel_samples = create_relabel_samples(uni_normal_fit_single_parameter,
                                             uni_normal_aux_distribution_single_parameter,
                                             single_parameter=init_parameter_index,
                                             eps=eps)

    return relabel_samples(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress=progress,
    )


def _validate_initialization_param_index(num_params: int,
                                         initialization_param_index: int | None) -> None:
    """
    Check that `initialization_param_index` addresses a column of the chain.

    Parameters
    ----------
    num_params : int
        Number of parameters per source in the chain.
    initialization_param_index : int or None
        Index of the parameter the univariate initialization pass sorts on, or
        ``None`` to skip that pass.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `initialization_param_index` is neither ``None`` nor an integer in
        ``[0, num_params)``.

    Notes
    -----
    The message names ``initialization_param_index``, which is the keyword the
    caller typed on the entry point, rather than any parameter spelling used
    between here and there.

    Examples
    --------
    >>> _validate_initialization_param_index(3, 2)
    >>> _validate_initialization_param_index(3, None)
    >>> _validate_initialization_param_index(3, 3)
    Traceback (most recent call last):
        ...
    ValueError: initialization_param_index must be None or in [0, 3), got 3; the chain has 3 parameters per source.
    """
    if initialization_param_index is None:
        return
    # A negative index would not raise: numpy would quietly wrap it round to a
    # column at the other end of the source, so the univariate pass would sort
    # on a parameter the caller never named and hand back a plausible-looking
    # wrong labeling.  An index past the end raises, but only from inside the
    # fit, several frames from the keyword that is wrong.
    if (not isinstance(initialization_param_index, (int, np.integer))
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
                              param_index: Optional[int],
                              num_iterations: int,
                              resume_from: str | Path | None = None,
                              progress: bool = True) -> PosteriorChain:
    """
    Run the cheap pre-relabeling the ``make_catalog_*`` entry points share.

    Up to two passes, each independently switchable: a univariate relabeling of
    the parameter `param_index`, then a full multivariate-normal one.  Both are
    warm starts -- they cost no training -- and the chain that comes back is
    what the caller's own method then refines.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain to initialize, already widened by
        :func:`petra.relabel.prepare_chain`.  Not modified: each pass returns a
        new chain.
    max_num_sources : int
        Number of source slots, passed on to both passes.
    univariate_relabeler : callable
        :func:`relabel_univariate_normal`, or a stand-in with its signature.
        Called as ``f(chain, max_num_sources=, num_iterations=,
        init_parameter_index=, progress=)``.
    mv_normal_relabeler : callable
        :func:`petra.make_catalog.relabel_mv_normal`, or a stand-in with its
        signature.  Called as ``f(chain, max_num_sources=, num_iterations=,
        progress=)``.
    with_mv_normal : bool
        Run the multivariate-normal pass.  This is
        ``initialization.with_mv_normal``, reached flat as
        ``init_with_mv_normal``.
    param_index : int or None
        Parameter the univariate pass sorts on, or ``None`` to skip that pass.
        This is the entry points' `initialization_param_index`, and it is the
        *only* thing that decides whether the univariate pass runs; see Notes.
        Checked against the chain here, which is the one boundary the two
        entry points that share this preamble all pass through.
    num_iterations : int
        Ceiling on the iterations of *each* pass, so the budget is spent twice
        when both run.  This is ``initialization.num_iterations``, reached flat
        as ``init_num_iterations``.  No default here on purpose: the default
        belongs to :class:`petra.options.Initialization`, and a second copy of
        it is how the defaults drifted apart before.
    resume_from : str or Path, optional
        Checkpoint the caller is resuming from.  Anything but ``None`` skips
        both passes, because the checkpoint already reflects them.
    progress : bool, default True
        Show a tqdm progress bar over the samples of each pass.

    Returns
    -------
    PosteriorChain
        The chain after whichever passes ran; `posterior_chain` itself if none
        did.

    Raises
    ------
    ValueError
        If `param_index` is neither ``None`` nor a column of `posterior_chain`.
        Raised before the `resume_from` check, so a nonsense index is reported
        on a resumed run too rather than only on the run that would have used
        it.

    Notes
    -----
    The two knobs are independent, and that is the decision this function
    exists to make once.  `param_index` names a column of the caller's chain --
    which parameter separates the sources -- and selects the univariate pass;
    `with_mv_normal` switches the multivariate pass.  Neither overrides the
    other, so ``init_with_mv_normal=False`` with the default
    ``initialization_param_index=0`` still runs the univariate pass.

    The copula-flow entry point used to gate *both* passes on
    ``init_with_mv_normal``, so that keyword silently disabled a pass a
    different keyword had asked for.  That coupling was not load-bearing: the
    preamble started life in
    :func:`petra.make_catalog.make_catalog_mv_normal`, where the univariate
    pass has always been gated on `initialization_param_index` alone and there
    is no ``init_with_mv_normal`` at all, and it acquired the extra gate only
    when it was copied into the flow entry points.  Nor does the univariate
    pass need the multivariate one behind it: it runs the same
    :func:`petra.relabel.run_relabeling_loop` as every other relabeler and
    returns a complete labeling, which is exactly what ``make_catalog_mv_normal``
    has always handed straight to its own method.

    Every taken and every skipped pass is logged, because a silently skipped
    pass is what made the divergence survive across entry points.

    The two relabelers are handed in rather than imported here.
    :mod:`petra.make_catalog` imports this module, so a module-scope import of
    :func:`~petra.make_catalog.relabel_mv_normal` would close a cycle; and
    taking both as arguments keeps each entry point's own namespace the place
    either pass can be substituted, which is what lets a caller -- or a test --
    watch one method's initialization without patching every other method's.

    Examples
    --------
    Stand-ins for the two relabelers, so the example shows the gating rather
    than the labeling:

    >>> import numpy as np
    >>> from petra.initialization import run_initialization_passes
    >>> from petra.posterior_chain import PosteriorChain
    >>> pc = PosteriorChain(np.zeros((4, 2, 1)), 2, 1)
    >>> def univariate(chain, **kwargs):
    ...     print('univariate pass on parameter', kwargs['init_parameter_index'])
    ...     return chain
    >>> def mv_normal(chain, **kwargs):
    ...     print('mv-normal pass')
    ...     return chain
    >>> def initialize(**settings):
    ...     return run_initialization_passes(
    ...         pc, 2, univariate_relabeler=univariate,
    ...         mv_normal_relabeler=mv_normal, num_iterations=5,
    ...         progress=False, **settings)

    Both knobs on, both passes run:

    >>> _ = initialize(with_mv_normal=True, param_index=0)
    univariate pass on parameter 0
    mv-normal pass

    Turning off the multivariate pass leaves the univariate one alone:

    >>> _ = initialize(with_mv_normal=False, param_index=0)
    univariate pass on parameter 0

    ...and turning off the univariate pass leaves the multivariate one alone:

    >>> _ = initialize(with_mv_normal=True, param_index=None)
    mv-normal pass

    With both off, and when resuming, the chain comes back untouched:

    >>> initialize(with_mv_normal=False, param_index=None) is pc
    True
    >>> initialize(with_mv_normal=True, param_index=0, resume_from='run/iter_3.feather') is pc
    True

    An index that is not a column of the chain is refused here rather than
    inside the fit:

    >>> initialize(with_mv_normal=True, param_index=-1)
    Traceback (most recent call last):
        ...
    ValueError: initialization_param_index must be None or in [0, 1), got -1; the chain has 1 parameters per source.
    """
    _validate_initialization_param_index(posterior_chain.shape[2], param_index)

    if resume_from is not None:
        logger.info("Resuming from %s; skipping both initialization passes.", resume_from)
        return posterior_chain

    # The univariate pass is gated on `param_index` alone and the multivariate
    # pass on `with_mv_normal` alone: one keyword, one pass.  Gating the
    # univariate pass on `with_mv_normal` too -- which the copula entry
    # point did -- means `init_with_mv_normal=False` silently overrides
    # `initialization_param_index`, so the same pair of keywords initialized
    # differently depending on which method was called.
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
        )
    else:
        logger.info("init_with_mv_normal is False; skipping the multivariate-normal initialization.")

    return posterior_chain
