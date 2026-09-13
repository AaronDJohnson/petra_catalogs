"""
The multivariate-normal catalog method, and the shape every entry point copies.

This is the simplest of petra's three catalog methods and the one to reach for
first: it fits a full multivariate normal to each source label and needs no JAX,
no flows and no training.  :func:`relabel_mv_normal` is the relabeler itself;
:func:`make_catalog_mv_normal` is the user-facing entry point that wraps it in
the standard preamble --- optional shuffle, widen to `max_num_sources`, optional
univariate initialization --- and is exported as ``petra.make_catalog_mv_normal``.

Entry point conventions
-----------------------
Every ``make_catalog_*`` function in the package shares this contract, so the
methods can be swapped without rewriting the call:

- the signature is ``(posterior_chain, max_num_sources, *, ...)``; everything
  after `max_num_sources` is keyword-only;
- the shared named prefix is ``num_iterations``,
  ``initialization_param_index``, ``shuffle_seed``, ``rng_seed``,
  ``checkpoint_dir``, ``resume_from``, ``progress`` and ``eps``, and
  ``init_num_iterations`` is spelled the same way everywhere -- a named
  parameter here, a field of :class:`petra.options.Initialization` on the other
  two.  This entry point omits ``init_with_mv_normal`` entirely, because the
  multivariate normal *is* its method, and accepts ``rng_seed`` only for
  symmetry;
- the other two entry points group their flow and initialization
  hyperparameters into the frozen dataclasses of :mod:`petra.options`.  This one
  takes none of them, and that is deliberate rather than unfinished: it trains
  no flow and runs no MCMC, and it cannot accept an
  :class:`~petra.options.Initialization`, because that object carries
  ``with_mv_normal`` -- taking it would advertise the one alias this entry point
  has to reject.  Twelve arguments is the whole signature; there is nothing left
  to group;
- legacy keyword spellings are accepted through a ``**deprecated`` catch-all and
  translated by :func:`petra.utils.resolve_deprecated_kwargs`, which raises
  :class:`TypeError` on anything that is neither a real parameter nor a known
  alias;
- the input chain is never modified, and a new
  :class:`~petra.posterior_chain.PosteriorChain` is returned.

Conventions
-----------
Chains are ``(n_samples, n_sources, n_params_per_source)`` and ``NaN`` means
"this source is absent from this sample"; see :mod:`petra.posterior_chain`.
``cost_dict`` holds negated log-likelihood, so lower is better; see
:mod:`petra.relabel`.  Diagnostics go to ``logging.getLogger("petra.make_catalog")``
and progress bars are controlled by the shared ``progress`` keyword.
"""

from typing import Any, Optional

from petra.posterior_chain import PosteriorChain
from petra.relabel import _validate_relabel_settings, create_relabel_samples, prepare_chain
from petra.aux_distributions import mv_normal_aux_distribution
from petra.parametric_fits import mv_normal_fit
from petra.initialization import _validate_initialization_param_index, relabel_univariate_normal
from petra.utils import get_logger, resolve_deprecated_kwargs

logger = get_logger(__name__)


def relabel_mv_normal(posterior_chain: PosteriorChain,
                      max_num_sources: int | None = None,
                      num_iterations: int = 20,
                      eps: float = 1e-6,
                      checkpoint_dir: Optional[str] = None,
                      resume_from: Optional[str] = None,
                      progress: bool = True) -> PosteriorChain:
    """
    Iteratively relabels a posterior chain using multivariate normal fits.

    At each iteration, fits a multivariate normal distribution to each source,
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
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities used *inside* the cost matrix,
        keeping them inside ``[eps, 1 - eps]`` so the logarithms there stay
        finite.  It does not reach the returned ``prob_in_model``; see Notes.
    checkpoint_dir : str, optional
        Directory to save a checkpoint of the chain after every iteration.
    resume_from : str, optional
        Checkpoint file, or directory of checkpoints, to resume an interrupted
        run from.  See :func:`petra.relabel.load_checkpoint`.
    progress : bool, default True
        Show a tqdm progress bar over the samples of each iteration.

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain instance with relabeled samples.  Its
        ``prob_in_model`` is recomputed from the returned samples *without*
        clipping, so that ``find_prob_in_model(relabeled_chain.get_chain(),
        max_num_sources, eps=0)`` reproduces it exactly, and its ``cost_dict``
        records the cost of the assignment that produced them.

    Notes
    -----
    `eps` governs only the clipping applied inside the cost matrix during
    iteration, where a probability of exactly 0 or 1 would make a ``log`` term
    infinite.  Only the returned array is unclipped; see
    :func:`petra.relabel.run_relabeling_loop`.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.make_catalog import relabel_mv_normal
    >>> from petra.posterior_chain import PosteriorChain
    >>> from petra.utils import find_prob_in_model
    >>> rng = np.random.default_rng(0)
    >>> chain_array = rng.normal(size=(20, 2, 1)) + np.array([[0.0], [5.0]])
    >>> pc = PosteriorChain(chain_array, 2, 1, True, None, {})
    >>> relabeled_pc = relabel_mv_normal(pc, max_num_sources=2, progress=False)
    >>> isinstance(relabeled_pc, PosteriorChain)
    True
    >>> relabeled_pc.prob_in_model
    array([1., 1.])
    >>> bool(np.array_equal(relabeled_pc.prob_in_model,
    ...                     find_prob_in_model(relabeled_pc.get_chain(), 2, eps=0)))
    True
    """

    relabel_samples = create_relabel_samples(mv_normal_fit,
                                             mv_normal_aux_distribution,
                                             eps=eps)

    # No prob_in_model fix-up here any more: the shared loop recomputes it, unclipped,
    # from the labeling it returns.  The fix-up that used to live here reached only
    # this one relabeler, which is how the other two came to hand back a clipped
    # array describing the labeling of the iteration *before* the one returned.
    return relabel_samples(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress=progress,
    )


# Two things here diverge from the other two entry points and both are
# load-bearing, so neither is the leftover it looks like.  `init_num_iterations`
# is a named parameter instead of a field of an `Initialization`, and the
# catch-all is `**deprecated` instead of `**flat_keywords`.  Taking an
# `Initialization` would put its fields into `option_field_keywords`, and
# `resolve_deprecated_kwargs` would then accept `mv_normal_init` -- the one alias
# this entry point has to reject, because the multivariate normal *is* its
# method and there is nothing left for it to initialize with.  With no options
# object there is also nothing to flatten, so only a deprecated alias can ever
# reach the catch-all and its name says exactly that.  Both are pinned:
# tests/test_invariants.py fixes the parameter list, catch-all name included, and
# tests/test_deprecated_keywords.py fixes the reject/accept pair.
def make_catalog_mv_normal(posterior_chain: PosteriorChain,
                           max_num_sources: int,
                           *,
                           num_iterations: int = 200,
                           init_num_iterations: int = 200,
                           initialization_param_index: int | None = 0,
                           shuffle_seed: int | None = None,
                           rng_seed: int = 999,
                           checkpoint_dir: Optional[str] = None,
                           resume_from: Optional[str] = None,
                           progress: bool = True,
                           eps: float = 1e-6,
                           **deprecated: Any) -> PosteriorChain:
    """
    Build a catalog by relabeling samples using multivariate normal fits,
    with optional univariate initialization and shuffling.

    The chain is first shuffled (if a seed is provided), then expanded
    to `max_num_sources`. If `initialization_param_index` is set,
    a univariate normal relabeling is run for `init_num_iterations` to
    initialize labels. Finally, multivariate relabeling is run for
    `num_iterations`.

    Every argument after `max_num_sources` is keyword-only, and the leading
    keywords are shared with the other ``make_catalog_*`` entry points so the
    methods can be swapped without rewriting the call.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        The chain of posterior samples to process.
    max_num_sources : int
        Target number of sources in the catalog.  Must be at least the number
        of entries already in the chain.
    num_iterations : int, default 200
        Maximum number of relabeling iterations.  One iteration refits a full
        multivariate normal -- mean and covariance -- to every source slot from
        that slot's non-NaN samples, then builds one cost matrix per sample from
        those fits and solves one Hungarian assignment per sample.  Nothing is
        trained, so an iteration costs a covariance and a Cholesky per slot plus
        ``n_samples`` assignment solves; that is why the default is 200 where the
        copula-flow method uses 50.  The loop stops early as soon as
        an iteration fails to lower the assignment cost, so this is a ceiling,
        not a schedule.
    init_num_iterations : int, default 200
        Number of iterations for the optional univariate initialization.  A
        named parameter here, and a field of
        :class:`petra.options.Initialization` on the two entry points that have
        a multivariate-normal initialization step; the flat spelling is the same
        on all three, so a call can still be moved between methods unchanged.
    initialization_param_index : int or None, default 0
        Index of the parameter for the univariate initialization step.  Pass
        ``None`` to skip that step entirely.  Any other value must be a column
        of `posterior_chain`, and is checked here rather than inside the fit.
    shuffle_seed : int, optional
        Seed for random shuffling of chain entries (for reproducibility).
    rng_seed : int, default 999
        Accepted for symmetry with the other ``make_catalog_*`` entry points
        and otherwise unused: multivariate-normal relabeling is deterministic
        given `shuffle_seed`.
    checkpoint_dir : str, optional
        Directory to save a checkpoint of the chain after every iteration of
        the final multivariate relabeling.
    resume_from : str, optional
        Checkpoint file, or directory of checkpoints, to resume an interrupted
        multivariate relabeling from.  The initialization step is skipped when
        resuming, since the checkpoint already reflects it.
    progress : bool, default True
        Show a tqdm progress bar over the samples of each iteration.
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities used *inside* the cost matrix,
        keeping them inside ``[eps, 1 - eps]`` so the logarithms there stay
        finite.  Shared with the other ``make_catalog_*`` entry points: it sets
        the scale of every ``log(1 - p)`` in the cost matrix, so two catalogs
        built with different values are not comparable.  It does not reach the
        returned ``prob_in_model``; see Notes.
    **deprecated
        Legacy keyword names, translated with a `DeprecationWarning`.  See
        :data:`petra.utils.DEPRECATED_KEYWORDS`.

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain instance with relabeled samples.  Its
        ``prob_in_model`` is recomputed from those samples *without* clipping,
        so that ``find_prob_in_model(relabeled_chain.get_chain(),
        max_num_sources, eps=0)`` reproduces it exactly.

    Raises
    ------
    ValueError
        If `initialization_param_index` is neither ``None`` nor a column of the
        chain, or if `max_num_sources` is smaller than the number of entries
        already in it.
    TypeError
        If a keyword is neither a real parameter nor a known deprecated alias.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.make_catalog import make_catalog_mv_normal
    >>> from petra.posterior_chain import PosteriorChain
    >>> rng = np.random.default_rng(0)
    >>> chain_array = rng.normal(size=(20, 3, 1)) + np.array([[0.0], [5.0], [10.0]])
    >>> pc = PosteriorChain(chain_array, 3, 1, True, None, {})
    >>> catalog = make_catalog_mv_normal(
    ...     pc,
    ...     max_num_sources=3,
    ...     num_iterations=5,
    ...     init_num_iterations=5,
    ...     initialization_param_index=0,
    ...     shuffle_seed=42,
    ...     progress=False,
    ... )
    >>> isinstance(catalog, PosteriorChain)
    True
    >>> catalog.chain.shape
    (20, 3, 1)
    >>> catalog.prob_in_model
    array([1., 1., 1.])
    """

    # `current=` enables the "not both" check: without it an alias silently wins
    # over a value the caller passed under the current name.
    resolved = resolve_deprecated_kwargs(
        make_catalog_mv_normal, deprecated,
        current={"num_iterations": num_iterations,
                 "init_num_iterations": init_num_iterations},
    )
    num_iterations = resolved.get("num_iterations", num_iterations)
    init_num_iterations = resolved.get("init_num_iterations", init_num_iterations)
    _validate_relabel_settings(num_iterations, eps)

    # Checked before any work, because neither way of getting this keyword wrong
    # reports itself: a negative index is wrapped round by numpy onto a
    # different parameter, and an index past the end raises an IndexError from
    # inside the fit.  The other two entry points get the same check from
    # `run_initialization_passes`, which this one does not call.
    _validate_initialization_param_index(posterior_chain.shape[2], initialization_param_index)

    # shuffle the entries and make sure the chain has the right shape
    posterior_chain = prepare_chain(posterior_chain, max_num_sources, shuffle_seed=shuffle_seed)

    # initialize here:
    if initialization_param_index is not None and resume_from is None:
        logger.info("Initializing with univariate normal distribution.")
        initial_posterior_chain = relabel_univariate_normal(
            posterior_chain,
            max_num_sources=max_num_sources,
            num_iterations=init_num_iterations,
            init_parameter_index=initialization_param_index,
            eps=eps,
            progress=progress,
        )

    else:
        initial_posterior_chain = posterior_chain

    # relabel using mv normal
    logger.info("Relabeling with multivariate normal distribution.")
    relabeled_chain = relabel_mv_normal(initial_posterior_chain,
                                        max_num_sources=max_num_sources,
                                        num_iterations=num_iterations,
                                        eps=eps,
                                        checkpoint_dir=checkpoint_dir,
                                        resume_from=resume_from,
                                        progress=progress)

    return relabeled_chain
