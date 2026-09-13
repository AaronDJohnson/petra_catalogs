"""
Copula-based normalizing flows for petra catalogs.

Each source is fit with a Gaussian-copula marginal transform followed by a
normalizing flow (``coppuccino.copula_flows.normalizing_flows_fit``), which
handles strong non-Gaussian dependence between parameters while keeping the
marginals well behaved.

Entry point
-----------
:func:`make_catalog_copula_flows` follows the shared ``make_catalog_*`` keyword
contract: ``posterior_chain, max_num_sources`` positionally, then keyword-only
``num_iterations``, ``initialization_param_index``, ``shuffle_seed``,
``rng_seed``, ``checkpoint_dir``, ``resume_from``, ``progress`` and ``eps``, and
finally this method's ``threshold_samples`` and its two settings objects.

Settings objects
----------------
The entry point takes this method's flow hyperparameters as a
:class:`petra.options.CopulaFlowFit` and the optional pre-relabeling as a
:class:`petra.options.Initialization`, rather than as a dozen loose keywords.
Every field of both is still accepted as a plain keyword -- ``knots=8`` and
``flow_fit=CopulaFlowFit(knots=8)`` are the same call -- but an object *and* one
of its fields in the same call is a :class:`TypeError`, never a merge.

The mid-level :func:`relabel_copula_flows` and :func:`make_copula_flows_fit` keep
their flat signatures; the entry point unpacks the object when it forwards.

Conventions
-----------
* Chains are ``(n_samples, n_sources, n_params_per_source)`` and a NaN row means
  "this source is absent from this sample"; every fit here drops those rows.
* ``eps`` is the clip bound on the inclusion probabilities and is ``1e-6``
  throughout petra, and ``log_prob_floor`` is
  :data:`petra.flow_utils.DEFAULT_LOG_PROB_FLOOR`, so the assignment costs of
  the different entry points are directly comparable.
* The early-stopping patience is spelled ``max_patience`` here. ``patience``,
  the spelling ``coppuccino`` itself uses, is still accepted with a
  :class:`DeprecationWarning`.
* Diagnostics go through :func:`petra.utils.get_logger`, never ``print``, and
  every progress bar is silenced by ``progress=False``.

64-bit JAX
----------
This module used to call ``jax.config.update("jax_enable_x64", True)`` at import
time.  That mutates process-global JAX state, so the result depended on import
order and silently overrode the choice of whatever application imported petra.
It is gone.  Enabling x64 is the caller's decision and must be made *before* any
JAX array is created, so it cannot be a function argument here; set the
environment variable ``JAX_ENABLE_X64=1``, or call
``jax.config.update("jax_enable_x64", True)`` yourself at the top of your
script, if the copula fits need double precision.
"""

import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Sequence

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from coppuccino.copula_flows import normalizing_flows_fit

from petra.flow_utils import DEFAULT_LOG_PROB_FLOOR, quiet_external_fit, stack_flow_log_probs
from petra.options import CopulaFlowFit, Initialization
from petra.posterior_chain import PosteriorChain
from petra.relabel import _validate_relabel_settings, create_relabel_samples, prepare_chain
# Shared with the other make_catalog_* entry points so that every one of them
# accepts the same legacy keyword spellings and routes the same flat keywords
# onto its settings objects, implemented once.
from petra.utils import get_logger, make_uniform_prior, resolve_entry_point_kwargs

logger = get_logger(__name__)

#: Default early-stopping patience for the copula flows. Read off
#: :class:`petra.options.CopulaFlowFit` rather than restated, so that the flat
#: defaults below and the options-object default cannot drift apart.
DEFAULT_MAX_PATIENCE = CopulaFlowFit().max_patience


def _resolve_max_patience(func_name: str, max_patience: int, patience: int | None) -> int:
    """
    Accept the legacy ``patience`` spelling of `max_patience`.

    ``coppuccino.copula_flows.normalizing_flows_fit`` calls this ``patience``,
    which is where the old name came from, but every other petra relabeler calls
    it ``max_patience``; the two spellings of one concept made the entry points
    non-interchangeable.

    Used by :func:`make_copula_flows_fit` and :func:`relabel_copula_flows`, which
    keep flat signatures.  :func:`make_catalog_copula_flows` translates
    ``patience`` itself: its `max_patience` now lives on a
    :class:`petra.options.CopulaFlowFit`, so "the caller passed it too" is no
    longer the comparison against `DEFAULT_MAX_PATIENCE` made below -- a caller
    who spells the default out loud means it -- and it has a second collision to
    reject, ``patience`` alongside a whole ``flow_fit``.

    Parameters
    ----------
    func_name : str
        Name of the calling function, used in the warning and error messages.
    max_patience : int
        Value passed under the current name.
    patience : int or None
        Value passed under the deprecated name, or ``None`` if it was not used.

    Returns
    -------
    max_patience : int
        The value to use.

    Raises
    ------
    TypeError
        If both spellings were supplied.

    Warns
    -----
    DeprecationWarning
        If the deprecated spelling was supplied.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     _resolve_max_patience("f", DEFAULT_MAX_PATIENCE, 5)
    5
    >>> _resolve_max_patience("f", 7, None)
    7

    Both spellings at once is a mistake, not a precedence question:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     _resolve_max_patience("f", 9, 5)
    Traceback (most recent call last):
        ...
    TypeError: Pass either 'patience' or 'max_patience', not both.
    """
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
    """
    Log-densities of every copula flow, evaluated on every source of one sample.

    Parameters
    ----------
    sample : ndarray, shape (n_sources, n_params_per_source)
        One posterior sample.  NaN rows mark absent sources.
    aux_parameters : list
        Fitted flows, one per source label.
    source_index : int or array-like of int
        Label indices to return rows for.
    floor : float, default `petra.flow_utils.DEFAULT_LOG_PROB_FLOOR`
        Value substituted for a non-finite log-density.  This used to be
        ``-1e10`` here and ``-50.0`` everywhere else, which made the assignment
        costs of this entry point incomparable with the others.

    Returns
    -------
    log_probs : jax.Array, shape (len(source_index), n_sources)
        Row `i` holds the log-densities flow ``source_index[i]`` assigns to each
        source slot of `sample`.

    Notes
    -----
    The hand-rolled broadcast-repair block this function used to carry -- for
    flows that return a scalar where others return an array -- now lives in
    :func:`petra.flow_utils.safe_flow_log_prob`, which broadcasts every result
    to one value per sample row.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.utils import UniformPrior
    >>> flows = [UniformPrior([0.0], [np.e]), UniformPrior([0.0], [np.e ** 2])]
    >>> sample = np.array([[1.0], [5.0]])
    >>> out = copula_flows_aux_distribution(sample, flows, np.arange(2))
    >>> np.round(np.asarray(out), 3).tolist()
    [[-1.0, -50.0], [-2.0, -2.0]]
    """
    # Ensure source_index is array-like.
    source_indices = jnp.atleast_1d(np.asarray(source_index))  # shape: (n,)
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
    """
    Build a function that fits a copula flow to each source of a chain.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples, possibly containing NaNs for missing sources.  Only
        used to build the uniform-prior fallback; the flows themselves are fit
        to whatever chain is passed to the returned function.
    rng_seed : int, default 999
        Random seed for reproducibility.
    threshold_samples : int, default 50
        A source slot with at most this many usable (non-NaN) samples is not
        worth a flow; it falls back to the uniform prior over the range of the
        whole chain.  This is a statistical decision, not a detail: raising it
        makes more slots uninformative in the assignment.
    knots : int, default 16
        Number of spline knots per flow layer.
    max_patience : int, default `DEFAULT_MAX_PATIENCE`
        Early-stopping patience, in epochs without validation improvement.
    learning_rate : float, default 1e-3
        Adam learning rate.
    max_epochs : int, default 800
        Maximum training epochs per flow.
    flow_layers : int, default 8
        Number of flow layers.
    progress : bool, default True
        Show the per-flow training progress bar.
    patience : int, optional
        Deprecated spelling of `max_patience`.

    Returns
    -------
    copula_flows_fit : callable
        Function ``(chain, max_num_sources) -> list of fitted distributions``.

    Raises
    ------
    TypeError
        If both `max_patience` and `patience` were supplied.

    Warns
    -----
    DeprecationWarning
        If `patience` was supplied.
    """
    max_patience = _resolve_max_patience("make_copula_flows_fit", max_patience, patience)
    # Direct factory callers get the same validation as the catalog entry point.
    CopulaFlowFit(knots=knots, flow_layers=flow_layers, max_epochs=max_epochs,
                  max_patience=max_patience, learning_rate=learning_rate)
    uniform_prior = make_uniform_prior(chain)

    def copula_flows_fit(chain: np.ndarray, max_num_sources: int) -> List:
        """
        Fit one copula flow per source slot of `chain`.

        Parameters
        ----------
        chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
            The chain under its *current* labeling.  Every slot is refit from
            scratch, because relabeling changes which samples it holds.
        max_num_sources : int
            Number of source slots to fit.

        Returns
        -------
        fitted_distributions : list
            One entry per source label: either a fitted copula flow or the
            shared :class:`petra.utils.UniformPrior` fallback.
        """
        fitted_distributions: List = [None] * max_num_sources
        rng_key = jr.key(rng_seed + 1)
        fit_seeds = jr.randint(rng_key, shape=(max_num_sources,), minval=0, maxval=999999)

        for i in range(max_num_sources):
            chain_entry = chain[:, i, :]  # Get the i-th source across all samples
            chain_entry = chain_entry[~np.isnan(chain_entry).any(axis=1)]  # Drop samples missing this source

            if chain_entry.shape[0] <= threshold_samples:  # Not enough samples: fall back to the prior
                logger.info("Source %d has %d usable samples (<= threshold_samples=%d); "
                            "falling back to the uniform prior.",
                            i, chain_entry.shape[0], threshold_samples)
                fitted_distributions[i] = uniform_prior

            else:  # Everything else gets an NF fit
                logger.debug("Fitting a copula flow for source %d on %d samples.", i, chain_entry.shape[0])
                # coppuccino's normalizing_flows_fit has no show_progress keyword, so
                # `progress` is honoured by silencing the bar rather than by not
                # drawing it. See petra.flow_utils.quiet_external_fit.
                with quiet_external_fit(not progress):
                    # coppuccino spells the early-stopping patience `patience`.
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
    """
    Relabel samples using Gaussian copula marginals plus normalizing flows.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain to relabel.
    max_num_sources : int, optional
        Maximum number of sources (defaults to ``chain.num_sources``).
    num_iterations : int, default 20
        Number of relabeling iterations.
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities used *inside* the cost matrix,
        keeping them inside ``[eps, 1 - eps]`` so the logarithms there stay
        finite.  ``1e-6`` throughout petra; this relabeler used to default to
        ``1e-2``, which made its costs incomparable with the other methods'.
        It does not reach the returned ``prob_in_model``; see Notes.
    rng_seed : int, default 999
        Seed for the flow initialization and training.
    threshold_samples : int, default 50
        Slots with at most this many usable samples fall back to the uniform
        prior; see :func:`make_copula_flows_fit`.
    knots, max_patience, learning_rate, max_epochs, flow_layers
        Flow hyperparameters, passed to
        ``coppuccino.copula_flows.normalizing_flows_fit``; see
        :func:`make_copula_flows_fit`.
    log_prob_floor : float, default `petra.flow_utils.DEFAULT_LOG_PROB_FLOOR`
        Value substituted for a non-finite flow log-density.  It sets how hard
        the assignment refuses a label, so it must match across entry points for
        their costs to be comparable.
    checkpoint_dir : str or Path, optional
        Directory to save a checkpoint of the chain after every iteration.
    resume_from : str or Path, optional
        Checkpoint file, or directory of checkpoints, to resume from.  See
        :func:`petra.relabel.load_checkpoint`.
    progress : bool, default True
        Show progress bars, both over samples and over flow training epochs.
    patience : int, optional
        Deprecated spelling of `max_patience`.

    Returns
    -------
    relabeled_chain : PosteriorChain
        Relabeled chain.  Its ``prob_in_model`` is recomputed from the returned
        samples *without* clipping, so that
        ``find_prob_in_model(relabeled_chain.get_chain(), max_num_sources,
        eps=0)`` reproduces it exactly.

    Raises
    ------
    TypeError
        If both `max_patience` and `patience` were supplied.

    Warns
    -----
    DeprecationWarning
        If `patience` was supplied.

    Notes
    -----
    `eps` governs only the clipping applied inside the cost matrix during
    iteration, where a probability of exactly 0 or 1 would make a ``log`` term
    infinite.  Only the returned array is unclipped; see
    :func:`petra.relabel.run_relabeling_loop`.
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

    relabel_samples = create_relabel_samples(copula_flows_fit,
                                             partial(copula_flows_aux_distribution, floor=log_prob_floor),
                                             eps=eps)

    return relabel_samples(
        posterior_chain,
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress=progress,
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
                              initialization: Initialization | None = None,
                              flow_fit: CopulaFlowFit | None = None,
                              **flat_keywords: Any) -> PosteriorChain:
    """
    Create a catalog using Gaussian copula marginal transforms + normalizing flows.

    By default the labels are first initialized with a fast multivariate normal
    relabeling step, itself preceded by a univariate one, which avoids the
    cold-start problem where flows are trained on randomly mixed source slots.
    The two passes are switched independently:
    ``initialization=Initialization(with_mv_normal=False)`` -- or the flat
    ``init_with_mv_normal=False`` -- drops the multivariate pass,
    ``initialization_param_index=None`` drops the univariate one, and both
    together go straight to the flows.

    Everything from `num_iterations` to `eps` is the shared ``make_catalog_*``
    contract, spelled and defaulted the same way by every entry point;
    `threshold_samples` and the two settings objects configure this method.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        The chain of posterior samples to process.
    max_num_sources : int
        Target number of sources in the catalog.  Must be at least the number of
        entries already in the chain.
    num_iterations : int, default 50
        Maximum number of relabeling iterations.  One iteration refits every
        populated source slot -- a Gaussian-copula marginal transform plus a
        ``flow_fit.flow_layers``-layer normalizing flow, trained for up to
        ``flow_fit.max_epochs`` epochs -- and then solves one Hungarian
        assignment per sample against the refitted flows. Training makes each
        iteration more expensive than the Gaussian methods. The loop stops
        early as soon as an iteration fails to lower
        the assignment cost, so this is a ceiling, not a schedule.
    initialization_param_index : int or None, default 0
        Index of the parameter for the univariate normal pre-initialization.
        ``None`` skips the univariate step and goes straight to the MV normal
        one.  It is the only thing that decides whether that step runs:
        ``initialization.with_mv_normal`` switches the multivariate pass alone
        (see Notes).  It names a column of *your* chain -- which parameter
        separates the sources -- rather than tuning the initializer, which is
        why it is here rather than on `initialization`.
    shuffle_seed : int, optional
        Seed for random shuffling of chain entries (for reproducibility).
    rng_seed : int, default 999
        Seed for the flow initialization and training.
    checkpoint_dir : str or Path, optional
        Directory to save iteration checkpoints of the flow relabeling. If None,
        no checkpointing is performed.
    resume_from : str or Path, optional
        Checkpoint file, or directory of checkpoints, to resume the flow
        relabeling from.  The initialization steps are skipped when resuming,
        since the checkpoint already reflects them.
    progress : bool, default True
        Show progress bars, both over samples and over flow training epochs.
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities used *inside* the cost matrix,
        keeping them inside ``[eps, 1 - eps]`` so the logarithms there stay
        finite.  The same value as every other ``make_catalog_*`` entry point,
        so the recorded costs are comparable between methods.  It does not reach
        the returned ``prob_in_model``; see Notes.
    threshold_samples : int, default 50
        A source slot with at most this many usable (non-NaN) samples falls back
        to the uniform prior instead of getting a flow; see
        :func:`make_copula_flows_fit`.  Raising it makes more slots
        uninformative in the assignment.  It decides which slots get a flow at
        all rather than how one trains, which is why it is here rather than on
        `flow_fit`.
    initialization : Initialization, optional
        The pre-relabeling passes -- ``with_mv_normal`` switches the
        multivariate one and ``num_iterations`` bounds each pass that runs.
        The univariate pass is switched by `initialization_param_index`
        instead, not by anything on this object.  ``None`` means
        ``Initialization()``.
    flow_fit : CopulaFlowFit, optional
        Flow training hyperparameters -- ``knots``, ``flow_layers``,
        ``max_epochs``, ``max_patience``, ``learning_rate``,
        ``log_prob_floor``. ``None`` means ``CopulaFlowFit()``.

    Returns
    -------
    relabeled_chain : PosteriorChain
        A new PosteriorChain with relabeled samples using the hybrid approach.
        Its ``prob_in_model`` is recomputed from those samples *without*
        clipping, so that ``find_prob_in_model(relabeled_chain.get_chain(),
        max_num_sources, eps=0)`` reproduces it exactly.

    Other Parameters
    ----------------
    init_with_mv_normal, init_num_iterations
        The fields of `initialization`, still accepted as flat keywords with
        unchanged meaning and no warning.  They keep the same spelling on all
        three ``make_catalog_*`` entry points, so a call can still be moved
        between methods without being rewritten.
    knots, flow_layers, max_epochs, max_patience, learning_rate, log_prob_floor
        The fields of `flow_fit`, likewise.
    patience : int, optional
        Deprecated spelling of ``max_patience``, which is what ``coppuccino``
        itself calls it.
    n_phases : int, optional
        Deprecated alias for `num_iterations`.
    mv_normal_init : bool, optional
        Deprecated alias for ``init_with_mv_normal``.
    mv_normal_init_iterations : int, optional
        Deprecated alias for ``init_num_iterations``.

    Raises
    ------
    ValueError
        If `max_num_sources` is smaller than the number of entries in the chain,
        if `initialization_param_index` is neither ``None`` nor a column of the
        chain, or if a flat keyword carries a value its options class rejects.
    TypeError
        If an unknown keyword argument is supplied, or if an options object is
        passed alongside a flat keyword that would overwrite one of its fields.

    Notes
    -----
    `eps` governs only the clipping applied inside the cost matrix during
    iteration, where a probability of exactly 0 or 1 would make a ``log`` term
    infinite.  Only the returned array is unclipped; see
    :func:`petra.relabel.run_relabeling_loop`.

    An options object and one of its flat fields is a :class:`TypeError`, never
    a merge -- ``CopulaFlowFit(knots=8)`` alongside ``max_epochs=20`` would
    otherwise raise the question of which defaults the object contributes, and
    the answer would have to be remembered rather than read.

    Changed in 1.1: ``init_with_mv_normal`` no longer switches the univariate
    pass off as well.  ``init_with_mv_normal=False`` -- equivalently
    ``initialization=Initialization(with_mv_normal=False)`` or the deprecated
    ``mv_normal_init=False`` -- together with an `initialization_param_index`
    that is not ``None`` (the default is 0) used to skip *both* initialization
    passes here, while
    :func:`petra.bayesian_gaussian.make_catalog_bayesian_gaussian` skipped only
    the multivariate one for the same pair of keywords. Both initialized entry points
    now go through :func:`petra.initialization.run_initialization_passes` and
    skip only the multivariate pass; add ``initialization_param_index=None`` to
    reproduce a run made with an older version.

    Examples
    --------
    A three-source chain, relabeled without any flow training: with
    ``threshold_samples`` above the number of samples every slot falls back to
    the uniform prior, which keeps the example fast.

    >>> import numpy as np
    >>> from petra.copula_flows import make_catalog_copula_flows
    >>> from petra.options import CopulaFlowFit, Initialization
    >>> from petra.posterior_chain import PosteriorChain
    >>> rng = np.random.default_rng(0)
    >>> chain_array = rng.normal(size=(20, 3, 1)) + np.array([[0.0], [5.0], [10.0]])
    >>> pc = PosteriorChain(chain_array, 3, 1, True, None, {})
    >>> flat = make_catalog_copula_flows(
    ...     pc,
    ...     max_num_sources=3,
    ...     num_iterations=2,
    ...     init_num_iterations=2,
    ...     knots=4,
    ...     threshold_samples=1000,
    ...     progress=False,
    ... )
    >>> flat.chain.shape
    (20, 3, 1)

    Spelling the same call with the settings objects is the same run:

    >>> grouped = make_catalog_copula_flows(
    ...     pc,
    ...     max_num_sources=3,
    ...     num_iterations=2,
    ...     initialization=Initialization(num_iterations=2),
    ...     flow_fit=CopulaFlowFit(knots=4),
    ...     threshold_samples=1000,
    ...     progress=False,
    ... )
    >>> bool(np.array_equal(flat.get_chain(), grouped.get_chain()))
    True

    With checkpointing enabled, and resumable:

    >>> catalog = make_catalog_copula_flows(          # doctest: +SKIP
    ...     posterior_chain, max_num_sources=3, checkpoint_dir="./my_checkpoints")
    """
    # `patience` is local to this module's flow backend, so it is not in the
    # package-wide DEPRECATED_KEYWORDS table; translate it here, before the
    # shared resolver sees it and rejects it as unknown.  Warn first and raise
    # afterwards, so a caller who used two spellings is still told which of them
    # is the deprecated one.
    patience = flat_keywords.pop("patience", None)
    if patience is not None:
        warnings.warn(
            "make_catalog_copula_flows(): keyword 'patience' is deprecated, "
            "use 'max_patience' instead.",
            DeprecationWarning, stacklevel=2,
        )
        if "max_patience" in flat_keywords or flow_fit is not None:
            raise TypeError("make_catalog_copula_flows(): pass either 'patience' or "
                            "'max_patience', not both.")
        flat_keywords["max_patience"] = patience

    # `current=` enables the "not both" check: without it an alias silently wins
    # over a value the caller passed under the current name.
    resolved, options = resolve_entry_point_kwargs(
        make_catalog_copula_flows, flat_keywords,
        options={"initialization": initialization, "flow_fit": flow_fit},
        current={"num_iterations": num_iterations},
    )
    # Every key the resolver can return has to be read back out.
    # `num_iterations` (the replacement for `n_phases`) used to be left in the
    # dict, so that alias warned and was then silently dropped -- the caller got
    # the default instead of the value they passed.
    num_iterations = resolved.get("num_iterations", num_iterations)
    _validate_relabel_settings(num_iterations, eps)
    init: Initialization = options["initialization"]
    fit: CopulaFlowFit = options["flow_fit"]

    from petra.make_catalog import relabel_mv_normal
    from petra.initialization import relabel_univariate_normal, run_initialization_passes

    # shuffle the entries and make sure the chain has the right shape
    posterior_chain = prepare_chain(posterior_chain, max_num_sources, shuffle_seed=shuffle_seed)

    # The preamble is shared across the initialized catalog builders: this one had
    # drifted from `make_catalog_bayesian_gaussian`'s, so `init_with_mv_normal`
    # decided both passes here and only one there.
    posterior_chain = run_initialization_passes(
        posterior_chain,
        max_num_sources,
        univariate_relabeler=relabel_univariate_normal,
        mv_normal_relabeler=relabel_mv_normal,
        with_mv_normal=init.with_mv_normal,
        param_index=initialization_param_index,
        num_iterations=init.num_iterations,
        resume_from=resume_from,
        progress=progress,
    )

    # relabel using the copula-flow hybrid approach.  The mid-level relabeler
    # keeps its flat signature, so the settings object is unpacked here rather
    # than pushed down.
    logger.info("Relabeling with Gaussian copula marginals + normalizing flows.")
    return relabel_copula_flows(posterior_chain,
                                max_num_sources=max_num_sources,
                                num_iterations=num_iterations,
                                eps=eps,
                                rng_seed=rng_seed,
                                threshold_samples=threshold_samples,
                                knots=fit.knots,
                                max_patience=fit.max_patience,
                                learning_rate=fit.learning_rate,
                                max_epochs=fit.max_epochs,
                                flow_layers=fit.flow_layers,
                                log_prob_floor=fit.log_prob_floor,
                                checkpoint_dir=checkpoint_dir,
                                resume_from=resume_from,
                                progress=progress)
