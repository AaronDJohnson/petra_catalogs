"""
Shared helpers for evaluating normalizing flows, in one place.

Every flow-based entry point needs the same thing: evaluate ``flow.log_prob``
on a sample and turn whatever non-finite garbage comes back into a finite
number, so that the Hungarian solver in :mod:`petra.relabel` sees a usable cost
matrix.  The floor used for that replacement is *not* cosmetic -- it sets how
strongly the assignment refuses to give a label to a sample that the
corresponding flow considers impossible -- so it lives here, once, as
:data:`DEFAULT_LOG_PROB_FLOOR`, instead of being open-coded differently in each
module.

Unlike :mod:`petra.utils`, importing this module pulls in ``jax``.

Examples
--------
>>> import numpy as np
>>> from petra.utils import UniformPrior
>>> prior = UniformPrior([0.0], [np.e])                  # density 1 / e
>>> round(float(safe_flow_log_prob(prior, np.array([1.0]))), 3)
-1.0
>>> float(safe_flow_log_prob(prior, np.array([5.0])))    # outside the support
-50.0
"""

import contextlib
import os
from typing import Any, Iterator, Optional, Sequence

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from petra.utils import UniformPrior

__all__ = [
    "DEFAULT_LOG_PROB_FLOOR",
    "quiet_external_fit",
    "safe_flow_log_prob",
    "stack_flow_log_probs",
]


#: Value substituted for a non-finite (``-inf`` or ``NaN``) flow log-density.
#: Large enough that the assignment solver will avoid such a pairing, small
#: enough that it does not swamp the finite entries of the cost matrix the way
#: a value like ``-1e10`` does.
DEFAULT_LOG_PROB_FLOOR = -50.0


@contextlib.contextmanager
def quiet_external_fit(quiet: bool) -> Iterator[None]:
    """
    Suppress a third-party fitter's progress bar.

    ``coppuccino.copula_flows.normalizing_flows_fit`` takes no ``show_progress``
    keyword; it forwards to ``flowjax.train.fit_to_data``, whose own
    ``show_progress`` defaults to ``True``.  There is therefore no argument
    petra can pass to honour ``progress=False`` for those fits, and the only
    remaining option is to discard what the bar writes.  ``tqdm`` writes to
    stderr, so that is what is redirected.

    Anything the fit writes to stderr while suppressed -- including warnings --
    is discarded.  That is the price of ``progress=False``, and it is why this
    is a narrow context manager wrapped around the single offending call rather
    than something applied broadly.

    Parameters
    ----------
    quiet : bool
        Suppress output when True; do nothing at all when False.

    Yields
    ------
    None

    Examples
    --------
    >>> import sys
    >>> with quiet_external_fit(True):
    ...     print("swallowed", file=sys.stderr)
    >>> with quiet_external_fit(False):
    ...     _ = 1  # stderr passes through untouched
    """
    if not quiet:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        yield


def safe_flow_log_prob(flow: Optional[Any], x: npt.ArrayLike,
                       floor: float = DEFAULT_LOG_PROB_FLOOR) -> Array:
    """
    Evaluate ``flow.log_prob(x)``, replacing non-finite results by `floor`.

    Handles every case the per-module copies of this helper used to handle:

    * a trained ``flowjax`` distribution, evaluated on one point or on a batch;
    * a :class:`petra.utils.UniformPrior` fallback (``-inf`` outside its
      support becomes `floor`, so the two kinds of distribution stay
      comparable in the same cost matrix);
    * an empty slot, ``flow is None`` or ``x`` with no samples, which yields
      `floor` everywhere;
    * rows of `x` that are NaN because the source is absent from that sample,
      which the flow maps to NaN and which are floored elementwise;
    * a distribution that returns a scalar for a batched `x`, which is
      broadcast to one value per row.

    Parameters
    ----------
    flow : object or None
        Anything with a ``log_prob`` method -- a ``flowjax`` distribution or a
        :class:`petra.utils.UniformPrior`.  ``None`` marks an unfitted slot.
    x : array-like, shape (n_params,) or (n_points, n_params)
        Point or batch of points to evaluate.
    floor : float, default `DEFAULT_LOG_PROB_FLOOR`
        Value substituted for every non-finite log-density.

    Returns
    -------
    log_prob : jax.Array
        Scalar for a 1-D `x`, shape ``(n_points,)`` for a 2-D `x`.  Always
        finite.

    Notes
    -----
    Only :class:`FloatingPointError` is caught -- that is what jax raises for a
    NaN produced inside the flow when ``jax_debug_nans`` is enabled.  The
    previous implementations caught bare ``Exception`` and returned the floor,
    which silently turned shape mismatches and mis-wired flows into a slightly
    worse assignment cost instead of a traceback.  Every other exception now
    propagates.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.utils import UniformPrior
    >>> prior = UniformPrior([0.0, 0.0], [np.e, 1.0])
    >>> round(float(safe_flow_log_prob(prior, np.array([1.0, 0.5]))), 3)
    -1.0
    >>> float(safe_flow_log_prob(prior, np.array([np.nan, np.nan])))
    -50.0
    >>> float(safe_flow_log_prob(None, np.array([0.5, 0.5])))
    -50.0
    >>> batch = np.array([[1.0, 0.5], [3.0, 0.5]])       # second point is outside
    >>> np.round(np.asarray(safe_flow_log_prob(prior, batch)), 3).tolist()
    [-1.0, -50.0]
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
    """
    Evaluate every flow on every source of a sample.

    Parameters
    ----------
    flows : sequence of flow objects, length n_flows
        One fitted distribution per source label; entries may be ``None`` or a
        :class:`petra.utils.UniformPrior`, see :func:`safe_flow_log_prob`.
    sample : ndarray, shape (n_sources, n_params)
        One posterior sample, i.e. the parameters currently sitting in each
        source slot.  NaN rows mark absent sources.
    floor : float, default `DEFAULT_LOG_PROB_FLOOR`
        Value substituted for every non-finite log-density.

    Returns
    -------
    log_probs : jax.Array, shape (n_flows, n_sources)
        ``log_probs[i, j]`` is the log-density flow `i` assigns to the data in
        slot `j`.  Always finite.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.utils import UniformPrior
    >>> narrow = UniformPrior([0.0], [np.e])
    >>> wide = UniformPrior([0.0], [np.e ** 2])
    >>> sample = np.array([[1.0], [5.0]])   # 5.0 is outside the narrow prior
    >>> np.round(np.asarray(stack_flow_log_probs([narrow, wide], sample)), 3).tolist()
    [[-1.0, -50.0], [-2.0, -2.0]]
    """
    points = jnp.asarray(sample)
    if points.ndim != 2:
        raise ValueError(f"sample must be 2-D of shape (n_sources, n_params), got {points.shape}.")
    rows = [safe_flow_log_prob(flow, points, floor=floor) for flow in flows]
    return jnp.stack(rows, axis=0)
