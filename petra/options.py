"""
Grouped, validated settings for the ``make_catalog_*`` entry points.

The settings describe flow training and optional initialization passes.  This module gives each concern a frozen dataclass, so an entry point
takes one object per concern instead of a flat block of keywords, and so the
defaults are declared once instead of being restated on the entry point, on the
mid-level relabeler and on the fitter factory.

Every flat keyword these classes absorb is still accepted by the entry point
that accepted it before, silently and with unchanged meaning; the routing is
:func:`petra.utils.resolve_entry_point_kwargs`, which reads the
:data:`FLAT_KEYWORDS` map each class carries.  What is *not* offered is a merge:
passing an object and one of its fields at the same time is a :class:`TypeError`
rather than a precedence question.

Import cost
-----------
This module imports nothing from :mod:`petra` and nothing outside the standard
library, which is what lets ``petra/__init__.py`` export these two classes
eagerly.  In particular it must not import
:data:`petra.flow_utils.DEFAULT_LOG_PROB_FLOOR`, because
:mod:`petra.flow_utils` imports ``jax`` at module scope and ``import petra``
promises not to.  ``-50.0`` is therefore written out twice on purpose; a test
pins the two spellings together.

Recognising an options class
----------------------------
:mod:`petra.utils` decides whether a parameter's annotation is an options object
by asking whether the class is a dataclass *and* carries a
:data:`FLAT_KEYWORDS` mapping.  "Is a dataclass" alone would not do:
:class:`petra.posterior_chain.PosteriorChain` is one too, and every entry point
takes one, so a dataclass-only rule would quietly accept ``num_sources=`` or
``cost_dict=`` as keywords and rewrite the caller's chain.  The marker is
duck-typed, so a new options class needs no registration anywhere.

Examples
--------
>>> CopulaFlowFit().max_epochs
800
>>> Initialization.FLAT_KEYWORDS['init_num_iterations']
'num_iterations'
"""

import dataclasses
import math
from numbers import Integral
from types import MappingProxyType
from typing import ClassVar, Mapping

__all__ = [
    "CopulaFlowFit",
    "Initialization",
]


def _check_at_least_one(name: str, value: int) -> None:
    """
    Reject a non-integer count or a count below one.

    Parameters
    ----------
    name : str
        Field name, used in the message so the caller knows which knob is wrong.
    value : int
        Value to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `value` is not an integer, is a boolean, or is smaller than 1.

    Examples
    --------
    >>> _check_at_least_one('knots', 1) is None
    True
    >>> _check_at_least_one('knots', 0)
    Traceback (most recent call last):
        ...
    ValueError: knots must be at least 1, got 0.
    """
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")


def _check_positive(name: str, value: float) -> None:
    """
    Reject a scale that is not finite and strictly positive.

    Parameters
    ----------
    name : str
        Field name, used in the message.
    value : float
        Value to check.  NaN fails, because ``not NaN > 0``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `value` is not finite or not strictly greater than zero.

    Examples
    --------
    >>> _check_positive('interval', 4.0) is None
    True
    >>> _check_positive('interval', 0.0)
    Traceback (most recent call last):
        ...
    ValueError: interval must be strictly positive, got 0.0.
    """
    if not value > 0:
        raise ValueError(f"{name} must be strictly positive, got {value}.")
    _check_finite(name, value)


def _check_finite(name: str, value: float) -> None:
    """
    Reject a non-finite replacement value.

    Parameters
    ----------
    name : str
        Field name, used in the message.
    value : float
        Value to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `value` is NaN, ``inf`` or ``-inf``.

    Examples
    --------
    >>> _check_finite('log_prob_floor', -50.0) is None
    True
    >>> _check_finite('log_prob_floor', float('-inf'))
    Traceback (most recent call last):
        ...
    ValueError: log_prob_floor must be finite, got -inf.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")


@dataclasses.dataclass(frozen=True)
class CopulaFlowFit:
    """
    Training hyperparameters for the ``coppuccino`` copula-flow backend.

    These settings are forwarded to the coppuccino fitter by
    :func:`petra.copula_flows.make_catalog_copula_flows`.

    Parameters
    ----------
    knots : int, default 16
        Spline knots per flow layer.
    flow_layers : int, default 8
        Coupling layers per flow.
    max_epochs : int, default 800
        Maximum training epochs per flow.
    max_patience : int, default 20
        Early-stopping patience, in epochs without validation improvement.
    learning_rate : float, default 1e-3
        Adam learning rate.
    log_prob_floor : float, default -50.0
        Value substituted for a non-finite flow log-density.

    Attributes
    ----------
    FLAT_KEYWORDS : mapping of str to str
        Flat keyword spelling to field name. The entry point accepts each
        spelling directly as an alternative to the settings object.

    Raises
    ------
    ValueError
        If `knots`, `flow_layers`, `max_epochs` or `max_patience` is not a
        positive integer, if `learning_rate` is not finite and strictly
        positive, or if `log_prob_floor` is not finite.

    Examples
    --------
    >>> CopulaFlowFit().max_epochs, CopulaFlowFit().max_patience
    (800, 20)
    >>> CopulaFlowFit(flow_layers=4).flow_layers
    4
    >>> CopulaFlowFit(flow_layers=0)
    Traceback (most recent call last):
        ...
    ValueError: flow_layers must be at least 1, got 0.
    """

    FLAT_KEYWORDS: ClassVar[Mapping[str, str]] = MappingProxyType({
        "knots": "knots",
        "flow_layers": "flow_layers",
        "max_epochs": "max_epochs",
        "max_patience": "max_patience",
        "learning_rate": "learning_rate",
        "log_prob_floor": "log_prob_floor",
    })

    knots: int = 16
    flow_layers: int = 8
    max_epochs: int = 800
    max_patience: int = 20
    learning_rate: float = 1e-3
    log_prob_floor: float = -50.0

    def __post_init__(self) -> None:
        """
        Validate every field that has a valid range.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            As documented on the class.
        """
        _check_at_least_one("knots", self.knots)
        _check_at_least_one("flow_layers", self.flow_layers)
        _check_at_least_one("max_epochs", self.max_epochs)
        _check_at_least_one("max_patience", self.max_patience)
        _check_positive("learning_rate", self.learning_rate)
        _check_finite("log_prob_floor", self.log_prob_floor)


@dataclasses.dataclass(frozen=True)
class Initialization:
    """
    The optional pre-relabeling passes run before the real method.

    Shared by the two entry points that have one.  There are *two* passes -- a
    univariate relabeling of one parameter, then a full multivariate-normal one
    -- and two independent switches for them, only one of which is here.
    ``initialization_param_index`` is deliberately *not* a field: it names a
    column of the caller's chain -- which parameter separates the sources --
    rather than tuning the initializer, and it is tuned often enough to stay a
    named keyword.  See
    :func:`petra.initialization.run_initialization_passes`, which is where the
    two gates are read.

    Parameters
    ----------
    with_mv_normal : bool, default True
        Run the multivariate-normal pass, and *only* that pass.  The univariate
        pass is gated on the entry point's ``initialization_param_index``
        alone, so ``with_mv_normal=False`` at the default
        ``initialization_param_index=0`` still runs it; pass
        ``initialization_param_index=None`` to drop that one too.  This field
        read "run the initialization at all" until 1.1, which described the
        coupling that the copula-flow entry point had and that the shared
        preamble removed.  Reached flat as ``init_with_mv_normal`` and, with a
        :class:`DeprecationWarning`, as ``mv_normal_init``.
    num_iterations : int, default 200
        Ceiling on the iterations of *each* initialization pass.  Both passes go
        through :func:`petra.relabel.run_relabeling_loop`, so the budget is
        spent twice: once on the univariate pass, once on the multivariate one.

    Attributes
    ----------
    FLAT_KEYWORDS : mapping of str to str
        Flat keyword spelling to field name.  The only non-identity map in the
        package, and it is load-bearing: the field is `num_iterations`, but the
        flat spelling must be ``init_num_iterations``, because every entry point
        already has a `num_iterations` of its own.  An identity map here would
        let the caller's top-level iteration budget be routed into the
        initializer instead.

    Raises
    ------
    ValueError
        If `num_iterations` is not a positive integer.

    Examples
    --------
    >>> Initialization().num_iterations
    200
    >>> Initialization(with_mv_normal=False).with_mv_normal
    False
    >>> Initialization(num_iterations=0)
    Traceback (most recent call last):
        ...
    ValueError: init_num_iterations must be at least 1, got 0.
    """

    FLAT_KEYWORDS: ClassVar[Mapping[str, str]] = MappingProxyType({
        "init_with_mv_normal": "with_mv_normal",
        "init_num_iterations": "num_iterations",
    })

    with_mv_normal: bool = True
    num_iterations: int = 200

    def __post_init__(self) -> None:
        """
        Validate the iteration budget.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `num_iterations` is below 1, matching
            :func:`petra.relabel.run_relabeling_loop`'s own check -- raising
            here means the message arrives before a univariate pass has run
            rather than after.  The message names the flat spelling
            ``init_num_iterations``, because the entry point's own
            `num_iterations` is a different budget and saying "num_iterations"
            would point at the wrong one.
        """
        _check_at_least_one("init_num_iterations", self.num_iterations)
