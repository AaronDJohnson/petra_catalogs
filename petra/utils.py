"""
Dependency-light helpers shared by every part of :mod:`petra`.

This module deliberately imports only numpy at module scope: it is pulled in by
``import petra`` (through :mod:`petra.samples_io`), and the package promises
that a bare import does not drag in ``jax``/``flowjax``.  The one place that
needs ``jax.numpy`` (:meth:`UniformPrior.log_prob`) imports it lazily.

Logging convention
------------------
The package logs through a single root logger, ``logging.getLogger("petra")``,
exposed here as :data:`logger` and reachable per-module via :func:`get_logger`.
Following the usual library convention it carries a :class:`logging.NullHandler`
so that petra prints nothing unless the *application* configures logging::

    import logging
    logging.basicConfig(level=logging.INFO)   # opt in to petra's progress log

Use ``logger.info`` for per-iteration progress, ``logger.warning`` for
recoverable surprises (a resumed checkpoint, a degenerate fit) and never
``print``.

Progress-bar convention
-----------------------
Any function that drives a ``tqdm`` bar takes a keyword argument

    ``progress : bool, default True``

and threads it into every bar it creates as ``tqdm(..., disable=not progress)``.
Functions that merely delegate to such a function accept the same keyword and
pass it straight through, so a caller can silence a whole pipeline with a single
``progress=False``.
"""

import dataclasses
import inspect
import logging
import types
import warnings
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEPRECATED_KEYWORDS",
    "SECONDS_PER_YEAR",
    "UniformPrior",
    "count_swaps",
    "deprecated_keyword",
    "fill_missing_indices",
    "find_prob_in_model",
    "find_uniform_bounds",
    "frequency_bin_width",
    "get_logger",
    "logger",
    "make_uniform_prior",
    "option_field_keywords",
    "process_array",
    "resolve_deprecated_kwargs",
    "resolve_entry_point_kwargs",
    "sort_by_number",
    "source_present",
]


#: Seconds in a 365-day year (525600 minutes x 60 seconds).  Used to convert an
#: observation time in years into the width of a Fourier frequency bin.
SECONDS_PER_YEAR = 525600 * 60

#: Package-wide logger.  See the module docstring for the logging convention.
logger = logging.getLogger("petra")
logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return the petra logger, or a named child of it.

    Parameters
    ----------
    name : str, optional
        Child logger name.  A bare name such as ``"relabel"`` is placed under
        the package logger (``"petra.relabel"``); a name that already starts
        with ``"petra"`` is used as-is.  ``None`` returns the package logger.

    Returns
    -------
    logging.Logger
        Logger whose records propagate to the ``"petra"`` logger.

    Examples
    --------
    >>> get_logger().name
    'petra'
    >>> get_logger('relabel').name
    'petra.relabel'
    >>> get_logger('petra.relabel').name
    'petra.relabel'
    """
    if name is None or name == "petra":
        return logger
    if name.startswith("petra."):
        return logging.getLogger(name)
    return logging.getLogger(f"petra.{name}")


#: Legacy keyword spellings accepted by the ``make_catalog_*`` entry points,
#: mapped to the name they were harmonized to.  Kept working (with a
#: :class:`DeprecationWarning`) so existing scripts and notebooks do not break.
#: Lives here rather than in a flow module so that the pure-numpy entry points
#: can share it without importing ``jax``.
DEPRECATED_KEYWORDS = {
    "mv_normal_init": "init_with_mv_normal",
    "mv_normal_init_iterations": "init_num_iterations",
    "n_phases": "num_iterations",
}


def deprecated_keyword(deprecated: dict, old: str, new: str,
                       current: Any, default: Any) -> Any:
    """
    Resolve one renamed keyword argument.

    Translates a *single* alias inline.  Superseded by
    :func:`resolve_deprecated_kwargs`, which translates a whole ``**kwargs``
    catch-all at once and checks each alias against the callee's signature.

    Parameters
    ----------
    deprecated : dict
        The ``**kwargs`` catch-all of the calling function.  The old name is
        removed from it when present.
    old, new : str
        Old and new spellings of the keyword.
    current : object
        Value the caller passed under the *new* name.
    default : object
        Default of the new keyword, used to detect that the caller passed both.

    Returns
    -------
    value : object
        `current`, or the value passed under the old name.

    Raises
    ------
    TypeError
        If both spellings were supplied.

    Warns
    -----
    DeprecationWarning
        Always, because this helper is itself deprecated; and a second time if
        the old spelling was actually supplied.

    Warnings
    --------
    Deprecated in favour of :func:`resolve_deprecated_kwargs` and scheduled for
    removal in petra 1.2.  This helper is handed `old` and `new` as bare strings
    and takes them on trust, so it will happily announce "use `new` instead" for
    a keyword the callee does not accept -- and the caller who follows that
    advice then gets ``TypeError: unexpected keyword argument``.  That is the
    concrete bug that motivated unifying the three deprecation mechanisms:
    ``make_catalog_mv_normal`` has no ``init_with_mv_normal`` parameter, because
    that entry point *is* the multivariate-normal method.
    :func:`resolve_deprecated_kwargs` reads the callee's real signature and
    refuses to recommend a keyword it would reject.

    Examples
    --------
    >>> import warnings
    >>> kwargs = {'mv_normal_init': False}
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter('always')
    ...     deprecated_keyword(kwargs, 'mv_normal_init', 'init_with_mv_normal', True, True)
    False
    >>> kwargs
    {}

    Two warnings fire: one for the helper, one for the spelling it resolved.

    >>> for warning in caught:
    ...     print(str(warning.message))
    petra.utils.deprecated_keyword() is deprecated in favour of resolve_deprecated_kwargs() and is scheduled for removal in petra 1.2.
    'mv_normal_init' is deprecated and will be removed; use 'init_with_mv_normal' instead.
    """
    warnings.warn(
        "petra.utils.deprecated_keyword() is deprecated in favour of "
        "resolve_deprecated_kwargs() and is scheduled for removal in petra 1.2.",
        DeprecationWarning,
        # stacklevel=2 blames whoever called this helper. The keyword warning
        # below uses 3 because its audience is the end user two frames out; this
        # one's audience is the petra developer who wrote the call itself.
        stacklevel=2,
    )
    if old not in deprecated:
        return current
    value = deprecated.pop(old)
    warnings.warn(
        f"{old!r} is deprecated and will be removed; use {new!r} instead.",
        DeprecationWarning, stacklevel=3,
    )
    if current != default:
        raise TypeError(f"Pass either {old!r} or {new!r}, not both.")
    return value


class _Unset:
    """Sentinel type for "this parameter has no declared default"."""

    def __repr__(self) -> str:
        """Return a readable marker, so a stray sentinel is obvious in a message."""
        return "<unset>"


#: Singleton marking a parameter that has no default in the callee's signature.
_UNSET = _Unset()


def _differs_from_default(value: Any, default: Any) -> bool:
    """
    Report whether a caller-supplied value is something other than the default.

    Parameters
    ----------
    value : object
        Value the caller currently holds under the new keyword spelling.
    default : object
        The callee's declared default, or :data:`_UNSET` if it has none.

    Returns
    -------
    bool
        True when `value` should be treated as explicitly passed.  A parameter
        with no declared default is always treated as explicitly passed.
    """
    if default is _UNSET:
        return True
    if value is default:
        return False
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return not np.array_equal(value, default)
    return bool(value != default)


def _as_options_class(annotation: Any) -> Optional[type]:
    """
    Return the petra options class an annotation names, if it names one.

    Parameters
    ----------
    annotation : object
        A parameter's annotation, already resolved to a real object.
        ``Optional[X]`` and ``X | None`` are unwrapped.

    Returns
    -------
    options_class : type or None
        The class, or ``None`` when the annotation is anything else.

    Notes
    -----
    The test is *not* ``dataclasses.is_dataclass``.
    :class:`petra.posterior_chain.PosteriorChain` is a dataclass and is the
    first parameter of every entry point, so a dataclass-only rule would make
    ``chain``, ``num_sources``, ``prob_in_model`` and ``cost_dict`` accepted
    keywords everywhere and silently ``replace()`` the caller's chain.  An
    explicit ``FLAT_KEYWORDS`` mapping is the marker instead, which also keeps
    this module free of any import from :mod:`petra.options`.

    Examples
    --------
    >>> from petra.options import CopulaFlowFit
    >>> _as_options_class(Optional[CopulaFlowFit]) is CopulaFlowFit
    True
    >>> _as_options_class(CopulaFlowFit | None) is CopulaFlowFit
    True
    >>> from petra.posterior_chain import PosteriorChain
    >>> _as_options_class(PosteriorChain) is None      # a dataclass, but not an options one
    True
    >>> _as_options_class(int) is None
    True
    """
    candidates = (get_args(annotation)
                  if get_origin(annotation) in (Union, types.UnionType)
                  else (annotation,))
    for candidate in candidates:
        if (isinstance(candidate, type)
                and dataclasses.is_dataclass(candidate)
                and isinstance(getattr(candidate, "FLAT_KEYWORDS", None), Mapping)):
            return candidate
    return None


def _options_parameters(func: Callable[..., Any]) -> dict[str, type]:
    """
    Map each of `func`'s options-object parameters to the class it takes.

    Parameters
    ----------
    func : callable
        Function to introspect.

    Returns
    -------
    parameters : dict of str to type
        Parameter name to options class, in signature order.  Empty for a
        function that takes no options object.

    Examples
    --------
    >>> from petra.options import CopulaFlowFit
    >>> def f(chain, *, flow_fit: Optional[CopulaFlowFit] = None, **flat): pass
    >>> _options_parameters(f)
    {'flow_fit': <class 'petra.options.CopulaFlowFit'>}
    >>> def g(a, *, num_iterations=1, **deprecated): pass
    >>> _options_parameters(g)
    {}
    """
    hints: Optional[dict[str, Any]] = None
    found: dict[str, type] = {}
    for parameter in inspect.signature(func).parameters.values():
        if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            continue
        annotation = parameter.annotation
        if isinstance(annotation, str):
            # A module with `from __future__ import annotations` hands every
            # annotation over as a string; resolve the whole signature once
            # rather than once per parameter.
            if hints is None:
                hints = get_type_hints(func)
            annotation = hints.get(parameter.name, annotation)
        options_class = _as_options_class(annotation)
        if options_class is not None:
            found[parameter.name] = options_class
    return found


def option_field_keywords(func: Callable[..., Any]) -> dict[str, tuple[str, str, type]]:
    """
    Index the flat keywords reachable through `func`'s options objects.

    An entry point that takes ``flow_fit: Optional[CopulaFlowFit]`` still accepts
    ``knots=8``; this is what says so, and where the keyword has to go.

    Parameters
    ----------
    func : callable
        Function to introspect.

    Returns
    -------
    index : dict of str to (str, str, type)
        Flat keyword name to ``(parameter_name, field_name, options_class)``.
        Empty for a function that takes no options object, which is why adding
        this to :func:`resolve_deprecated_kwargs` cannot change how any
        flat-signature function behaves.

    Examples
    --------
    >>> from petra.options import CopulaFlowFit
    >>> def f(chain, *, flow_fit: Optional[CopulaFlowFit] = None, **flat): pass
    >>> option_field_keywords(f)['knots']
    ('flow_fit', 'knots', <class 'petra.options.CopulaFlowFit'>)
    >>> 'num_iterations' in option_field_keywords(f)
    False
    >>> def g(a, *, num_iterations=1, **deprecated): pass
    >>> option_field_keywords(g)
    {}
    """
    index: dict[str, tuple[str, str, type]] = {}
    for parameter_name, options_class in _options_parameters(func).items():
        # getattr rather than attribute access: the marker is duck-typed so that
        # this module never imports petra.options (see `_as_options_class`), and
        # `type` is therefore all a static checker can know about the class here.
        flat_map: Mapping[str, str] = getattr(options_class, "FLAT_KEYWORDS")
        for flat_name, field_name in flat_map.items():
            index[flat_name] = (parameter_name, field_name, options_class)
    return index


def resolve_deprecated_kwargs(func: Union[Callable[..., Any], str],
                              deprecated_kwargs: dict,
                              current: Optional[Mapping[str, Any]] = None,
                              *,
                              stacklevel: int = 3) -> dict:
    """
    Translate legacy keyword names into their harmonized replacements.

    ``DEPRECATED_KEYWORDS`` is shared by every entry point, but not every alias in
    it is meaningful everywhere: ``mv_normal_init`` renames to
    ``init_with_mv_normal``, which ``make_catalog_mv_normal`` does not have, because
    that entry point *is* the multivariate-normal method. Pass the function object
    rather than its name and this checks the alias against the real signature, so a
    deprecation can never point at a keyword the callee would reject.

    Parameters
    ----------
    func : callable or str
        The calling entry point. Pass the function itself so its accepted keywords
        can be read from its signature; a bare name is accepted for backwards
        compatibility but skips that check.
    deprecated_kwargs : dict
        The ``**kwargs`` the entry point swallowed, i.e. everything that did not
        match a real parameter.
    current : mapping, optional
        Values the caller currently holds under the *new* spellings, i.e.
        ``{"num_iterations": num_iterations, ...}``. Supplying it enables the
        "not both" check: passing an alias *and* its replacement is then an error
        instead of letting the alias silently override an explicitly passed value.
        A value equal to the parameter's declared default counts as not passed.
    stacklevel : int, default 3
        Frame the :class:`DeprecationWarning` is blamed on.  The default points
        at the code that called `func`, which is the audience: warn/error, this
        helper and `func` itself are the three frames in between.  Raise it by
        one per extra frame -- :func:`resolve_entry_point_kwargs` passes 4 --
        or the warning names a line inside petra and a caller filtering
        ``DeprecationWarning`` by module never sees it.

    Returns
    -------
    resolved : dict
        Mapping from *current* keyword name to the value that was passed under
        the deprecated name. Every key is guaranteed to be a parameter `func`
        actually accepts.

    Raises
    ------
    TypeError
        If a keyword is not a known deprecated alias, if it is an alias whose
        replacement this particular entry point does not accept, or if `current`
        shows that both spellings were supplied.

    Notes
    -----
    The returned mapping must be *read*, not merely popped from: the usual
    idiom is one ``resolved.get(name, name_value)`` per key the entry point
    understands. Dropping a key on the floor reinstates exactly the bug this
    function exists to prevent -- warning about an alias and then ignoring it.

    Warns
    -----
    DeprecationWarning
        Once per deprecated keyword, naming its replacement.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     resolve_deprecated_kwargs("f", {"mv_normal_init": False})
    {'init_with_mv_normal': False}
    >>> str(caught[0].message)
    "f(): keyword 'mv_normal_init' is deprecated, use 'init_with_mv_normal' instead."
    >>> resolve_deprecated_kwargs("f", {"nonsense": 1})
    Traceback (most recent call last):
        ...
    TypeError: f() got an unexpected keyword argument 'nonsense'

    An alias whose replacement the callee does not accept is rejected outright,
    rather than warning and then silently discarding the value:

    >>> def g(a, *, num_iterations=1, **deprecated): pass
    >>> resolve_deprecated_kwargs(g, {"mv_normal_init": True})
    Traceback (most recent call last):
        ...
    TypeError: g() does not accept 'mv_normal_init' (renamed to 'init_with_mv_normal'), which is not applicable to this entry point

    Pass `current` to reject a call that used both spellings at once, the way
    :func:`deprecated_keyword` does:

    >>> resolve_deprecated_kwargs(g, {"n_phases": 7}, current={"num_iterations": 3})
    Traceback (most recent call last):
        ...
    TypeError: g(): pass either 'n_phases' or 'num_iterations', not both.

    A value that is simply the declared default is not "passed":

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     resolve_deprecated_kwargs(g, {"n_phases": 7}, current={"num_iterations": 1})
    {'num_iterations': 7}
    """
    defaults: dict = {}
    if callable(func):
        func_name = func.__name__
        parameters = inspect.signature(func).parameters
        # A keyword that now lives on an options object is still accepted, so it
        # has to count as accepted here too. Without this, `init_with_mv_normal`
        # stopped being a named parameter of two entry points and their
        # `mv_normal_init` alias started reporting itself "not applicable" --
        # while the keyword it renames to kept working.
        accepted = {
            p.name
            for p in parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        } | set(option_field_keywords(func))
        defaults = {
            p.name: p.default
            for p in parameters.values()
            if p.default is not inspect.Parameter.empty
        }
    else:
        func_name = func
        accepted = None

    resolved = {}
    for name, value in deprecated_kwargs.items():
        if name not in DEPRECATED_KEYWORDS:
            raise TypeError(f"{func_name}() got an unexpected keyword argument {name!r}")
        new_name = DEPRECATED_KEYWORDS[name]
        if accepted is not None and new_name not in accepted:
            raise TypeError(
                f"{func_name}() does not accept {name!r} (renamed to {new_name!r}), "
                f"which is not applicable to this entry point"
            )
        # Warn *before* the "not both" check, matching `deprecated_keyword`: the
        # caller used a deprecated spelling, and that is worth saying even when
        # the call also turns out to be an error. Reversing the order makes the
        # warning disappear exactly when two spellings collide.
        warnings.warn(
            f"{func_name}(): keyword {name!r} is deprecated, use {new_name!r} instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        if current is not None and new_name in current:
            # `deprecated_keyword` raises on both spellings; without this the alias
            # would silently win over a value the caller passed explicitly.
            if _differs_from_default(current[new_name], defaults.get(new_name, _UNSET)):
                raise TypeError(
                    f"{func_name}(): pass either {name!r} or {new_name!r}, not both."
                )
        resolved[new_name] = value
    return resolved


def resolve_entry_point_kwargs(func: Callable[..., Any],
                               kwargs: dict,
                               *,
                               options: Mapping[str, Any],
                               current: Optional[Mapping[str, Any]] = None,
                               ) -> tuple[dict, dict]:
    """
    Resolve an entry point's catch-all into named values and options objects.

    The single place the ``make_catalog_*`` entry points call.  It does three
    jobs at once: translate deprecated spellings (through
    :func:`resolve_deprecated_kwargs`), route the flat keywords that now live on
    an options object into that object, and reject a call that supplies both an
    object and one of its fields.

    Parameters
    ----------
    func : callable
        The entry point itself, so its signature and its options-object
        parameters can be read.
    kwargs : dict
        The ``**kwargs`` catch-all the entry point swallowed.
    options : mapping of str to object
        Options-parameter name to the value the caller passed, ``None`` when
        they passed nothing.  Every options parameter of `func` is resolved,
        whether or not it appears here.
    current : mapping, optional
        Values the caller holds under the *named* keywords, for
        :func:`resolve_deprecated_kwargs`' "not both" check.  It must contain
        only real named parameters: a keyword that now lives on an options
        object is checked by this function instead, from the catch-all.

    Returns
    -------
    resolved_named : dict
        Named keyword to the value a deprecated spelling supplied, exactly as
        :func:`resolve_deprecated_kwargs` returns it, minus the keys that were
        routed onto an options object.
    resolved_options : dict
        Options-parameter name to a fully built options object.  Never
        ``None``-valued, so the caller can read fields off it unconditionally.

    Raises
    ------
    TypeError
        If a keyword is unknown, if a deprecated alias is not applicable to
        `func`, if two spellings of the same setting were supplied -- either an
        alias and its replacement, or an options object and one of its fields --
        or if an options object is not an instance of the class its parameter is
        annotated with.
    ValueError
        If a flat keyword carries a value the owning options class rejects.
        ``dataclasses.replace`` re-runs ``__post_init__``, so ``knots=0`` fails
        identically whether it was spelled flat or inside a ``CopulaFlowFit``.

    Warns
    -----
    DeprecationWarning
        Once per deprecated keyword.

    Notes
    -----
    The order is load-bearing.  Splitting the options keywords out of `kwargs`
    has to happen *before* :func:`resolve_deprecated_kwargs` runs, because they
    are not aliases and it would reject them as unknown; raising on them has to
    happen *after*, or the deprecation warning would vanish exactly when two
    spellings collide -- which is the case where knowing which one is deprecated
    matters most.

    Landing in the catch-all is the only signal used, so there is no
    default-comparison anywhere in the options path: ``knots=16`` spelled out is
    treated exactly like ``knots=32``.  And an options object is never merged
    field-by-field with flat keywords, so ``replace`` is only ever applied to
    the class's own no-arg default and there is no precedence rule to get wrong
    later.

    Examples
    --------
    >>> from petra.options import CopulaFlowFit
    >>> def entry(chain, *, num_iterations=5, flow_fit: Optional[CopulaFlowFit] = None, **flat):
    ...     pass
    >>> named, resolved = resolve_entry_point_kwargs(
    ...     entry, {"knots": 8}, options={"flow_fit": None},
    ...     current={"num_iterations": 5})
    >>> named
    {}
    >>> resolved["flow_fit"].knots, resolved["flow_fit"].max_epochs
    (8, 800)

    Nothing passed at all still yields a usable object:

    >>> resolve_entry_point_kwargs(entry, {}, options={"flow_fit": None})[1]["flow_fit"]
    CopulaFlowFit(knots=16, flow_layers=8, max_epochs=800, max_patience=20, learning_rate=0.001, log_prob_floor=-50.0)

    Passing the object and one of its fields is an error, not a merge:

    >>> resolve_entry_point_kwargs(entry, {"knots": 8}, options={"flow_fit": CopulaFlowFit()})
    Traceback (most recent call last):
        ...
    TypeError: entry(): pass either 'knots' or 'flow_fit', not both.

    So is an options object of the wrong class, which the annotation names and
    the caller's ``import`` line does not:

    >>> from petra.options import Initialization
    >>> resolve_entry_point_kwargs(entry, {}, options={"flow_fit": Initialization()})
    Traceback (most recent call last):
        ...
    TypeError: entry(): 'flow_fit' must be a CopulaFlowFit, got Initialization.
    """
    func_name = func.__name__
    index = option_field_keywords(func)
    classes = _options_parameters(func)

    flat: dict[str, Any] = {}
    typed_as: dict[str, str] = {}
    rest: dict[str, Any] = {}
    for name, value in kwargs.items():
        if name in index:
            flat[name] = value
            typed_as[name] = name
        else:
            rest[name] = value

    # stacklevel=4, not the default 3: this frame sits between the entry point
    # and the warning, so the default would blame a line inside petra for the
    # caller's deprecated keyword.
    resolved_named = resolve_deprecated_kwargs(func, rest, current=current, stacklevel=4)

    # Everything left in `rest` survived the resolver, so it is a known alias and
    # this inversion cannot miss: it recovers the spelling the caller actually
    # typed, so the "not both" message names their keyword rather than ours.
    alias_of = {DEPRECATED_KEYWORDS[old]: old for old in rest}
    for field_keyword in [key for key in resolved_named if key in index]:
        value = resolved_named.pop(field_keyword)
        if field_keyword in flat:
            raise TypeError(
                f"{func_name}(): pass either {alias_of[field_keyword]!r} or "
                f"{field_keyword!r}, not both."
            )
        flat[field_keyword] = value
        typed_as[field_keyword] = alias_of[field_keyword]

    for name in flat:
        parameter = index[name][0]
        if options.get(parameter) is not None:
            raise TypeError(
                f"{func_name}(): pass either {typed_as[name]!r} or {parameter!r}, not both."
            )

    resolved_options: dict[str, Any] = {}
    for parameter, options_class in classes.items():
        overrides = {index[name][1]: value for name, value in flat.items()
                     if index[name][0] == parameter}
        passed = options.get(parameter)
        if passed is not None and not isinstance(passed, options_class):
            # Reject a settings object under the wrong parameter before its
            # missing fields surface inside initialization or flow training.
            raise TypeError(
                f"{func_name}(): {parameter!r} must be a {options_class.__name__}, "
                f"got {type(passed).__name__}."
            )
        if overrides:
            resolved_options[parameter] = dataclasses.replace(options_class(), **overrides)
        elif passed is not None:
            resolved_options[parameter] = passed
        else:
            resolved_options[parameter] = options_class()
    return resolved_named, resolved_options


def frequency_bin_width(obs_time_yrs: float) -> float:
    """
    Width of one Fourier frequency bin for a given observation time.

    Parameters
    ----------
    obs_time_yrs : float
        Observation time in years.  Must be strictly positive.

    Returns
    -------
    delta_freq : float
        ``1 / (obs_time_yrs * SECONDS_PER_YEAR)`` in Hz.

    Raises
    ------
    ValueError
        If `obs_time_yrs` is not strictly positive.

    Examples
    --------
    >>> round(frequency_bin_width(1.0), 12)
    3.171e-08
    >>> frequency_bin_width(0.0)
    Traceback (most recent call last):
        ...
    ValueError: obs_time_yrs must be positive, got 0.0
    """
    if not obs_time_yrs > 0:
        raise ValueError(f"obs_time_yrs must be positive, got {obs_time_yrs}")
    return 1.0 / (obs_time_yrs * SECONDS_PER_YEAR)


def source_present(source_data: npt.ArrayLike) -> np.ndarray:
    """
    Test which source rows are present, under petra's all-or-nothing NaN rule.

    This is the single reading of the convention stated in :mod:`petra` and in
    :mod:`petra.posterior_chain`: a source row -- the ``n_params_per_source``
    values of one source in one sample -- is either **all** ``NaN``, meaning
    that source is absent from that sample, or **all** finite, meaning it is
    present.  Every fit, cost matrix and inclusion probability in the package
    decides presence through this function, so they cannot drift apart.

    A row that violates the rule, i.e. one that mixes ``NaN`` with real values,
    is reported **absent**: presence requires the whole row, because a fit
    cannot use half a source.  Such a row is not supposed to exist, and
    :class:`~petra.posterior_chain.PosteriorChain` rejects one on construction
    rather than let it reach here; this is what happens to an array that never
    passed through that check.

    Parameters
    ----------
    source_data : array_like
        Array whose **last** axis is the parameter axis.  Any leading axes are
        kept, so this accepts a whole chain ``(n_samples, n_sources,
        n_params_per_source)``, one source's samples ``(n_samples,
        n_params_per_source)``, one sample ``(n_sources,
        n_params_per_source)``, or a single row ``(n_params_per_source,)``.

    Returns
    -------
    present : ndarray of bool
        `source_data` with its last axis reduced away: ``True`` wherever that
        row is entirely free of ``NaN``.  A single row gives a 0-d array.

    Examples
    --------
    >>> chain = np.array([
    ...     [[1.0, 2.0], [np.nan, np.nan]],
    ...     [[3.0, 4.0], [5.0, 6.0]],
    ... ])
    >>> source_present(chain).tolist()          # a whole chain
    [[True, False], [True, True]]
    >>> source_present(chain[:, 1, :]).tolist()  # one source, all samples
    [False, True]
    >>> source_present(chain[0]).tolist()        # one sample, all sources
    [True, False]
    >>> bool(source_present(np.array([1.0, np.nan])))   # half a row is absent
    False
    """
    return ~np.isnan(np.asarray(source_data)).any(axis=-1)


def _validate_eps(eps: float) -> None:
    """Require finite, ordered probability clipping bounds, including eps=0."""
    if not np.isfinite(eps) or not 0 <= eps <= 0.5:
        raise ValueError(f"eps must be finite and between 0 and 0.5, got {eps!r}.")


def find_prob_in_model(chain: np.ndarray, max_num_sources: int, eps: float = 1e-6) -> np.ndarray:
    """
    Compute the probability that each source is present in the model.

    For each source index i, counts the fraction of samples in which that source
    is present -- :func:`source_present`, i.e. the whole row is free of ``NaN``
    -- then clips to [eps, 1-eps] to avoid log-domain issues.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples, possibly containing NaNs for missing sources.
    max_num_sources : int
        Maximum number of sources to consider.
    eps : float, optional
        Clip bound keeping the returned probabilities inside ``[eps, 1 - eps]``,
        so that ``log(p)`` and ``log(1 - p)`` stay finite (default 1e-6).
        Pass ``eps=0`` for unclipped probabilities.

    Returns
    -------
    prob_in_model : ndarray, shape (max_num_sources,)
        Probability each source index is active in the samples.

    Raises
    ------
    ValueError
        If `eps` is nonfinite or outside ``[0, 0.5]``, or if
        `max_num_sources` exceeds the number of source slots in `chain`.
        Slicing would otherwise return a *shorter* array than the caller asked
        for, and callers index the result by source label.

    Examples
    --------
    >>> chain = np.array([
    ...     [[1.0], [np.nan], [2.0]],
    ...     [[0.5], [ 2.1], [np.nan]],
    ...     [[np.nan], [1.8], [2.3]],
    ... ])
    >>> # chain.shape = (3 samples, 3 sources, 1 param); every source is
    >>> # present in exactly 2 of the 3 samples
    >>> np.round(find_prob_in_model(chain, max_num_sources=3), 3)
    array([0.667, 0.667, 0.667])
    >>> find_prob_in_model(chain, max_num_sources=4)
    Traceback (most recent call last):
        ...
    ValueError: max_num_sources (4) exceeds the 3 source slots of a chain of shape (3, 3, 1).
    """
    _validate_eps(eps)
    chain = np.asarray(chain)
    if max_num_sources > chain.shape[1]:
        raise ValueError(
            f"max_num_sources ({max_num_sources}) exceeds the {chain.shape[1]} source "
            f"slots of a chain of shape {chain.shape}."
        )
    prob_in_model = source_present(chain[:, :max_num_sources, :]).mean(axis=0)
    # clip the probabilities to avoid division by zero in np.log
    prob_in_model = np.clip(prob_in_model, eps, 1 - eps)
    return prob_in_model


def fill_missing_indices(total_sources: int, given_indices: Sequence[int]) -> np.ndarray:
    """
    Fill in missing indices by appending those not in `given_indices`.

    Takes the list of selected indices and appends all other indices
    from 0 to total_sources-1 in ascending order.

    Parameters
    ----------
    total_sources : int
        The total number of source indices desired.
    given_indices : array-like of int
        Indices that are already assigned or filled.

    Returns
    -------
    filled_indices : ndarray, shape (total_sources,)
        Array starting with `given_indices`, then the missing indices.

    Examples
    --------
    >>> fill_missing_indices(5, [2, 4])
    array([2, 4, 0, 1, 3])
    >>> fill_missing_indices(3, [])
    array([0, 1, 2])
    """
    # Step 1: Generate a list of all indices from 0 to total_sources - 1
    all_indices = np.arange(total_sources)
    # Step 2: Convert given_indices to a set for faster operations
    given_indices_set = set(given_indices)
    # Step 3: Filter out the given indices from all_indices to get missing indices
    missing_indices = [index for index in all_indices if index not in given_indices_set]
    # Step 4: Combine the given indices with the missing indices
    filled_indices = list(given_indices) + missing_indices

    return np.array(filled_indices)


def count_swaps(arr: np.ndarray) -> int:
    """
    Count the minimum number of transpositions needed to sort an array.

    The permutation that sorts `arr` is decomposed into disjoint cycles; a
    cycle of length k costs k-1 transpositions, so the total is
    ``n - number_of_cycles``.  Counting mismatches with the sorted array and
    halving is only correct when every cycle has length 2.

    Parameters
    ----------
    arr : ndarray of shape (n,)
        Input array of comparable elements.  Ties are broken by position
        (stable sort), which is the cheapest of the equivalent orderings.

    Returns
    -------
    swaps : int
        Minimum number of pairwise swaps to sort `arr`.

    Warns
    -----
    DeprecationWarning
        Always, because this helper is itself deprecated.

    Warnings
    --------
    Deprecated with no replacement and scheduled for removal in petra 1.2.
    Nothing in petra calls it, and nothing in petra can supply its argument: the
    per-sample assignment permutation is built and consumed inside
    :func:`petra.relabel.relabel_samples_one_iteration`, which returns the
    reordered samples rather than the ordering, so a caller has no relabeling
    permutation to count.  What petra does report for how good a labeling is, is
    the assignment cost recorded in
    :attr:`petra.posterior_chain.PosteriorChain.cost_dict`, which is comparable
    across labelings and across methods in a way a transposition count is not --
    a swap count is a distance from whatever order the slots happened to arrive
    in, and the slot order is arbitrary.  Deprecating rather than deleting
    matches :func:`deprecated_keyword`: this is exported public API, so it keeps
    working for one release.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter('always')
    ...     count_swaps(np.array([2, 1, 3]))
    ...     count_swaps(np.array([3, 1, 2]))       # a single 3-cycle: two swaps
    ...     count_swaps(np.array([1, 0, 3, 2]))    # two disjoint 2-cycles
    ...     count_swaps(np.array([1, 2, 3]))
    1
    2
    2
    0

    The retirement notice fires once per call, and names the release that drops
    the function:

    >>> len(caught)
    4
    >>> print(str(caught[0].message))
    petra.utils.count_swaps() is deprecated with no replacement and is scheduled for removal in petra 1.2.
    """
    warnings.warn(
        "petra.utils.count_swaps() is deprecated with no replacement and is "
        "scheduled for removal in petra 1.2.",
        DeprecationWarning,
        # stacklevel=2 blames the call site rather than this line: the audience is
        # whoever still reaches for the helper, and a notice pointing at utils.py
        # tells them nothing they can act on.
        stacklevel=2,
    )
    arr = np.asarray(arr)
    n = arr.shape[0]

    # destination[i] is the position element i ends up in once sorted
    order = np.argsort(arr, kind="stable")
    destination = np.empty(n, dtype=np.int64)
    destination[order] = np.arange(n)

    seen = np.zeros(n, dtype=bool)
    num_cycles = 0
    for start in range(n):
        if seen[start]:
            continue
        num_cycles += 1
        i = start
        while not seen[i]:
            seen[i] = True
            i = destination[i]

    return int(n - num_cycles)


def sort_by_number(filenames: Sequence[str]) -> list[str]:
    """
    Sort filenames by the integer suffix after the final dot.

    Assumes each filename ends with ".<number>".

    Parameters
    ----------
    filenames : list of str
        Filenames to sort.

    Returns
    -------
    sorted_list : list of str
        Filenames sorted in ascending order of their numeric suffix.

    Examples
    --------
    >>> sort_by_number(['file.10', 'file.2', 'file.1'])
    ['file.1', 'file.2', 'file.10']
    """
    # Extract the number after the dot and convert it to an integer
    def extract_number(filename: str) -> int:
        """Return the integer suffix following the final dot of `filename`."""
        return int(filename.split(".")[-1])

    # Sort the filenames using the extracted number
    return sorted(filenames, key=extract_number)


class UniformPrior:
    """
    Uniform density over the hyper-rectangle spanned by a chain.

    The log density is a property of the *bounds* only,
    ``-sum(log(maxs - mins))``, and is therefore well defined even for a
    trans-dimensional chain whose first sample is all NaN.  Points outside the
    hyper-rectangle (and points containing NaN) get ``-inf``.

    Use ``isinstance(x, UniformPrior)`` to test whether a fitted distribution
    is the fallback prior rather than a trained flow; this replaces the older
    ``flow is uniform_prior`` and ``hasattr(flow, "bijection")`` sentinels.

    Parameters
    ----------
    mins : array-like, shape (n_params,)
        Lower bound of the support in each dimension.
    maxs : array-like, shape (n_params,)
        Upper bound of the support in each dimension.

    Attributes
    ----------
    mins : ndarray, shape (n_params,)
        Lower bounds.
    maxs : ndarray, shape (n_params,)
        Upper bounds.
    log_density : float
        Constant log density inside the support, ``-sum(log(maxs - mins))``
        over the non-degenerate dimensions.

    Raises
    ------
    ValueError
        If the bounds are not 1-D arrays of the same length, contain
        non-finite values, or if any ``maxs < mins``.

    Notes
    -----
    A dimension with zero width (every sample took the same value) carries no
    information and would make the density infinite; such dimensions are
    dropped from both the log density and the support test, and a warning is
    logged when the prior is built.

    Examples
    --------
    >>> prior = UniformPrior([0.0, 0.0], [2.0, 4.0])
    >>> round(float(prior.log_prob(np.array([1.0, 1.0]))), 6)  # -log(2 * 4)
    -2.079442
    >>> float(prior.log_prob(np.array([3.0, 1.0])))  # outside the support
    -inf
    >>> [round(float(v), 6) for v in prior.log_prob(np.array([[1.0, 1.0], [1.0, 9.0]]))]
    [-2.079442, -inf]
    """

    def __init__(self, mins: npt.ArrayLike, maxs: npt.ArrayLike) -> None:
        """Validate the bounds and precompute the constant log density."""
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)
        if mins.ndim != 1 or maxs.ndim != 1 or mins.shape != maxs.shape:
            raise ValueError(
                f"mins and maxs must be 1-D arrays of the same length, got shapes "
                f"{mins.shape} and {maxs.shape}."
            )
        if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs))):
            raise ValueError("UniformPrior bounds must be finite; got NaN or inf.")
        if np.any(maxs < mins):
            raise ValueError("UniformPrior requires maxs >= mins in every dimension.")

        widths = maxs - mins
        self._active = widths > 0
        if not np.all(self._active):
            logger.warning(
                "UniformPrior: %d of %d dimensions have zero width and are "
                "ignored by the prior.",
                int(np.sum(~self._active)),
                widths.size,
            )
        if not np.any(self._active):
            raise ValueError("UniformPrior requires at least one dimension of non-zero width.")

        self.mins = mins
        self.maxs = maxs
        self.log_density = float(-np.sum(np.log(widths[self._active])))

    @property
    def n_params(self) -> int:
        """int : Number of dimensions the prior is defined over."""
        return self.mins.shape[0]

    def __repr__(self) -> str:
        """Return a representation naming the dimension and the log density."""
        return f"UniformPrior(n_params={self.n_params}, log_density={self.log_density:.6g})"

    def log_prob(self, sample: npt.ArrayLike) -> Any:
        """
        Log density of the prior at one point or a batch of points.

        Parameters
        ----------
        sample : array-like, shape (n_params,) or (n_points, n_params)
            Point(s) to evaluate.

        Returns
        -------
        log_prob : jax.Array
            Scalar for a 1-D `sample`, shape ``(n_points,)`` for a 2-D one.
            ``self.log_density`` inside the support and ``-inf`` outside it
            (NaN entries count as outside).

        Raises
        ------
        ValueError
            If `sample` is neither 1-D nor 2-D, or its trailing dimension does
            not match the number of parameters.
        """
        # Imported lazily so that ``import petra`` stays jax-free; numpy is a
        # perfectly good fallback if jax is not installed.
        xnp: Any
        try:
            import jax.numpy as jax_numpy
            xnp = jax_numpy
        except ImportError:  # pragma: no cover - jax is a heavy optional dep
            xnp = np

        x = xnp.asarray(sample)
        if x.ndim not in (1, 2):
            raise ValueError("Sample must be 1D or 2D array.")
        if x.shape[-1] != self.n_params:
            raise ValueError(
                f"Sample has {x.shape[-1]} parameters but the prior is defined over "
                f"{self.n_params}."
            )

        active = xnp.asarray(self._active)
        inside_dim = (x >= xnp.asarray(self.mins)) & (x <= xnp.asarray(self.maxs))
        inside = xnp.all(inside_dim | ~active, axis=-1)
        return xnp.where(inside, self.log_density, -xnp.inf)


def make_uniform_prior(chain: np.ndarray) -> UniformPrior:
    """
    Build a :class:`UniformPrior` spanning the range observed in a chain.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples array; NaNs indicate missing entries and are ignored
        when taking the per-parameter minima and maxima.

    Returns
    -------
    UniformPrior
        Prior over the hyper-rectangle ``[mins, maxs]``, where `mins` and
        `maxs` are the per-parameter minima and maxima over all samples and
        sources.

    Raises
    ------
    ValueError
        If a parameter is NaN in every sample of every source, so that no
        bounds can be inferred.

    Examples
    --------
    >>> chain = np.array([
    ...     [[0.0, 0.0], [np.nan, np.nan]],
    ...     [[2.0, 4.0], [1.0, 1.0]],
    ... ])
    >>> prior = make_uniform_prior(chain)
    >>> prior
    UniformPrior(n_params=2, log_density=-2.07944)
    >>> round(float(prior.log_prob(np.array([1.0, 1.0]))), 6)
    -2.079442
    """
    mins, maxs = find_uniform_bounds(chain)
    if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs))):
        raise ValueError(
            "Cannot build a uniform prior: at least one parameter is NaN in every "
            "entry of the chain."
        )
    return UniformPrior(mins, maxs)


def process_array(arr: np.ndarray) -> np.ndarray:
    """
    Break ties in an array so that every value is strictly unique.

    Duplicated values are nudged apart by a fraction of the smallest positive gap
    in the data, which keeps the perturbation below the resolution of the sample
    while making the sequence strictly increasing.  An empirical CDF built from
    the result is then strictly monotonic, which the spline interpolators used by
    the flow transforms require.

    Parameters
    ----------
    arr : ndarray, shape (n,)
        Values to de-duplicate.  Sorted on entry, so the input need not be.

    Returns
    -------
    processed : ndarray, shape (n,)
        Sorted copy of `arr` with duplicates perturbed apart.  The input array is
        never modified in place.

    Raises
    ------
    ValueError
        If the input is not a one-dimensional array of finite real numbers,
        or duplicates cannot be separated without overflowing a finite float.

    Examples
    --------
    >>> process_array(np.array([2.0, 1.0, 1.0]))
    array([1.  , 1.25, 2.  ])
    >>> process_array(np.array([3.0, 1.0, 2.0]))   # already unique: only sorted
    array([1., 2., 3.])
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("process_array requires a one-dimensional array.")
    if not (np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.integer)):
        raise ValueError("process_array requires real numeric values.")
    # Validate after conversion as well: a wider float can overflow float64.
    with np.errstate(over="ignore"):
        arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("process_array requires finite values.")
    arr = np.sort(arr)

    # Check for duplicates
    _, counts = np.unique(arr, return_counts=True)
    if np.all(counts <= 1):
        return arr

    # Determine epsilon for perturbation
    max_count = np.max(counts)
    with np.errstate(over="ignore"):
        diffs = np.diff(arr)
    pos_diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(pos_diffs) > 0:
        min_pos_diff = np.min(pos_diffs)
        epsilon = min_pos_diff / (max_count * 2)
    else:
        epsilon = 1e-10

    # Copy array for modification
    new_arr = arr.copy()

    # Perturb duplicates
    i = 0
    while i < len(new_arr):
        val = new_arr[i]
        j = i
        while j < len(new_arr) and new_arr[j] == val:
            j += 1
        group_size = j - i
        if group_size > 1:
            with np.errstate(over="ignore"):
                for k in range(group_size):
                    new_arr[i + k] += k * epsilon
        i = j

    # A fractional gap can be smaller than floating-point spacing. Advance by
    # at least one representable value, including any following value reached
    # by the nudged group, so the result is strictly increasing throughout.
    with np.errstate(over="ignore"):
        for i in range(1, len(new_arr)):
            if new_arr[i] <= new_arr[i - 1]:
                new_arr[i] = np.nextafter(new_arr[i - 1], np.inf)
    if not np.all(np.isfinite(new_arr)):
        raise ValueError("process_array cannot separate these values into finite representable floats.")

    return new_arr


def find_uniform_bounds(chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-parameter minima and maxima over every sample and source of a chain.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples; NaNs are ignored.

    Returns
    -------
    lower_bound : ndarray, shape (n_params_per_source,)
        Per-parameter minimum.
    upper_bound : ndarray, shape (n_params_per_source,)
        Per-parameter maximum.

    Examples
    --------
    >>> chain = np.array([[[0.0], [np.nan]], [[2.0], [1.0]]])
    >>> find_uniform_bounds(chain)
    (array([0.]), array([2.]))
    """
    all_entries = chain.reshape(-1, chain.shape[2])
    lower_bound = np.nanmin(all_entries, axis=0)
    upper_bound = np.nanmax(all_entries, axis=0)
    return lower_bound, upper_bound
