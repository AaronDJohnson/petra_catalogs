"""
The container every other module in :mod:`petra` passes around.

:class:`PosteriorChain` bundles a trans-dimensional posterior chain with the
metadata needed to interpret it, and knows how to round-trip itself through a
Feather file.  It deliberately depends only on numpy, pandas and pyarrow, so
``import petra`` does not drag in the JAX stack.

Conventions
-----------
**Shape.** The chain is always stored as ``(n_samples, n_sources,
n_params_per_source)``.  Flat and two dimensional inputs are reshaped to that
form on construction; anything whose shape contradicts `num_sources` /
`num_params_per_source` is rejected rather than silently reinterpreted.

**NaN means absent.** A source slot whose parameters are ``NaN`` in a given
sample is not present in that sample.  This is how a fixed-width array
represents a variable number of sources, and it is why every fit in the package
filters on :meth:`PosteriorChain.get_valid_chain_entry` rather than assuming
every slot is populated.  The rule is all-or-nothing -- a source row is either
entirely ``NaN`` or entirely finite -- and it is read in exactly one place,
:func:`petra.utils.source_present`.  A chain carrying a row that is neither is
rejected on construction by `_validate_nan_convention`, because the readers
would otherwise disagree about it in silence.

**Views, not copies.** The chain is *not* copied by default; see the Notes of
:class:`PosteriorChain`.  Pass ``copy=True`` for an instance that owns its data.

**Metadata describes *this* chain.** ``prob_in_model`` is indexed by source
label and ``cost_dict`` is keyed by number of sources, so both are statements
about the labeling the chain is carrying right now.  Any method that changes
that labeling re-derives the first and drops the second rather than passing them
on: see :meth:`PosteriorChain.expand_chain` and
:meth:`PosteriorChain.randomize_entries`.  ``prob_in_model`` is recoverable --
it is ``find_prob_in_model(chain, num_sources, eps=0)`` by contract, so it can
always be recomputed from the samples -- while a cost cannot be recomputed
without redoing the fit that produced it, which is why one is re-derived and the
other is dropped.

**Cost is negated log-likelihood.** ``cost_dict`` maps a number of sources to
the cost of the best labeling found at that width, so lower is better; see
:mod:`petra.relabel`.

Feather layout
--------------
:meth:`PosteriorChain.to_feather` writes the chain as one float column per
``(source, parameter)`` pair and everything else --- `num_sources`,
`num_params_per_source`, `trans_dimensional`, `prob_in_model`, `cost_dict` ---
as JSON under the :data:`FEATHER_METADATA_KEY` schema key.  What is written is
what *describes* the chain; the two fields that only record how the constructor
was called, ``copy`` and ``validate_nan_convention``, are construction options
and are taken by :meth:`PosteriorChain.read_feather` instead -- which is a
construction.  Files written by the older layout, which padded the metadata into
extra columns, are still readable; see :meth:`PosteriorChain.read_feather`.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from petra.utils import find_prob_in_model, get_logger, source_present

logger = get_logger(__name__)

#: Key under which the PosteriorChain metadata is stored in the Feather schema.
FEATHER_METADATA_KEY = b"petra_posterior_chain"

#: Version of the Feather metadata layout written by :meth:`PosteriorChain.to_feather`.
FEATHER_METADATA_VERSION = 1

#: Column names written by the pre-metadata (legacy) version of ``to_feather``.
_LEGACY_METADATA_COLUMNS = ("num_sources", "num_params_per_source",
                            "transdimensional", "prob_in_model")


def _encode_cost_dict(cost_dict: dict) -> list:
    """
    Encode a cost dictionary as a JSON-serializable list of ``[key, value]`` pairs.

    A list of pairs is used rather than a JSON object because JSON object keys
    are always strings, whereas ``cost_dict`` is keyed by the (integer) number
    of sources.

    Parameters
    ----------
    cost_dict : dict
        Mapping of number of sources to total assignment cost.  A
        :class:`PosteriorChain` always has one, empty at worst.

    Returns
    -------
    list
        List of ``[key, value]`` pairs.

    Examples
    --------
    >>> from petra.posterior_chain import _encode_cost_dict
    >>> _encode_cost_dict({3: -1.5})
    [[3, -1.5]]
    >>> _encode_cost_dict({})
    []
    """
    encoded = []
    for key, value in cost_dict.items():
        try:
            key = int(key)
        except (TypeError, ValueError):
            key = str(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
        encoded.append([key, value])
    return encoded


def _normalize_cost_dict(cost_dict: dict | None) -> dict:
    """
    Turn a possibly-``None`` cost dictionary into a dictionary.

    :attr:`PosteriorChain.cost_dict` is declared non-optional so that readers can
    index it directly, but callers written against the older optional field --
    and chains rebuilt from metadata that predates the key -- still pass
    ``cost_dict=None``.  Normalizing on construction is what keeps that ``None``
    from reaching a reader as an ``AttributeError`` on ``None.items()``.

    Parameters
    ----------
    cost_dict : dict or None
        Mapping of number of sources to total assignment cost, or ``None``.

    Returns
    -------
    dict
        `cost_dict` itself, or a new empty dictionary if it was ``None``.

    Examples
    --------
    >>> from petra.posterior_chain import _normalize_cost_dict
    >>> _normalize_cost_dict(None)
    {}
    >>> _normalize_cost_dict({3: -1.5})
    {3: -1.5}
    """
    return {} if cost_dict is None else cost_dict


def _decode_cost_dict(encoded: list | None) -> dict:
    """
    Decode the output of `_encode_cost_dict` back into a dictionary.

    Parameters
    ----------
    encoded : list or None
        List of ``[key, value]`` pairs as written by `_encode_cost_dict`.

    Returns
    -------
    dict
        The reconstructed cost dictionary.

    Examples
    --------
    >>> from petra.posterior_chain import _decode_cost_dict
    >>> _decode_cost_dict([[3, -1.5]])
    {3: -1.5}
    """
    return {key: value for key, value in (encoded or [])}


def _validate_dimension(name: str, value: Any) -> int:
    """
    Check that a chain dimension is a positive integer, and normalize its type.

    Parameters
    ----------
    name : str
        Name of the attribute, used in the error message.
    value : object
        Candidate value, either a Python ``int`` or a NumPy integer.

    Returns
    -------
    int
        `value` as a builtin ``int``.

    Raises
    ------
    ValueError
        If `value` is not an integer, or is not at least 1.

    Examples
    --------
    >>> from petra.posterior_chain import _validate_dimension
    >>> _validate_dimension('num_sources', np.int64(3))
    3
    >>> _validate_dimension('num_sources', 0)
    Traceback (most recent call last):
        ...
    ValueError: num_sources must be a positive integer, got 0.
    """
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _validate_numeric_array(chain: Any, num_sources: int, num_params_per_source: int) -> np.ndarray:
    """
    Coerce the chain to an array and reject non-real or infinite data.

    Parameters
    ----------
    chain : array_like
        Candidate chain.
    num_sources, num_params_per_source : int
        Validated chain dimensions, quoted in the error message so the user is
        told what shape was expected.

    Returns
    -------
    ndarray
        `chain` as a NumPy array; a view whenever NumPy can provide one.

    Raises
    ------
    ValueError
        If the array does not hold floats or integers, or contains infinity.
        NaN is allowed here because whole NaN rows represent absent sources.

    Examples
    --------
    >>> from petra.posterior_chain import _validate_numeric_array
    >>> _validate_numeric_array([[1.0, 2.0]], 2, 1).shape
    (1, 2)
    >>> _validate_numeric_array(np.array([['a', 'b']]), 2, 1)
    Traceback (most recent call last):
        ...
    ValueError: chain must be a numeric array, got dtype dtype('<U1'); expected a float or integer array of shape (n_samples, 2, 1).
    """
    array = np.asarray(chain)
    if not (np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.integer)):
        raise ValueError(
            f"chain must be a numeric array, got dtype {array.dtype!r}; expected "
            f"a float or integer array of shape (n_samples, {num_sources}, "
            f"{num_params_per_source})."
        )
    if np.isinf(array).any():
        raise ValueError("chain must not contain infinite values; use NaN for absent source rows.")
    return array


def _validate_chain_shape(chain: np.ndarray, num_sources: int, num_params_per_source: int) -> None:
    """
    Check that a chain's shape can mean what `num_sources` says it means.

    Three layouts are accepted, and each is checked against the declared
    dimensions rather than being reshaped on trust: the canonical
    ``(n_samples, num_sources, num_params_per_source)``; the flattened-sample
    form ``(n_samples, num_sources * num_params_per_source)``; and a fully flat
    array whose size is a whole number of samples.  Without these checks a
    ``(100, 3, 4)`` array would silently reshape under
    ``num_sources=4, num_params_per_source=3``, swapping the meaning of the
    source and parameter axes.

    Parameters
    ----------
    chain : ndarray
        Numeric chain array, as returned by `_validate_numeric_array`.
    num_sources, num_params_per_source : int
        Validated chain dimensions.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `chain` has more than three dimensions, or if its shape contradicts
        `num_sources` and `num_params_per_source`.

    Examples
    --------
    >>> from petra.posterior_chain import _validate_chain_shape
    >>> _validate_chain_shape(np.zeros((10, 2, 3)), 2, 3)     # canonical
    >>> _validate_chain_shape(np.zeros((10, 6)), 2, 3)        # flattened sample
    >>> _validate_chain_shape(np.zeros(60), 2, 3)             # fully flat
    >>> _validate_chain_shape(np.zeros((10, 5)), 2, 3)
    Traceback (most recent call last):
        ...
    ValueError: chain has shape (10, 5), but a two dimensional chain must have num_sources * num_params_per_source = 2 * 3 = 6 columns; expected shape (n_samples, 6) or (n_samples, 2, 3).
    """
    entries_per_sample = num_sources * num_params_per_source
    if chain.ndim == 3:
        if chain.shape[1:] != (num_sources, num_params_per_source):
            raise ValueError(
                f"chain has shape {chain.shape}, which is inconsistent with "
                f"num_sources={num_sources} and "
                f"num_params_per_source={num_params_per_source}; expected shape "
                f"(n_samples, {num_sources}, {num_params_per_source})."
            )
    elif chain.ndim == 2:
        if chain.shape[1] != entries_per_sample:
            raise ValueError(
                f"chain has shape {chain.shape}, but a two dimensional chain must have "
                f"num_sources * num_params_per_source = {num_sources} * "
                f"{num_params_per_source} = {entries_per_sample} columns; expected "
                f"shape (n_samples, {entries_per_sample}) or (n_samples, "
                f"{num_sources}, {num_params_per_source})."
            )
    elif chain.ndim == 1:
        if chain.size % entries_per_sample != 0:
            raise ValueError(
                f"chain has shape {chain.shape} ({chain.size} values), which is not a "
                f"multiple of num_sources * num_params_per_source = "
                f"{num_sources} * {num_params_per_source} = "
                f"{entries_per_sample}; expected shape (n_samples, "
                f"{num_sources}, {num_params_per_source})."
            )
    else:
        raise ValueError(
            f"chain must have 1, 2, or 3 dimensions, got {chain.ndim}; expected shape "
            f"(n_samples, {num_sources}, {num_params_per_source})."
        )


def _validate_nan_convention(chain: np.ndarray) -> None:
    """
    Reject a chain holding a source row that is neither all-NaN nor all-finite.

    The whole package is built on the all-or-nothing rule of
    :func:`petra.utils.source_present`, and it is built on it *inconsistently
    enough to matter*: ``find_prob_in_model`` would count a half-NaN row as a
    present source while every fit drops it, so the inclusion probability fed
    to the spike-slab cost matrix would describe data no fit ever saw.  Nothing
    inside petra can create such a row -- every writer moves whole rows -- so it
    can only enter from outside, which is why it is caught here, at the
    boundary, instead of being handled downstream.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Chain already reshaped to its canonical form, so that a violating row
        can be named by its sample and source index.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any source row mixes ``NaN`` with finite values.

    Examples
    --------
    >>> from petra.posterior_chain import _validate_nan_convention
    >>> _validate_nan_convention(np.array([[[1.0, 2.0], [np.nan, np.nan]]]))
    >>> _validate_nan_convention(np.array([[[1.0, 2.0], [3.0, np.nan]]]))
    Traceback (most recent call last):
        ...
    ValueError: chain violates petra's all-or-nothing NaN convention: sample 0, source 1 has 1 NaN among its 2 parameters, so it is neither absent (all NaN) nor present (all finite). 1 source row is like this. A NaN row means "this source is absent from this sample", and a partial one would be counted present by find_prob_in_model but dropped by every fit. Pass validate_nan_convention=False to accept the chain anyway; the chains PosteriorChain and petra.relabel derive from it inherit the setting.
    """
    # Counting NaNs is one pass over the array; deriving the same answer from
    # `source_present` and a separate all-NaN mask needs two, which measured
    # 29 ms against 10 ms on a 100k x 15 x 8 chain -- and this runs on every
    # construction, including the one per relabeling iteration.
    num_params = chain.shape[-1]
    nan_count = np.isnan(chain).sum(axis=-1)
    partial = (nan_count != 0) & (nan_count != num_params)
    if not partial.any():
        return

    sample_index, source_index = (int(i) for i in np.argwhere(partial)[0])
    num_partial = int(partial.sum())
    raise ValueError(
        f"chain violates petra's all-or-nothing NaN convention: sample "
        f"{sample_index}, source {source_index} has {nan_count[sample_index, source_index]} "
        f"NaN among its {num_params} parameters, so it is neither absent (all NaN) nor "
        f"present (all finite). {num_partial} source row"
        f"{'' if num_partial == 1 else 's'} "
        f"{'is' if num_partial == 1 else 'are'} like this. A NaN row means \"this source "
        f"is absent from this sample\", and a partial one would be counted present by "
        f"find_prob_in_model but dropped by every fit. Pass validate_nan_convention=False "
        f"to accept the chain anyway; the chains PosteriorChain and petra.relabel derive "
        f"from it inherit the setting."
    )


def _validate_prob_in_model(prob_in_model: Any, num_sources: int) -> None:
    """
    Reject an inclusion-probability array that does not have one entry per source slot.

    ``prob_in_model`` is indexed by source label everywhere it is read -- the
    cost matrix takes ``log(prob_in_model[i])`` for row `i` -- so an array
    of the wrong length either raises ``IndexError`` deep
    inside a fit or, worse, silently scores the wrong slot.  It is checked here,
    beside the shape and NaN rules, because it is the same kind of statement: a
    chain that carries a `prob_in_model` describing a *different* number of
    sources is not a chain anyone can interpret.

    Parameters
    ----------
    prob_in_model : array_like or None
        Candidate array of per-source inclusion probabilities.  ``None`` means
        "not computed" and is always accepted.
    num_sources : int
        Validated number of source slots the array has to describe.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `prob_in_model` is not ``None`` and its shape is not
        ``(num_sources,)``.

    Examples
    --------
    >>> from petra.posterior_chain import _validate_prob_in_model
    >>> _validate_prob_in_model(None, 3)
    >>> _validate_prob_in_model(np.array([1.0, 0.5]), 2)
    >>> _validate_prob_in_model(np.array([1.0, 0.5]), 4)
    Traceback (most recent call last):
        ...
    ValueError: prob_in_model has shape (2,), but it holds one inclusion probability per source slot and is indexed by source label, so a chain with num_sources=4 needs shape (4,). Pass prob_in_model=None, or recompute it from the samples with petra.utils.find_prob_in_model(chain, 4, eps=0); a Feather file whose stored array is this shape is repaired that way on read.
    """
    if prob_in_model is None:
        return
    shape = np.asarray(prob_in_model).shape
    if shape != (num_sources,):
        raise ValueError(
            f"prob_in_model has shape {shape}, but it holds one inclusion probability "
            f"per source slot and is indexed by source label, so a chain with "
            f"num_sources={num_sources} needs shape ({num_sources},). Pass "
            f"prob_in_model=None, or recompute it from the samples with "
            f"petra.utils.find_prob_in_model(chain, {num_sources}, eps=0); a Feather "
            f"file whose stored array is this shape is repaired that way on read."
        )


def _prob_in_model_from_file(prob_in_model: Any, chain: np.ndarray, num_sources: int,
                             feather_filepath: str) -> np.ndarray | None:
    """
    Return a stored inclusion-probability array, recomputing it if it cannot describe the chain.

    A Feather file is not a caller making a mistake that can be handed back to
    them, it is a run already on disk: a checkpoint written by a petra whose
    :meth:`PosteriorChain.expand_chain` widened the chain without widening this
    array carries one that is *shorter* than the chain's own ``num_sources``,
    and rejecting it would make a half-finished multi-day run unresumable over
    metadata alone.  The samples in the file are intact, and ``prob_in_model``
    is a function of them -- every entry point in petra returns
    ``find_prob_in_model(chain, num_sources, eps=0)`` -- so the array is
    recovered rather than trusted or refused.

    Recomputing, not padding with zeros for the missing slots: padding is right
    only if the stored array still describes the file's chain restricted to its
    first ``len(prob_in_model)`` slots, and a checkpoint is written *after* a
    relabeling iteration, which permutes the source axis of every sample.  On a
    chain widened from two slots to four and then relabeled once, padding gives
    ``[1.0, 0.667, 0.0, 0.0]`` where the samples say ``[0.233, 0.433, 0.5,
    0.5]``.  Nothing in the file says which of those two situations produced it,
    so the only repair that is right for every file is the one derived from the
    samples.

    The repair is confined to this boundary on purpose.  :class:`PosteriorChain`
    itself still rejects a mismatched array outright, because in memory the
    mismatch is a bug in the code that built the chain -- exactly the bug
    ``expand_chain`` had -- and repairing it there would hide the next one.

    Parameters
    ----------
    prob_in_model : array_like or None
        Array as stored in the file.  ``None`` -- "never computed" -- stays
        ``None``; a file that recorded no probabilities does not acquire some
        by being read.
    chain : ndarray, shape (n_samples, num_sources, n_params_per_source)
        Samples read from the same file, already in canonical form.
    num_sources : int
        Number of source slots the file declares.
    feather_filepath : str
        Path the array came from, named in the warning so the file that needs
        rewriting can be identified.

    Returns
    -------
    ndarray or None
        The stored array when it has shape ``(num_sources,)``, ``None`` when
        nothing was stored, and otherwise the array recomputed from `chain`.

    Examples
    --------
    >>> from petra.posterior_chain import _prob_in_model_from_file
    >>> chain = np.array([[[1.0], [2.0], [np.nan]], [[3.0], [np.nan], [np.nan]]])
    >>> _prob_in_model_from_file([1.0, 0.5, 0.0], chain, 3, 'ok.feather')
    array([1. , 0.5, 0. ])
    >>> _prob_in_model_from_file(None, chain, 3, 'none.feather') is None
    True
    >>> _prob_in_model_from_file([1.0], chain, 3, 'short.feather')
    array([1. , 0.5, 0. ])
    """
    if prob_in_model is None:
        return None
    prob_in_model = np.asarray(prob_in_model)
    if prob_in_model.shape == (num_sources,):
        return prob_in_model
    repaired = find_prob_in_model(chain, num_sources, eps=0)
    logger.warning(
        "%s stores a prob_in_model of shape %s for a chain of %d source slots, so it "
        "cannot be indexed by source label as it stands; a short one is what "
        "PosteriorChain.expand_chain wrote before it learned to widen the array. The "
        "samples in the file are intact and prob_in_model is a function of them, so it "
        "has been recomputed from them as %s. The chain, cost_dict and every other "
        "field are unchanged, and the file itself is not rewritten; save the chain "
        "again to fix it on disk.",
        feather_filepath, prob_in_model.shape, num_sources, repaired)
    return repaired


@dataclass
class PosteriorChain:
    """
    A dataclass to store the chains, number of sources, and number of parameters per source.

    Parameters
    ----------
    chain : ndarray
        The chain of samples. It may be given as ``(num_samples, num_sources,
        num_params_per_source)``, as ``(num_samples, num_sources *
        num_params_per_source)``, or flat; it is reshaped to the three
        dimensional form on initialization.
    num_sources : int
        The number of sources.
    num_params_per_source : int
        The number of parameters per source.
    trans_dimensional : bool, optional
        Whether the chain has variable number of sources. Default is False.
    prob_in_model : ndarray, optional
        Probability of a source being in the model, one entry per source slot:
        it is indexed by source label, so it must have shape ``(num_sources,)``
        and describe *this* chain.  ``None``, the default, means "not
        computed".
    cost_dict : dict, optional
        A dictionary of total cost for the labeling, keyed by number of sources.
        Defaults to an empty dictionary; an explicit ``None`` is normalized to
        one, so ``chain.cost_dict`` is always a dict and never needs guarding.
        Like `prob_in_model` it describes the labeling this chain is carrying,
        so :meth:`expand_chain` and :meth:`randomize_entries` return a chain
        with an empty one.
    copy : bool, optional
        If True, store a copy of `chain` instead of a view onto it.
        Default is False. See Notes.
    validate_nan_convention : bool, optional
        If True (the default), reject a chain containing a source row that is
        neither all-``NaN`` nor all-finite; see `_validate_nan_convention`.  The
        check is one NaN pass over the array -- 11 ms on a 100k x 15 x 8 chain,
        about 120 ms per gigabyte -- so it is on by default and can be turned
        off for a chain already known to satisfy the convention.  The setting is
        *sticky*: every chain derived from this one -- :meth:`expand_chain`,
        :meth:`randomize_entries`, :func:`petra.relabel.prepare_chain` and the
        per-iteration rebuilds inside :func:`petra.relabel.run_relabeling_loop`
        -- inherits it, because every writer in petra moves whole source rows,
        so a derived chain can only fail the check in exactly the places its
        parent already did.  Without that, the opt-out dead-ended one call later
        on the very error that recommended it.  :meth:`read_feather` takes the
        same keyword rather than inheriting it, because a file is a boundary
        rather than a derivation; see its Notes. This switch only permits
        partial NaN rows; complex and infinite values are always rejected.

    Attributes
    ----------
    chain : ndarray
        Reshaped chain after initialization.

    Raises
    ------
    ValueError
        If `chain` is not a real numeric array whose shape is consistent with
        `num_sources` and `num_params_per_source`, if it holds a source row
        that mixes ``NaN`` with finite values, contains infinity, or if `prob_in_model` does not
        hold exactly `num_sources` probabilities.

    Notes
    -----
    **The chain is not copied by default.** ``__post_init__`` reshapes the array
    that is passed in, and ``numpy`` returns a *view* whenever it can, so
    ``pc[i] = value`` will usually write through to the caller's original array::

        arr = np.zeros((10, 2, 3))
        pc = PosteriorChain(arr, 2, 3)
        pc[0] = 1.0        # arr[0] is now 1.0 as well

    Whether a view or a copy is produced depends on the memory layout of the
    input (a non-contiguous array cannot be reshaped in place), so the aliasing
    is not even consistent. Pass ``copy=True`` to get an instance that owns its
    data. The default remains ``False`` because production chains are multi-GB
    and an unconditional copy would double peak memory for every derived chain.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.posterior_chain import PosteriorChain
    >>> arr = np.random.randn(50, 2, 4)
    >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=4)
    >>> pc.chain.shape
    (50, 2, 4)
    >>> pc.cost_dict
    {}
    >>> PosteriorChain(arr, 2, 4, cost_dict=None).cost_dict   # None is normalized away
    {}
    >>> pc = PosteriorChain(arr, num_sources=4, num_params_per_source=2)
    Traceback (most recent call last):
        ...
    ValueError: chain has shape (50, 2, 4), which is inconsistent with num_sources=4 and num_params_per_source=2; expected shape (n_samples, 4, 2).
    >>> PosteriorChain(arr, 2, 4, prob_in_model=np.array([1.0]))
    Traceback (most recent call last):
        ...
    ValueError: prob_in_model has shape (1,), but it holds one inclusion probability per source slot and is indexed by source label, so a chain with num_sources=2 needs shape (2,). Pass prob_in_model=None, or recompute it from the samples with petra.utils.find_prob_in_model(chain, 2, eps=0); a Feather file whose stored array is this shape is repaired that way on read.
    """

    chain: np.ndarray
    num_sources: int
    num_params_per_source: int
    trans_dimensional: bool = False
    prob_in_model: np.ndarray | None = None
    cost_dict: dict = field(default_factory=dict)
    copy: bool = False
    validate_nan_convention: bool = True

    def __post_init__(self) -> None:
        """
        Validate the arguments and reshape `chain` to its canonical form.

        Each rule is delegated to a named helper so that it can be read, and
        tested, on its own: `_validate_dimension` for the two counts,
        `_validate_numeric_array` for the dtype, `_validate_chain_shape` for
        the layout, `_validate_prob_in_model` for the width of the inclusion
        probabilities and `_validate_nan_convention` for the all-or-nothing NaN
        rule.  Only once the first three pass is the chain reshaped, so a
        mismatched shape is reported rather than silently reinterpreted; the
        NaN rule is checked after the reshape, which is what lets it name the
        offending sample and source.  `cost_dict` is normalized the same way,
        by `_normalize_cost_dict`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `num_sources` or `num_params_per_source` is not a positive
            integer, if `chain` is not a numeric array of a shape consistent
            with them, if it holds a source row that mixes ``NaN`` with finite
            values, or if `prob_in_model` does not hold exactly `num_sources`
            probabilities.
        """
        self.cost_dict = _normalize_cost_dict(self.cost_dict)

        self.num_sources = _validate_dimension("num_sources", self.num_sources)
        self.num_params_per_source = _validate_dimension("num_params_per_source",
                                                         self.num_params_per_source)
        _validate_prob_in_model(self.prob_in_model, self.num_sources)

        chain = _validate_numeric_array(self.chain, self.num_sources, self.num_params_per_source)
        _validate_chain_shape(chain, self.num_sources, self.num_params_per_source)

        chain = chain.reshape(-1, self.num_sources, self.num_params_per_source)
        if self.validate_nan_convention:
            _validate_nan_convention(chain)
        self.chain = chain.copy() if self.copy else chain

    def __repr__(self) -> str:
        """
        Return the underlying array's repr, so a chain prints like its data.

        Returns
        -------
        str
            ``repr`` of `chain`.
        """
        return self.chain.view().__repr__()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the stored chain.

        Returns
        -------
        tuple of int
            ``(n_samples, num_sources, num_params_per_source)``.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> PosteriorChain(np.zeros((10, 2, 3)), 2, 3).shape
        (10, 2, 3)
        """
        return self.chain.shape

    def __getitem__(self, index: Any) -> np.ndarray:
        """
        Index the stored chain directly, i.e. ``pc[i] is pc.chain[i]``.

        Parameters
        ----------
        index : object
            Any NumPy index expression.

        Returns
        -------
        ndarray
            The selected part of `chain`.  Basic indexing returns a *view*, so
            writing through it modifies the chain.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> PosteriorChain(np.arange(6.0), 2, 3)[0, 1, 2]
        np.float64(5.0)
        """
        return self.chain[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        """
        Assign into the stored chain, i.e. ``pc[i] = v`` writes ``pc.chain[i]``.

        Because the chain is not copied by default, this usually writes through
        to the array the caller constructed the instance from; see the class
        Notes.

        Parameters
        ----------
        index : object
            Any NumPy index expression.
        value : object
            Value to assign, broadcast by NumPy as usual.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> pc = PosteriorChain(np.zeros((2, 2, 1)), 2, 1, copy=True)
        >>> pc[0] = 1.0
        >>> pc.chain[0].ravel()
        array([1., 1.])
        """
        self.chain[index] = value

    def get_chain(self, burn: int = 0, thin: int = 1) -> np.ndarray:
        """
        Retrieve a sub-chain by discarding initial samples and thinning.

        Parameters
        ----------
        burn : int, default 0
            Number of initial samples to discard.
        thin : int, default 1
            Keep every `thin`-th sample.

        Returns
        -------
        ndarray
            Array of shape (ceil((n_samples - burn) / thin), n_sources, n_params_per_source).

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.arange(60).reshape(20, 3, 1)
        >>> pc = PosteriorChain(arr, num_sources=3, num_params_per_source=1)
        >>> pc.get_chain(burn=5, thin=2).shape
        (8, 3, 1)
        """
        return self.chain[burn::thin]

    def get_valid_chain_entry(self, entry_index: int, burn: int = 0, thin: int = 1) -> np.ndarray:
        """
        Retrieve the samples of a single source that contain no NaNs.

        Parameters
        ----------
        entry_index : int
            Index of the source (second axis of `chain`) to extract.
        burn : int, default 0
            Number of initial samples to discard before filtering.
        thin : int, default 1
            Keep every `thin`-th sample after discarding `burn` samples.

        Returns
        -------
        ndarray
            Array of shape (n_valid_samples, n_params_per_source) containing
            only those samples in which source `entry_index` is present, as
            decided by :func:`petra.utils.source_present`.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.array([
        ...     [[np.nan, np.nan], [2.0, 3.0]],
        ...     [[4.0, 5.0],       [6.0, 7.0]],
        ... ])
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=2)
        >>> pc.get_valid_chain_entry(0).shape
        (1, 2)
        """
        chain = self.get_chain(burn=burn, thin=thin)[:, entry_index, :]
        return chain[source_present(chain)]

    def expand_chain(self, max_num_sources: int) -> "PosteriorChain":
        """
        Expand the chain to a larger number of sources, padding with NaNs.

        Parameters
        ----------
        max_num_sources : int
            Desired number of sources after expansion.

        Returns
        -------
        PosteriorChain
            New instance with `chain.shape == (n_samples, max_num_sources, n_params_per_source)`,
            an empty `cost_dict`, and -- when this chain carries one -- a
            `prob_in_model` recomputed from the widened samples without
            clipping.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.random.randn(10, 2, 3)
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=3)
        >>> pc2 = pc.expand_chain(4)
        >>> pc2.chain.shape
        (10, 4, 3)

        The added slots are absent from every sample, so their inclusion
        probability is exactly zero, and the slots that were already there keep
        the probability their samples imply:

        >>> gappy = np.array([[[1.0], [2.0]], [[3.0], [np.nan]]])
        >>> pc = PosteriorChain(gappy, 2, 1, trans_dimensional=True,
        ...                     prob_in_model=np.array([1.0, 0.5]),
        ...                     cost_dict={2: -3.5})
        >>> pc.expand_chain(4).prob_in_model
        array([1. , 0.5, 0. , 0. ])

        The cost is *not* carried over: ``cost_dict[2]`` was the cost of the
        best labeling of a two-slot chain, and this chain has four slots.

        >>> pc.expand_chain(4).cost_dict
        {}
        >>> pc.cost_dict                        # the caller's chain is untouched
        {2: -3.5}
        """
        expanded_chain = np.zeros((self.chain.shape[0], max_num_sources, self.num_params_per_source)) + np.nan
        expanded_chain[:, :self.num_sources, :] = self.chain
        # `prob_in_model` is indexed by source label, so it has to grow with the chain.
        # Carrying the caller's array through unchanged produced a chain whose
        # `prob_in_model` was shorter than its own `num_sources` -- and since
        # `petra.relabel.prepare_chain` widens on the way into every entry point, that
        # chain reached the fits, the checkpoints and the feather files.
        #
        # Recomputed rather than right-padded with zeros.  Padding is correct only when
        # the caller's array already satisfies the package contract for the *input*
        # chain, and two reachable inputs do not: an array clipped by the default
        # `find_prob_in_model(chain, n)` reports `eps` for a slot that is empty in every
        # sample, and an array read from a legacy Feather file can hold a NaN.  Padding
        # either one produces a widened chain whose first entries still lie and whose
        # last entries are exactly 0 -- inconsistent with each other, never mind with the
        # samples.  The pass this costs is 17 ms on a 100k x 15 x 8 chain, the same as
        # the array copy two lines up that this method already pays.
        prob_in_model = (None if self.prob_in_model is None
                         else find_prob_in_model(expanded_chain, max_num_sources, eps=0))
        # `cost_dict` is keyed by number of sources and cannot be re-derived without
        # redoing the fit, so widening drops it rather than reporting the cost of a
        # labeling at a width this chain no longer has.
        return PosteriorChain(expanded_chain, max_num_sources, self.num_params_per_source, True,
                              prob_in_model, {},
                              validate_nan_convention=self.validate_nan_convention)

    def randomize_entries(self, seed: int | None = None) -> "PosteriorChain":
        """
        Shuffle the entries (second dimension) of each sample without repetition.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible shuffling.

        Returns
        -------
        PosteriorChain
            New instance with entries randomized per sample, an empty
            `cost_dict`, and -- when this chain carries one -- a
            `prob_in_model` recomputed from the shuffled samples without
            clipping.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.arange(12).reshape(3, 2, 2)
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=2)
        >>> pc2 = pc.randomize_entries(seed=0)
        >>> pc2.chain.shape
        (3, 2, 2)

        Shuffling moves absent sources between slots, so the inclusion
        probabilities move with them -- here the only present source lands in
        slot 1 in both samples -- and the cost of the labeling that was just
        scrambled does not come along:

        >>> gappy = np.array([[[1.0], [np.nan]], [[2.0], [np.nan]]])
        >>> pc = PosteriorChain(gappy, 2, 1, trans_dimensional=True,
        ...                     prob_in_model=np.array([1.0, 0.0]),
        ...                     cost_dict={2: -3.5})
        >>> pc.randomize_entries(seed=5).prob_in_model
        array([0., 1.])
        >>> pc.randomize_entries(seed=5).cost_dict
        {}
        >>> pc.cost_dict                        # the caller's chain is untouched
        {2: -3.5}
        """
        # Shuffle along axis=1 (the second dimension) for each index in the first dimension
        rng = np.random.default_rng(seed=seed)

        # Make a copy of the chain to avoid modifying the original
        chain = self.chain.copy()

        for i in range(self.chain.shape[0]):
            rng.shuffle(chain[i])
        # A shuffle permutes the source axis per sample, so slot `i` of the result is not
        # populated by the same samples as slot `i` of the input: the array carried over
        # unchanged described the labeling that no longer exists.  Recomputed rather than
        # permuted because there is no single permutation to apply -- each sample got its
        # own -- and unclipped, so a slot that ended up empty in every sample reads 0.
        prob_in_model = (None if self.prob_in_model is None
                         else find_prob_in_model(chain, self.num_sources, eps=0))
        # And `cost_dict` is dropped for the same reason, one step further: a cost is
        # the score of one specific labeling, the shuffle has just replaced it, and
        # unlike `prob_in_model` it cannot be re-derived without redoing the fit.
        # Carrying it would let `run_relabeling_loop`'s resume seed take the score of a
        # labeling this chain no longer has as the cost the run must beat.
        return PosteriorChain(chain, self.num_sources, self.num_params_per_source,
                              self.trans_dimensional, prob_in_model, {},
                              validate_nan_convention=self.validate_nan_convention)

    def to_feather(self, feather_filepath: str) -> None:
        """
        Save the PosteriorChain to a Feather file including chain data and metadata.

        The chain itself is written as one column per (source, parameter) pair.
        Everything else -- `num_sources`, `num_params_per_source`,
        `trans_dimensional`, `prob_in_model` and `cost_dict` -- is stored in the
        Feather schema metadata, so that arrays of any length (including
        ``None``) survive the round trip untouched.

        That is every field that describes the chain.  The remaining two, `copy`
        and `validate_nan_convention`, describe how the constructor was called
        rather than what was constructed, and reading is its own construction:
        :meth:`read_feather` takes `validate_nan_convention` as an argument, so
        a chain built with the check turned off can be read back with it turned
        off too.

        Parameters
        ----------
        feather_filepath : str
            Path to the output Feather file.

        Returns
        -------
        None

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.random.randn(5, 2, 3)
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=3)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = os.path.join(tmpdir, 'test_pc.feather')
        ...     pc.to_feather(path)
        ...     os.path.exists(path)
        True
        """
        flat_chain = self.chain.reshape(self.chain.shape[0], self.num_sources * self.num_params_per_source)
        # pyarrow warns about non-string column names, so name them explicitly.
        df = pd.DataFrame(flat_chain, columns=[str(i) for i in range(flat_chain.shape[1])])

        metadata = {
            "version": FEATHER_METADATA_VERSION,
            "num_sources": int(self.num_sources),
            "num_params_per_source": int(self.num_params_per_source),
            "trans_dimensional": bool(self.trans_dimensional),
            "prob_in_model": (None if self.prob_in_model is None
                              else np.asarray(self.prob_in_model).tolist()),
            "cost_dict": _encode_cost_dict(self.cost_dict),
        }

        table = pa.Table.from_pandas(df, preserve_index=False)
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata[FEATHER_METADATA_KEY] = json.dumps(metadata).encode("utf-8")
        table = table.replace_schema_metadata(schema_metadata)
        feather.write_feather(table, feather_filepath)
        logger.debug("Wrote PosteriorChain with shape %s to %s", self.chain.shape, feather_filepath)

    @staticmethod
    def read_feather(feather_filepath: str,
                     validate_nan_convention: bool = True) -> "PosteriorChain":
        """
        Load a PosteriorChain from a Feather file with stored chain data and metadata.

        Files written by `to_feather` are restored losslessly. Files written by
        the older version of `to_feather`, which stored the metadata in padded
        columns, are still readable; for those, `cost_dict` is empty (it was
        never written) and `prob_in_model` is recovered by trusting the stored
        `num_sources`, since the legacy padding is otherwise
        indistinguishable from a trailing ``NaN``. Legacy files written before
        probabilities were stored return ``prob_in_model=None``.

        A file whose stored `prob_in_model` cannot describe its own chain --
        the file the pre-fix `expand_chain` wrote, and the legacy file whose
        `prob_in_model` column is shorter than `num_sources` because it was
        padded to the *sample* count -- is repaired rather than refused: the
        array is recomputed from the samples in the file and a warning naming
        the file is logged. See `_prob_in_model_from_file` for why recomputing
        beats padding and why the repair stops at this boundary.

        Parameters
        ----------
        feather_filepath : str
            Path to the Feather file produced by `to_feather`.
        validate_nan_convention : bool, optional
            Forwarded to :class:`PosteriorChain`; see Notes for why the flag is
            asked of the reader instead of being read out of the file.

        Returns
        -------
        PosteriorChain
            A new PosteriorChain instance reconstructed from the file.

        Raises
        ------
        ValueError
            If the file is not a PosteriorChain Feather file or its chain
            columns do not reshape to the stored dimensions, or -- unless
            `validate_nan_convention` is ``False`` -- if it holds a source row
            mixing ``NaN`` with finite values.

        Notes
        -----
        `to_feather` persists what *describes* the chain -- its dimensions, its
        `trans_dimensional` flag, `prob_in_model` and `cost_dict` -- and not the
        two fields that only record how the constructor was called, `copy` and
        `validate_nan_convention`.  Reading is a construction, so it takes the
        construction option itself.

        The flag is deliberately not stored in the file and honoured on read.  A
        Feather file is exactly the boundary the NaN check exists for: it may
        have been written on another machine, by another version, or by an
        opted-out chain whose owner is not the person now reading it, and a file
        that could switch off the validation of its own contents would leave
        that reader with no check at all.  Nothing is lost silently -- a chain
        that really does violate the convention fails loudly here, naming the
        offending row and this keyword.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.random.randn(5, 2, 3)
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=3)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = os.path.join(tmpdir, 'test_pc.feather')
        ...     pc.to_feather(path)
        ...     pc2 = PosteriorChain.read_feather(path)
        >>> isinstance(pc2, PosteriorChain)
        True
        >>> pc2.chain.shape
        (5, 2, 3)
        >>> bool(np.array_equal(pc.chain, pc2.chain))
        True

        A chain constructed with ``validate_nan_convention=False`` can be read
        back the same way, so the documented opt-out survives a round trip:

        >>> half = np.array([[[1.0, np.nan]]])
        >>> pc = PosteriorChain(half, 1, 2, validate_nan_convention=False)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = os.path.join(tmpdir, 'half.feather')
        ...     pc.to_feather(path)
        ...     back = PosteriorChain.read_feather(path, validate_nan_convention=False)
        >>> bool(np.array_equal(back.chain, half, equal_nan=True))
        True
        """
        table = feather.read_table(feather_filepath)
        raw_metadata = (table.schema.metadata or {}).get(FEATHER_METADATA_KEY)
        df = table.to_pandas()

        if raw_metadata is None:
            return PosteriorChain._from_legacy_dataframe(df, feather_filepath,
                                                         validate_nan_convention)

        metadata = json.loads(raw_metadata.decode("utf-8"))
        num_sources = int(metadata["num_sources"])
        num_params_per_source = int(metadata["num_params_per_source"])

        chain = df.to_numpy()
        _validate_chain_shape(chain, num_sources, num_params_per_source)
        chain = chain.reshape(len(df), num_sources, num_params_per_source)
        return PosteriorChain(
            chain,
            num_sources,
            num_params_per_source,
            bool(metadata["trans_dimensional"]),
            _prob_in_model_from_file(metadata["prob_in_model"], chain, num_sources,
                                     feather_filepath),
            _decode_cost_dict(metadata.get("cost_dict")),
            validate_nan_convention=validate_nan_convention,
        )

    @staticmethod
    def _from_legacy_dataframe(df: pd.DataFrame, feather_filepath: str,
                               validate_nan_convention: bool = True) -> "PosteriorChain":
        """
        Rebuild a PosteriorChain from a Feather file written before the metadata schema.

        Parameters
        ----------
        df : DataFrame
            Contents of the legacy Feather file.
        feather_filepath : str
            Path the DataFrame was read from, used for the log message.
        validate_nan_convention : bool, optional
            Forwarded to :class:`PosteriorChain`; see
            :meth:`PosteriorChain.read_feather`.

        Returns
        -------
        PosteriorChain
            The reconstructed chain, with an empty `cost_dict` and, when the
            `prob_in_model` column is too short to hold one probability per
            source slot, an array recomputed from the samples; see
            `_prob_in_model_from_file`. If probabilities were never stored,
            ``prob_in_model`` remains ``None``.

        Raises
        ------
        ValueError
            If the file has neither the metadata nor the legacy columns.
        """
        required = ("num_sources", "num_params_per_source", "transdimensional")
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(
                f"{feather_filepath!r} is not a PosteriorChain Feather file: it has no petra "
                f"schema metadata and is missing the legacy column(s) {missing}."
            )
        logger.info("Reading %s in the legacy format; cost_dict was never stored and will be "
                    "empty.", feather_filepath)

        num_sources = int(df['num_sources'][0])
        num_params_per_source = int(df['num_params_per_source'][0])
        transdimensional = bool(df['transdimensional'][0])

        # The legacy writer right-padded prob_in_model with NaNs to the number of
        # samples, so the column cannot say where the array ended -- but the array is
        # one probability per source slot, and `num_sources` is stored, so the length
        # is not in fact lost.  Cutting at the first NaN instead would drop a genuine
        # trailing NaN and hand back a prob_in_model shorter than the chain it
        # describes, which the constructor rejects.
        #
        # The slice can still come up short: the column is only `n_samples` long, so a
        # file with fewer samples than source slots cannot hold one probability per
        # slot at all.  That is not a length this reader can guess, and it is why the
        # slice goes through `_prob_in_model_from_file` rather than straight into the
        # constructor -- the samples are right here, and the array is a function of
        # them.
        stored_prob_in_model = (df['prob_in_model'].to_numpy()[:num_sources]
                                if 'prob_in_model' in df.columns else None)

        df = df.drop(columns=[name for name in _LEGACY_METADATA_COLUMNS if name in df.columns])
        chain = df.to_numpy()
        _validate_chain_shape(chain, num_sources, num_params_per_source)
        chain = chain.reshape(len(df), num_sources, num_params_per_source)
        return PosteriorChain(chain, num_sources, num_params_per_source, transdimensional,
                              _prob_in_model_from_file(stored_prob_in_model, chain,
                                                       num_sources, feather_filepath),
                              validate_nan_convention=validate_nan_convention)
