"""PosteriorChain construction, derived chains, the Feather format, and the presence helpers in petra.utils."""

import json
import logging
import re
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from petra.fits import uni_normal_fit
from petra.initialization import label_chain_by_histogram
from petra.posterior_chain import FEATHER_METADATA_KEY, PosteriorChain, _encode_cost_dict
from petra.utils import find_prob_in_model, source_present
from tests.helpers import LISA_EXAMPLE


def _partially_nan_chain() -> np.ndarray:
    """A (4, 2, 3) chain with a legal absent row at sample 1, source 0 and one NaN at sample 3, source 1."""
    chain = np.arange(24, dtype=float).reshape(4, 2, 3)
    chain[1, 0, :] = np.nan
    chain[3, 1, 2] = np.nan
    return chain


def _legal_chain(seed: int = 0, n_samples: int = 40, n_sources: int = 4, n_params: int = 5) -> np.ndarray:
    """A chain whose rows are all NaN or all finite; slot 0 is never present and slot 1 always."""
    rng = np.random.default_rng(seed)
    chain = rng.standard_normal((n_samples, n_sources, n_params))
    chain[rng.random((n_samples, n_sources)) < 0.4] = np.nan
    chain[:, 0, :] = np.nan
    chain[:, 1, :] = rng.standard_normal((n_samples, n_params))
    return chain


def _refuse_non_standard_json(token: str) -> None:
    """``parse_constant`` hook: strict JSON has no NaN or Infinity."""
    raise ValueError(f"non-standard JSON token {token!r}")


def _stored_metadata(path) -> dict:
    """Parse the metadata `to_feather` wrote, refusing bare NaN and Infinity tokens."""
    raw = feather.read_table(str(path)).schema.metadata[FEATHER_METADATA_KEY]
    return json.loads(raw.decode("utf-8"), parse_constant=_refuse_non_standard_json)


def _write_legacy(path, flat: np.ndarray, num_sources: int, num_params: int, prob_column=None) -> None:
    """Write `flat` in the legacy padded-column layout, with a `prob_in_model` column if given."""
    df = pd.DataFrame(flat, columns=[str(i) for i in range(flat.shape[1])])
    df["num_sources"] = num_sources
    df["num_params_per_source"] = num_params
    df["transdimensional"] = 1
    if prob_column is not None:
        df["prob_in_model"] = prob_column
    df.to_feather(path)


def _write_with_metadata(path, flat: np.ndarray, metadata: dict) -> None:
    """Write `flat` under hand-written petra metadata, serialized with json.dumps defaults (bare tokens allowed)."""
    table = pa.Table.from_pandas(pd.DataFrame(flat, columns=[str(i) for i in range(flat.shape[1])]),
                                 preserve_index=False)
    feather.write_feather(table.replace_schema_metadata({FEATHER_METADATA_KEY: json.dumps(metadata).encode()}), path)


def _write_with_stored_prob(path, chain: np.ndarray, stored_prob_in_model: list) -> None:
    """Write `chain` with to_feather, then replace its stored prob_in_model, as the pre-fix expand_chain left it."""
    PosteriorChain(chain, chain.shape[1], chain.shape[2], trans_dimensional=True).to_feather(path)
    table = feather.read_table(path)
    metadata = dict(table.schema.metadata)
    stored = json.loads(metadata[FEATHER_METADATA_KEY].decode("utf-8"))
    stored["prob_in_model"] = list(stored_prob_in_model)
    metadata[FEATHER_METADATA_KEY] = json.dumps(stored).encode("utf-8")
    feather.write_feather(table.replace_schema_metadata(metadata), path)


LAYOUTS = {
    "3-D": (np.arange(60.0).reshape(10, 2, 3), 2, 3, None),
    "2-D": (np.arange(60.0).reshape(10, 6), 2, 3, None),
    "1-D": (np.arange(60.0), 2, 3, None),
    "transposed": (np.zeros((100, 3, 4)), 4, 3, "inconsistent with num_sources="),
    "bad-column-count": (np.zeros((10, 5)), 2, 2, "columns"),
    "1-D-leftover": (np.zeros(7), 2, 3, re.escape("(7,)")),
    "4-D": (np.zeros((4, 2, 3, 1)), 2, 3, "must have 1, 2, or 3 dimensions"),
}


@pytest.mark.parametrize("chain, num_sources, num_params, anchor", LAYOUTS.values(), ids=LAYOUTS.keys())
def test_construction_reshapes_valid_layouts_and_refuses_bad_shapes(chain, num_sources, num_params, anchor):
    """Valid layouts reshape to (n, S, P) and print as their array; bad ones raise naming the problem."""
    if anchor is not None:
        with pytest.raises(ValueError, match=anchor):
            PosteriorChain(chain, num_sources, num_params)
        return
    pc = PosteriorChain(chain, num_sources, num_params)
    assert pc.shape == (10, 2, 3)
    np.testing.assert_array_equal(pc.chain, np.arange(60.0).reshape(10, 2, 3))
    assert repr(pc) == repr(np.arange(60.0).reshape(10, 2, 3))


@pytest.mark.parametrize("field", ["num_sources", "num_params_per_source"])
@pytest.mark.parametrize("value", [True, 0, 2.0, "3"], ids=["True", "0", "2.0", "str"])
def test_dimensions_must_be_integers_of_at_least_one(field, value):
    """A bool, zero, float or string dimension is refused; a NumPy integer is stored as int."""
    dimensions = {"num_sources": 2, "num_params_per_source": 2}
    with pytest.raises(ValueError, match=f"{field} must be an integer >="):
        PosteriorChain(np.zeros((4, 2, 2)), **{**dimensions, field: value})

    pc = PosteriorChain(np.zeros((4, 2, 2)), **{**dimensions, field: np.int64(2)})
    assert type(getattr(pc, field)) is int


@pytest.mark.parametrize("dtype", ["f2", "f4", np.longdouble, ">f8", "i4", ">i4", "i8", "u1"],
                         ids=["f2", "f4", "longdouble", ">f8", "i4", ">i4", "i8", "u1"])
def test_every_numeric_dtype_is_stored_as_native_float64(tmp_path, dtype):
    """A converted chain is native float64, a copy, can mark a source absent and round-trips through Feather."""
    values = np.arange(24, dtype=float).reshape(4, 2, 3)
    arr = values.astype(dtype)
    snapshot = arr.copy()

    pc = PosteriorChain(arr, 2, 3, trans_dimensional=True)

    assert pc.chain.dtype.type is np.float64 and pc.chain.dtype.isnative
    np.testing.assert_array_equal(pc.chain, values)
    assert not np.shares_memory(pc.chain, arr)
    pc[0, 1] = np.nan
    np.testing.assert_array_equal(find_prob_in_model(pc.chain, 2, eps=0), [1.0, 0.75])
    np.testing.assert_array_equal(arr, snapshot)
    assert arr.dtype == snapshot.dtype

    path = tmp_path / "chain.feather"
    pc.to_feather(path)
    np.testing.assert_array_equal(PosteriorChain.read_feather(path).chain, pc.chain)

    native = values.copy()
    assert np.shares_memory(PosteriorChain(native, 2, 3).chain, native)
    assert not np.shares_memory(PosteriorChain(native, 2, 3, copy=True).chain, native)


NOT_STORABLE = {
    "bool": (np.ones((2, 1, 1), dtype=bool), "chain must be a numeric array"),
    "complex": (np.full((2, 1, 1), 1 + 2j), "chain must be a numeric array"),
    "object": (np.ones((2, 1, 1), dtype=object), "chain must be a numeric array"),
    "str": (np.full((2, 1, 1), "a"), "chain must be a numeric array"),
    "inf": (np.array([[[np.inf]], [[1.0]]]), "must not contain infinite values"),
}


@pytest.mark.parametrize("validate_nan_convention", [True, False])
@pytest.mark.parametrize("chain, anchor", NOT_STORABLE.values(), ids=NOT_STORABLE.keys())
def test_non_numeric_and_infinite_chains_are_refused(chain, anchor, validate_nan_convention):
    """The dtype and inf checks hold whether or not the NaN-convention check is on."""
    with pytest.raises(ValueError, match=anchor):
        PosteriorChain(chain, 1, 1, validate_nan_convention=validate_nan_convention)


def test_a_masked_array_stores_masked_rows_as_nan():
    """Masked rows become absent rows, a mask over part of a row breaks the convention, and str is refused."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(8, 3, 2)) + np.array([0.0, 5.0, 10.0])[None, :, None]
    chain[:4, 2, :] = np.nan
    chain[:2, 1, :] = np.nan
    for masked in (np.ma.array(np.nan_to_num(chain), mask=np.isnan(chain)),
                   np.ma.masked_values(np.where(np.isnan(chain), -999.0, chain), -999.0),
                   np.ma.masked_invalid(chain)):
        data, mask = masked.data.copy(), np.ma.getmaskarray(masked).copy()
        pc = PosteriorChain(masked, 3, 2, trans_dimensional=True)
        assert type(pc.chain) is np.ndarray
        np.testing.assert_array_equal(pc.chain, chain)
        np.testing.assert_array_equal(find_prob_in_model(pc.chain, 3, eps=0), [1.0, 0.75, 0.5])
        np.testing.assert_array_equal(masked.data, data)
        np.testing.assert_array_equal(np.ma.getmaskarray(masked), mask)

    integer = PosteriorChain(np.ma.array(np.arange(8).reshape(2, 2, 2), mask=[[[0, 0], [1, 1]], [[0, 0], [0, 0]]]),
                             2, 2)
    assert integer.chain.dtype == np.float64
    np.testing.assert_array_equal(integer.chain[0, 1], [np.nan, np.nan])

    with pytest.raises(ValueError, match="sample 0, source 1 has 1 NaN"):
        PosteriorChain(np.ma.array(np.zeros((2, 2, 2)), mask=[[[0, 0], [0, 1]], [[0, 0], [0, 0]]]), 2, 2)
    with pytest.raises(ValueError, match="chain must be a numeric array"):
        PosteriorChain(np.ma.array(np.array([["a", "b"]])), 1, 2)


def test_a_partial_nan_row_is_named_and_the_opt_out_is_inherited():
    """The convention error names the row and the readers, the readers act as it says, and derived chains opt out."""
    for layout in (_partially_nan_chain(), _partially_nan_chain().reshape(4, 6)):
        with pytest.raises(ValueError, match="all-or-nothing NaN convention") as excinfo:
            PosteriorChain(layout, 2, 3, trans_dimensional=True)
        message = str(excinfo.value)
        assert "sample 3, source 1 has 1 NaN" in message
        for name in ("find_prob_in_model", "uni_normal_fit", "label_chain_by_histogram",
                     "validate_nan_convention=False"):
            assert name in message, name

    opted_out = PosteriorChain(_partially_nan_chain(), 2, 3, trans_dimensional=True, validate_nan_convention=False)
    prob_in_model = find_prob_in_model(opted_out.get_chain(), 2, eps=0)
    assert prob_in_model[1] == pytest.approx(0.75)
    assert prob_in_model[1] * opted_out.shape[0] == len(opted_out.get_valid_chain_entry(1))
    for derived in (opted_out.expand_chain(4), opted_out.randomize_entries(seed=0)):
        assert derived.validate_nan_convention is False
    assert PosteriorChain(np.zeros((4, 2, 3)), 2, 3).validate_nan_convention is True

    # Sample 29 holds the one partial row: parameter 0 is an outlying 100.0, parameter 1 is NaN.
    rng = np.random.default_rng(0)
    chain = np.empty((30, 1, 2))
    chain[:, 0, 0] = 1.0 + 0.01 * rng.standard_normal(30)
    chain[:, 0, 1] = 2.0 + 0.01 * rng.standard_normal(30)
    chain[-1, 0] = [100.0, np.nan]
    pc = PosteriorChain(chain, 1, 2, validate_nan_convention=False)

    assert not source_present(pc.chain[-1, 0])
    assert find_prob_in_model(pc.chain, 1, eps=0)[0] == pytest.approx(29 / 30)
    uni_means, _ = uni_normal_fit(pc.chain, 1, 0)
    assert uni_means[0] == pytest.approx(chain[:, 0, 0].mean())

    with pytest.warns(UserWarning, match="mixes NaN with finite values"):
        labeled = label_chain_by_histogram(pc, obs_time_yrs=1e-8, low_num_samples=1, num_extra_entries=2)
    assert 100.0 not in labeled.chain[:, :, 0], "the message says the histogram labeler leaves a partial row out"


_OBJECT_KEY = object()
BAD_COST_ENTRIES = [
    ({True: -1.0}, "cost_dict key True"),
    ({2.5: -1.0}, "cost_dict key 2.5"),
    ({float("nan"): -1.0}, "cost_dict key nan"),
    ({(1, 2): -1.0}, "cost_dict key (1, 2)"),
    ({_OBJECT_KEY: -1.0}, f"cost_dict key {_OBJECT_KEY!r}"),
    ({3: True}, "cost_dict[3]"),
    ({3: np.array([1.0, 2.0])}, "cost_dict[3]"),
    ({3: object()}, "cost_dict[3]"),
]


@pytest.mark.parametrize("cost_dict, named", BAD_COST_ENTRIES,
                         ids=["bool-key", "fractional-key", "nan-key", "tuple-key", "object-key", "bool-value",
                              "array-value", "object-value"])
def test_cost_dict_and_prob_in_model_are_checked_on_construction(cost_dict, named):
    """An entry Feather cannot store unchanged is refused naming its key; prob_in_model needs one entry per slot."""
    zeros = np.zeros((4, 2, 1))
    with pytest.raises(ValueError, match=re.escape(named)):
        PosteriorChain(zeros, 2, 1, cost_dict=cost_dict)

    assert PosteriorChain(zeros, 2, 1, cost_dict=None).cost_dict == {}
    encoded = _encode_cost_dict({3.0: -1.0, "total": None})
    assert encoded == [[3, -1.0], ["total", None]] and type(encoded[0][0]) is int
    assert PosteriorChain(zeros, 2, 1, cost_dict={2: np.inf}).cost_dict == {2: np.inf}

    first, second = PosteriorChain(zeros, 2, 1), PosteriorChain(zeros, 2, 1)
    first.cost_dict[2] = -1.0
    assert second.cost_dict == {}

    with pytest.raises(ValueError, match=re.escape("prob_in_model has shape")):
        PosteriorChain(np.zeros((5, 3, 2)), 3, 2, prob_in_model=np.array([0.5, 0.5]))


DERIVATIONS = {
    "expand": lambda pc: pc.expand_chain(4),
    "randomize": lambda pc: pc.randomize_entries(seed=0),
}


@pytest.mark.parametrize("derive", DERIVATIONS.values(), ids=DERIVATIONS.keys())
def test_derived_chains_rederive_prob_in_model_and_drop_costs(derive):
    """prob_in_model is recomputed unclipped if the parent had one, costs are dropped, and rows move whole."""
    chain = np.full((12, 2, 2), np.nan)
    chain[:, 0, :] = np.arange(24.0).reshape(12, 2)
    stored = find_prob_in_model(chain, 2)      # clipped by the default eps: slot 1 reads eps, not 0
    pc = PosteriorChain(chain, 2, 2, prob_in_model=stored, cost_dict={2: -12.5})

    derived = derive(pc)

    width = derived.num_sources
    np.testing.assert_array_equal(derived.prob_in_model, find_prob_in_model(derived.chain, width, eps=0))
    assert not np.array_equal(derived.prob_in_model[:2], stored)
    assert derived.cost_dict == {}
    assert pc.prob_in_model is stored and pc.cost_dict == {2: -12.5}
    assert derive(PosteriorChain(chain, 2, 2)).prob_in_model is None
    np.testing.assert_array_equal(derive(pc).chain, derived.chain)

    present = source_present(derived.chain)
    for k in range(derived.chain.shape[2]):
        np.testing.assert_array_equal(present, ~np.isnan(derived.chain[:, :, k]))
    for before, after in zip(pc.chain, derived.chain):
        assert sorted(map(tuple, before[source_present(before)])) == sorted(map(tuple, after[source_present(after)]))

    if width > pc.num_sources:
        np.testing.assert_array_equal(derived.prob_in_model, [1.0, 0.0, 0.0, 0.0])
        assert derived.trans_dimensional is True
        with pytest.raises(ValueError, match="max_num_sources must be an integer >="):
            pc.expand_chain(2.5)
        with pytest.raises(ValueError, match="cannot be less than the number of entries in the chain"):
            pc.expand_chain(1)
        assert pc.expand_chain(np.int64(3)).num_sources == 3


@pytest.mark.parametrize("burn, thin, name", [
    (-1, 1, "burn"), (None, 1, "burn"), (True, 1, "burn"), (1.5, 1, "burn"), (0, 0, "thin"), (0, True, "thin"),
], ids=["negative-burn", "None-burn", "bool-burn", "float-burn", "zero-thin", "bool-thin"])
def test_get_chain_checks_burn_and_thin_and_valid_entries_drop_absent_rows(burn, thin, name):
    """Invalid burn or thin raises in both readers; valid values slice [burn::thin] and absent rows are dropped."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(30, 2, 1)) + np.array([[0.0], [6.0]])
    chain[::4, 1] = np.nan
    pc = PosteriorChain(chain, 2, 1)

    with pytest.raises(ValueError, match=f"{name} must be an integer >="):
        pc.get_chain(burn=burn, thin=thin)
    with pytest.raises(ValueError, match=f"{name} must be an integer >="):
        pc.get_valid_chain_entry(0, burn=burn, thin=thin)

    np.testing.assert_array_equal(pc.get_chain(burn=np.int64(5), thin=np.int64(2)), chain[5::2])
    kept = chain[2::3, 1]
    entry = pc.get_valid_chain_entry(1, burn=2, thin=3)
    np.testing.assert_array_equal(entry, kept[~np.isnan(kept).any(axis=-1)])
    assert 0 < len(entry) < len(kept)


ROUND_TRIP_OPTIONS = {
    "plain-str": (False, False, str),
    "trans-dimensional-opt-out-Path": (True, True, lambda path: path),
}


@pytest.mark.parametrize("trans_dimensional, opt_out, as_target", ROUND_TRIP_OPTIONS.values(),
                         ids=ROUND_TRIP_OPTIONS.keys())
@pytest.mark.parametrize("n_samples", [0, 5])
@pytest.mark.parametrize("cost_dict", [None, {2: 1.5, "4": "x", 3: None}], ids=["no-costs", "costs"])
@pytest.mark.parametrize("prob_in_model", [None, [np.nan, np.inf, -np.inf]], ids=["no-prob", "non-finite-prob"])
def test_feather_round_trip_keeps_everything_as_strict_json(tmp_path, prob_in_model, cost_dict, n_samples,
                                                            trans_dimensional, opt_out, as_target):
    """Every field comes back equal, keys keep their type, and the metadata is strict JSON."""
    chain = np.arange(n_samples * 6, dtype=float).reshape(n_samples, 3, 2)
    if n_samples:
        chain[1, 0, :] = np.nan
    if n_samples and opt_out:
        chain[3, 1, 1] = np.nan
    prob = None if prob_in_model is None else np.array(prob_in_model)
    pc = PosteriorChain(chain, 3, 2, trans_dimensional, prob, cost_dict, validate_nan_convention=not opt_out)
    path = tmp_path / "chain.feather"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pc.to_feather(as_target(path))
    assert [str(w.message) for w in caught if "column names" in str(w.message)] == []
    metadata = _stored_metadata(path)
    loaded = PosteriorChain.read_feather(as_target(path), validate_nan_convention=not opt_out)

    np.testing.assert_array_equal(loaded.chain, chain)
    assert (loaded.num_sources, loaded.num_params_per_source) == (3, 2)
    assert loaded.trans_dimensional is trans_dimensional
    assert loaded.validate_nan_convention is (not opt_out)
    if prob is None:
        assert loaded.prob_in_model is None and metadata["prob_in_model"] is None
    else:
        np.testing.assert_array_equal(loaded.prob_in_model, prob)
    expected_costs = cost_dict or {}
    assert loaded.cost_dict == expected_costs
    assert [type(key) for key in loaded.cost_dict] == [type(key) for key in expected_costs]


@pytest.mark.parametrize("key, value, named", [
    (2, np.nan, "cost_dict[2]"),
    (2, np.inf, "cost_dict[2]"),
    (2.5, -1.0, "cost_dict key 2.5"),
], ids=["nan", "inf", "bad-key-assigned-later"])
def test_to_feather_refuses_a_cost_it_cannot_write_and_writes_nothing(tmp_path, key, value, named):
    """A non-finite cost or a later bad key raises naming the key, writes no file and leaves the chain usable."""
    pc = PosteriorChain(np.arange(8.0).reshape(4, 2, 1), 2, 1, cost_dict={1: -1.0})
    pc.cost_dict[key] = value
    path = tmp_path / "chain.feather"

    with pytest.raises(ValueError, match=re.escape(named)):
        pc.to_feather(path)

    assert not path.exists()
    del pc.cost_dict[key]
    pc.to_feather(path)
    loaded = PosteriorChain.read_feather(path)
    np.testing.assert_array_equal(loaded.chain, pc.chain)
    assert loaded.cost_dict == {1: -1.0}


def _legacy_with_prob(path, caplog):
    """The legacy layout reads its chain, flag and NaN-padded prob_in_model column."""
    chain = np.arange(12, dtype=float).reshape(3, 2, 2)
    _write_legacy(path, chain.reshape(3, 4), 2, 2, prob_column=[0.5, 0.25, np.nan])
    loaded = PosteriorChain.read_feather(path)
    np.testing.assert_array_equal(loaded.chain, chain)
    assert loaded.trans_dimensional is True
    np.testing.assert_array_equal(loaded.prob_in_model, [0.5, 0.25])
    assert loaded.cost_dict == {}


def _legacy_without_prob(path, caplog):
    """A legacy file may omit the prob_in_model column."""
    chain = np.array([[[1.0, 2.0], [np.nan, np.nan]], [[3.0, 4.0], [5.0, 6.0]]])
    _write_legacy(path, chain.reshape(2, 4), 2, 2)
    loaded = PosteriorChain.read_feather(path)
    np.testing.assert_array_equal(loaded.chain, chain)
    assert loaded.prob_in_model is None and loaded.cost_dict == {}


def _legacy_trailing_nan(path, caplog):
    """The stored num_sources says where a legacy prob_in_model ends, so a real trailing NaN survives."""
    _write_legacy(path, np.arange(12, dtype=float).reshape(3, 4), 2, 2, prob_column=[0.5, np.nan, np.nan])
    loaded = PosteriorChain.read_feather(path)
    assert loaded.prob_in_model.shape == (2,)
    np.testing.assert_array_equal(loaded.prob_in_model, [0.5, np.nan])


def _bare_tokens(path, caplog):
    """Metadata written with bare NaN and Infinity tokens before it became strict JSON still reads."""
    _write_with_metadata(path, np.array([[1.0, 2.0], [3.0, np.nan]]),
                         {"version": 1, "num_sources": 2, "num_params_per_source": 1, "trans_dimensional": True,
                          "prob_in_model": [1.0, float("nan")], "cost_dict": [[2, float("inf")]]})
    loaded = PosteriorChain.read_feather(path)
    np.testing.assert_array_equal(loaded.prob_in_model, [1.0, np.nan])
    assert loaded.cost_dict == {2: np.inf}


def _lisa_example(path, caplog):
    """The shipped LISA example reads whole, with no prob_in_model and no costs."""
    if not LISA_EXAMPLE.is_file():
        pytest.skip(f"the LISA example {LISA_EXAMPLE} is missing")
    loaded = PosteriorChain.read_feather(LISA_EXAMPLE)
    assert loaded.shape == (2000, 20, 8)
    assert loaded.prob_in_model is None and loaded.cost_dict == {}


def _widened_then_relabeled() -> tuple[np.ndarray, np.ndarray]:
    """A 2-slot chain widened to 4 and permuted per sample, with the 2-slot prob_in_model it had before."""
    rng = np.random.default_rng(0)
    narrow = rng.normal(size=(30, 2, 1))
    narrow[::3, 1, :] = np.nan
    widened = np.full((30, 4, 1), np.nan)
    widened[:, :2, :] = narrow
    relabeled = np.stack([widened[i][rng.permutation(4)] for i in range(30)])
    return relabeled, find_prob_in_model(narrow, 2, eps=0)


def _repaired(path, caplog, stored: list, chain: np.ndarray) -> None:
    """Read `path`, whose stored prob_in_model has the wrong length, and check it is recomputed with a warning."""
    with caplog.at_level(logging.WARNING, logger="petra.posterior_chain"):
        loaded = PosteriorChain.read_feather(path)
    truth = find_prob_in_model(chain, chain.shape[1], eps=0)
    np.testing.assert_array_equal(loaded.chain, chain)
    np.testing.assert_array_equal(loaded.prob_in_model, truth)
    assert path.name in caplog.text
    with pytest.raises(ValueError, match=re.escape("prob_in_model has shape")):
        PosteriorChain(chain, chain.shape[1], chain.shape[2], prob_in_model=np.array(stored))


def _short_prob(path, caplog):
    """A short stored prob_in_model is recomputed from the samples, not padded with zeros."""
    chain, stored = _widened_then_relabeled()
    assert not np.array_equal(np.concatenate([stored, np.zeros(2)]), find_prob_in_model(chain, 4, eps=0))
    _write_with_stored_prob(path, chain, stored.tolist())
    _repaired(path, caplog, stored.tolist(), chain)


def _long_prob(path, caplog):
    """A long stored prob_in_model is recomputed from the samples."""
    chain, _ = _widened_then_relabeled()
    stored = [1.0] * 6
    _write_with_stored_prob(path, chain, stored)
    _repaired(path, caplog, stored, chain)


def _fewer_samples_than_slots(path, caplog):
    """A legacy prob column shorter than num_sources, because there are fewer samples than slots, is recomputed."""
    chain = np.arange(12, dtype=float).reshape(2, 3, 2)
    chain[1, 2, :] = np.nan
    _write_legacy(path, chain.reshape(2, 6), 3, 2, prob_column=[1.0, 1.0])
    with caplog.at_level(logging.WARNING, logger="petra.posterior_chain"):
        loaded = PosteriorChain.read_feather(path)
    np.testing.assert_array_equal(loaded.chain, chain)
    np.testing.assert_array_equal(loaded.prob_in_model, [1.0, 1.0, 0.5])
    assert path.name in caplog.text


def _unrelated_file(path, caplog):
    """A Feather file without petra metadata or legacy columns is refused, named as a str."""
    pd.DataFrame({"a": [1.0, 2.0]}).to_feather(path)
    with pytest.raises(ValueError, match="is not a PosteriorChain Feather file") as excinfo:
        PosteriorChain.read_feather(path)
    assert repr(str(path)) in str(excinfo.value) and "Path(" not in str(excinfo.value)


def _width_mismatch_legacy(path, caplog):
    """Legacy dimensions whose S*P disagrees with the columns are refused."""
    _write_legacy(path, np.arange(6.0).reshape(2, 3), 1, 2)
    with pytest.raises(ValueError, match="columns"):
        PosteriorChain.read_feather(path)


def _width_mismatch_metadata(path, caplog):
    """Metadata dimensions whose S*P disagrees with the columns are refused."""
    _write_with_metadata(path, np.arange(6.0).reshape(2, 3),
                         {"version": 1, "num_sources": 1, "num_params_per_source": 2, "trans_dimensional": False,
                          "prob_in_model": None})
    with pytest.raises(ValueError, match="columns"):
        PosteriorChain.read_feather(path)


def _partial_row(path, caplog):
    """A partial-NaN row in a petra file raises unless the reader opts out."""
    pc = PosteriorChain(_partially_nan_chain(), 2, 3, trans_dimensional=True, validate_nan_convention=False)
    pc.to_feather(path)
    with pytest.raises(ValueError, match="sample 3, source 1"):
        PosteriorChain.read_feather(path)
    np.testing.assert_array_equal(PosteriorChain.read_feather(path, validate_nan_convention=False).chain, pc.chain)


def _partial_row_legacy(path, caplog):
    """A partial-NaN row in a legacy file raises unless the reader opts out."""
    _write_legacy(path, _partially_nan_chain().reshape(4, 6), 2, 3, prob_column=[0.75, 1.0, np.nan, np.nan])
    with pytest.raises(ValueError, match="sample 3, source 1"):
        PosteriorChain.read_feather(path)
    loaded = PosteriorChain.read_feather(path, validate_nan_convention=False)
    np.testing.assert_array_equal(loaded.chain, _partially_nan_chain())


FEATHER_FILES = [_legacy_with_prob, _legacy_without_prob, _legacy_trailing_nan, _bare_tokens, _lisa_example,
                 _short_prob, _long_prob, _fewer_samples_than_slots, _unrelated_file, _width_mismatch_legacy,
                 _width_mismatch_metadata, _partial_row, _partial_row_legacy]


@pytest.mark.parametrize("case", FEATHER_FILES, ids=[case.__name__.strip("_") for case in FEATHER_FILES])
def test_read_feather_reads_legacy_files_and_repairs_or_refuses_bad_ones(tmp_path, caplog, case):
    """Each file layout is read, repaired with a warning, or refused, as its case function states."""
    case(tmp_path / "chain.feather", caplog)


def test_source_present_and_find_prob_in_model_read_whole_rows():
    """Presence reduces the last axis only and needs the whole row; find_prob_in_model clips a per-slot share."""
    chain = _legal_chain()
    present = source_present(chain)
    assert present.shape == chain.shape[:2]
    np.testing.assert_array_equal(source_present(chain[:, 2, :]), present[:, 2])
    np.testing.assert_array_equal(source_present(chain[7]), present[7])
    assert bool(source_present(chain[7, 2])) == bool(present[7, 2])
    assert not source_present(np.array([1.0, 2.0, np.nan]))

    whole_rows = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, np.nan], [np.nan, np.nan]]])
    np.testing.assert_array_equal(find_prob_in_model(whole_rows, 2, eps=0), [0.5, 0.5])

    legal = _legal_chain(seed=3, n_samples=64, n_sources=6, n_params=8)
    for max_num_sources in range(1, legal.shape[1] + 1):
        expected = np.array([np.sum(~np.isnan(legal[:, i, 0])) / legal.shape[0] for i in range(max_num_sources)])
        for eps in (0.0, 1e-6, 1e-2):
            np.testing.assert_array_equal(find_prob_in_model(legal, max_num_sources, eps=eps),
                                          np.clip(expected, eps, 1 - eps))

    rng = np.random.default_rng(0)
    halves = np.full((20_000, 2, 1), np.nan)
    halves[::2, :, 0] = rng.uniform(0, 1, size=(10_000, 2))
    np.testing.assert_allclose(find_prob_in_model(halves, max_num_sources=2), [0.5, 0.5], rtol=1e-8)
    np.testing.assert_allclose(find_prob_in_model(halves, max_num_sources=1), [0.5], rtol=1e-8)

    one_empty_slot = np.array([[[1.0], [np.nan]], [[2.0], [np.nan]]])
    np.testing.assert_array_equal(find_prob_in_model(one_empty_slot, 2, eps=0), [1.0, 0.0])
    np.testing.assert_array_equal(find_prob_in_model(one_empty_slot, 2, eps=0.5), [0.5, 0.5])
    for eps in (-0.1, 0.6, np.nan):
        with pytest.raises(ValueError, match="eps must be finite and between 0 and 0.5"):
            find_prob_in_model(one_empty_slot, 2, eps=eps)
    with pytest.raises(ValueError, match="exceeds the"):
        find_prob_in_model(np.zeros((5, 2, 3)), max_num_sources=3)
