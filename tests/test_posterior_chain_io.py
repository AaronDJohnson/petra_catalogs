"""Round-trip and argument-forwarding tests for PosteriorChain I/O and samples_io."""
import json
import logging
import os

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest

from petra.make_catalog import relabel_mv_normal
from petra.posterior_chain import FEATHER_METADATA_KEY, PosteriorChain
from petra.relabel import prepare_chain
from petra.samples_io import (load_samples, load_samples_product_space,
                              load_samples_ucbmcmc)
from petra.utils import find_prob_in_model


# ---------------------------------------------------------------------------
# PosteriorChain feather round trips
# ---------------------------------------------------------------------------

def test_default_prob_in_model_none_round_trip(tmp_path):
    """A hand-built chain (prob_in_model=None) must save and reload unchanged."""
    rng = np.random.default_rng(0)
    chain = rng.standard_normal((7, 2, 3))
    pc = PosteriorChain(chain, 2, 3)
    path = os.path.join(tmp_path, "default.feather")

    pc.to_feather(path)  # used to raise TypeError: object of type NoneType has no len()
    loaded = PosteriorChain.read_feather(path)

    assert loaded.prob_in_model is None
    assert loaded.num_sources == 2
    assert loaded.num_params_per_source == 3
    assert loaded.trans_dimensional is False
    assert loaded.cost_dict == {}
    assert np.array_equal(loaded.chain, pc.chain)


def test_nan_chain_round_trip(tmp_path):
    """NaNs in the chain survive the round trip, in place."""
    chain = np.arange(24, dtype=float).reshape(4, 2, 3)
    # Whole rows: the all-or-nothing convention makes a half-NaN row illegal,
    # so a partial one here would be rejected before it ever reached the file.
    chain[1, 0, :] = np.nan
    chain[3, 1, :] = np.nan
    pc = PosteriorChain(chain, 2, 3, trans_dimensional=True)
    path = os.path.join(tmp_path, "nans.feather")

    pc.to_feather(path)
    loaded = PosteriorChain.read_feather(path)

    assert np.array_equal(loaded.chain, pc.chain, equal_nan=True)
    assert np.array_equal(np.isnan(loaded.chain), np.isnan(chain))


def test_prob_in_model_and_cost_dict_preserved(tmp_path):
    """prob_in_model keeps its full length (NaNs included) and cost_dict survives."""
    chain = np.zeros((10, 3, 2))
    prob_in_model = np.array([1.0, np.nan, 0.25])
    cost_dict = {2: -12.5, 3: -9.75}
    pc = PosteriorChain(chain, 3, 2, True, prob_in_model, cost_dict)
    path = os.path.join(tmp_path, "meta.feather")

    pc.to_feather(path)
    loaded = PosteriorChain.read_feather(path)

    assert loaded.prob_in_model.shape == (3,)
    assert np.array_equal(loaded.prob_in_model, prob_in_model, equal_nan=True)
    assert loaded.cost_dict == cost_dict
    assert all(isinstance(key, int) for key in loaded.cost_dict)


def test_cost_dict_none_round_trips_as_an_empty_dict(tmp_path):
    """An explicit cost_dict=None is normalized on construction, so it writes as {}."""
    chain = np.arange(24, dtype=float).reshape(6, 2, 2)
    pc = PosteriorChain(chain, 2, 2, cost_dict=None)
    path = os.path.join(tmp_path, "none_cost.feather")

    pc.to_feather(path)
    loaded = PosteriorChain.read_feather(path)

    assert pc.cost_dict == {}
    assert loaded.cost_dict == {}
    assert np.array_equal(loaded.chain, chain)


def test_cost_dict_is_a_dict_on_every_construction_path(tmp_path):
    """No construction path may hand a reader None: readers index cost_dict unguarded."""
    zeros = np.zeros((4, 2, 2))
    default = PosteriorChain(zeros, 2, 2)
    explicit_none = PosteriorChain(zeros, 2, 2, cost_dict=None)
    populated = PosteriorChain(zeros, 2, 2, cost_dict={2: -1.0})

    path = os.path.join(tmp_path, "paths.feather")
    explicit_none.to_feather(path)
    from_metadata = PosteriorChain.read_feather(path)

    for pc in (default, explicit_none, populated, from_metadata,
               populated.expand_chain(3), populated.randomize_entries(seed=0)):
        assert isinstance(pc.cost_dict, dict)

    assert default.cost_dict == {}
    assert explicit_none.cost_dict == {}
    assert from_metadata.cost_dict == {}
    # Empty rather than inherited: see
    # test_a_derived_chain_does_not_report_the_cost_of_a_labeling_it_no_longer_has.
    assert populated.expand_chain(3).cost_dict == {}
    assert populated.randomize_entries(seed=0).cost_dict == {}


def test_default_cost_dicts_are_not_shared_between_chains():
    """The default comes from a factory, so writing one chain's costs cannot leak."""
    first = PosteriorChain(np.zeros((4, 2, 2)), 2, 2)
    second = PosteriorChain(np.zeros((4, 2, 2)), 2, 2)

    first.cost_dict[2] = -1.0

    assert second.cost_dict == {}


def test_trans_dimensional_preserved(tmp_path):
    """The trans_dimensional flag round-trips in both states."""
    for flag in (False, True):
        pc = PosteriorChain(np.zeros((5, 2, 2)), 2, 2, trans_dimensional=flag)
        path = os.path.join(tmp_path, f"td_{flag}.feather")
        pc.to_feather(path)
        assert PosteriorChain.read_feather(path).trans_dimensional is flag


def test_to_feather_does_not_warn_about_column_names(tmp_path, recwarn):
    """Column names are stringified, so pyarrow stops warning on every save."""
    pc = PosteriorChain(np.zeros((4, 2, 2)), 2, 2)
    pc.to_feather(os.path.join(tmp_path, "cols.feather"))
    assert [w for w in recwarn.list if "column names" in str(w.message)] == []


def test_read_feather_reads_legacy_files(tmp_path):
    """Files written by the old padded-column writer are still readable."""
    chain = np.arange(12, dtype=float).reshape(3, 2, 2)
    prob_in_model = np.array([0.5, 0.25])

    df = pd.DataFrame(chain.reshape(-1, 4))
    df.columns = [str(c) for c in df.columns]
    df['num_sources'] = 2
    df['num_params_per_source'] = 2
    df['transdimensional'] = 1
    df['prob_in_model'] = np.pad(prob_in_model, (0, df.shape[0] - len(prob_in_model)),
                                 constant_values=np.nan)
    path = os.path.join(tmp_path, "legacy.feather")
    df.to_feather(path)

    loaded = PosteriorChain.read_feather(path)
    assert np.array_equal(loaded.chain, chain)
    assert loaded.trans_dimensional is True
    assert np.array_equal(loaded.prob_in_model, prob_in_model)
    assert isinstance(loaded.cost_dict, dict)
    assert loaded.cost_dict == {}


def test_read_feather_rejects_unrelated_files(tmp_path):
    path = os.path.join(tmp_path, "unrelated.feather")
    pd.DataFrame({"a": [1.0, 2.0]}).to_feather(path)
    with pytest.raises(ValueError, match="not a PosteriorChain Feather file"):
        PosteriorChain.read_feather(path)


def test_legacy_prob_in_model_keeps_a_trailing_nan(tmp_path):
    """
    The stored `num_sources` says how long the array is, so the padding is not a guess.

    The legacy writer right-padded `prob_in_model` with NaNs to the number of
    samples, and the reader used to cut at the last non-NaN value -- which drops
    a genuine trailing NaN and hands back an array shorter than the chain it is
    attached to.  There is one probability per source slot, and how many slots
    there are is written in the file.
    """
    chain = np.arange(12, dtype=float).reshape(3, 2, 2)
    prob_in_model = np.array([0.5, np.nan])

    df = pd.DataFrame(chain.reshape(-1, 4))
    df.columns = [str(c) for c in df.columns]
    df['num_sources'] = 2
    df['num_params_per_source'] = 2
    df['transdimensional'] = 1
    df['prob_in_model'] = np.pad(prob_in_model, (0, df.shape[0] - len(prob_in_model)),
                                 constant_values=np.nan)
    path = os.path.join(tmp_path, "legacy_trailing_nan.feather")
    df.to_feather(path)

    loaded = PosteriorChain.read_feather(path)
    assert loaded.prob_in_model.shape == (2,)
    assert np.array_equal(loaded.prob_in_model, prob_in_model, equal_nan=True)


def test_a_legacy_file_with_fewer_samples_than_source_slots_is_still_readable(tmp_path, caplog):
    """
    The legacy column is `n_samples` long, which is not always `num_sources` long.

    Trusting the stored `num_sources` and slicing to it is right whenever the
    column is long enough to hold one probability per slot -- and it is the only
    way to keep a genuine trailing ``NaN``, which the test above pins.  When the
    chain has fewer samples than source slots the column cannot hold them all,
    the slice comes up short, and the file became unreadable rather than
    slightly wrong.  The samples are in the file, so the array is recovered from
    them.
    """
    chain = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    chain[1, 2, :] = np.nan
    df = pd.DataFrame(chain.reshape(2, -1))
    df.columns = [str(c) for c in df.columns]
    df['num_sources'] = 3               # three slots, but only two samples to store in
    df['num_params_per_source'] = 2
    df['transdimensional'] = 1
    df['prob_in_model'] = [1.0, 1.0]
    path = os.path.join(tmp_path, "legacy_short_column.feather")
    df.to_feather(path)

    with caplog.at_level(logging.WARNING, logger="petra.posterior_chain"):
        loaded = PosteriorChain.read_feather(path)

    assert np.array_equal(loaded.chain, chain, equal_nan=True)
    assert np.array_equal(loaded.prob_in_model, find_prob_in_model(chain, 3, eps=0))
    assert loaded.prob_in_model[2] == 0.5
    assert "legacy_short_column.feather" in caplog.text


# ---------------------------------------------------------------------------
# prob_in_model describes the chain it is attached to
# ---------------------------------------------------------------------------

def test_prob_in_model_must_have_one_entry_per_source_slot():
    """
    It is indexed by source label, so a mismatched length is uninterpretable.

    The cost matrix reads ``prob_in_model[i]`` for source row `i`; an array describing a
    different number of sources either raises ``IndexError`` deep inside a fit
    or scores the wrong slot.
    """
    with pytest.raises(ValueError, match="one inclusion probability per source slot"):
        PosteriorChain(np.zeros((5, 3, 2)), 3, 2, prob_in_model=np.array([0.5, 0.5]))


def test_expanding_a_chain_widens_its_prob_in_model():
    """
    A widened chain has to describe itself: the slots it gains are empty everywhere.

    ``expand_chain`` used to pass the caller's array straight through, so a
    chain widened from two slots to four reported ``num_sources == 4`` alongside
    two probabilities -- and :func:`petra.relabel.prepare_chain` widens on the
    way into every entry point, so that chain reached the fits, the checkpoints
    and the Feather files.
    """
    rng = np.random.default_rng(0)
    pc = PosteriorChain(rng.standard_normal((10, 2, 3)), 2, 3, trans_dimensional=True,
                        prob_in_model=np.array([1.0, 1.0]))

    widened = pc.expand_chain(4)

    assert widened.num_sources == 4
    assert len(widened.prob_in_model) == 4
    # Unclipped and reproduced exactly by recomputing from the chain it is on:
    # the padded slots read 0.0, not eps.
    assert np.array_equal(widened.prob_in_model, np.array([1.0, 1.0, 0.0, 0.0]))
    assert np.array_equal(widened.prob_in_model,
                          find_prob_in_model(widened.get_chain(), 4, eps=0))
    # A chain that never carried the array does not acquire one by being padded.
    assert PosteriorChain(np.zeros((4, 2, 3)), 2, 3).expand_chain(4).prob_in_model is None


def test_shuffling_entries_moves_prob_in_model_with_them():
    """
    ``randomize_entries`` permutes the source axis per sample, one permutation each.

    Slot ``i`` of the result is therefore populated by a different set of
    samples than slot ``i`` of the input, so the array carried through unchanged
    described a labeling that no longer exists.  Here source 0 is present in
    every sample and source 1 in none, and after the shuffle both slots hold
    roughly half the sources.
    """
    chain = np.tile(np.array([[[1.0], [np.nan]]]), (12, 1, 1))
    pc = PosteriorChain(chain, 2, 1, trans_dimensional=True,
                        prob_in_model=np.array([1.0, 0.0]))

    shuffled = pc.randomize_entries(seed=0)

    assert np.array_equal(shuffled.prob_in_model,
                          find_prob_in_model(shuffled.get_chain(), 2, eps=0))
    assert not np.array_equal(shuffled.prob_in_model, pc.prob_in_model)
    assert pc.prob_in_model.tolist() == [1.0, 0.0]      # the caller's chain is untouched


#: A chain whose second slot is absent from every sample, so its true inclusion
#: probability is exactly ``0`` and the clip bound is visible.
_ONE_EMPTY_SLOT = np.array([[[1.0], [np.nan]], [[2.0], [np.nan]]])


@pytest.mark.parametrize("build_stored, why", [
    (lambda: find_prob_in_model(_ONE_EMPTY_SLOT, 2),
     "clipped by find_prob_in_model's default eps"),
    (lambda: np.array([1.0, np.nan]),
     "a NaN, as a legacy Feather column can hand back"),
], ids=["clipped", "nan"])
def test_widening_re_derives_prob_in_model_rather_than_keeping_the_input(build_stored, why):
    """
    Padding with zeros is the recomputed array only when the input already is one.

    ``expand_chain`` used to keep the caller's entries and append zeros, on the
    stated ground that the result equals
    ``find_prob_in_model(widened, max_num_sources, eps=0)``.  That holds only if
    the caller's array already satisfies the contract for the *input* chain, and
    two inputs a live pipeline produces do not: an array clipped by the default
    ``find_prob_in_model(chain, n)`` -- the spelling every docstring shows, and
    what every entry point's own ``eps`` produces -- and an array read back from
    a legacy file, which can carry a ``NaN``.  Padding the clipped one is
    self-contradicting on its face: slot 1 is absent from every sample and reads
    ``eps``, while slots 2 and 3 are absent from every sample and read ``0``.
    """
    stored = build_stored()
    pc = PosteriorChain(_ONE_EMPTY_SLOT, 2, 1, trans_dimensional=True, prob_in_model=stored)

    widened = pc.expand_chain(4)

    padded = np.concatenate([stored, np.zeros(2)])
    assert not np.array_equal(padded, widened.prob_in_model, equal_nan=True), (
        f"the fixture no longer holds {why}"
    )
    assert np.array_equal(widened.prob_in_model, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(widened.prob_in_model,
                          find_prob_in_model(widened.get_chain(), 4, eps=0))
    assert pc.prob_in_model is stored        # the caller's array is untouched


def test_a_derived_chain_does_not_report_the_cost_of_a_labeling_it_no_longer_has():
    """
    ``cost_dict`` is keyed by source count, so it goes stale the same way ``prob_in_model``
    does.

    A cost is the score of one specific labeling at one specific width.
    Widening changes the width, so no key names the chain's own any more;
    shuffling replaces the labeling outright, so the score belongs to something
    that is gone.  Unlike ``prob_in_model`` a cost cannot be re-derived without
    redoing the fit that produced it, so it is dropped rather than corrected.
    """
    chain = np.array([[[1.0], [2.0]], [[3.0], [np.nan]]])
    pc = PosteriorChain(chain, 2, 1, trans_dimensional=True, cost_dict={2: -12.5})

    assert pc.expand_chain(4).cost_dict == {}
    assert pc.randomize_entries(seed=0).cost_dict == {}
    assert pc.cost_dict == {2: -12.5}       # the caller's chain is untouched


def _write_short_prob_in_model_file(path, chain, num_sources, stored_prob_in_model):
    """
    Write the file the pre-fix ``expand_chain`` produced: a chain plus a short array.

    ``expand_chain`` widened the chain without widening `prob_in_model`, and
    ``to_feather`` wrote that pair out faithfully, so the inconsistency survived
    on disk.  The metadata is edited after the fact because the constructor now
    refuses to build such a chain in memory at all -- which is the point: the
    files exist, and the objects can no longer be made.
    """
    PosteriorChain(chain, num_sources, chain.shape[2], trans_dimensional=True,
                   prob_in_model=find_prob_in_model(chain, num_sources, eps=0)
                   ).to_feather(path)
    table = feather.read_table(path)
    metadata = dict(table.schema.metadata)
    stored = json.loads(metadata[FEATHER_METADATA_KEY].decode("utf-8"))
    stored["prob_in_model"] = list(stored_prob_in_model)
    metadata[FEATHER_METADATA_KEY] = json.dumps(stored).encode("utf-8")
    feather.write_feather(table.replace_schema_metadata(metadata), path)
    return path


def test_read_feather_repairs_metadata_that_contradicts_its_own_chain(tmp_path, caplog):
    """
    The file is a run already on disk, so it is repaired rather than refused.

    Rejecting it made every checkpoint the pre-fix ``expand_chain`` fed into
    ``to_feather`` unloadable -- ``resume_from`` hard-failing on precisely the
    files the bug produced, over metadata, with the samples intact beside it.
    ``prob_in_model`` is a function of those samples by contract, so it is
    recovered from them, and the repair is announced: the warning names the file
    so the run that has to be saved again can be found.
    """
    chain = np.zeros((6, 4, 1))
    path = _write_short_prob_in_model_file(
        os.path.join(tmp_path, "short_prob.feather"), chain, 4, [1.0, 1.0])

    with caplog.at_level(logging.WARNING, logger="petra.posterior_chain"):
        loaded = PosteriorChain.read_feather(path)

    assert np.array_equal(loaded.prob_in_model, find_prob_in_model(chain, 4, eps=0))
    assert np.array_equal(loaded.chain, chain)
    assert "short_prob.feather" in caplog.text


def test_a_short_prob_in_model_is_recomputed_from_the_chain_not_padded_with_zeros(tmp_path):
    """
    Padding is right only for a file that was widened and never relabeled since.

    Zero is the answer for a slot that is ``NaN`` in every sample, but a
    checkpoint is written *after* a relabeling iteration, and relabeling permutes
    the source axis of every sample: the empty slots move.  Here a two-slot chain
    was widened to four and relabeled once, so padding the stored array reports
    the two original slots' probabilities and two zeros, while the samples in the
    file say something else entirely.  Nothing in the file distinguishes the two
    situations, so the repair has to be the one that is right for both.
    """
    rng = np.random.default_rng(0)
    narrow = rng.normal(size=(30, 2, 1))
    narrow[::3, 1, :] = np.nan
    stored = find_prob_in_model(narrow, 2, eps=0)

    widened = np.full((30, 4, 1), np.nan)
    widened[:, :2, :] = narrow
    permutations = np.array([rng.permutation(4) for _ in range(30)])
    relabeled = np.stack([widened[i][permutations[i]] for i in range(30)])

    path = _write_short_prob_in_model_file(
        os.path.join(tmp_path, "widened_then_relabeled.feather"), relabeled, 4, stored)
    loaded = PosteriorChain.read_feather(path)

    truth = find_prob_in_model(relabeled, 4, eps=0)
    padded = np.concatenate([stored, np.zeros(2)])
    assert not np.array_equal(padded, truth), "the fixture no longer distinguishes the two"
    assert np.array_equal(loaded.prob_in_model, truth)


def test_the_constructor_still_refuses_the_pair_the_reader_repairs():
    """
    The repair is a file-boundary concession, not a relaxation of the invariant.

    In memory a `prob_in_model` that cannot describe its chain is a bug in the
    code that built it -- exactly the bug ``expand_chain`` had -- and repairing
    it there would hide the next one.  On disk it is a run that already
    happened.
    """
    with pytest.raises(ValueError, match=r"needs shape \(4,\)"):
        PosteriorChain(np.zeros((6, 4, 1)), 4, 1, prob_in_model=np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# PosteriorChain validation
# ---------------------------------------------------------------------------

def test_transposed_axes_raise_value_error():
    """(n, 3, 4) with num_sources=4 used to be silently reshaped."""
    with pytest.raises(ValueError, match=r"expected shape \(n_samples, 4, 3\)"):
        PosteriorChain(np.zeros((100, 3, 4)), num_sources=4, num_params_per_source=3)


def test_bad_column_count_raises_value_error():
    with pytest.raises(ValueError, match="columns"):
        PosteriorChain(np.zeros((10, 5)), num_sources=2, num_params_per_source=2)


def test_non_numeric_chain_raises_value_error():
    with pytest.raises(ValueError, match="numeric array"):
        PosteriorChain(np.array([["a", "b"]]), num_sources=1, num_params_per_source=2)


def test_flat_and_two_dimensional_chains_still_reshape():
    assert PosteriorChain(np.zeros((10, 4)), 2, 2).shape == (10, 2, 2)
    assert PosteriorChain(np.zeros(40), 2, 2).shape == (10, 2, 2)


def test_copy_flag_controls_aliasing():
    arr = np.zeros((4, 2, 2))
    PosteriorChain(arr, 2, 2)[0] = 1.0
    assert arr[0, 0, 0] == 1.0  # documented default: a view

    arr2 = np.zeros((4, 2, 2))
    PosteriorChain(arr2, 2, 2, copy=True)[0] = 1.0
    assert arr2[0, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# The all-or-nothing NaN convention
# ---------------------------------------------------------------------------

def _partially_nan_chain():
    """A convention-respecting chain with one row spoiled: sample 3, source 1."""
    chain = np.arange(24, dtype=float).reshape(4, 2, 3)
    chain[1, 0, :] = np.nan     # legal: a whole absent row
    chain[3, 1, 2] = np.nan     # illegal: one parameter of a populated row
    return chain


def test_a_partially_nan_row_is_rejected_and_named():
    """The offending row has to be findable in a 100k-sample chain, so name it."""
    with pytest.raises(ValueError) as excinfo:
        PosteriorChain(_partially_nan_chain(), 2, 3, trans_dimensional=True)

    message = str(excinfo.value)
    assert "sample 3, source 1" in message
    assert "1 NaN among its 3 parameters" in message
    assert "validate_nan_convention=False" in message


def test_a_partially_nan_row_is_named_in_the_flattened_layout_too():
    """The reshape happens first, so the row is named in chain coordinates."""
    flat = _partially_nan_chain().reshape(4, 6)
    with pytest.raises(ValueError, match="sample 3, source 1"):
        PosteriorChain(flat, 2, 3, trans_dimensional=True)


def test_opting_out_of_the_convention_check_still_leaves_the_readers_agreeing():
    """
    The check is opt-out, not the convention: with it off, presence is still
    all-or-nothing.  ``find_prob_in_model`` used to read column 0 only, so it
    scored source 1 as present in all four samples while
    ``get_valid_chain_entry`` -- and every fit behind it -- saw only three.
    """
    pc = PosteriorChain(_partially_nan_chain(), 2, 3, trans_dimensional=True,
                        validate_nan_convention=False)

    prob_in_model = find_prob_in_model(pc.get_chain(), 2, eps=0)
    assert prob_in_model[1] == pytest.approx(0.75)      # was 1.0 reading column 0
    assert len(pc.get_valid_chain_entry(1)) == 3
    assert prob_in_model[1] * pc.shape[0] == len(pc.get_valid_chain_entry(1))


def test_the_convention_opt_out_is_inherited_by_every_derived_chain():
    """
    An escape hatch that dead-ends one call later is worse than no escape hatch.

    Opting out bought nothing end to end: the next thing to touch the chain
    rebuilt it with the default, so the caller got the very message that had
    told them to pass the flag -- from a chain they had already accepted.  The
    setting is a property of the chain, and every writer in petra moves whole
    source rows, so a derived chain can only fail the check exactly where its
    parent already did; re-running it can report nothing new.
    """
    pc = PosteriorChain(_partially_nan_chain(), 2, 3, trans_dimensional=True,
                        validate_nan_convention=False)

    derived = [pc.expand_chain(4), pc.randomize_entries(seed=0),
               prepare_chain(pc, 4, shuffle_seed=1)]
    for chain in derived:
        assert chain.validate_nan_convention is False
    # And the default is untouched: a chain nobody opted out for still validates.
    assert PosteriorChain(np.zeros((4, 2, 3)), 2, 3).validate_nan_convention is True


def test_an_opted_out_chain_survives_a_whole_relabeling_run():
    """
    The per-iteration rebuild is the other place the flag was dropped.

    ``run_relabeling_loop`` builds a fresh chain on every iteration through
    :func:`petra.relabel.relabel_posterior_chain_one_iteration`, so an opted-out
    chain used to be rejected on the first one even when the preamble let it
    through.  This is the end-to-end statement: widened, shuffled, relabeled and
    handed back, with the offending row still in it and no exception anywhere.
    """
    rng = np.random.default_rng(0)
    chain = rng.normal(scale=0.2, size=(20, 2, 2)) + np.array([[0.0, 0.0], [5.0, 5.0]])
    chain[3, 1, 1] = np.nan             # the row that violates the convention
    pc = PosteriorChain(chain, 2, 2, trans_dimensional=True, validate_nan_convention=False)

    result = relabel_mv_normal(pc, max_num_sources=3, num_iterations=2, progress=False)

    assert result.validate_nan_convention is False
    assert result.num_sources == 3
    # The half-NaN row came through rather than being silently repaired or dropped.
    nan_counts = np.isnan(result.get_chain()).sum(axis=-1)
    assert np.count_nonzero((nan_counts == 1)) == 1


def test_read_feather_rejects_a_partially_nan_row(tmp_path):
    """A file written elsewhere is exactly the boundary the check exists for."""
    pc = PosteriorChain(_partially_nan_chain(), 2, 3, trans_dimensional=True,
                        validate_nan_convention=False)
    path = os.path.join(tmp_path, "partial.feather")
    pc.to_feather(path)

    with pytest.raises(ValueError, match="sample 3, source 1"):
        PosteriorChain.read_feather(path)


def test_the_convention_opt_out_survives_a_feather_round_trip(tmp_path):
    """
    The escape hatch the error message advertises must not dead-end at the file.

    ``to_feather`` does not persist the flag and never will: the file is the
    boundary the check exists for, and a file that could switch off the
    validation of its own contents would leave the reader with no check at all.
    Reading is a construction instead, so it takes the construction option --
    which is also what makes the advertised ``validate_nan_convention=False``
    true of the call that raises here.
    """
    pc = PosteriorChain(_partially_nan_chain(), 2, 3, trans_dimensional=True,
                        validate_nan_convention=False)
    path = os.path.join(tmp_path, "opt_out.feather")
    pc.to_feather(path)

    # The default is unchanged: the flag was not smuggled into the file.
    with pytest.raises(ValueError, match="validate_nan_convention=False"):
        PosteriorChain.read_feather(path)

    loaded = PosteriorChain.read_feather(path, validate_nan_convention=False)
    assert np.array_equal(loaded.chain, pc.chain, equal_nan=True)
    assert loaded.num_sources == 2
    assert loaded.trans_dimensional is True


def test_the_convention_opt_out_reaches_the_legacy_reader_too(tmp_path):
    """The legacy layout is a read path as well, so the keyword has to reach it."""
    chain = _partially_nan_chain()
    df = pd.DataFrame(chain.reshape(-1, 6))
    df.columns = [str(c) for c in df.columns]
    df['num_sources'] = 2
    df['num_params_per_source'] = 3
    df['transdimensional'] = 1
    df['prob_in_model'] = [0.75, 1.0, np.nan, np.nan]
    path = os.path.join(tmp_path, "legacy_partial.feather")
    df.to_feather(path)

    with pytest.raises(ValueError, match="sample 3, source 1"):
        PosteriorChain.read_feather(path)

    loaded = PosteriorChain.read_feather(path, validate_nan_convention=False)
    assert np.array_equal(loaded.chain, chain, equal_nan=True)


def test_load_samples_rejects_a_stray_nan_in_a_populated_source(tmp_path):
    """
    Nothing in petra writes a half-NaN row, but a sampler that emitted one into
    its chain file used to be loaded without complaint.
    """
    rows = np.arange(2 * 2 * 8, dtype=float).reshape(4, 8)
    rows[2, 5] = np.nan     # sample 1, source 0 of a two-source file
    path = os.path.join(tmp_path, "dimension_chain.dat.2")
    np.savetxt(path, rows)

    with pytest.raises(ValueError, match="sample 1, source 0"):
        load_samples_ucbmcmc(str(tmp_path))


# ---------------------------------------------------------------------------
# samples_io argument threading
# ---------------------------------------------------------------------------

def _write_ucbmcmc_file(folder, nsources, nsamples, num_params_per_source=8, start=0.0):
    rows = np.arange(nsamples * nsources * num_params_per_source, dtype=float) + start
    rows = rows.reshape(nsamples * nsources, num_params_per_source)
    path = os.path.join(folder, f"dimension_chain.dat.{nsources}")
    np.savetxt(path, rows)
    return path


def test_ucbmcmc_applies_burn_per_file(tmp_path):
    """burn/thin are applied per file, not only to the first one."""
    _write_ucbmcmc_file(tmp_path, nsources=1, nsamples=10)
    _write_ucbmcmc_file(tmp_path, nsources=2, nsamples=10, start=1000.0)

    pc = load_samples_ucbmcmc(str(tmp_path), burn=5)
    assert pc.chain.shape == (10, 2, 8)  # 5 kept from each file, not 15 total

    thinned = load_samples_ucbmcmc(str(tmp_path), burn=0, thin=2)
    assert thinned.chain.shape == (10, 2, 8)


def test_load_samples_forwards_num_params_per_source(tmp_path):
    """The directory branch must not hardcode 8 parameters per source."""
    _write_ucbmcmc_file(tmp_path, nsources=2, nsamples=4, num_params_per_source=3)
    pc = load_samples(str(tmp_path), num_params_per_source=3)
    assert pc.chain.shape == (4, 2, 3)
    assert pc.num_params_per_source == 3


def test_ucbmcmc_trailing_newline_does_not_break_reshape(tmp_path):
    """nsamples comes from the parsed array, so extra blank lines are harmless."""
    path = _write_ucbmcmc_file(tmp_path, nsources=2, nsamples=3)
    with open(path, "a") as f:
        f.write("\n")
    pc = load_samples_ucbmcmc(str(tmp_path))
    assert pc.chain.shape == (3, 2, 8)


def _write_product_space_file(path, nsamples, num_params_per_source=3, max_sources=2):
    n_data = max_sources * num_params_per_source
    data = np.arange(nsamples * n_data, dtype=float).reshape(nsamples, n_data)
    meta = np.zeros((nsamples, 5))
    meta[:, 0] = max_sources - 1  # column -5 holds (num_sources - 1)
    np.savetxt(path, np.hstack([data, meta]))


def test_product_space_honours_burn_and_thin(tmp_path):
    path = os.path.join(tmp_path, "chain.ps.txt")
    _write_product_space_file(path, nsamples=20)

    full = load_samples_product_space(path, 3)
    assert full.chain.shape == (20, 2, 3)

    burned = load_samples(path, num_params_per_source=3, burn=10, thin=2)
    assert burned.chain.shape == (5, 2, 3)
    assert np.array_equal(burned.chain[0], full.chain[10])


def test_dispatcher_handles_single_row_file(tmp_path):
    """Dispatching reads only the first data line, so one-row files work."""
    path = os.path.join(tmp_path, "one_row.dat")
    np.savetxt(path, np.arange(8, dtype=float).reshape(1, 8))
    pc = load_samples(path, num_params_per_source=2)
    assert pc.chain.shape == (1, 2, 2)
