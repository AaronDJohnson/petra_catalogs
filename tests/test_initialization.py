"""Tests for petra.initialization: the histogram labeling, the univariate warm start and the initialization passes."""

import logging
import warnings
from copy import deepcopy

import numpy as np
import pytest

from petra.checkpoint import find_latest_checkpoint
from petra.initialization import (check_param_index, label_chain_by_histogram, relabel_univariate_normal,
                                  run_initialization_passes)
from petra.make_catalog import make_catalog_mv_normal
from petra.posterior_chain import PosteriorChain
from petra.utils import source_present
from tests.helpers import run_state

#: The width of one frequency bin of a one-year observation, in Hz.
DELTA_FREQ = 1.0 / (525600 * 60)

#: The anchors of the three skip messages of `run_initialization_passes`.
RESUME_SKIP = "skipping both initialization passes"
UNIVARIATE_SKIP = "initialization_param_index is None; skipping"
MV_NORMAL_SKIP = "init_with_mv_normal is False; skipping"

PARAM_INDEX_ERROR = r"initialization_param_index must be None or in \[0,"


# The per-sample labeler of petra 1.1.0, kept to prove the vectorized one bit-identical (F030)


def _get_max_count_bin_center(chain, delta_freq, relabeling_parameter):
    """Return the centre of the fullest bin of the parameter's non-NaN values."""
    x = chain[:, :, relabeling_parameter].reshape(-1)
    x = x[~np.isnan(x)]
    k = np.rint(x / delta_freq).astype(np.int64)
    return float(np.bincount(k).argmax() * delta_freq)


def _tetris_rise_nd(a, axis=1):
    """Move the non-NaN values of every line along `axis` to its front, in order."""
    order = np.argsort(np.isnan(a), axis=axis, kind="stable")
    return np.take_along_axis(a, order, axis=axis)


def _count_nonnan_samples(chain, source_index, relabeling_parameter):
    """Return how many samples hold the slot."""
    return int(np.count_nonzero(~np.isnan(chain[:, source_index, relabeling_parameter])))


def _sort_array_by_sample_count(sources_array, relabeling_parameter):
    """Order the slots by decreasing sample count, equal counts in reverse slot order."""
    counts = [_count_nonnan_samples(sources_array, i, relabeling_parameter) for i in range(sources_array.shape[1])]
    sorted_source_indices = np.argsort(counts, kind="stable")[::-1]
    sources_array_sorted = np.zeros_like(sources_array)
    for new_index, old_index in enumerate(sorted_source_indices):
        sources_array_sorted[:, new_index, :] = sources_array[:, old_index, :]
    return sources_array_sorted


def _reference_label_chain_by_histogram(chain, obs_time_yrs=1, num_surrounding_bins=1, relabeling_parameter=0,
                                        num_extra_entries=1000, low_num_samples=100):
    """Return the labels of petra 1.1.0's per-sample loop as a raw array."""
    delta_freq = 1.0 / (obs_time_yrs * 525600 * 60)

    chain = deepcopy(chain)
    sources_array = np.zeros((chain.shape[0], chain.shape[1] + num_extra_entries, chain.shape[2]))
    sources_array[:] = np.nan

    source_index = 0
    while np.any(~np.isnan(chain.chain)):
        if source_index >= sources_array.shape[1]:
            raise ValueError("Ran out of source labels; increase num_extra_entries.")
        max_count_bin_center = _get_max_count_bin_center(chain.chain, delta_freq, relabeling_parameter)
        for i in range(chain.shape[0]):
            distances = np.abs(chain[i, :, relabeling_parameter] - max_count_bin_center)
            if np.all(np.isnan(distances)):
                sources_array[i, source_index, :] = np.nan
                continue
            elif np.nanmin(distances) <= (num_surrounding_bins + 0.5) * delta_freq:
                min_index = np.nanargmin(distances)
            else:
                sources_array[i, source_index, :] = np.nan
                continue
            sources_array[i, source_index, :] = chain[i, min_index, :]
            chain[i, min_index, :] = np.nan
        source_index += 1

    sources_array = _sort_array_by_sample_count(sources_array, relabeling_parameter)
    low_sample_count_source_indices = []
    for source_index in range(sources_array.shape[1]):
        if _count_nonnan_samples(sources_array, source_index, relabeling_parameter) < low_num_samples:
            low_sample_count_source_indices.append(source_index)
    low_sample_count_sources = sources_array[:, low_sample_count_source_indices, :]
    sources_array[:, low_sample_count_source_indices, :] = _tetris_rise_nd(low_sample_count_sources)

    empty_source_indices = []
    for source_index in range(sources_array.shape[1]):
        if _count_nonnan_samples(sources_array, source_index, relabeling_parameter) == 0:
            empty_source_indices.append(source_index)
    non_empty_source_indices = [i for i in range(sources_array.shape[1]) if i not in empty_source_indices]
    return _sort_array_by_sample_count(sources_array[:, non_empty_source_indices, :], relabeling_parameter)


# Chains


def _lisa_like_chain(seed, num_samples, num_sources, num_params, absent_fraction, spread_bins):
    """Well and badly separated frequencies, absent rows, slots shuffled per sample."""
    rng = np.random.default_rng(seed)
    gaps = rng.choice([1, 2, 3, 40], size=num_sources)
    centres = 3e-3 + np.cumsum(gaps) * DELTA_FREQ
    arr = rng.normal(size=(num_samples, num_sources, num_params))
    arr[:, :, 0] = centres + rng.normal(scale=spread_bins * DELTA_FREQ, size=(num_samples, num_sources))
    arr[rng.random((num_samples, num_sources)) < absent_fraction] = np.nan
    order = np.argsort(rng.random((num_samples, num_sources)), axis=1)
    return PosteriorChain(np.take_along_axis(arr, order[:, :, None], axis=1), num_sources, num_params,
                          trans_dimensional=True)


def _ties_chain():
    """Thirty-two sources ten bins apart, each in its first 2, 1, 0 or 2 samples, cycling: many equal counts."""
    chain = np.full((2, 32, 2), np.nan)
    for slot, count in enumerate([2, 1, 0, 2] * 8):
        chain[:count, slot, :] = [(slot + 1) * 10 * DELTA_FREQ, 100 + slot]
    return chain


def _tied_sparse_chain():
    """Two sources at bins 10 and 20 in all four samples, and four one-sample sources at bins 30 to 60, one each."""
    chain = np.full((4, 6, 2), np.nan)
    chain[:, 0] = [10 * DELTA_FREQ, 100]
    chain[:, 1] = [20 * DELTA_FREQ, 101]
    for i in range(4):
        chain[i, 2 + i] = [(30 + 10 * i) * DELTA_FREQ, 102 + i]
    return chain


def _two_source_chain(num_samples=6):
    """Two sources, at bins 100 and 400, in every sample."""
    arr = np.empty((num_samples, 2, 2))
    arr[:, 0, :] = [100 * DELTA_FREQ, 0.5]
    arr[:, 1, :] = [400 * DELTA_FREQ, 0.7]
    return arr


def _two_parameter_chain():
    """A 12-sample, two-source, two-parameter chain."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 2)) + np.array([[0.0, 0.0], [8.0, 8.0]])
    return PosteriorChain(chain, 2, 2, trans_dimensional=True)


def _label_counts(chain):
    return np.count_nonzero(~np.isnan(chain[:, :, 0]), axis=0).tolist()


# The histogram labeling


def test_histogram_labeling_separates_two_frequencies():
    rng = np.random.default_rng(0)
    n_samples = 40
    chain = np.empty((n_samples, 2, 1))
    chain[:, 0, 0] = 1e-3 + 0.1 * DELTA_FREQ * rng.normal(size=n_samples)
    chain[:, 1, 0] = 1e-3 + 50 * DELTA_FREQ + 0.1 * DELTA_FREQ * rng.normal(size=n_samples)
    for i in range(n_samples):
        chain[i] = chain[i, rng.permutation(2), :]

    labeled = label_chain_by_histogram(PosteriorChain(chain, 2, 1, trans_dimensional=True), obs_time_yrs=1.0,
                                       low_num_samples=1, num_extra_entries=4)

    assert labeled.num_sources == 2
    assert np.count_nonzero(~np.isnan(labeled.chain)) == np.count_nonzero(~np.isnan(chain))
    slot_means = np.sort([np.nanmean(labeled.chain[:, i, 0]) for i in range(2)])
    assert slot_means[0] == pytest.approx(1e-3, abs=DELTA_FREQ)
    assert slot_means[1] == pytest.approx(1e-3 + 50 * DELTA_FREQ, abs=DELTA_FREQ)

    # Bin 102 is the fullest, so it claims the value one bin below it before bin 100 can.
    fullest = np.full((14, 1, 1), 102 * DELTA_FREQ)
    fullest[:3, 0, 0] = 100 * DELTA_FREQ
    fullest[3, 0, 0] = 101 * DELTA_FREQ
    labeled = label_chain_by_histogram(PosteriorChain(fullest, 1, 1), low_num_samples=1, num_extra_entries=1)
    assert _label_counts(labeled.chain) == [11, 3]
    assert labeled.chain[3, 0, 0] == 101 * DELTA_FREQ

    # One year is 525,600 minutes: a value just inside 1.5 bins of the centre is claimed, one just outside is not.
    edge = np.full((4, 1, 1), 1000 * DELTA_FREQ)
    edge[2, 0, 0] = (1001.5 - 1e-5) * DELTA_FREQ
    edge[3, 0, 0] = (1001.5 + 1e-5) * DELTA_FREQ
    labeled = label_chain_by_histogram(PosteriorChain(edge, 1, 1), low_num_samples=1, num_extra_entries=1)
    assert _label_counts(labeled.chain) == [3, 1]
    assert np.isnan(labeled.chain[3, 0, 0])

    no_sources = np.full((6, 2, 1), np.nan)
    no_sources[:, 0, 0] = 100 * DELTA_FREQ
    no_sources[:, 1, 0] = 400 * DELTA_FREQ
    no_sources[3, :, 0] = np.nan
    labeled = label_chain_by_histogram(PosteriorChain(no_sources, 2, 1, trans_dimensional=True), low_num_samples=1,
                                       num_extra_entries=2)
    assert labeled.num_sources == 2
    assert np.all(np.isnan(labeled.chain[3]))

    integers = PosteriorChain(np.tile(np.array([[1], [5]]), (10, 1, 1)), 2, 1, trans_dimensional=True)
    labeled = label_chain_by_histogram(integers, obs_time_yrs=1e-8, low_num_samples=1, num_extra_entries=2)
    assert labeled.chain.dtype == np.float64

    for obs_time_yrs in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            label_chain_by_histogram(integers, obs_time_yrs=obs_time_yrs)


@pytest.mark.parametrize("case", [0, 1, 2, 3, 4, 5, "ties"])
def test_histogram_labeling_matches_the_reference_loop(case):
    if case == "ties":
        chains = [PosteriorChain(_ties_chain(), 32, 2, trans_dimensional=True),
                  PosteriorChain(_tied_sparse_chain(), 6, 2, trans_dimensional=True)]
        settings_list = [dict(low_num_samples=low_num_samples) for low_num_samples in (1, 2, 100)]
    else:
        rng = np.random.default_rng(100 + case)
        chains = [_lisa_like_chain(case, num_samples=int(rng.integers(20, 200)), num_sources=int(rng.integers(1, 7)),
                                   num_params=int(rng.integers(1, 4)), absent_fraction=float(rng.uniform(0.0, 0.6)),
                                   spread_bins=float(rng.choice([0.1, 0.6, 2.0])))]
        settings_list = [dict(num_surrounding_bins=num_surrounding_bins, low_num_samples=low_num_samples,
                              num_extra_entries=60)
                         for num_surrounding_bins, low_num_samples in [(1, 100), (0, 5), (2, 30)]]

    for pc in chains:
        for settings in settings_list:
            expected = _reference_label_chain_by_histogram(pc, **settings)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                labeled = label_chain_by_histogram(pc, **settings)

            assert labeled.num_sources == expected.shape[1]
            np.testing.assert_array_equal(labeled.chain, expected)

    if case == "ties":
        # Sorting by count reverses equal counts, so the four one-sample sources rise into the slot after the full
        # ones, and that merged label, now tied with them, is sorted first; every row moves as a whole vector.
        sparse = _tied_sparse_chain()
        labeled = label_chain_by_histogram(chains[1], low_num_samples=2)
        np.testing.assert_array_equal(labeled.chain, np.stack([sparse[np.arange(4), np.arange(2, 6)], sparse[:, 0],
                                                               sparse[:, 1]], axis=1))


def test_histogram_labeling_compacts_sparse_labels_and_caps_headroom():
    sparse = np.full((8, 3, 1), np.nan)
    sparse[:, 0, 0] = 100 * DELTA_FREQ
    sparse[:2, 1, 0] = 400 * DELTA_FREQ
    sparse[2:4, 2, 0] = 800 * DELTA_FREQ
    labeled = label_chain_by_histogram(PosteriorChain(sparse, 3, 1, trans_dimensional=True), low_num_samples=5,
                                       num_extra_entries=2)
    assert _label_counts(labeled.chain) == [8, 4]
    np.testing.assert_array_equal(labeled.chain[:4, 1, 0], np.array([400, 400, 800, 800]) * DELTA_FREQ)

    apart = np.array([[[1e-3 + 0 * DELTA_FREQ], [1e-3 + 100 * DELTA_FREQ]],
                      [[1e-3 + 200 * DELTA_FREQ], [1e-3 + 300 * DELTA_FREQ]]])
    with pytest.raises(ValueError, match="Ran out of source labels"):
        label_chain_by_histogram(PosteriorChain(apart, 2, 1, trans_dimensional=True), obs_time_yrs=1.0,
                                 num_extra_entries=0, low_num_samples=1)

    tiled = np.tile(np.array([[1e-3], [1e-3 + 50 * DELTA_FREQ]]), (30, 1, 1))
    labeled = label_chain_by_histogram(PosteriorChain(tiled, 2, 1, trans_dimensional=True),
                                       num_extra_entries=10 ** 9, low_num_samples=1)
    assert labeled.num_sources == 2

    values = np.full((2, 1, 1), 0.001000001)
    labeled = label_chain_by_histogram(PosteriorChain(values, 1, 1), num_surrounding_bins=0, num_extra_entries=1,
                                       low_num_samples=1)
    assert labeled.num_sources == 1
    np.testing.assert_array_equal(labeled.chain, values)


def test_histogram_labeling_leaves_partial_rows_out_and_keeps_the_opt_out():
    for nan_parameter in (1, 0):
        arr = _two_source_chain()
        arr[2, 1, nan_parameter] = np.nan
        pc = PosteriorChain(arr, 2, 2, trans_dimensional=True, validate_nan_convention=False)

        with pytest.warns(UserWarning, match="mixes NaN with finite values"):
            labeled = label_chain_by_histogram(pc, low_num_samples=1, num_extra_entries=2)

        assert labeled.validate_nan_convention is False and labeled.trans_dimensional is True
        assert np.isnan(labeled.chain[2, 1]).all()
        assert np.count_nonzero(source_present(labeled.chain)) == np.count_nonzero(source_present(arr))
        assert np.array_equal(source_present(labeled.chain), ~np.isnan(labeled.chain[:, :, 1]))

    arr = np.full((10, 3, 2), np.nan)
    arr[:, 0, :] = [100 * DELTA_FREQ, 0.1]
    arr[0, 1, :] = [400 * DELTA_FREQ, np.nan]
    arr[1, 1, :] = [400 * DELTA_FREQ, 0.4]
    arr[0, 2, :] = [800 * DELTA_FREQ, 0.9]
    with pytest.warns(UserWarning, match="mixes NaN with finite values"):
        labeled = label_chain_by_histogram(PosteriorChain(arr, 3, 2, trans_dimensional=True,
                                                          validate_nan_convention=False),
                                           low_num_samples=5, num_extra_entries=3)
    input_rows = {tuple(row) for row in arr.reshape(-1, 2) if not np.isnan(row).any()}
    for row in labeled.chain.reshape(-1, 2):
        assert np.isnan(row).all() or tuple(row) in input_rows
    sample_zero = [tuple(row) for row in labeled.chain[0] if not np.isnan(row).any()]
    assert sorted(sample_zero) == sorted([(100 * DELTA_FREQ, 0.1), (800 * DELTA_FREQ, 0.9)])

    opted_out = PosteriorChain(_two_source_chain(), 2, 2, trans_dimensional=True, validate_nan_convention=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labeled = label_chain_by_histogram(opted_out, low_num_samples=1, num_extra_entries=2)
    assert labeled.validate_nan_convention is False and labeled.trans_dimensional is True
    assert labeled.num_sources == 2


# The univariate warm start


def test_univariate_warm_start_keeps_a_narrow_source_apart_and_checkpoints_into_a_path(tmp_path):
    rng = np.random.default_rng(0)
    rng.standard_normal((4000, 4))
    rng.standard_normal((4000, 3))
    mu = np.array([1e-4, 1e-2, 5e-3, 5e-3 + 3e-7])
    sd = np.array([1e-8, 1e-8, 1e-9, 1e-7])
    x = mu + sd * rng.standard_normal((4000, 4))
    truth = np.tile(np.arange(4), (4000, 1))
    x[rng.random(4000) >= 0.7, 2] = np.nan
    x[rng.random(4000) >= 0.7, 3] = np.nan
    order = np.array([rng.permutation(4) for _ in range(4000)])
    scrambled = np.take_along_axis(x, order, axis=1)
    source_of = np.take_along_axis(truth, order, axis=1)
    checkpoint_dir = tmp_path / "univariate"

    out = relabel_univariate_normal(PosteriorChain(scrambled[:, :, None], 4, 1, trans_dimensional=True),
                                    max_num_sources=4, num_iterations=30, checkpoint_dir=checkpoint_dir,
                                    progress=False)

    values = out.get_chain()[:, :, 0]
    for label in range(4):
        present = ~np.isnan(values[:, label])
        sources = [source_of[i][scrambled[i] == values[i, label]][0] for i in np.flatnonzero(present)]
        majority = np.bincount(sources).argmax()
        assert np.count_nonzero(np.array(sources) != majority) <= 5
    latest = find_latest_checkpoint(checkpoint_dir)
    assert latest is not None
    state = run_state(latest[0])
    assert state["method"] == "univariate_normal"
    assert state["settings"] == {"init_parameter_index": 0}


# The initialization passes


def _stand_in(name, passes):
    """Return a relabeler that records its call and hands back a new chain."""
    def relabeler(posterior_chain, **kwargs):
        relabeled = PosteriorChain(posterior_chain.chain + 1.0, posterior_chain.num_sources,
                                   posterior_chain.num_params_per_source)
        passes.append((name, posterior_chain, kwargs, relabeled))
        return relabeled
    return relabeler


@pytest.mark.parametrize("param_index, valid", [
    (-1, False), (2, False), (True, False), (False, False), (np.True_, False), (1.5, False),
    (None, True), (0, True), (1, True),
], ids=["wrapped", "past-the-end", "true", "false", "numpy-true", "float", "none", "first", "last"])
def test_initialization_param_index_is_none_or_a_column_of_the_chain(param_index, valid):
    chain = _two_parameter_chain()
    passes = []

    def run(resume_from):
        return run_initialization_passes(chain, 2, univariate_relabeler=_stand_in("univariate", passes),
                                         mv_normal_relabeler=_stand_in("mv_normal", passes), with_mv_normal=False,
                                         param_index=param_index, num_iterations=3, resume_from=resume_from,
                                         progress=False)

    if valid:
        assert check_param_index(2, param_index) is None
        run(None)
        run("run/iteration_3.feather")
        return
    calls = [
        lambda: check_param_index(2, param_index),
        lambda: run(None),
        lambda: run("run/iteration_3.feather"),
        lambda: make_catalog_mv_normal(chain, 2, num_iterations=1, initialization_param_index=param_index,
                                       progress=False),
    ]
    for call in calls:
        with pytest.raises(ValueError, match=PARAM_INDEX_ERROR):
            call()
    assert passes == []


@pytest.mark.parametrize("with_mv_normal, param_index, resume_from, expected, logged", [
    (True, 0, None, ["univariate", "mv_normal"], []),
    (True, None, None, ["mv_normal"], [UNIVARIATE_SKIP]),
    (False, 0, None, ["univariate"], [MV_NORMAL_SKIP]),
    (False, None, None, [], [UNIVARIATE_SKIP, MV_NORMAL_SKIP]),
    (True, 0, "run/iteration_3.feather", [], [RESUME_SKIP]),
], ids=["both", "mv-normal-only", "univariate-only", "neither", "resume"])
def test_run_initialization_passes_runs_exactly_the_passes_selected(
        caplog, with_mv_normal, param_index, resume_from, expected, logged):
    chain = PosteriorChain(np.zeros((4, 2, 1)), 2, 1)
    passes = []

    with caplog.at_level(logging.INFO, logger="petra.initialization"):
        result = run_initialization_passes(chain, 2, univariate_relabeler=_stand_in("univariate", passes),
                                           mv_normal_relabeler=_stand_in("mv_normal", passes),
                                           with_mv_normal=with_mv_normal, param_index=param_index, num_iterations=3,
                                           resume_from=resume_from, progress=False)

    assert [name for name, _, _, _ in passes] == expected
    expected_kwargs = {
        "univariate": {"max_num_sources": 2, "num_iterations": 3, "init_parameter_index": param_index,
                       "progress": False},
        "mv_normal": {"max_num_sources": 2, "num_iterations": 3, "progress": False, "split_moves": False},
    }
    received = chain
    for name, posterior_chain, kwargs, relabeled in passes:
        assert posterior_chain is received
        assert kwargs == expected_kwargs[name]
        received = relabeled
    assert result is received
    text = "\n".join(record.getMessage() for record in caplog.records)
    for anchor in (RESUME_SKIP, UNIVARIATE_SKIP, MV_NORMAL_SKIP):
        assert (anchor in text) == (anchor in logged)
