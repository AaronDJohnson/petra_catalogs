"""Tests of petra.match on a fake waveform model: the match, the clustering and the split and merge passes."""

import logging
import warnings

import numpy as np
import pytest

from petra.match import WaveformBank, bank_matches, cluster_by_match, greedy_clusters, present_rows, refine_catalog
from petra.posterior_chain import PosteriorChain
from petra.utils import find_prob_in_model

DF = 1e-6
N_BINS = 16


def fake_waveforms(params: np.ndarray) -> WaveformBank:
    """Gaussian bumps 3 bins wide at bin round(f0 / DF) with phase exp(i phi0); the second channel is half the first."""
    centre = np.round(params[:, 0] / DF).astype(np.int64)
    bump = np.exp(-0.5 * ((np.arange(N_BINS) - N_BINS // 2) / 3.0) ** 2) * np.exp(1j * params[:, 7:8])
    return WaveformBank(centre - N_BINS // 2, np.stack([bump, 0.5 * bump], axis=1), DF)


def source(f_bin: int, n: int, phase: float = 0.0, scatter: float = 0.05, seed: int = 0) -> np.ndarray:
    """Return `n` rows of one fake source at bin `f_bin`, the phase scattered about `phase`, other columns random."""
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n, 8))
    rows[:, 0] = f_bin * DF
    rows[:, 7] = phase + scatter * rng.normal(size=n)
    return rows


def mixed_label_chain(n_labels: int) -> np.ndarray:
    """Label 0 holds source A in samples 0-29 and B in 30-41, label 1 source C in all 42; the others are empty."""
    chain = np.full((42, n_labels, 8), np.nan)
    chain[:30, 0] = source(1000, 30, seed=1)
    chain[30:, 0] = source(1020, 12, seed=2)
    chain[:, 1] = source(2000, 42, seed=3)
    return chain


def dense_matches(bank: WaveformBank, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Return the matches computed on one grid spanning every waveform of the bank."""
    lo = bank.start.min()
    grid = np.zeros((len(bank.start), 2, bank.start.max() - lo + N_BINS), dtype=complex)
    for i, offset in enumerate(bank.start - lo):
        grid[i, :, offset:offset + N_BINS] = bank.data[i]
    flat = grid.reshape(len(grid), -1)
    norm = np.linalg.norm(flat, axis=1)
    unit = flat / np.where(norm > 0, norm, 1.0)[:, None]
    return (unit[rows].conj() @ unit[columns].T).real


def assert_rows_permuted(result: np.ndarray, original: np.ndarray) -> None:
    """Assert that every sample of `result` holds the rows of the same sample of `original`, NaN rows included."""
    def rows(chain: np.ndarray) -> list:
        return [sorted(row.tobytes() for row in sample) for sample in chain]
    assert rows(result) == rows(original)


def summary(result) -> list:
    """Return each action of a MatchResult without its match."""
    return [(a.kind, a.source_label, a.target_label, a.rows) for a in result.actions]


def test_bank_matches():
    params = np.vstack([source(1000, 1, scatter=0), source(1001, 1, scatter=0), source(1000, 1, np.pi, scatter=0),
                        source(1100, 1, scatter=0), source(1000, 1, scatter=0)])
    bank = fake_waveforms(params)
    bank.data[4] = 0
    matches = bank_matches(bank, np.arange(5))
    assert matches[0, 0] == pytest.approx(1, abs=1e-12) and 0 < matches[0, 1] < 1
    assert matches[0, 2] == pytest.approx(-1, abs=1e-12)
    # Row 3's window does not overlap the others, so it sits in a band of its own.
    assert np.all(matches[3, [0, 1, 2, 4]] == 0) and np.all(matches[4] == 0)
    rows, columns = np.array([3, 0, 1]), np.array([2, 3, 4, 0])
    np.testing.assert_allclose(bank_matches(bank, rows, columns), dense_matches(bank, rows, columns), atol=1e-12)
    np.testing.assert_allclose(bank_matches(bank, rows, columns), bank_matches(bank, columns, rows).T, atol=1e-15)
    np.testing.assert_array_equal(bank_matches(bank, rows), bank_matches(bank, rows, rows))
    assert bank_matches(bank, np.array([], dtype=int), rows).shape == (0, 3)


def test_greedy_clusters():
    params = np.vstack([source(1000, 5, seed=1), source(1030, 3, seed=2), source(1060, 2, seed=3)])
    rng = np.random.default_rng(0)
    order = rng.permutation(10)
    truth = np.repeat([0, 1, 2], [5, 3, 2])[order]
    matches = bank_matches(fake_waveforms(params[order]), np.arange(10))
    clusters = greedy_clusters(matches, 0.5)
    assert [set(truth[cluster]) for cluster in clusters] == [{0}, {1}, {2}]
    assert [cluster.size for cluster in clusters] == [5, 3, 2]
    relabel = rng.permutation(10)
    permuted = greedy_clusters(matches[np.ix_(relabel, relabel)], 0.5)
    assert ([(relabel[c[0]], sorted(relabel[c])) for c in permuted] == [(c[0], sorted(c)) for c in clusters])
    highest = matches[~np.eye(10, dtype=bool)].max()
    assert [cluster.tolist() for cluster in greedy_clusters(matches, highest)] == [[i] for i in range(10)]


def test_cluster_by_match_is_greedy_clusters_when_every_row_is_its_own_sample():
    params = np.vstack([source(1000, 5, seed=1), source(1030, 3, seed=2), source(1060, 2, seed=3),
                        source(1500, 4, seed=4)])[np.random.default_rng(0).permutation(14)]
    clusters = greedy_clusters(bank_matches(fake_waveforms(params), np.arange(14)), 0.5)
    catalog = cluster_by_match(PosteriorChain(params[:, None], 1, 8), fake_waveforms, batch_rows=3)
    assert catalog.shape == (14, 4, 8) and catalog.trans_dimensional and catalog.cost_dict == {}
    for label, cluster in enumerate(clusters):
        np.testing.assert_array_equal(np.flatnonzero(~np.isnan(catalog.chain[:, label, 0])), np.sort(cluster))
    np.testing.assert_array_equal(catalog.prob_in_model, [5 / 14, 4 / 14, 3 / 14, 2 / 14])


def test_cluster_by_match_takes_one_row_per_sample_and_ignores_the_labels():
    # Sample s holds source A twice (phases 0.3 and 0.05) and source B once, in shuffled slots.
    chain = np.full((20, 4, 8), np.nan)
    rng = np.random.default_rng(1)
    for s in range(20):
        chain[s, rng.permutation(4)[:3]] = np.vstack([source(1000, 1, 0.3, scatter=0, seed=s),
                                                      source(1000, 1, 0.05, scatter=0, seed=50 + s),
                                                      source(1040, 1, scatter=0, seed=100 + s)])
    catalog = cluster_by_match(PosteriorChain(chain, 4, 8, trans_dimensional=True), fake_waveforms)
    assert catalog.shape == (20, 3, 8)
    np.testing.assert_array_equal(catalog.prob_in_model, [1, 1, 1])
    # Each sample gives A's two rows to two labels: a phase matches itself (1) better than the other (cos 0.25).
    phases = [set(np.round(catalog.chain[:, label, 7], 2)) for label in range(3)]
    assert sorted(phases[:2], key=min) == [{0.05}, {0.3}] and phases[2] == {0.0}
    assert np.all(np.round(catalog.chain[:, 2, 0] / DF) == 1040)
    assert_rows_permuted(np.concatenate([catalog.chain, np.full((20, 1, 8), np.nan)], axis=1), chain)


@pytest.mark.parametrize("options, message", [
    ({"threshold": -1.0}, "threshold"),
    ({"batch_rows": 0}, "batch_rows"),
    ({"waveforms": lambda params: fake_waveforms(params[1:])}, r"data of shape \(83, 2, 16\).*84 present rows"),
])
def test_cluster_by_match_rejects_bad_arguments(options, message):
    with pytest.raises(ValueError, match=message):
        cluster_by_match(PosteriorChain(mixed_label_chain(3), 3, 8), **{"waveforms": fake_waveforms, **options})


def test_cluster_by_match_rejects_an_empty_chain():
    with pytest.raises(ValueError, match="at least one present row"):
        cluster_by_match(PosteriorChain(np.full((3, 1, 8), np.nan), 1, 8), fake_waveforms)


def test_present_rows():
    chain = np.ones((2, 3, 2))
    chain[0, 1] = np.nan
    chain[1, 0, 0] = np.nan
    samples, slots = present_rows(chain)
    assert samples.tolist() == [0, 0, 1, 1] and slots.tolist() == [0, 2, 1, 2]


def test_split_moves_the_second_source_to_the_empty_label(caplog):
    caplog.set_level(logging.INFO, logger="petra.match")
    catalog = PosteriorChain(mixed_label_chain(3), 3, 8, trans_dimensional=True)
    snapshot = catalog.chain.copy()
    result = refine_catalog(catalog, fake_waveforms)
    refined = result.catalog.chain
    assert summary(result) == [("split", 0, 2, 12)] and 0.9 < result.actions[0].match <= 1
    assert "Split: label 0 gave 12 rows" in caplog.text
    np.testing.assert_array_equal(refined[30:, 2], snapshot[30:, 0])
    assert np.isnan(refined[30:, 0]).all() and np.isnan(refined[:30, 2]).all()
    np.testing.assert_array_equal(refined[:, 1], snapshot[:, 1])
    assert_rows_permuted(refined, snapshot)
    np.testing.assert_array_equal(result.catalog.prob_in_model, find_prob_in_model(refined, 3, eps=0))
    assert result.catalog.cost_dict == {} and result.catalog.trans_dimensional
    np.testing.assert_array_equal(catalog.chain, snapshot)


def test_split_without_a_free_label_leaves_the_rows_and_warns_once():
    chain = mixed_label_chain(2)
    with pytest.warns(UserWarning, match="12 rows.*max_num_sources") as record:
        result = refine_catalog(PosteriorChain(chain.copy(), 2, 8, trans_dimensional=True), fake_waveforms)
    assert len(record) == 1 and result.actions == () and result.rounds == 1
    np.testing.assert_array_equal(result.catalog.chain, chain)


def test_grow_gives_a_split_with_no_free_label_a_new_one():
    chain = mixed_label_chain(2)
    widened = refine_catalog(PosteriorChain(chain.copy(), 2, 8, trans_dimensional=True).expand_chain(3),
                             fake_waveforms)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = refine_catalog(PosteriorChain(chain.copy(), 2, 8, trans_dimensional=True), fake_waveforms, grow=True)
    assert summary(result) == summary(widened) == [("split", 0, 2, 12)] and result.catalog.shape == (42, 3, 8)
    np.testing.assert_array_equal(result.catalog.chain, widened.catalog.chain)
    np.testing.assert_array_equal(result.catalog.prob_in_model, find_prob_in_model(result.catalog.chain, 3, eps=0))


def test_grow_lets_a_one_label_catalog_split():
    chain = mixed_label_chain(2)[:, :1]
    assert refine_catalog(PosteriorChain(chain.copy(), 1, 8), fake_waveforms).actions == ()
    result = refine_catalog(PosteriorChain(chain.copy(), 1, 8), fake_waveforms, grow=True)
    assert summary(result) == [("split", 0, 1, 12)] and result.catalog.num_sources == 2
    assert_rows_permuted(result.catalog.chain, np.concatenate([chain, np.full_like(chain, np.nan)], axis=1))


def test_split_prefers_the_label_that_matches_the_cluster_over_an_empty_one():
    chain = mixed_label_chain(3)
    chain[:, 1] = np.nan
    chain[:30, 1] = source(1020, 30, seed=4)  # label 1 is source B where label 0 is A, and absent where label 0 is B
    result = refine_catalog(PosteriorChain(chain.copy(), 3, 8, trans_dimensional=True), fake_waveforms)
    assert summary(result) == [("split", 0, 1, 12)] and result.rounds == 2
    refined = result.catalog.chain
    np.testing.assert_array_equal(refined[30:, 1], chain[30:, 0])
    assert np.isnan(refined[:, 2]).all() and np.isnan(refined[30:, 0]).all()
    assert_rows_permuted(refined, chain)


@pytest.mark.parametrize("min_cluster_rows, min_cluster_fraction", [(13, 0.05), (10, 0.3), (22, 0.0)])
def test_small_clusters_stay(min_cluster_rows, min_cluster_fraction):
    chain = mixed_label_chain(3)
    result = refine_catalog(PosteriorChain(chain.copy(), 3, 8, trans_dimensional=True), fake_waveforms,
                            min_cluster_rows=min_cluster_rows, min_cluster_fraction=min_cluster_fraction)
    assert result.actions == () and result.rounds == 1
    np.testing.assert_array_equal(result.catalog.chain, chain)


def test_merge_joins_labels_that_never_cooccur(caplog):
    caplog.set_level(logging.INFO, logger="petra.match")
    chain = np.full((30, 4, 8), np.nan)
    chain[:12, 0] = source(1000, 12, 0.0, scatter=0, seed=0)
    chain[12:20, 1] = source(1000, 8, 0.1, scatter=0, seed=1)
    chain[20:, 2] = source(1000, 10, 0.3, scatter=0, seed=2)
    # Label 3 has label 0's waveform but shares its samples: a sample cannot hold one source twice, so it stays.
    chain[:, 3] = source(1000, 30, 0.0, scatter=0, seed=3)
    result = refine_catalog(PosteriorChain(chain, 4, 8, trans_dimensional=True), fake_waveforms)
    # Label 0 merges once per round: label 1 (match cos 0.1) in round 1, label 2 (cos 0.3) in round 2.
    assert summary(result) == [("merge", 1, 0, 8), ("merge", 2, 0, 10)] and result.rounds == 3
    np.testing.assert_allclose([action.match for action in result.actions], np.cos([0.1, 0.3]), atol=1e-12)
    assert "Merge: label 1 gave its 8 rows to label 0" in caplog.text
    np.testing.assert_array_equal(result.catalog.prob_in_model, [1, 0, 0, 1])
    np.testing.assert_array_equal(result.catalog.chain[:, 3], chain[:, 3])
    assert_rows_permuted(result.catalog.chain, chain)


@pytest.mark.parametrize("shift, freq_gate_bins, merged", [(1, 0, False), (1, 1, True), (0, 0, True)])
def test_merge_frequency_gate(shift, freq_gate_bins, merged):
    chain = np.full((20, 2, 8), np.nan)
    chain[:10, 0] = source(1000, 10, seed=1)
    chain[10:, 1] = source(1000 + shift, 10, seed=2)
    result = refine_catalog(PosteriorChain(chain, 2, 8, trans_dimensional=True), fake_waveforms,
                            freq_gate_bins=freq_gate_bins)
    assert [action.kind for action in result.actions] == (["merge"] if merged else [])


@pytest.mark.parametrize("options, kinds, rounds", [
    ({}, ["split", "merge"], 2),
    ({"max_rounds": 1}, ["split", "merge"], 1),
    ({"split": False}, [], 1),
    ({"merge": False}, ["split"], 2),
])
def test_rounds(options, kinds, rounds):
    # Label 0 holds A then B, label 1 holds B only where label 0 holds A, label 2 holds A only where label 0 holds B:
    # B splits into label 1, after which labels 0 and 2 never co-occur and merge.
    chain = mixed_label_chain(3)
    chain[:, 1] = np.nan
    chain[:20, 1] = source(1020, 20, seed=4)
    chain[30:, 2] = source(1000, 12, seed=5)
    result = refine_catalog(PosteriorChain(chain, 3, 8, trans_dimensional=True), fake_waveforms, **options)
    assert [action.kind for action in result.actions] == kinds and result.rounds == rounds


@pytest.mark.parametrize("options, message", [
    ({"threshold": 1.0}, "threshold"),
    ({"threshold": float("nan")}, "threshold"),
    ({"min_cluster_fraction": 1.0}, "min_cluster_fraction"),
    ({"max_rounds": 0}, "max_rounds"),
    ({"waveforms": lambda params: fake_waveforms(params[1:])}, r"data of shape \(83, 2, 16\).*84 present rows"),
])
def test_refine_catalog_rejects_bad_arguments(options, message):
    with pytest.raises(ValueError, match=message):
        refine_catalog(PosteriorChain(mixed_label_chain(3), 3, 8), **{"waveforms": fake_waveforms, **options})
