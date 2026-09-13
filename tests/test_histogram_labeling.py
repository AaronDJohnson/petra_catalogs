"""
Tests for the histogram-based initialization in :mod:`petra.initialization`.

:func:`petra.initialization.label_chain_by_histogram` is the cheap first guess
that gets similar samples into the same slot before the iterative relabelers
refine it.  It is not itself posterior-preserving in the "one slot per sample"
sense -- it may hand out more or fewer slots than it was given -- but it must
never create or destroy a sample, and it must separate two well-resolved
frequencies.
"""

import numpy as np
import pytest

from petra.initialization import (count_nonnan_samples, get_max_count_bin_center,
                                  label_chain_by_histogram, sort_array_by_sample_count,
                                  tetris_rise_nd)
from petra.posterior_chain import PosteriorChain
from petra.utils import frequency_bin_width


def test_get_max_count_bin_center_finds_the_most_populated_bin():
    delta_freq = 1e-8
    # Twelve entries in bin 100, four in bin 300, plus NaNs that must be ignored.
    chain = np.full((8, 2, 1), np.nan)
    chain[:6, 0, 0] = 100 * delta_freq
    chain[:6, 1, 0] = 100 * delta_freq
    chain[6:, 0, 0] = 300 * delta_freq
    chain[6:, 1, 0] = 300 * delta_freq

    assert get_max_count_bin_center(chain, delta_freq, 0) == pytest.approx(100 * delta_freq)


def test_tetris_rise_moves_filled_values_to_the_front():
    a = np.array([[np.nan, 1.0, np.nan, 2.0],
                  [3.0, np.nan, 4.0, np.nan]])
    risen = tetris_rise_nd(a)
    assert np.array_equal(risen, np.array([[1.0, 2.0, np.nan, np.nan],
                                           [3.0, 4.0, np.nan, np.nan]]), equal_nan=True)
    # Nothing is created or destroyed, only moved.
    assert np.count_nonzero(~np.isnan(risen)) == np.count_nonzero(~np.isnan(a))


def test_tetris_rise_honours_a_non_nan_empty_marker():
    a = np.array([[0.0, 5.0, 0.0, 7.0]])
    assert np.array_equal(tetris_rise_nd(a, empty=0.0), np.array([[5.0, 7.0, 0.0, 0.0]]))


def test_count_and_sort_by_sample_count():
    chain = np.full((10, 3, 1), np.nan)
    chain[:2, 0, 0] = 1.0
    chain[:9, 1, 0] = 2.0
    chain[:5, 2, 0] = 3.0

    assert count_nonnan_samples(chain, 0, 0) == 2
    assert count_nonnan_samples(chain, 1, 0) == 9
    assert count_nonnan_samples(chain, 2, 0) == 5

    sorted_chain = sort_array_by_sample_count(chain, 0)
    counts = [count_nonnan_samples(sorted_chain, i, 0) for i in range(3)]
    assert counts == [9, 5, 2]   # descending


def test_sample_count_ties_keep_reverse_slot_order_and_complete_source_vectors():
    # Include enough populated and empty slots to expose unstable tie ordering.
    chain = np.full((2, 32, 2), np.nan)
    for source_index, count in enumerate([2, 1, 0, 2] * 8):
        chain[:count, source_index, :] = [source_index + 1, 100 + source_index]
    original = chain.copy()

    sorted_chain = sort_array_by_sample_count(chain, relabeling_parameter=0)

    # Decreasing population, then decreasing original slot index within ties.
    expected_slots = [
        31, 28, 27, 24, 23, 20, 19, 16, 15, 12, 11, 8, 7, 4, 3, 0,
        29, 25, 21, 17, 13, 9, 5, 1,
        30, 26, 22, 18, 14, 10, 6, 2,
    ]
    np.testing.assert_array_equal(sorted_chain, original[:, expected_slots, :])
    np.testing.assert_array_equal(chain, original)


def test_label_chain_by_histogram_separates_two_frequencies():
    """Two well-resolved frequencies end up one per slot, however they arrived."""
    rng = np.random.default_rng(0)
    delta_freq = frequency_bin_width(1.0)
    n_samples = 40

    chain = np.empty((n_samples, 2, 1))
    chain[:, 0, 0] = 1e-3 + 0.1 * delta_freq * rng.normal(size=n_samples)
    chain[:, 1, 0] = 1e-3 + 50 * delta_freq + 0.1 * delta_freq * rng.normal(size=n_samples)
    for i in range(n_samples):
        chain[i] = chain[i, rng.permutation(2), :]
    pc = PosteriorChain(chain, 2, 1, trans_dimensional=True)

    labeled = label_chain_by_histogram(pc, obs_time_yrs=1.0, low_num_samples=1,
                                       num_extra_entries=4)

    assert labeled.num_sources == 2
    # No sample was created or destroyed by the labeling.
    assert np.count_nonzero(~np.isnan(labeled.chain)) == np.count_nonzero(~np.isnan(chain))
    # Each slot now holds exactly one of the two frequencies.
    slot_means = np.sort([np.nanmean(labeled.chain[:, i, 0]) for i in range(2)])
    assert slot_means[0] == pytest.approx(1e-3, abs=delta_freq)
    assert slot_means[1] == pytest.approx(1e-3 + 50 * delta_freq, abs=delta_freq)


def test_label_chain_by_histogram_refuses_to_run_out_of_labels():
    """
    Too little head-room must raise, not silently index past the array.

    Each sample sits at its own pair of frequencies, so the labeling needs four
    labels for a chain that only has two source slots -- exactly the case
    `num_extra_entries` exists for.
    """
    delta_freq = frequency_bin_width(1.0)
    chain = np.array([
        [[1e-3 + 0 * delta_freq], [1e-3 + 100 * delta_freq]],
        [[1e-3 + 200 * delta_freq], [1e-3 + 300 * delta_freq]],
    ])
    pc = PosteriorChain(chain, 2, 1, trans_dimensional=True)

    with pytest.raises(ValueError, match="num_extra_entries"):
        label_chain_by_histogram(pc, obs_time_yrs=1.0, num_extra_entries=0,
                                 low_num_samples=1)
