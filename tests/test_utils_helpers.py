"""
Small pure helpers in :mod:`petra.utils`, and the guards they carry.

None of these need a chain or a flow, so they are the cheapest place to pin down
behaviour that the rest of the package silently depends on: the uniform-prior
fallback used whenever a source slot is too sparse to fit, and the
duplicate-breaking that keeps the empirical CDF spline monotonic.

`count_swaps` is the exception: it is exported, it is deprecated, and it has no
caller inside petra at all, so these are the only tests it has.  They pin the
arithmetic *and* the retirement notice, because a deprecation nobody can hear is
the same as no deprecation.
"""

import numpy as np
import pytest

from petra.utils import (UniformPrior, count_swaps, fill_missing_indices,
                         find_prob_in_model, find_uniform_bounds, frequency_bin_width,
                         make_uniform_prior, process_array, sort_by_number,
                         source_present)


def _trans_dimensional_chain(seed=0, n_samples=40, n_sources=4, n_params=5):
    """A chain that respects the convention: every source row is all-NaN or all-finite."""
    rng = np.random.default_rng(seed)
    chain = rng.standard_normal((n_samples, n_sources, n_params))
    absent = rng.random((n_samples, n_sources)) < 0.4
    chain[absent] = np.nan
    # Slots that are never, and always, present: the ends of the range matter.
    chain[:, 0, :] = np.nan
    chain[:, 1, :] = rng.standard_normal((n_samples, n_params))
    return chain


# ---------------------------------------------------------------------------
# UniformPrior
# ---------------------------------------------------------------------------

def test_uniform_prior_normalizes_over_its_support():
    prior = UniformPrior([0.0, 0.0], [2.0, 4.0])
    assert prior.n_params == 2
    assert float(prior.log_prob(np.array([1.0, 1.0]))) == pytest.approx(-np.log(8.0))
    assert float(prior.log_prob(np.array([3.0, 1.0]))) == -np.inf
    assert float(prior.log_prob(np.array([np.nan, 1.0]))) == -np.inf
    assert "UniformPrior(n_params=2" in repr(prior)


def test_uniform_prior_ignores_a_zero_width_dimension():
    """A parameter that never varies carries no information and must not blow up."""
    prior = UniformPrior([0.0, 1.0], [2.0, 1.0])
    assert np.isfinite(prior.log_density)
    assert float(prior.log_prob(np.array([1.0, 1.0]))) == pytest.approx(-np.log(2.0))
    # The degenerate dimension no longer constrains the support.
    assert float(prior.log_prob(np.array([1.0, 5.0]))) == pytest.approx(-np.log(2.0))


@pytest.mark.parametrize("mins,maxs,message", [
    ([0.0, 0.0], [1.0], "same length"),
    ([0.0], [np.inf], "finite"),
    ([1.0], [0.0], "maxs >= mins"),
    ([1.0], [1.0], "non-zero width"),
])
def test_uniform_prior_rejects_impossible_bounds(mins, maxs, message):
    with pytest.raises(ValueError, match=message):
        UniformPrior(mins, maxs)


def test_uniform_prior_rejects_a_mis_shaped_sample():
    prior = UniformPrior([0.0], [1.0])
    with pytest.raises(ValueError, match="1D or 2D"):
        prior.log_prob(np.zeros((2, 2, 1)))
    with pytest.raises(ValueError, match="parameters"):
        prior.log_prob(np.zeros(3))


def test_make_uniform_prior_spans_the_chain_and_ignores_nans():
    chain = np.array([
        [[0.0, 0.0], [np.nan, np.nan]],
        [[2.0, 4.0], [1.0, 1.0]],
    ])
    lower, upper = find_uniform_bounds(chain)
    assert np.array_equal(lower, [0.0, 0.0])
    assert np.array_equal(upper, [2.0, 4.0])
    assert make_uniform_prior(chain).log_density == pytest.approx(-np.log(8.0))


# find_uniform_bounds reaches np.nanmin on the all-NaN column before
# make_uniform_prior gets a chance to raise, so numpy warns on the way past.
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_make_uniform_prior_refuses_an_all_nan_parameter():
    chain = np.array([[[0.0, np.nan]], [[2.0, np.nan]]])
    with pytest.raises(ValueError, match="NaN in every"):
        make_uniform_prior(chain)


# ---------------------------------------------------------------------------
# count_swaps, process_array and friends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("array,expected", [
    ([1, 2, 3], 0),
    ([2, 1, 3], 1),
    ([3, 1, 2], 2),         # one 3-cycle
    ([1, 0, 3, 2], 2),      # two disjoint 2-cycles
    ([4, 3, 2, 1], 2),
    ([1, 1, 1], 0),         # ties are broken by position, so nothing moves
])
def test_count_swaps(array, expected):
    """Deprecating it must not change what it computes for the release it survives."""
    with pytest.warns(DeprecationWarning, match="scheduled for removal in petra 1.2"):
        assert count_swaps(np.array(array)) == expected


def test_count_swaps_announces_its_own_scheduled_removal():
    """
    The trap this closes: public, tested, and with zero callers inside petra.

    Same treatment as `deprecated_keyword`, deliberately -- a dead public helper
    is announced and dropped on a named release rather than left to rot or
    deleted out from under whoever imported it.
    """
    with pytest.warns(DeprecationWarning) as caught:
        count_swaps(np.array([2, 1, 3]))
    assert [str(w.message) for w in caught] == [
        "petra.utils.count_swaps() is deprecated with no replacement and is "
        "scheduled for removal in petra 1.2."
    ]


def test_count_swaps_points_its_retirement_notice_at_the_caller():
    """stacklevel must blame the call site, not utils.py, or the notice is useless."""
    def caller():
        """Stand-in for the user script that still reaches for the helper."""
        return count_swaps(np.array([2, 1, 3]))

    with pytest.warns(DeprecationWarning, match="count_swaps") as caught:
        caller()
    assert caught[0].filename == __file__
    assert caught[0].lineno == caller.__code__.co_firstlineno + 2


def test_process_array_breaks_ties_without_reordering():
    """Duplicates get nudged apart, which is what the CDF spline needs."""
    arr = np.array([1.0, 1.0, 1.0, 2.0])
    processed = process_array(arr)
    assert len(np.unique(processed)) == len(processed)
    assert np.all(np.diff(processed) > 0)
    assert processed[0] == pytest.approx(1.0)
    assert processed[-1] == pytest.approx(2.0)


def test_process_array_sorts_and_leaves_unique_input_alone():
    arr = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(process_array(arr), np.array([1.0, 2.0, 3.0]))


def test_process_array_handles_an_all_identical_input():
    processed = process_array(np.full(4, 5.0))
    assert len(np.unique(processed)) == 4
    assert np.all(np.diff(processed) > 0)


def test_fill_missing_indices_appends_what_the_assignment_left_out():
    assert np.array_equal(fill_missing_indices(5, [2, 0]), np.array([2, 0, 1, 3, 4]))


def test_sort_by_number_orders_numerically_not_lexicographically():
    assert sort_by_number(['f.10', 'f.2', 'f.1']) == ['f.1', 'f.2', 'f.10']


@pytest.mark.parametrize("obs_time_yrs", [0.0, -1.0, np.nan])
def test_frequency_bin_width_rejects_a_non_positive_observation_time(obs_time_yrs):
    with pytest.raises(ValueError):
        frequency_bin_width(obs_time_yrs)


def test_frequency_bin_width_of_one_year():
    assert frequency_bin_width(1.0) == pytest.approx(1.0 / (525600 * 60), rel=1e-12)


# ---------------------------------------------------------------------------
# source_present: the one reading of the all-or-nothing NaN convention
# ---------------------------------------------------------------------------

def test_source_present_agrees_with_every_old_spelling():
    """
    Presence used to be spelled three ways -- column 0 only, all columns, and one
    chosen column -- which agree exactly as long as the convention holds.  Pin
    that down, because the shared predicate has to be a drop-in for all three.
    """
    chain = _trans_dimensional_chain()
    present = source_present(chain)
    assert present.shape == chain.shape[:2]

    for i in range(chain.shape[1]):
        column_zero = ~np.isnan(chain[:, i, 0])                    # find_prob_in_model
        all_columns = ~np.isnan(chain[:, i, :]).any(axis=1)        # every fitter
        assert np.array_equal(present[:, i], column_zero)
        assert np.array_equal(present[:, i], all_columns)
        for k in range(chain.shape[2]):                            # any parameter column
            assert np.array_equal(present[:, i], ~np.isnan(chain[:, i, k]))


def test_source_present_reduces_only_the_parameter_axis():
    """One predicate has to serve a whole chain, one source, one sample, one row."""
    chain = _trans_dimensional_chain()
    present = source_present(chain)

    assert np.array_equal(source_present(chain[:, 2, :]), present[:, 2])   # one source
    assert np.array_equal(source_present(chain[7]), present[7])            # one sample
    assert bool(source_present(chain[7, 2])) == bool(present[7, 2])        # one row


def test_source_present_calls_a_partial_row_absent():
    """
    A row that violates the convention is where the three old spellings parted
    company: column 0 called it present, the fitters dropped it.  The predicate
    sides with the fitters -- no fit can use half a source.
    """
    row = np.array([1.0, 2.0, np.nan])
    assert bool(source_present(row)) is False
    assert bool(~np.isnan(row[0])) is True          # the reading it replaces


def test_find_prob_in_model_reads_the_whole_row():
    """The inclusion probability must describe the samples the fits actually saw."""
    chain = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, np.nan], [np.nan, np.nan]],
    ])
    assert np.array_equal(find_prob_in_model(chain, 2, eps=0), np.array([0.5, 0.5]))


def test_find_prob_in_model_matches_the_column_zero_loop_on_a_legal_chain():
    """Converting to the shared predicate must not move a single number."""
    chain = _trans_dimensional_chain(seed=3, n_samples=64, n_sources=6, n_params=8)
    num_samples = chain.shape[0]
    for max_num_sources in range(1, chain.shape[1] + 1):
        expected = np.zeros(max_num_sources)
        for i in range(max_num_sources):
            expected[i] = np.sum(~np.isnan(chain[:, i, 0])) / num_samples
        for eps in (0.0, 1e-6, 1e-2):
            assert np.array_equal(find_prob_in_model(chain, max_num_sources, eps=eps),
                                  np.clip(expected, eps, 1 - eps))


def test_find_prob_in_model_rejects_more_sources_than_the_chain_has():
    """Slicing would return a short array that callers then index by source label."""
    with pytest.raises(ValueError, match="exceeds the 2 source slots"):
        find_prob_in_model(np.zeros((5, 2, 3)), max_num_sources=3)
