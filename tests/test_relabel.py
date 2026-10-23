"""Tests of petra.relabel: one assignment, chain preparation, the shared loop and its split moves."""

import logging
import warnings
from functools import partial

import numpy as np
import pytest
from scipy import optimize

from petra import make_catalog
from petra import relabel as relabel_module
from petra.checkpoint import load_checkpoint
from petra.fits import compute_cost_matrix, mv_normal_fit, mv_normal_log_prob
from petra.make_catalog import make_catalog_mv_normal, relabel_mv_normal
from petra.posterior_chain import PosteriorChain
from petra.relabel import (
    _lloyd_two_means,
    _one_dimensional_two_means,
    _opened_label_cost,
    _propose_split,
    _split_candidates,
    _two_means_split,
    check_chain,
    prepare_chain,
    relabel_samples_one_iteration,
    run_relabeling_loop,
)
from petra.utils import find_prob_in_model, source_present
from tests.helpers import (
    EMPTY_LABEL_WARNING,
    REF,
    budget_message,
    chain_hash,
    dense_chain,
    empty_slot_chain,
    golden_chain,
    never_cooccurring_chain,
    packed_chain,
    same_catalog,
    separated_chain,
    three_full_labels_chain,
    trans_dimensional_chain,
    two_clusters,
    widening_chain,
)

MV_COST = partial(compute_cost_matrix, log_prob=mv_normal_log_prob)

#: The rows the old one-iteration test pinned at the two ends of the golden chain.
GOLDEN_FIRST3 = [[[0.5488135], [0.74826798]], [[0.71518937], [0.18020271]], [[0.60276338], [0.38902314]]]
GOLDEN_LAST3 = [[[0.44645576], [0.75842952]], [[0.36012661], [0.02378743]], [[0.62588665], [0.81357508]]]


def shuffled_three_clusters() -> np.ndarray:
    """Three clusters in two parameters, the third absent from about half the samples, slots permuted per sample."""
    rng = np.random.default_rng(0)
    chain = rng.normal(loc=[[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]], scale=0.3, size=(80, 3, 2))
    chain[rng.random(80) < 0.5, 2, :] = np.nan
    for i in range(80):
        chain[i] = chain[i, rng.permutation(3), :]
    return chain


def assert_rows_permuted(result: np.ndarray, original: np.ndarray) -> None:
    """Assert that every sample of `result` holds exactly the whole rows of the same sample of `original`."""
    assert result.shape == original.shape
    for got, want in zip(result, original):
        assert np.array_equal(got[np.lexsort(got.T[::-1])], want[np.lexsort(want.T[::-1])], equal_nan=True)


def fit_spy() -> tuple:
    """Return the multivariate-normal fit wrapped to record each call, and the list it records into."""
    calls = []

    def fit(chain, max_num_sources):
        calls.append(np.array(chain, copy=True))
        return mv_normal_fit(chain, max_num_sources)

    return fit, calls


def relabel_messages(caplog: pytest.LogCaptureFixture, anchor: str) -> list:
    """Return the petra.relabel log records whose message contains `anchor`."""
    return [record for record in caplog.records if record.name == "petra.relabel" and anchor in record.getMessage()]


def recorded(run) -> tuple:
    """Call `run()` and return its result and the text of every warning it gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    return result, [str(warning.message) for warning in caught]


def run_state(directory) -> tuple:
    """Return (iteration, stopped, stop_reason) of the newest checkpoint in `directory`."""
    state = load_checkpoint(str(directory)).state
    return state["iteration"], state["stopped"], state["stop_reason"]


# One assignment


def test_one_iteration_reproduces_the_golden_values():
    """One assignment of the golden chain gives the reference cost and labeling."""
    chain = golden_chain().get_chain()
    relabeled, total_cost = relabel_samples_one_iteration(chain, mv_normal_fit(chain, 2),
                                                          find_prob_in_model(chain, 2, eps=0), 2, MV_COST,
                                                          progress=False)

    assert abs(total_cost - REF["golden_low_level"]["total_cost"]) <= 1e-12
    assert chain_hash(relabeled) == REF["golden_low_level"]["chain_hash"]
    assert np.allclose(relabeled[:3], GOLDEN_FIRST3) and np.allclose(relabeled[-3:], GOLDEN_LAST3)


def test_one_iteration_is_a_permutation_and_keeps_every_source_of_a_wider_chain():
    """Each sample is reordered, never changed; slots beyond K survive; a square sample takes the solver's order."""
    chain = shuffled_three_clusters()
    relabeled, _ = relabel_samples_one_iteration(chain, mv_normal_fit(chain, 3), find_prob_in_model(chain, 3), 3,
                                                 MV_COST, progress=False)
    assert_rows_permuted(relabeled, chain)

    wide = np.random.default_rng(0).normal(scale=0.2, size=(25, 3, 1)) + np.array([[0.0], [5.0], [10.0]])
    relabeled, cost = relabel_samples_one_iteration(wide, mv_normal_fit(wide, 2), find_prob_in_model(wide, 2), 2,
                                                    MV_COST, progress=False)
    assert_rows_permuted(relabeled, wide)
    assert np.isfinite(cost)

    square = np.random.default_rng(1).normal(scale=0.2, size=(25, 3, 1)) + np.array([[0.0], [5.0], [10.0]])
    fit, prob_in_model = mv_normal_fit(square, 3), find_prob_in_model(square, 3)
    relabeled, _ = relabel_samples_one_iteration(square, fit, prob_in_model, 3, MV_COST, progress=False)
    for sample, got in zip(square, relabeled):
        _, col_ind = optimize.linear_sum_assignment(MV_COST(sample, fit, prob_in_model, 3), maximize=True)
        assert np.array_equal(got, sample[col_ind])


# Preparing the chain and checking the arguments


@pytest.mark.parametrize("widen_first", [False, True])
def test_prepare_chain_widens_and_shuffles_in_the_requested_order(widen_first, monkeypatch):
    """Nothing to do returns the input; added slots stay empty unless widened first; the input is untouched."""
    as_wide = PosteriorChain(np.zeros((5, 3, 2)), 3, 2)
    assert prepare_chain(as_wide, widen_first=widen_first) is as_wide
    assert prepare_chain(as_wide, 3, widen_first=widen_first) is as_wide

    rng = np.random.default_rng(10)
    chain = rng.normal(size=(300, 2, 3))
    chain[rng.random((300, 2)) < 0.3] = np.nan
    snapshot = chain.copy()
    input_prob = find_prob_in_model(chain, 2, eps=0)
    pc = PosteriorChain(chain, 2, 3, trans_dimensional=True, prob_in_model=input_prob.copy(), cost_dict={2: -1.0})

    prepared = prepare_chain(pc, 4, shuffle_seed=1, widen_first=widen_first)

    if widen_first:
        assert np.all(prepared.prob_in_model[2:] > 0.0)
        assert np.all(source_present(prepared.chain)[:, 2:].mean(axis=0) > 0.2)
    else:
        assert np.array_equal(prepared.prob_in_model[2:], [0.0, 0.0])
    padded = np.concatenate([snapshot, np.full((300, 2, 3), np.nan)], axis=1)
    assert not np.array_equal(prepared.chain, padded, equal_nan=True)
    assert_rows_permuted(prepared.chain, padded)
    assert np.array_equal(prepared.prob_in_model, find_prob_in_model(prepared.get_chain(), 4, eps=0))
    assert prepared.cost_dict == {}
    assert np.array_equal(chain, snapshot, equal_nan=True)
    assert np.array_equal(pc.prob_in_model, input_prob) and pc.cost_dict == {2: -1.0}

    if widen_first:
        seen = []
        real = make_catalog.prepare_chain

        def spy(*args, **kwargs):
            seen.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(make_catalog, "prepare_chain", spy)
        make_catalog_mv_normal(PosteriorChain(two_clusters(), 2, 1), 2, shuffle_seed=3, num_iterations=2,
                               progress=False)
        assert seen == [{"shuffle_seed": 3, "widen_first": True}]


#: The three relabel functions that check a chain and its width, each reduced to ``(chain, width, fit) -> width``.
LEADING = {
    "check_chain": lambda pc, width, fit: check_chain(pc, width, required=True),
    "prepare_chain": lambda pc, width, fit: prepare_chain(pc, width).num_sources,
    "run_relabeling_loop": lambda pc, width, fit: next(iter(run_relabeling_loop(
        pc, fit, MV_COST, width, num_iterations=2, progress=False).cost_dict)),
}


@pytest.mark.parametrize("name", sorted(LEADING))
def test_leading_arguments_are_checked_before_any_work(name):
    """A bare array, an empty chain and a bad width are refused before any fit; a NumPy width becomes an int."""
    call = LEADING[name]
    fit, fitted = fit_spy()
    pc = PosteriorChain(separated_chain(), 2, 1)

    with pytest.raises(TypeError, match="posterior_chain must be a petra.PosteriorChain"):
        call(pc.get_chain(), 2, fit)
    with pytest.raises(ValueError, match="holds no samples"):
        call(PosteriorChain(np.zeros((0, 2, 1)), 2, 1), 2, fit)
    for width in [3.0, True, "2"] + ([None] if name == "check_chain" else []):
        with pytest.raises(ValueError, match="max_num_sources must be an integer >="):
            call(pc, width, fit)
    with pytest.raises(ValueError, match="cannot be less than the number of entries in the chain"):
        call(pc, 1, fit)
    assert fitted == []

    width = call(pc, np.int64(3), fit)
    assert type(width) is int and width == 3


def test_shuffle_seed_is_none_or_a_non_negative_integer():
    """Bools, floats, lists and Generators are refused without being consumed; a NumPy integer reproduces."""
    pc = PosteriorChain(separated_chain(), 2, 1)
    for bad in (True, 1.5, [1]):
        with pytest.raises(TypeError, match="shuffle_seed must be None or a non-negative integer"):
            prepare_chain(pc, 2, shuffle_seed=bad)
    generator = np.random.default_rng(0)
    state = generator.bit_generator.state
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        prepare_chain(pc, 2, shuffle_seed=generator)
    assert generator.bit_generator.state == state
    with pytest.raises(ValueError, match="shuffle_seed"):
        prepare_chain(pc, 2, shuffle_seed=-1)

    first = prepare_chain(pc, 2, shuffle_seed=np.int64(3)).get_chain()
    assert np.array_equal(first, prepare_chain(pc, 2, shuffle_seed=np.int64(3)).get_chain())
    assert np.array_equal(first, prepare_chain(pc, 2, shuffle_seed=3).get_chain())
    assert not np.array_equal(first, pc.get_chain())


#: Stands for the test's checkpoint directory in the parameters below.
CHECKPOINTS = object()


@pytest.mark.parametrize("settings, anchor", [
    ({"num_iterations": 0}, "num_iterations must be an integer >="),
    ({"num_iterations": 1.5}, "num_iterations must be an integer >="),
    ({"num_iterations": True}, "num_iterations must be an integer >="),
    ({"eps": -0.1}, "eps must be finite and between 0 and 0.5"),
    ({"eps": 0.6}, "eps must be finite and between 0 and 0.5"),
    ({"eps": np.nan}, "eps must be finite and between 0 and 0.5"),
    ({"keep_checkpoints": 0}, "keep_checkpoints must be an integer >="),
    ({"checkpoint_dir": CHECKPOINTS}, "needs method="),
    ({"resume_from": CHECKPOINTS}, "needs method="),
    ({"eps": 0.0}, None),
    ({"eps": 0.5}, None),
], ids=["iterations-0", "iterations-1.5", "iterations-True", "eps--0.1", "eps-0.6", "eps-nan", "keep-0",
        "checkpoint_dir-without-method", "resume_from-without-method", "eps-0", "eps-0.5"])
def test_loop_settings_are_checked_before_any_work(settings, anchor, tmp_path):
    """Every bad setting is refused before the first fit and before a directory is made; eps 0 and 0.5 run."""
    target = tmp_path / "checkpoints"
    kwargs = {key: str(target) if value is CHECKPOINTS else value for key, value in settings.items()}
    fit, fitted = fit_spy()

    def run():
        return run_relabeling_loop(PosteriorChain(separated_chain(), 2, 1), fit, MV_COST, 2,
                                   **{"num_iterations": 2, "progress": False, **kwargs})

    if anchor is None:
        assert run().cost_dict.keys() == {2}
        return
    with pytest.raises(ValueError, match=anchor):
        run()
    assert fitted == []
    assert not target.exists()


# The loop


@pytest.mark.parametrize("costs, budget, returned", [
    ([3.0, 1.0, 2.0], 10, 2),
    ([3.0, 2.0, 2.0], 10, 2),
    ([3.0, 2.0, 1.0], 3, 3),
], ids=["a-rise-stops-it", "converged", "budget-spent"])
def test_loop_returns_the_cheapest_labeling_of_a_scripted_trajectory(scripted_costs, costs, budget, returned):
    """The cheapest labeling is returned, the earlier of a tie, with its cost recorded under the width."""
    calls = scripted_costs(costs)

    result = run_relabeling_loop(PosteriorChain(separated_chain(), 2, 1), mv_normal_fit, MV_COST, 2,
                                 num_iterations=budget, progress=False)

    assert len(calls) == 3
    assert np.array_equal(result.get_chain(), separated_chain(offset=float(returned)))
    assert result.cost_dict == {2: min(costs)}


def test_each_iteration_is_assigned_under_the_fit_of_the_labeling_before_it(scripted_costs):
    """Iteration k + 1 gets the fit of the chain iteration k returned, and every iteration fits once."""
    calls = scripted_costs([-1.0, -2.0, -3.0, -4.0])
    fit, fitted = fit_spy()
    start = PosteriorChain(separated_chain(), 2, 1)

    run_relabeling_loop(start, fit, MV_COST, 2, num_iterations=4, progress=False)

    assert len(calls) == 4 and len(fitted) == 5
    assert np.array_equal(fitted[0], start.get_chain())
    for k in range(3):
        returned = separated_chain(offset=k + 1.0)
        assert np.array_equal(fitted[k + 1], returned)
        for given, expected in zip(calls[k + 1]["aux_parameters"], mv_normal_fit(returned, 2)):
            assert np.array_equal(given, expected)


@pytest.mark.parametrize("build, expected", [
    (dense_chain, [1.0, 1.0, 1.0]),
    (empty_slot_chain, [1.0, 1.0, 1.0, 0.0]),
    (trans_dimensional_chain, None),
], ids=lambda value: getattr(value, "__name__", ""))
def test_returned_prob_in_model_is_unclipped_while_the_cost_matrix_sees_it_clipped(build, expected, monkeypatch):
    """The result's prob_in_model is read off its chain unclipped; each cost matrix sees it clipped by eps."""
    seen = []
    real = relabel_module.relabel_samples_one_iteration

    def spy(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix, progress=True):
        seen.append(np.array(prob_in_model, copy=True))
        return real(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix, progress=progress)

    monkeypatch.setattr(relabel_module, "relabel_samples_one_iteration", spy)
    pc = build()
    width = pc.num_sources

    result = run_relabeling_loop(pc, mv_normal_fit, MV_COST, width, num_iterations=5, eps=1e-3, progress=False)

    assert np.array_equal(result.prob_in_model, find_prob_in_model(result.get_chain(), width, eps=0))
    if expected is None:
        assert np.all((result.prob_in_model > 0.0) & (result.prob_in_model < 1.0))
    else:
        assert np.array_equal(result.prob_in_model, expected)
    assert seen and all(prob.min() >= 1e-3 and prob.max() <= 1.0 - 1e-3 for prob in seen)

    for copy, validate in [(True, True), (False, False)]:
        source = PosteriorChain(pc.get_chain(), width, pc.num_params_per_source, trans_dimensional=True,
                                prob_in_model=np.full(width, 0.5), cost_dict={width: -1.0}, copy=copy,
                                validate_nan_convention=validate)
        unclipped = relabel_module._with_unclipped_prob_in_model(source, width)
        assert np.shares_memory(unclipped.chain, source.chain)
        assert unclipped.cost_dict == source.cost_dict and unclipped.cost_dict is not source.cost_dict
        assert (unclipped.copy, unclipped.validate_nan_convention) == (copy, validate)
        assert unclipped.trans_dimensional is True
        assert np.array_equal(unclipped.prob_in_model, find_prob_in_model(source.get_chain(), width, eps=0))
        assert np.array_equal(source.prob_in_model, np.full(width, 0.5))


def test_the_recorded_cost_is_the_previous_fits_cost_reproduced_only_at_a_fixed_point(caplog):
    """A refit of the result rescores it to cost_dict[K] after a converged run, not after one cut short."""
    eps = 1e-2
    rng = np.random.default_rng(4)
    chain = rng.normal(scale=1.0, size=(60, 3, 2)) + np.array([[0.0, 0.0], [4.0, 4.0], [8.0, 0.0]])
    pc = PosteriorChain(np.stack([chain[i, rng.permutation(3)] for i in range(60)]), 3, 2)

    def rescore(result):
        labeling = result.get_chain()
        return relabel_samples_one_iteration(labeling, mv_normal_fit(labeling, 3),
                                             find_prob_in_model(labeling, 3, eps=eps), 3, MV_COST, progress=False)

    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        converged = run_relabeling_loop(pc, mv_normal_fit, MV_COST, 3, num_iterations=20, eps=eps, progress=False)

    lines = [record.getMessage() for record in relabel_messages(caplog, "cost of assignment is")]
    assert len(lines) > 2
    assert "there is no previous cost" in lines[0]
    assert not any("there is no previous cost" in line for line in lines[1:])
    relabeled, cost = rescore(converged)
    assert np.array_equal(relabeled, converged.get_chain())
    assert cost == pytest.approx(converged.cost_dict[3], rel=1e-12, abs=1e-12)

    cut_short = run_relabeling_loop(pc, mv_normal_fit, MV_COST, 3, num_iterations=1, eps=eps, progress=False)
    relabeled, cost = rescore(cut_short)
    assert not np.array_equal(relabeled, cut_short.get_chain())
    assert cost != pytest.approx(cut_short.cost_dict[3], rel=1e-6)


def test_the_loop_leaves_its_input_alone_keeps_the_opt_out_and_ignores_trans_dimensional():
    """The input survives a shuffled, widened run; an opted-out partial row comes through; the flag changes nothing."""
    chain = shuffled_three_clusters()
    snapshot, input_prob = chain.copy(), np.array([1.0, 1.0, 0.6])
    pc = PosteriorChain(chain, 3, 2, trans_dimensional=True, prob_in_model=input_prob.copy(), cost_dict={3: -1.0})

    catalog = make_catalog_mv_normal(pc, 4, shuffle_seed=7, num_iterations=10, init_num_iterations=10,
                                     progress=False)

    assert chain.tobytes() == snapshot.tobytes() and pc.chain.tobytes() == snapshot.tobytes()
    assert pc.prob_in_model.tobytes() == input_prob.tobytes() and pc.cost_dict == {3: -1.0}
    assert_rows_permuted(catalog.chain, np.concatenate([snapshot, np.full((80, 1, 2), np.nan)], axis=1))

    rng = np.random.default_rng(0)
    partial_row = rng.normal(scale=0.2, size=(20, 2, 2)) + np.array([[0.0, 0.0], [5.0, 5.0]])
    partial_row[3, 1, 1] = np.nan
    opted_out = relabel_mv_normal(PosteriorChain(partial_row, 2, 2, trans_dimensional=True,
                                                 validate_nan_convention=False),
                                  max_num_sources=3, num_iterations=2, progress=False)
    assert opted_out.validate_nan_convention is False and opted_out.num_sources == 3
    assert np.count_nonzero(np.isnan(opted_out.get_chain()).sum(axis=-1) == 1) == 1
    whole_rows = ~np.isnan(opted_out.get_chain()).any(axis=-1)
    assert np.array_equal(opted_out.prob_in_model, whole_rows.mean(axis=0))

    rng = np.random.default_rng(42)
    flagged = rng.normal(loc=[[1.0, 2.0], [4.0, 5.0], [7.0, 1.5]], scale=0.3, size=(60, 3, 2))
    flagged[:20, 2, :] = np.nan
    for sample in flagged:
        rng.shuffle(sample, axis=0)
    for width in (3, 5):
        declared, declared_warnings = recorded(lambda: make_catalog_mv_normal(
            PosteriorChain(flagged, 3, 2, trans_dimensional=True), width, num_iterations=5, init_num_iterations=5,
            progress=False))
        undeclared, undeclared_warnings = recorded(lambda: make_catalog_mv_normal(
            PosteriorChain(flagged, 3, 2, trans_dimensional=False), width, num_iterations=5, init_num_iterations=5,
            progress=False))
        assert same_catalog(declared, undeclared) and declared_warnings == undeclared_warnings
        assert declared.trans_dimensional is True and undeclared.trans_dimensional is (width > 3)


# Split moves


def _separation_tells_merged_sources_from_one_source(monkeypatch):
    """One normal cloud scores below the threshold; two or four merged sources clear it."""
    rng = np.random.default_rng(7)
    one = rng.normal(size=(500, 3))
    two = np.concatenate([rng.normal(size=(300, 3)), rng.normal(size=(100, 3)) + 6.0])
    four = np.concatenate([k * 10.0 + rng.normal(size=(n, 2)) for k, n in enumerate((280, 236, 150, 99))])
    assert _two_means_split(one)[1] < relabel_module._SPLIT_MIN_SEPARATION
    assert _two_means_split(two)[1] > relabel_module._SPLIT_MIN_SEPARATION
    assert _two_means_split(four)[1] > 10


def _a_non_dyadic_constant_is_constant_to_the_split(monkeypatch):
    """A column held at 0.3 neither blocks nor fakes a split."""
    rng = np.random.default_rng(8)
    rows = np.column_stack([np.where(np.arange(100) < 60, 0.0, 9.0) + rng.normal(size=100), np.full(100, 0.3)])
    score, separation, smaller = _two_means_split(rows)
    assert separation > 10 and score > 0.4
    np.testing.assert_array_equal(np.flatnonzero(smaller), np.arange(60, 100))
    assert _two_means_split(np.full((400, 2), 0.3))[:2] == (0.0, 0.0)


def _a_split_that_rounding_empties_is_no_split(monkeypatch):
    """Rows differing from their mean in the last bit only give no split."""
    rows = np.r_[np.ones(399), 1.0 + np.finfo(float).eps][:, None]
    score, separation, smaller = _two_means_split(rows)
    assert (score, separation) == (0.0, 0.0) and not smaller.any()


def _lloyds_iterations_stop_at_their_cap(monkeypatch):
    """With one Lloyd iteration allowed, the first assignment is returned."""
    monkeypatch.setattr(relabel_module, "_TWO_MEANS_ITERATIONS", 1)
    within, in_second = _lloyd_two_means(np.array([[-3.0], [-1.0], [0.5], [1.0], [3.0]]), np.array([1.0]))
    assert in_second.tolist() == [True, True, False, False, False] and within > 0


def _the_one_dimensional_two_means_is_exact(monkeypatch):
    """The 1-D cut is the brute-force optimum."""
    values = np.random.default_rng(9).normal(size=40)
    low, high = _one_dimensional_two_means(values)
    ordered = np.sort(values)

    def within(cut):
        return ((ordered[:cut] - ordered[:cut].mean()) ** 2).sum() + ((ordered[cut:] - ordered[cut:].mean()) ** 2).sum()

    assert low.size + high.size == 40 and low.max() < high.min()
    assert within(low.size) == pytest.approx(min(within(cut) for cut in range(1, 40)))


@pytest.mark.parametrize("case", [
    _separation_tells_merged_sources_from_one_source,
    _a_non_dyadic_constant_is_constant_to_the_split,
    _a_split_that_rounding_empties_is_no_split,
    _lloyds_iterations_stop_at_their_cap,
    _the_one_dimensional_two_means_is_exact,
], ids=lambda case: case.__name__.strip("_"))
def test_two_means_split_scores_and_separates(case, monkeypatch):
    """The 2-means split's score, separation and smaller cluster on the old pinned cases."""
    case(monkeypatch)


@pytest.mark.parametrize("block_rows", [None, 4], ids=["default-block", "block-4"])
@pytest.mark.parametrize("exclusive", [False, True], ids=["together", "exclusive"])
def test_split_candidates_rank_donors_and_receivers_and_need_clusters_seen_together(
        exclusive, block_rows, monkeypatch, caplog):
    """Donors by score, receivers emptiest first; small, lopsided and unimodal labels never donate."""
    if block_rows is not None:
        monkeypatch.setattr(relabel_module, "_BLOCK_ROWS", block_rows)

    rng = np.random.default_rng(4)
    ranked = np.full((200, 5, 1), np.nan)
    ranked[:, 0, 0] = np.where(np.arange(200) < 100, 0.0, 40.0) + rng.normal(size=200)
    ranked[:, 1, 0] = np.where(np.arange(200) < 150, 0.0, 40.0) + rng.normal(size=200)
    ranked[:, 2, 0] = rng.normal(size=200)
    ranked[:20, 3, 0] = rng.normal(size=20)
    proposals = _split_candidates(ranked, exclusive)
    assert [(donor, receiver) for donor, receiver, _, _ in proposals] == [
        (0, 4), (0, 3), (0, 1), (1, 4), (1, 3), (1, 0)]
    assert proposals[0][3] > proposals[3][3]
    np.testing.assert_array_equal(np.sort(proposals[3][2]), np.arange(150, 200))
    assert proposals[0][2].size == 100

    rng = np.random.default_rng(5)
    no_donor = np.full((200, 4, 2), np.nan)
    no_donor[:7, 0] = rng.normal(size=(7, 2))
    no_donor[:, 1] = rng.normal(size=(200, 2))
    no_donor[:, 2] = rng.normal(size=(200, 2))
    no_donor[:3, 2] += 40.0
    assert _split_candidates(no_donor, exclusive) == []

    single = (np.where(np.arange(40)[:, None, None] < 20, 0.0, 30.0)
              + np.random.default_rng(6).normal(size=(40, 1, 1)))
    assert _two_means_split(single[:, 0])[1] > 10
    assert _split_candidates(single, exclusive) == []

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        never_together = _split_candidates(never_cooccurring_chain(), exclusive)
    kept_whole = [record for record in relabel_messages(caplog, "split_exclusive=True")
                  if record.levelno == logging.INFO]
    if exclusive:
        assert [(donor, receiver) for donor, receiver, _, _ in never_together] == [(1, 0)] and kept_whole == []
    else:
        assert never_together == [] and len(kept_whole) == 1

    rng = np.random.default_rng(11)
    beside = np.full((60, 2, 2), np.nan)
    beside[:, 0, 0] = np.where(np.arange(60) < 40, 0.0, 40.0) + rng.normal(size=60)
    beside[:, 0, 1] = 0.3
    smaller, samples = np.arange(60) >= 40, np.arange(60)

    def together():
        return relabel_module._clusters_appear_together(beside, 0, samples, smaller)

    beside[:5, 1] = [40.0, 0.3]
    assert together()
    beside[:5, 1, 1] = 0.31
    assert not together()
    beside[:5, 1] = [55.0, 0.3]
    assert not together()
    beside[:5, 1] = [0.0, 0.3]
    assert not together()
    beside[40:42, 1] = [0.0, 0.3]
    assert not together()
    beside[42, 1] = [0.0, 0.3]
    assert together()

    rng = np.random.default_rng(12)
    blocks = np.full((60, 2, 1), np.nan)
    blocks[:, 0, 0] = np.where(np.arange(60) < 40, 0.0, 40.0) + rng.normal(size=60)
    blocks[50:55, 1, 0] = 0.0
    assert relabel_module._clusters_appear_together(blocks, 0, samples, smaller)
    blocks[52:55, 1, 0] = np.nan
    assert not relabel_module._clusters_appear_together(blocks, 0, samples, smaller)


def test_a_proposal_swaps_two_labels_and_a_rejection_keeps_the_plain_result(caplog):
    """A proposal swaps whole rows in the chosen samples only; rejected proposals leave the plain result."""
    rng = np.random.default_rng(3)
    chain = rng.normal(size=(6, 3, 2))
    chain[[1, 4], 2] = np.nan
    chain[5, 0] = np.nan
    pc = PosteriorChain(chain, 3, 2, trans_dimensional=True, cost_dict={3: 1.5, 2: 0.5},
                        validate_nan_convention=False)
    before = chain.copy()
    samples = np.array([1, 2, 4])

    proposal = _propose_split(pc, 0, 2, samples)

    assert np.array_equal(pc.chain, before, equal_nan=True)
    assert_rows_permuted(proposal.chain, before)
    np.testing.assert_array_equal(proposal.chain[samples, 2], before[samples, 0])
    np.testing.assert_array_equal(proposal.chain[samples, 0], before[samples, 2])
    np.testing.assert_array_equal(proposal.chain[[0, 3, 5]], before[[0, 3, 5]])
    np.testing.assert_array_equal(proposal.prob_in_model, find_prob_in_model(proposal.chain, 3, eps=0))
    assert proposal.cost_dict == {2: 0.5}
    assert proposal.validate_nan_convention is False and proposal.trans_dimensional is True

    assert _opened_label_cost(1000, 2) == pytest.approx(0.5 * 6 * np.log(1000) / 1000)
    assert round(_opened_label_cost(200, 3), 4) == 0.1325

    full = PosteriorChain(three_full_labels_chain(), 3, 2)
    plain = relabel_mv_normal(full, 3, progress=False, split_moves=False)
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        result = relabel_mv_normal(full, 3, progress=False, split_exclusive=True)
    assert same_catalog(result, plain)
    rejections = relabel_messages(caplog, "Split move rejected after")
    assert len(rejections) == 2
    for record in rejections:
        _, _, bound, opened = record.args
        assert bound == plain.cost_dict[3] - opened * _opened_label_cost(300, 2)
    assert same_catalog(relabel_mv_normal(full, 3, progress=False), plain)


def _a_run_that_never_stalls_warns_with_split_moves(tmp_path, caplog):
    """Case A at 300 samples with a budget of 2 never stalls: the search did not finish."""
    pc = PosteriorChain(widening_chain(300)[0], 2, 2, True)
    _, messages = recorded(lambda: relabel_mv_normal(pc, 4, num_iterations=2, progress=False))
    assert messages == [budget_message(2)]


def _a_run_that_never_stalls_is_silent_without_split_moves(tmp_path, caplog):
    """The plain search's budget is a ceiling, and reaching it is silent."""
    pc = PosteriorChain(widening_chain(300)[0], 2, 2, True)
    _, messages = recorded(lambda: relabel_mv_normal(pc, 4, num_iterations=2, progress=False, split_moves=False))
    assert messages == []


def _a_budget_ending_right_after_an_accepted_move_warns(tmp_path, caplog):
    """A move accepted at iteration 13 of 13 has not been followed to a stall."""
    pc = PosteriorChain(widening_chain(300)[0], 2, 2, True)
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        result, messages = recorded(lambda: relabel_mv_normal(pc, 4, num_iterations=13, progress=False))
    assert messages == [budget_message(13)]
    assert result.cost_dict[4] == pytest.approx(6.663825334279802, rel=1e-12)
    assert relabel_messages(caplog, "ran out before the split pass finished") == []


def _a_pass_the_budget_cuts_short_is_not_a_stop(tmp_path, caplog):
    """The first proposal takes the last iteration; the run is not over, and a resume finishes it."""
    pc = PosteriorChain(three_full_labels_chain(), 3, 2)
    directory = str(tmp_path / "short")
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        result, messages = recorded(lambda: relabel_mv_normal(pc, 3, num_iterations=3, checkpoint_dir=directory,
                                                              progress=False, split_exclusive=True))
    assert messages == [budget_message(3)]
    assert [record.args[0] for record in relabel_messages(caplog, "Split move rejected after")] == [1]
    assert len(relabel_messages(caplog, "ran out before the split pass finished")) == 1
    assert same_catalog(result, relabel_mv_normal(pc, 3, num_iterations=3, progress=False, split_moves=False))
    assert run_state(directory) == (1, False, None)

    resumed, messages = recorded(lambda: relabel_mv_normal(pc, 3, num_iterations=20, resume_from=directory,
                                                           checkpoint_dir=directory, progress=False,
                                                           split_exclusive=True))
    assert messages == []
    assert same_catalog(resumed, relabel_mv_normal(pc, 3, num_iterations=20, progress=False, split_exclusive=True))
    assert run_state(directory)[1] is True


def _a_budget_that_cuts_the_last_proposal_short_is_not_a_stop(tmp_path, caplog):
    """Both proposals are tried, but the second gets 2 of the 4 iterations it needs."""
    pc = PosteriorChain(three_full_labels_chain(), 3, 2)
    directory = str(tmp_path / "short")
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        _, messages = recorded(lambda: relabel_mv_normal(pc, 3, num_iterations=6, checkpoint_dir=directory,
                                                         progress=False, split_exclusive=True))
    assert messages == [budget_message(6)]
    assert [record.args[0] for record in relabel_messages(caplog, "Split move rejected after")] == [2, 2]
    assert len(relabel_messages(caplog, "ran out before the split pass finished")) == 1
    assert run_state(directory) == (1, False, None)


@pytest.mark.parametrize("case", [
    _a_run_that_never_stalls_warns_with_split_moves,
    _a_run_that_never_stalls_is_silent_without_split_moves,
    _a_budget_ending_right_after_an_accepted_move_warns,
    _a_pass_the_budget_cuts_short_is_not_a_stop,
    _a_budget_that_cuts_the_last_proposal_short_is_not_a_stop,
], ids=lambda case: case.__name__.strip("_"))
def test_the_budget_warning_fires_only_when_the_budget_ends_a_split_search(case, tmp_path, caplog):
    """The budget warning and the stop as the old budget scenarios pinned them."""
    case(tmp_path, caplog)


def _a_stall_on_a_cost_increase_runs_a_split_pass(tmp_path, caplog, monkeypatch):
    """The plain run stops on a cost increase at iteration 6; the split pass there lowers the cost."""
    rng = np.random.default_rng(235)
    centres = rng.uniform(-8, 8, size=(3, 1))
    presence = rng.uniform(0.3, 0.95, size=3)
    pc = PosteriorChain(packed_chain(rng, centres, presence, 100, 2)[0], 2, 1, True)

    plain = relabel_mv_normal(pc, 3, checkpoint_dir=str(tmp_path), progress=False, split_moves=False)
    result = relabel_mv_normal(pc, 3, progress=False)

    assert run_state(tmp_path) == (6, True, "cost increased")
    assert plain.cost_dict[3] == pytest.approx(4.028711459641443, rel=1e-9)
    assert result.cost_dict[3] == pytest.approx(3.687053412186856, rel=1e-9)


def _nothing_to_split_on_the_last_iteration_is_a_stop(tmp_path, caplog, monkeypatch):
    """Two separated sources converge at iteration 2 with nothing to propose: a stop, at no extra cost."""
    pc = PosteriorChain(two_clusters(), 2, 1)
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        _, messages = recorded(lambda: relabel_mv_normal(pc, 2, num_iterations=2, checkpoint_dir=str(tmp_path / "two"),
                                                         progress=False))
    assert messages == []
    assert relabel_messages(caplog, "no label splits into two clusters")
    assert run_state(tmp_path / "two") == (2, True, "converged")
    with pytest.warns(UserWarning, match="already stopped after 2 iterations because"):
        relabel_mv_normal(pc, 2, num_iterations=20, resume_from=str(tmp_path / "two"), progress=False)

    relabel_mv_normal(pc, 2, checkpoint_dir=str(tmp_path / "on"), progress=False)
    relabel_mv_normal(pc, 2, checkpoint_dir=str(tmp_path / "off"), progress=False, split_moves=False)
    assert run_state(tmp_path / "on") == run_state(tmp_path / "off")


def _an_evaluation_ends_at_its_cheapest_labeling_and_hands_over_its_stall(tmp_path, caplog, monkeypatch):
    """The evaluation returns the labeling before its stall, and the stall is the next iteration from it."""
    pc = PosteriorChain(widening_chain(300)[0], 2, 2, True)
    with pytest.warns(UserWarning, match=EMPTY_LABEL_WARNING):
        plain = make_catalog_mv_normal(pc, 4, initialization_param_index=None, split_moves=False, progress=False)
    donor, receiver, samples, _ = _split_candidates(plain.get_chain())[0]
    costs = []

    def first_pays(labeling):
        costs.append(labeling.cost)
        return len(costs) == 1

    evaluation = relabel_module._evaluate_split(_propose_split(plain, donor, receiver, samples), 4, mv_normal_fit,
                                                MV_COST, 1e-6, 10, first_pays)

    assert evaluation.paid and len(costs) == 1
    assert (evaluation.iterations, evaluation.found_at) == (3, 2)
    assert costs[0] > evaluation.best.cost
    reason, stalled = evaluation.stall
    again = relabel_module._relabel_and_refit(evaluation.best.chain, evaluation.best.aux_parameters,
                                              evaluation.best.prob_in_model, 4, mv_normal_fit, MV_COST, 1e-6,
                                              progress=False)
    assert reason == "converged" and stalled.cost == again.cost == evaluation.best.cost
    assert np.array_equal(stalled.chain.get_chain(), again.chain.get_chain(), equal_nan=True)

    capped = relabel_module._evaluate_split(_propose_split(plain, donor, receiver, samples), 4, mv_normal_fit,
                                            MV_COST, 1e-6, 2, lambda labeling: False)
    assert (capped.iterations, capped.found_at, capped.paid, capped.stall) == (2, 2, False, None)
    assert capped.best.cost == evaluation.best.cost


def _an_evaluation_that_runs_to_its_cap_completes_the_pass(tmp_path, caplog, monkeypatch):
    """Rejected evaluations that ran their full 10 iterations end the search: the run stops on its stall."""
    real = relabel_module._evaluate_split

    def capped(proposal, max_num_sources, param_fit, compute_cost_matrix, eps, max_iterations, pays):
        first = real(proposal, max_num_sources, param_fit, compute_cost_matrix, eps, 1, lambda labeling: False).best
        return relabel_module._Evaluation(iterations=10, best=first, found_at=1, paid=False, stall=None)

    monkeypatch.setattr(relabel_module, "_evaluate_split", capped)
    pc = PosteriorChain(three_full_labels_chain(), 3, 2)
    _, messages = recorded(lambda: relabel_mv_normal(pc, 3, num_iterations=100, checkpoint_dir=str(tmp_path),
                                                     progress=False, split_exclusive=True))
    assert messages == []
    assert run_state(tmp_path) == (22, True, "converged")


@pytest.mark.parametrize("case", [
    _a_stall_on_a_cost_increase_runs_a_split_pass,
    _nothing_to_split_on_the_last_iteration_is_a_stop,
    _an_evaluation_ends_at_its_cheapest_labeling_and_hands_over_its_stall,
    _an_evaluation_that_runs_to_its_cap_completes_the_pass,
], ids=lambda case: case.__name__.strip("_"))
def test_a_stall_runs_a_split_pass_that_ends_in_a_stop_or_an_acceptance(case, tmp_path, caplog, monkeypatch):
    """A stall of either kind runs a split pass, and the pass ends the search or hands on an accepted labeling."""
    case(tmp_path, caplog, monkeypatch)
