"""
Tests for the shared relabeling loop in :func:`petra.relabel.run_relabeling_loop`.

The loop used to exist twice -- once inside ``create_relabel_samples`` and once
as ``petra.bayesian_gaussian.bayesian_relabel_loop`` -- and the two copies
disagreed about what ``prob_in_model`` on the returned chain means.  The copy in
``create_relabel_samples`` handed back whatever
:func:`petra.relabel.relabel_posterior_chain_one_iteration` had stored, which is
the array the *cost matrix* was built from: computed from the labeling of the
previous iteration and clipped into ``[eps, 1 - eps]``.  So a chain whose sources
were in every sample came back reporting ``0.99``, and one whose fourth slot was
never occupied came back reporting ``0.01``.

The contract now is the one the Bayesian loop always had, and the first test
below is what pins it: **the returned ``prob_in_model`` is
``find_prob_in_model(result.get_chain(), max_num_sources, eps=0)``, exactly, for
every entry point.**  ``eps`` still clips the copy used inside the cost matrix,
where ``log(0)`` would be infinite; it never reaches the returned array.

The rest of the file pins the behaviour the unification had to preserve: the
cheapest labeling is the one returned, the loop stops when the cost rises or
stops moving, ``cost_dict[max_num_sources]`` holds that best cost, and a run can
be checkpointed and resumed.
"""

import logging

import numpy as np
import pytest

from petra import relabel as relabel_module
from petra.aux_distributions import mv_normal_aux_distribution
from petra.bayesian_gaussian import bayesian_relabel_loop
from petra.copula_flows import relabel_copula_flows
from petra.cost_matrix import create_compute_cost_matrix
from petra.initialization import relabel_univariate_normal
from petra.make_catalog import relabel_mv_normal
from petra.parametric_fits import create_parametric_fit, mv_normal_fit
from petra.posterior_chain import PosteriorChain
from petra.relabel import (
    CHECKPOINT_TEMPLATE,
    create_relabel_samples,
    relabel_posterior_chain_one_iteration,
    relabel_samples_one_iteration,
    run_relabeling_loop,
)
from petra.utils import find_prob_in_model

#: Larger than any chain here, so every slot falls back to the uniform prior and
#: the flow relabelers exercise the loop rather than flowjax.
NO_FLOWS = 10 ** 6


def two_cluster_chain(n_samples: int = 40, offset: float = 0.0) -> np.ndarray:
    """Two well-separated one-dimensional clusters, shifted by `offset`."""
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.2, size=(n_samples, 2, 1)) + np.array([[0.0], [5.0]]) + offset


def dense_chain() -> PosteriorChain:
    """Three separated sources, every one present in every sample: truth is exactly 1.0."""
    rng = np.random.default_rng(0)
    chain = rng.normal(scale=0.3, size=(40, 3, 2)) + np.array([[0.0, 0.0],
                                                              [6.0, 0.0],
                                                              [0.0, 6.0]])
    return PosteriorChain(chain, 3, 2, trans_dimensional=True)


def empty_slot_chain() -> PosteriorChain:
    """Three separated sources in four slots: the spare slot's truth is exactly 0.0."""
    rng = np.random.default_rng(1)
    chain = np.full((50, 4, 2), np.nan)
    chain[:, :3, :] = rng.normal(scale=0.2, size=(50, 3, 2)) + np.array([[0.0, 0.0],
                                                                        [8.0, 0.0],
                                                                        [0.0, 8.0]])
    return PosteriorChain(chain, 4, 2, trans_dimensional=True)


def trans_dimensional_chain() -> PosteriorChain:
    """Overlapping sources with scattered NaN rows: every probability is inside (0, 1)."""
    rng = np.random.default_rng(119)
    chain = rng.normal(size=(60, 4, 2)) * rng.uniform(0.2, 3.0)
    chain[rng.random((60, 4)) < 0.5] = np.nan
    return PosteriorChain(chain, 4, 2, trans_dimensional=True)


#: The four relabeling entry points, each reduced to ``(chain, n_slots) -> chain``.
#: All four are the same loop with a different fit and cost matrix bound to it,
#: so all four owe the caller the same ``prob_in_model`` contract.
ENTRY_POINTS = {
    "relabel_univariate_normal": lambda pc, n: relabel_univariate_normal(
        pc, max_num_sources=n, num_iterations=5, progress=False),
    "relabel_mv_normal": lambda pc, n: relabel_mv_normal(
        pc, max_num_sources=n, num_iterations=5, progress=False),
    "relabel_copula_flows": lambda pc, n: relabel_copula_flows(
        pc, max_num_sources=n, num_iterations=3, threshold_samples=NO_FLOWS,
        progress=False),
    "bayesian_relabel_loop": lambda pc, n: bayesian_relabel_loop(
        pc, n, num_iterations=5, progress=False),
}


# ---------------------------------------------------------------------------
# The prob_in_model contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS), ids=str)
@pytest.mark.parametrize("build_chain",
                         [dense_chain, empty_slot_chain, trans_dimensional_chain],
                         ids=lambda f: f.__name__)
def test_returned_prob_in_model_describes_the_returned_chain(entry_point, build_chain):
    """
    The array a caller gets back must be derivable from the chain they got back.

    This is the test that would have caught the old bug: the four non-Bayesian
    relabelers returned the clipped array of the *previous* iteration's labeling,
    so the check below failed both because of the clip and because the labeling
    had moved on since.
    """
    posterior_chain = build_chain()
    n_slots = posterior_chain.num_sources

    result = ENTRY_POINTS[entry_point](posterior_chain, n_slots)

    expected = find_prob_in_model(result.get_chain(), n_slots, eps=0)
    assert np.array_equal(result.prob_in_model, expected), (
        f"{entry_point} returned {result.prob_in_model!r}, but its chain says {expected!r}"
    )


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS), ids=str)
def test_a_source_in_every_sample_is_reported_as_one_not_as_one_minus_eps(entry_point):
    """
    The unclipped and clipped values genuinely differ here, and the unclipped one wins.

    Every source is present in every sample, so the truth is exactly ``1.0``
    while the array the cost matrix used is ``1 - eps``.  Before the fix these
    relabelers returned ``0.999999`` (or ``0.99``, at ``create_relabel_samples``'
    own default ``eps=1e-2``); ``bayesian_relabel_loop`` already returned ``1.0``.
    """
    result = ENTRY_POINTS[entry_point](dense_chain(), 3)

    assert np.array_equal(result.prob_in_model, np.ones(3))
    # Not merely "close to 1": the clipped value is close to 1 too.
    assert not np.array_equal(result.prob_in_model, np.full(3, 1.0 - 1e-6))


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS), ids=str)
def test_a_source_in_no_sample_is_reported_as_zero_not_as_eps(entry_point):
    """The other end of the clip: an unoccupied slot has probability 0, not ``eps``."""
    result = ENTRY_POINTS[entry_point](empty_slot_chain(), 4)

    assert result.prob_in_model[3] == 0.0
    assert np.array_equal(result.prob_in_model, np.array([1.0, 1.0, 1.0, 0.0]))


def test_eps_still_clips_the_probabilities_the_cost_matrix_sees(monkeypatch):
    """
    Unclipping the *returned* array must not unclip the one used during iteration.

    A slot that is never occupied would otherwise contribute ``log(0)`` to its
    cost-matrix row, and the assignment would be comparing infinities.
    """
    seen = []
    real_one_iteration = relabel_module.relabel_posterior_chain_one_iteration

    def spy(posterior_chain, aux_parameters, prob_in_model, max_num_sources,
            compute_cost_matrix, progress=True):
        seen.append(np.asarray(prob_in_model).copy())
        return real_one_iteration(posterior_chain, aux_parameters, prob_in_model,
                                  max_num_sources, compute_cost_matrix, progress=progress)

    monkeypatch.setattr(relabel_module, "relabel_posterior_chain_one_iteration", spy)

    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution, eps=1e-3)
    result = relabel(empty_slot_chain(), max_num_sources=4, num_iterations=4,
                     progress=False)

    assert seen, "the loop never ran an iteration"
    for prob_in_model in seen:
        assert prob_in_model.min() >= 1e-3
        assert prob_in_model.max() <= 1.0 - 1e-3
    # ... and none of that clipping survives into the returned chain.
    assert result.prob_in_model[3] == 0.0
    assert np.array_equal(result.prob_in_model,
                          find_prob_in_model(result.get_chain(), 4, eps=0))


def test_a_resumed_run_with_nothing_left_to_do_still_normalizes_prob_in_model(tmp_path):
    """
    The no-op resume path returns a chain straight off disk, and owes the same contract.

    The checkpoint is written by hand carrying the clipped array, because the
    loop now normalizes a chain before checkpointing it: resuming from one of
    *its* checkpoints could no longer tell whether the no-op path normalizes on
    the way out.  A checkpoint from an older petra can still carry the clipped,
    pre-relabel array of the iteration that wrote it, and that is the input this
    path has to survive.
    """
    original = empty_slot_chain()
    clipped = find_prob_in_model(original.get_chain(), 4, eps=1e-2)
    # The spare slot is occupied in no sample, so the clip bound is the only
    # reason this is not 0.0 -- which is what makes the array detectably wrong.
    assert clipped[3] == 1e-2
    on_disk = PosteriorChain(original.get_chain(), 4, 2, trans_dimensional=True,
                             prob_in_model=clipped)
    on_disk.to_feather(str(tmp_path / CHECKPOINT_TEMPLATE.format(iteration=2)))

    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution, eps=1e-2)
    result = relabel(original, max_num_sources=4, num_iterations=2,
                     resume_from=str(tmp_path), progress=False)

    assert np.array_equal(result.prob_in_model,
                          find_prob_in_model(result.get_chain(), 4, eps=0))
    assert result.prob_in_model[3] == 0.0


# ---------------------------------------------------------------------------
# Which labeling comes back, and when the loop stops
# ---------------------------------------------------------------------------

@pytest.fixture
def scripted_costs(monkeypatch):
    """
    Replace one relabeling step with a stub that follows a scripted cost sequence.

    Returns a callable taking the list of costs to hand out, one per iteration,
    and returning the list of chains the stub produced, so a test can assert
    *which* chain came back.  The chains are compared by value, not by identity:
    the loop rebuilds the winner with a corrected ``prob_in_model`` before
    returning it, so the object that comes back is never the object the stub
    made.
    """
    def install(costs):
        produced = []

        def fake_one_iteration(posterior_chain, aux_parameters, prob_in_model,
                               max_num_sources, compute_cost_matrix, progress=True):
            index = len(produced)
            # A distinguishable, non-degenerate chain per iteration, so that the
            # parametric fit the loop runs on it succeeds and the test can tell
            # the iterations apart.
            chain = two_cluster_chain(offset=float(index))
            new = PosteriorChain(chain, 2, 1, prob_in_model=prob_in_model,
                                 cost_dict={max_num_sources: costs[index]})
            produced.append(new)
            return new

        monkeypatch.setattr(relabel_module, "relabel_posterior_chain_one_iteration",
                            fake_one_iteration)
        return produced

    return install


def assert_is_iteration(result, produced, index):
    """The loop returned the chain the stub produced on iteration `index`."""
    assert np.array_equal(result.get_chain(), produced[index].get_chain()), (
        f"expected the chain from iteration {index}"
    )
    assert result.cost_dict == produced[index].cost_dict


def test_loop_returns_the_cheapest_labeling_not_the_last(scripted_costs):
    """A cost that goes down and then up must not lose the good labeling."""
    produced = scripted_costs([-10.0, -20.0, -15.0])
    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)

    result = relabel(PosteriorChain(two_cluster_chain(), 2, 1),
                     max_num_sources=2, num_iterations=10, progress=False)

    # It stopped on the increase rather than running all ten iterations.
    assert len(produced) == 3
    # And it handed back the cheapest chain, not the most recent one.
    assert_is_iteration(result, produced, 1)
    assert result.cost_dict[2] == -20.0
    assert result.cost_dict[2] == min(chain.cost_dict[2] for chain in produced)


def test_loop_stops_when_the_cost_stops_changing(scripted_costs):
    """An unchanged cost is convergence, and returns the chain that achieved it."""
    produced = scripted_costs([-10.0, -20.0, -20.0])
    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)

    result = relabel(PosteriorChain(two_cluster_chain(), 2, 1),
                     max_num_sources=2, num_iterations=10, progress=False)

    assert len(produced) == 3
    # The comparison keeping the best is strict, so a tie keeps the earlier chain.
    assert_is_iteration(result, produced, 1)
    assert result.cost_dict[2] == -20.0


def test_loop_returns_the_best_of_a_run_that_exhausts_its_iterations(scripted_costs):
    """With a monotonically falling cost the best chain is also the last one."""
    produced = scripted_costs([-10.0, -20.0, -30.0])
    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)

    result = relabel(PosteriorChain(two_cluster_chain(), 2, 1),
                     max_num_sources=2, num_iterations=3, progress=False)

    assert len(produced) == 3
    assert_is_iteration(result, produced, -1)
    assert result.cost_dict[2] == -30.0


def test_bayesian_loop_also_reverts_to_its_best_labeling(scripted_costs):
    """
    ``bayesian_relabel_loop`` is now the shared loop, so it stops the same way.

    Its own copy of this logic is gone; this is what says the surviving one
    still behaves for the NIW relabeler.
    """
    produced = scripted_costs([-10.0, -20.0, -15.0])

    result = bayesian_relabel_loop(PosteriorChain(two_cluster_chain(), 2, 1), 2,
                                   num_iterations=10, progress=False)

    assert len(produced) == 3
    assert_is_iteration(result, produced, 1)
    assert result.cost_dict[2] == -20.0


def test_cost_recorded_under_max_num_sources_is_the_best_cost(scripted_costs):
    """
    ``cost_dict[max_num_sources]`` is the cost of the labeling returned.

    A run whose last iteration was worse must not record that last cost, or a
    caller comparing methods by ``cost_dict`` would be comparing a number to a
    labeling that was thrown away.
    """
    scripted_costs([-10.0, -20.0, -1.0])
    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)

    result = relabel(PosteriorChain(two_cluster_chain(), 2, 1),
                     max_num_sources=2, num_iterations=10, progress=False)

    assert result.cost_dict[2] == -20.0


def test_returned_cost_is_the_minimum_over_the_checkpointed_iterations(tmp_path):
    """
    On a real run, the cost that comes back is the best of the ones written out.

    Every iteration checkpoints itself, and ``cost_dict`` survives the Feather
    round trip, so the checkpoints are an independent record of what each
    iteration actually cost.
    """
    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    rng = np.random.default_rng(3)
    chain = rng.normal(scale=1.0, size=(60, 3, 2)) + np.array([[0.0, 0.0],
                                                              [4.0, 4.0],
                                                              [8.0, 0.0]])
    result = relabel(PosteriorChain(chain, 3, 2), max_num_sources=3,
                     num_iterations=6, checkpoint_dir=str(tmp_path), progress=False)

    checkpoint_costs = [
        PosteriorChain.read_feather(str(path)).cost_dict[3]
        for path in sorted(tmp_path.glob("posterior_chain_iteration_*.feather"))
    ]
    assert len(checkpoint_costs) > 1, "the run stopped too early to compare iterations"
    assert result.cost_dict[3] == pytest.approx(min(checkpoint_costs))


def test_checkpoints_are_written_and_resumed_from(tmp_path, caplog):
    """A walltime kill mid-run must not cost the iterations already done."""
    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    posterior_chain = PosteriorChain(two_cluster_chain(), 2, 1)

    relabel(posterior_chain, max_num_sources=2, num_iterations=2,
            checkpoint_dir=str(tmp_path), progress=False)
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == ["posterior_chain_iteration_001.feather",
                       "posterior_chain_iteration_002.feather"]

    with caplog.at_level(logging.WARNING, logger="petra.relabel"):
        resumed = relabel(posterior_chain, max_num_sources=2, num_iterations=5,
                          resume_from=str(tmp_path), progress=False)
    assert "after 2 completed iterations" in caplog.text
    assert resumed.chain.shape == posterior_chain.chain.shape

    # A checkpoint that already holds the whole budget is a no-op, not a rerun.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="petra.relabel"):
        done = relabel(posterior_chain, max_num_sources=2, num_iterations=2,
                       resume_from=str(tmp_path), progress=False)
    assert "nothing to do" in caplog.text
    stored = PosteriorChain.read_feather(str(tmp_path / written[-1]))
    assert np.array_equal(done.get_chain(), stored.get_chain(), equal_nan=True)


def test_loop_rejects_a_zero_iteration_budget():
    """A budget of zero iterations is a caller mistake, not a silent no-op."""
    with pytest.raises(ValueError, match="num_iterations must be at least 1"):
        run_relabeling_loop(PosteriorChain(two_cluster_chain(), 2, 1),
                            create_parametric_fit(mv_normal_fit),
                            create_compute_cost_matrix(mv_normal_aux_distribution),
                            max_num_sources=2, num_iterations=0, progress=False)


# ---------------------------------------------------------------------------
# The single-iteration step
# ---------------------------------------------------------------------------

def test_fewer_distributions_than_slots_keeps_every_source():
    """
    The ``fill_missing_indices`` call is load-bearing when the chain is wider
    than ``max_num_sources``: the Hungarian solver only reports one column per
    auxiliary distribution, and the slots it never looked at still have to come
    back.
    """
    rng = np.random.default_rng(0)
    chain = rng.normal(scale=0.2, size=(25, 3, 1)) + np.array([[0.0], [5.0], [10.0]])
    aux_parameters = create_parametric_fit(mv_normal_fit)(chain, max_num_sources=2)
    prob_in_model = find_prob_in_model(chain, max_num_sources=2)
    compute_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)

    relabeled, cost = relabel_samples_one_iteration(
        chain, aux_parameters, prob_in_model, 2, compute_cost_matrix, progress=False)

    assert relabeled.shape == chain.shape
    # Relabeling only permutes: every sample holds exactly the same values.
    assert np.allclose(np.sort(relabeled, axis=1), np.sort(chain, axis=1))
    assert np.isfinite(cost)


def test_the_posterior_chain_layer_says_which_argument_is_wrong_and_where_to_go():
    """
    The two layers disagree about ``max_num_sources``, and the wrapper has to say so.

    :func:`~petra.relabel.relabel_samples_one_iteration` accepts a chain wider
    than ``max_num_sources`` -- the test above is what that is for -- while
    :func:`~petra.relabel.relabel_posterior_chain_one_iteration` cannot: it
    returns a :class:`~petra.posterior_chain.PosteriorChain`, whose
    ``prob_in_model`` holds one entry per source slot, and the caller supplied
    one entry per *distribution*.  The narrower case reached that constructor and
    came back as a complaint about the shape of an array the caller had passed
    correctly, naming neither the argument that chose the shape nor the layer
    that does support it.
    """
    rng = np.random.default_rng(0)
    chain = rng.normal(scale=0.2, size=(25, 3, 1)) + np.array([[0.0], [5.0], [10.0]])
    aux_parameters = create_parametric_fit(mv_normal_fit)(chain, max_num_sources=2)
    prob_in_model = find_prob_in_model(chain, max_num_sources=2)
    compute_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)

    with pytest.raises(ValueError) as excinfo:
        relabel_posterior_chain_one_iteration(
            PosteriorChain(chain, 3, 1, trans_dimensional=True), aux_parameters,
            prob_in_model, 2, compute_cost_matrix, progress=False)

    message = str(excinfo.value)
    assert "max_num_sources (2) must equal the chain's num_sources (3)" in message
    assert "relabel_samples_one_iteration" in message
    # ... and that is a live route, not a consolation: the layer named works.
    relabeled, _ = relabel_samples_one_iteration(
        chain, aux_parameters, prob_in_model, 2, compute_cost_matrix, progress=False)
    assert relabeled.shape == chain.shape


def test_square_case_leaves_the_assignment_untouched():
    """
    When the chain is exactly ``max_num_sources`` wide -- the shared code path --
    the solver already returns a full permutation and the fill is the identity.
    """
    rng = np.random.default_rng(1)
    chain = rng.normal(scale=0.2, size=(25, 3, 1)) + np.array([[0.0], [5.0], [10.0]])
    aux_parameters = create_parametric_fit(mv_normal_fit)(chain, max_num_sources=3)
    prob_in_model = find_prob_in_model(chain, max_num_sources=3)
    compute_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)

    from scipy import optimize

    from petra.utils import fill_missing_indices

    for sample in chain:
        cost_matrix = compute_cost_matrix(sample, aux_parameters, prob_in_model, 3)
        _, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)
        assert np.array_equal(fill_missing_indices(3, col_ind), col_ind)
