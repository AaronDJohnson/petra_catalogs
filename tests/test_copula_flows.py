"""Tests for petra.copula_flows: flooring flow log-densities, the uniform prior, the entry point and its checkpoints."""

import contextlib
import inspect
import warnings
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from petra import copula_flows, initialization
from petra import relabel as relabel_module
from petra.checkpoint import find_latest_checkpoint
from petra.copula_flows import (DEFAULT_LOG_PROB_FLOOR, copula_flows_aux_distribution, make_catalog_copula_flows,
                                make_copula_flows_fit, relabel_copula_flows, safe_flow_log_prob,
                                stack_flow_log_probs)
from petra.fits import compute_cost_matrix
from petra.posterior_chain import PosteriorChain
from petra.utils import UniformPrior, make_uniform_prior
from tests.helpers import NO_FLOWS, run_state, two_clusters

FLOOR = DEFAULT_LOG_PROB_FLOOR
UNIT = UniformPrior([0.0], [1.0])


class _ScalarFlow:
    """A distribution that returns one scalar however many points it is given."""

    def log_prob(self, x):
        return -1.5


class _AngryFlow:
    """A distribution that raises what jax_debug_nans raises for an internal NaN."""

    def log_prob(self, x):
        raise FloatingPointError("nan in log_prob")


class _ImpossibleFlow:
    """A distribution that scores three points -1, NaN and -inf."""

    def log_prob(self, x):
        return jnp.array([-1.0, jnp.nan, -jnp.inf])


def check(call, expected):
    """Assert that `call()` raises `expected`, an exception class, or else returns an array of its shape and values."""
    if isinstance(expected, type):
        with pytest.raises(expected):
            call()
        return
    out = np.asarray(call())
    assert out.shape == np.shape(expected)
    np.testing.assert_allclose(out, expected)


@contextlib.contextmanager
def no_warnings():
    """Turn every warning raised inside the block into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# Flooring flow log-densities


@pytest.mark.parametrize("call, expected", [
    pytest.param(lambda: safe_flow_log_prob(_ScalarFlow(), np.zeros((4, 2))), np.full(4, -1.5), id="scalar"),
    pytest.param(lambda: safe_flow_log_prob(_AngryFlow(), np.zeros((3, 2))), np.full(3, FLOOR),
                 id="FloatingPointError"),
    pytest.param(lambda: safe_flow_log_prob(None, np.zeros((3, 2))), np.full(3, FLOOR), id="unfitted"),
    pytest.param(lambda: safe_flow_log_prob(UNIT, np.zeros((0, 1))), np.zeros(0), id="empty batch"),
    pytest.param(lambda: safe_flow_log_prob(_ImpossibleFlow(), np.zeros((3, 1))), [-1.0, FLOOR, FLOOR],
                 id="NaN and -inf"),
    pytest.param(lambda: safe_flow_log_prob(UNIT, np.array([np.nan])), FLOOR, id="absent point"),
    pytest.param(lambda: safe_flow_log_prob(UNIT, np.array([[0.5], [9.0], [np.nan]])), [0.0, FLOOR, FLOOR],
                 id="outside the uniform support"),
    pytest.param(lambda: safe_flow_log_prob(UNIT, np.zeros((2, 2, 1))), ValueError, id="3-D input"),
    pytest.param(lambda: stack_flow_log_probs([UNIT], np.zeros(1)), ValueError, id="stack of a 1-D sample"),
    pytest.param(lambda: stack_flow_log_probs([UNIT, None], np.array([[0.5], [9.0]]), floor=-7.0),
                 [[0.0, -7.0], [-7.0, -7.0]], id="stack with a custom floor"),
])
def test_safe_flow_log_prob_floors_what_a_flow_cannot_score(call, expected):
    """Every non-finite or unscorable log-density becomes the floor, finite ones are kept, and bad shapes raise."""
    check(call, expected)


@pytest.mark.parametrize("sample, flows, prob_in_model, expected", [
    pytest.param(np.array([[0.5], [np.nan]]), [UniformPrior([0.0], [np.exp(3.0)]), UNIT], np.array([0.9, 0.1]),
                 [[-3.1053605156578263, -2.302585092994046], [-2.3025850929940455, -0.10536051565782631]],
                 id="absent slot"),
    pytest.param(np.array([[2.0]]), [UNIT], np.array([0.25]), [[-50.0 + np.log(0.25)]],
                 id="present invalid point"),
])
def test_the_flow_cost_matrix_keeps_absence_and_floors_invalid_present_points(
        sample, flows, prob_in_model, expected):
    """An absent row scores log(1 - p) although its density is floored; a present point outside keeps the floor."""
    cost_matrix = partial(compute_cost_matrix, log_prob=partial(copula_flows_aux_distribution, floor=-50.0))
    cost = cost_matrix(sample, flows, prob_in_model, len(flows))
    np.testing.assert_allclose(cost, expected, atol=1e-6)
    rows, columns = linear_sum_assignment(cost, maximize=True)
    np.testing.assert_array_equal(columns, rows)


# The uniform prior


def bounds(prior: UniformPrior) -> np.ndarray:
    """Return a prior's lower and upper bounds as one (2, P) array."""
    return np.array([prior.mins, prior.maxs])


BOX = UniformPrior([0.0, 0.0], [2.0, 4.0])
SPANNED = np.array([[[0.0, 0.0], [np.nan, np.nan]], [[2.0, 4.0], [1.0, 1.0]]])


@pytest.mark.parametrize("call, expected", [
    pytest.param(lambda: BOX.log_density, -np.log(8.0), id="log density"),
    pytest.param(lambda: BOX.log_prob(np.array([[1.0, 1.0], [3.0, 1.0], [np.nan, 1.0]])),
                 [-np.log(8.0), -np.inf, -np.inf], id="support"),
    pytest.param(lambda: UniformPrior([0.0, 1.0], [2.0, 1.0]).log_prob(np.array([[1.0, 1.0], [1.0, 5.0]])),
                 [-np.log(2.0), -np.log(2.0)], id="zero-width dimension"),
    pytest.param(lambda: UniformPrior([0.0, 0.0], [1.0]), ValueError, id="lengths differ"),
    pytest.param(lambda: UniformPrior([0.0], [np.inf]), ValueError, id="infinite bound"),
    pytest.param(lambda: UniformPrior([1.0], [0.0]), ValueError, id="maxs below mins"),
    pytest.param(lambda: UniformPrior([1.0], [1.0]), ValueError, id="no width at all"),
    pytest.param(lambda: UNIT.log_prob(np.zeros((2, 2, 1))), ValueError, id="3-D sample"),
    pytest.param(lambda: UNIT.log_prob(np.zeros(3)), ValueError, id="sample of the wrong width"),
    pytest.param(lambda: bounds(make_uniform_prior(SPANNED)), [[0.0, 0.0], [2.0, 4.0]], id="spans the chain"),
    pytest.param(lambda: make_uniform_prior(SPANNED).log_density, -np.log(8.0), id="spanned log density"),
    pytest.param(lambda: make_uniform_prior(np.array([[[0.0, np.nan]], [[2.0, np.nan]]])), ValueError,
                 id="all-NaN parameter",
                 marks=pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")),
])
def test_the_uniform_prior_normalizes_over_its_support_and_rejects_bad_bounds(call, expected):
    """The prior's log density is -sum(log width) over the dimensions that vary; bad bounds and samples raise."""
    check(call, expected)


# The entry point


#: The flow defaults the docs and the README quote.
DEFAULT_FLOW_SETTINGS = {"knots": 16, "flow_layers": 8, "max_epochs": 800, "max_patience": 20,
                         "learning_rate": 1e-3, "log_prob_floor": -50.0}
FLAT = {"knots": 4, "flow_layers": 2, "max_epochs": 3, "max_patience": 2, "learning_rate": 2e-3,
        "log_prob_floor": -33.0}
NUMPY_COUNTS = {"knots": np.int64(4), "flow_layers": np.int64(2), "max_epochs": np.int64(3),
                "max_patience": np.int64(2), "init_num_iterations": np.int64(7)}


@pytest.mark.parametrize("keywords, expected", [
    pytest.param({}, {**DEFAULT_FLOW_SETTINGS, "with_mv_normal": True, "init_num_iterations": 200}, id="defaults"),
    pytest.param({**FLAT, "init_with_mv_normal": False, "init_num_iterations": 7},
                 {**FLAT, "with_mv_normal": False, "init_num_iterations": 7}, id="flat"),
    pytest.param(NUMPY_COUNTS, NUMPY_COUNTS, id="numpy integers"),
    pytest.param({"patience": 5}, {"max_patience": 5}, id="patience"),
    pytest.param({"patience": 5, "max_patience": 9}, TypeError, id="both patience spellings"),
    pytest.param({"knots": 0}, ValueError, id="knots=0"),
    pytest.param({"knots": 1.5}, ValueError, id="knots=1.5"),
    pytest.param({"knots": True}, ValueError, id="knots=True"),
    pytest.param({"learning_rate": 0.0}, ValueError, id="learning_rate=0"),
    pytest.param({"learning_rate": np.inf}, ValueError, id="learning_rate=inf"),
    pytest.param({"log_prob_floor": np.nan}, ValueError, id="log_prob_floor=nan"),
    pytest.param({"init_num_iterations": 0}, ValueError, id="init_num_iterations=0"),
    pytest.param({"flow_fit": None}, TypeError, id="flow_fit"),
    pytest.param({"initialization": None}, TypeError, id="initialization"),
    pytest.param({"n_phases": 2}, TypeError, id="n_phases"),
])
def test_make_catalog_copula_flows_routes_flat_keywords_and_the_patience_alias(
        monkeypatch, two_cluster_chain, keywords, expected):
    """Every flat keyword reaches the relabeler or the initialization; bad values and removed spellings raise first."""
    forwarded, initialized = {}, {}
    real_relabel = copula_flows.relabel_copula_flows

    def record_relabel(posterior_chain, **kwargs):
        forwarded.update(kwargs)
        return posterior_chain

    def record_initialization(posterior_chain, max_num_sources, **kwargs):
        initialized.update(kwargs)
        return posterior_chain

    monkeypatch.setattr(copula_flows, "relabel_copula_flows", record_relabel)
    monkeypatch.setattr(initialization, "run_initialization_passes", record_initialization)

    def expected_warnings():
        return pytest.warns(DeprecationWarning, match="patience") if "patience" in keywords else no_warnings()

    call = partial(make_catalog_copula_flows, two_cluster_chain, 2, threshold_samples=NO_FLOWS, progress=False,
                   **keywords)
    if isinstance(expected, dict):
        with expected_warnings():
            call()
        arrived = {**forwarded, "with_mv_normal": initialized["with_mv_normal"],
                   "init_num_iterations": initialized["num_iterations"]}
        assert {name: arrived[name] for name in expected} == expected
        return

    with pytest.raises(expected), expected_warnings():
        call()
    assert forwarded == initialized == {}
    # The direct callers that take these keywords refuse them too.
    direct = [partial(make_copula_flows_fit, two_cluster_chain.chain)]
    if "patience" in keywords:
        direct.append(partial(real_relabel, two_cluster_chain, 2))
    for function in direct:
        if set(keywords) <= set(inspect.signature(function.func).parameters):
            with pytest.raises(expected), expected_warnings():
                function(**keywords)


#: Every setting a copula checkpoint records; no flow is trained under them.
FLOW_SETTINGS = {"rng_seed": 1, "threshold_samples": NO_FLOWS, "knots": 8, "max_patience": 20,
                 "learning_rate": 1e-3, "max_epochs": 60, "flow_layers": 8, "log_prob_floor": -50.0}


@pytest.mark.parametrize("changed", [
    {"rng_seed": 2}, {"threshold_samples": 10 ** 7}, {"knots": 4}, {"max_patience": 3}, {"learning_rate": 1e-2},
    {"max_epochs": 3}, {"flow_layers": 2}, {"log_prob_floor": -40.0},
], ids=lambda changed: next(iter(changed)))
def test_a_copula_checkpoint_resumes_only_with_the_same_flow_settings(tmp_path, two_cluster_chain, changed):
    """A checkpoint records the eight flow settings; changing any refuses the resume, changing progress does not."""
    relabel_copula_flows(two_cluster_chain, 2, num_iterations=1, checkpoint_dir=tmp_path, progress=True,
                         **FLOW_SETTINGS)
    state = run_state(find_latest_checkpoint(tmp_path)[0])
    assert (state["method"], state["settings"]) == ("copula_flows", FLOW_SETTINGS)

    with pytest.raises(ValueError, match="was written with other settings"):
        relabel_copula_flows(two_cluster_chain, 2, num_iterations=3, resume_from=tmp_path, progress=False,
                             **{**FLOW_SETTINGS, **changed})
    with pytest.warns(UserWarning, match="already holds 1 completed iterations"):
        relabel_copula_flows(two_cluster_chain, 2, num_iterations=1, resume_from=tmp_path, progress=False,
                             **FLOW_SETTINGS)


def test_make_catalog_copula_flows_recovers_clusters_with_real_flows(monkeypatch):
    """Two scrambled clusters come back; the trained 1-D flows are finite, normalized, and see the caller's eps (F042)."""
    eps, seen = 1e-3, []
    real_one_iteration = relabel_module.relabel_samples_one_iteration

    def spy(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix, progress=True):
        seen.append((aux_parameters, np.array(prob_in_model, copy=True)))
        return real_one_iteration(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix,
                                  progress=progress)

    monkeypatch.setattr(relabel_module, "relabel_samples_one_iteration", spy)
    # coppuccino's marginal transform needs at least 21 rows, and progress=True lets the fitter's own bar through.
    scrambled = two_clusters(30)
    scrambled[::2] = scrambled[::2, ::-1]
    catalog = make_catalog_copula_flows(PosteriorChain(scrambled, 2, 1), 2, num_iterations=1, threshold_samples=20,
                                        knots=4, flow_layers=1, max_epochs=3, rng_seed=7, eps=eps, progress=True)

    # The clusters sit at 0 and 8 with scale 0.2, so each slot must hold one side of 4 only.
    values = catalog.chain[:, :, 0]
    low = int(np.argmin(values[0]))
    assert np.all(values[:, low] < 4.0) and np.all(values[:, 1 - low] > 4.0)
    np.testing.assert_array_equal(catalog.prob_in_model, [1.0, 1.0])

    # The copula pass is the last call; the initialization passes before it keep their own eps.
    flows, prob_in_model = seen[-1]
    np.testing.assert_array_equal(prob_in_model, np.full(2, 1.0 - eps))
    grid = np.linspace(-6.0, 14.0, 200_001)
    for flow in flows:
        assert not isinstance(flow, UniformPrior)
        log_prob = np.asarray(flow.log_prob(grid[:, None]))
        assert np.isfinite(log_prob).all()
        # A flow trained on n rows leaves about 1 / (n + 1) of its mass outside their range.
        assert np.trapezoid(np.exp(log_prob), grid) == pytest.approx(1.0, abs=0.05)
