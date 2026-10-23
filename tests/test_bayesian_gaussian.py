"""Tests for petra.bayesian_gaussian: the NIW prior and predictive, the entry point and the NIW relabeling loop."""

import warnings
from functools import partial

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from petra import bayesian_gaussian
from petra import relabel as relabel_module
from petra.bayesian_gaussian import (DEFAULT_KAPPA0, bayesian_relabel_loop, compute_niw_prior,
                                     make_catalog_bayesian_gaussian, niw_aux_distribution, niw_fit,
                                     precompute_t_params)
from petra.checkpoint import load_checkpoint
from petra.fits import compute_cost_matrix
from petra.posterior_chain import PosteriorChain
from petra.relabel import relabel_samples_one_iteration
from petra.utils import find_prob_in_model
from tests.helpers import (BUDGET_WARNING, empty_slot_chain, label_purity, separated_chain,
                           trans_dimensional_chain)

#: Three 2-D clusters at least 3 apart, twelve times their scale.
TRUE_MEANS = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 1.5]])


# Chains


def tiny_chain() -> PosteriorChain:
    """A 12-sample, two-source, one-parameter chain, the sources 8 apart and in label order."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 1)) + np.array([[0.0], [8.0]])
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


def shuffled_clusters(n_samples: int, seed: int) -> PosteriorChain:
    """Draws from TRUE_MEANS with scale 0.25, the source axis permuted in every sample; no source is ever absent."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(loc=TRUE_MEANS, scale=0.25, size=(n_samples, 3, 2))
    shuffled = np.stack([clean[i, rng.permutation(3), :] for i in range(n_samples)])
    return PosteriorChain(shuffled, 3, 2)


def independently_absent_chain(absent: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three sources 10 apart in two parameters, each absent with probability `absent`; the true ids are -1 there."""
    rng = np.random.default_rng(seed)
    values = np.arange(3)[None, :, None] * 10.0 + rng.normal(size=(200, 3, 2))
    order = np.array([rng.permutation(3) for _ in range(200)])
    chain = np.take_along_axis(values, order[:, :, None], axis=1)
    missing = rng.random((200, 3)) < absent
    chain[missing] = np.nan
    return chain, np.where(missing, -1, order)


def same_rows(relabeled: np.ndarray, original: np.ndarray) -> bool:
    """Whether every sample of `relabeled` holds exactly the rows, NaN rows included, of that sample of `original`."""
    def canonical(sample: np.ndarray) -> np.ndarray:
        return sample[np.lexsort(sample.T[::-1])]
    return relabeled.shape == original.shape and all(
        np.array_equal(canonical(new), canonical(old), equal_nan=True) for new, old in zip(relabeled, original))


def spy_on_the_initialization_passes(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record the num_iterations each initialization pass of the entry point is called with, keyed by pass."""
    seen: dict = {}
    real = {"univariate": bayesian_gaussian.relabel_univariate_normal, "mv_normal": bayesian_gaussian.relabel_mv_normal}

    def spy(name):
        def relabel(chain, **kwargs):
            seen[name] = kwargs["num_iterations"]
            return real[name](chain, **kwargs)
        return relabel

    monkeypatch.setattr(bayesian_gaussian, "relabel_univariate_normal", spy("univariate"))
    monkeypatch.setattr(bayesian_gaussian, "relabel_mv_normal", spy("mv_normal"))
    return seen


# The NIW prior and predictive


SPREAD = np.random.default_rng(0).normal(size=(50, 2, 1))
NO_COMPLETE_VECTOR = np.full((2, 2, 2), np.nan)
ONE_COMPLETE_VECTOR = np.array([[[1.0, 2.0], [np.nan, np.nan]], [[np.nan, np.nan], [np.nan, np.nan]]])
ONE_FINITE_VECTOR = np.array([[[np.inf, 2.0]], [[1.0, 2.0]]])
TWO_VECTORS = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])


@pytest.mark.parametrize("chain, settings, outcome", [
    pytest.param(SPREAD, {"kappa0": 0.0}, "refused", id="kappa0=0"),
    pytest.param(SPREAD, {"kappa0": -1.0}, "refused", id="kappa0=-1"),
    pytest.param(SPREAD, {"kappa0": np.inf}, "refused", id="kappa0=inf"),
    pytest.param(SPREAD, {"kappa0": np.nan}, "refused", id="kappa0=nan"),
    pytest.param(SPREAD, {"nu0": 2}, "refused", id="nu0=D+1"),
    pytest.param(SPREAD, {"nu0": np.inf}, "refused", id="nu0=inf"),
    pytest.param(SPREAD, {"nu0": np.nan}, "refused", id="nu0=nan"),
    pytest.param(ONE_FINITE_VECTOR, {}, "refused", id="one finite vector"),
    pytest.param(NO_COMPLETE_VECTOR, {}, "refused before initialization", id="no complete vector"),
    pytest.param(ONE_COMPLETE_VECTOR, {}, "refused before initialization", id="one complete vector"),
    pytest.param(TWO_VECTORS, {}, "finite", id="two complete vectors"),
])
def test_the_niw_prior_validates_its_hyperparameters_and_data(monkeypatch, chain, settings, outcome):
    """A non-finite or out-of-range kappa0 or nu0, or fewer than two complete vectors, is refused; two suffice."""
    if outcome == "finite":
        prior = compute_niw_prior(chain, max_num_sources=1)
        np.testing.assert_array_equal(prior["m0"], [2.0, 3.0])
        assert np.isfinite(np.linalg.cholesky(prior["Psi0"])).all()
        return

    with pytest.raises(ValueError) as refused:
        compute_niw_prior(chain, 2, **settings)
    assert all(name in str(refused.value) for name in settings)

    if outcome == "refused before initialization":
        seen = spy_on_the_initialization_passes(monkeypatch)
        with pytest.raises(ValueError) as public:
            make_catalog_bayesian_gaussian(PosteriorChain(chain, 2, 2, trans_dimensional=True), 2, progress=False)
        assert str(public.value) == str(refused.value)
        assert seen == {}


def test_kappa0_and_nu0_shape_the_posterior_predictive():
    """A larger kappa0 pulls each predictive mean towards the pooled mean; a larger nu0 widens a tight predictive."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(100, 2, 2)) + np.array([[0.0, 0.0], [8.0, 8.0]])
    pooled_mean = compute_niw_prior(chain, 2)["m0"]
    weak, strong = niw_fit(chain, 2, kappa0=DEFAULT_KAPPA0), niw_fit(chain, 2, kappa0=10_000.0)
    for source in range(2):
        assert (np.linalg.norm(strong[source]["loc"] - pooled_mean)
                < np.linalg.norm(weak[source]["loc"] - pooled_mean))

    # Source 1 is far tighter than the pooled spread, so shrinking towards it can only widen source 1.
    rng = np.random.default_rng(0)
    chain = np.concatenate([rng.normal(size=(100, 1, 1)) * 4.0, rng.normal(size=(100, 1, 1)) * 0.1 + 8.0], axis=1)
    assert niw_fit(chain, 2, nu0=500)[1]["log_scale_det"] > niw_fit(chain, 2)[1]["log_scale_det"]


@pytest.mark.parametrize("degenerate", [{"Psi_n": np.zeros((1, 1))}, {"nu_n": 0}],
                         ids=["scale not positive definite", "no degrees of freedom"])
def test_a_degenerate_niw_slot_falls_back_to_its_inclusion_probability(degenerate):
    """A degenerate slot's predictive is None and scores log p alone; with every slot degenerate the fit refuses."""
    valid = {"m_n": np.zeros(1), "kappa_n": 1.0, "nu_n": 3, "Psi_n": np.eye(1)}
    parameters = precompute_t_params([{**valid, **degenerate}, valid], 1)
    assert parameters[0] is None and parameters[1] is not None

    sample = np.array([[0.0], [np.nan]])
    probabilities = np.array([0.8, 0.2])
    log_density = niw_aux_distribution(sample, parameters, [0, 1])
    assert log_density[0, 0] == 0.0
    assert np.isnan(log_density[:, 1]).all()

    cost_matrix = partial(compute_cost_matrix, log_prob=niw_aux_distribution)
    cost = cost_matrix(sample, parameters, probabilities, 2)
    np.testing.assert_allclose(cost[0], [np.log(0.8), np.log(0.2)])
    assert np.isfinite(cost).all()
    relabeled, total_cost = relabel_samples_one_iteration(sample[None], parameters, probabilities, 2, cost_matrix,
                                                          progress=False)
    np.testing.assert_array_equal(relabeled[0], sample)
    assert np.isfinite(total_cost)

    with pytest.raises(np.linalg.LinAlgError):
        precompute_t_params([{**valid, **degenerate}] * 2, 1)


@pytest.mark.parametrize("case", ["catalog units", "density units", "rank deficient"])
def test_niw_results_survive_a_change_of_units_and_a_rank_deficient_fit_stays_finite(case):
    """A unit change keeps the NIW catalog and adds only the Jacobian to its density; a rank-deficient fit is finite."""
    if case == "catalog units":
        rng = np.random.default_rng(5)
        chain = rng.normal(size=(200, 2, 2)) * [1.0, 0.05] + [[0.0, -2.0], [0.0, 2.0]]
        for sample in chain:
            rng.shuffle(sample)
        catalogs = []
        for units in (1.0, 1e-9):
            scaled_chain = chain * [1.0, units]
            original = scaled_chain.copy()
            catalog = make_catalog_bayesian_gaussian(PosteriorChain(scaled_chain, 2, 2), 2, num_iterations=5,
                                                     initialization_param_index=1, init_num_iterations=5,
                                                     progress=False)
            restored = catalog.chain / [1.0, units]
            ordered = restored[:, np.argsort(restored[:, :, 1].mean(axis=0)), :]
            assert np.all(ordered[:, 0, 1] < 0)
            assert np.all(ordered[:, 1, 1] > 0)
            np.testing.assert_array_equal(scaled_chain, original)
            catalogs.append(ordered)
        np.testing.assert_allclose(catalogs[0], catalogs[1], rtol=1e-14, atol=0)

    elif case == "density units":
        rng = np.random.default_rng(7)
        chain = rng.normal(size=(30, 2, 2)) + [[0.0, -2.0], [0.0, 2.0]]
        units = np.array([1.0, 1e-22])
        ordinary = niw_aux_distribution(chain[0], niw_fit(chain, 2), [0, 1])
        rescaled = niw_aux_distribution(chain[0] * units, niw_fit(chain * units, 2), [0, 1])
        np.testing.assert_allclose(rescaled + np.log(units).sum(), ordinary, rtol=1e-12)

    else:
        x = np.linspace(-1.0, 1.0, 30)
        chain = np.stack([x, 1e-22 * x, np.full_like(x, 3.0)], axis=-1)[:, None, :]
        log_prob = niw_aux_distribution(chain[10], niw_fit(chain, 1), [0])
        assert log_prob.shape == (1,)
        assert np.isfinite(log_prob).all()


# The entry point


@pytest.mark.parametrize("case", ["complete", "absent 30%", "absent 50%", "eps", "prior"])
def test_make_catalog_bayesian_gaussian_recovers_clusters_as_a_permutation(monkeypatch, case):
    """Shuffled and gappy clusters come back sample by sample permuted; eps and the NIW prior reach the run (F042)."""
    if case == "complete":
        posterior_chain = shuffled_clusters(60, seed=2)
        original = posterior_chain.chain.copy()
        catalog = make_catalog_bayesian_gaussian(posterior_chain, 3, num_iterations=5, init_num_iterations=20,
                                                 rng_seed=7, progress=False)
        found = np.nanmean(catalog.chain, axis=0)
        slot, truth = linear_sum_assignment(np.linalg.norm(found[:, None] - TRUE_MEANS[None], axis=-1))
        np.testing.assert_allclose(found[slot], TRUE_MEANS[truth], atol=0.3)
        np.testing.assert_array_equal(catalog.prob_in_model, find_prob_in_model(catalog.chain, 3, eps=0))
        np.testing.assert_array_equal(catalog.prob_in_model, np.ones(3))
        assert same_rows(catalog.chain, original)

    elif case.startswith("absent"):
        chain, true_ids = independently_absent_chain(absent=0.3 if case == "absent 30%" else 0.5)
        catalog = make_catalog_bayesian_gaussian(PosteriorChain(chain, 3, 2, trans_dimensional=True), 3,
                                                 progress=False)
        assert label_purity(catalog.chain, chain, true_ids) >= 0.99
        true_occupancy = np.sort([np.count_nonzero(true_ids == k) / len(true_ids) for k in range(3)])
        np.testing.assert_allclose(np.sort(catalog.prob_in_model), true_occupancy, atol=0.01)
        assert same_rows(catalog.chain, chain)

    elif case == "eps":
        # Both initialization passes are off: they keep their own eps.
        eps, seen = 1e-3, []
        real = relabel_module.relabel_samples_one_iteration

        def spy(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix, progress=True):
            seen.append(np.array(prob_in_model, copy=True))
            return real(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix, progress=progress)

        monkeypatch.setattr(relabel_module, "relabel_samples_one_iteration", spy)
        make_catalog_bayesian_gaussian(empty_slot_chain(), 4, num_iterations=2, initialization_param_index=None,
                                       init_with_mv_normal=False, eps=eps, progress=False)
        assert seen
        for prob_in_model in seen:
            assert (prob_in_model.min(), prob_in_model.max()) == (eps, 1.0 - eps)

    else:
        settings = []
        real_fit = bayesian_gaussian.niw_fit

        def spy_fit(chain, max_num_sources, **kwargs):
            settings.append(kwargs)
            return real_fit(chain, max_num_sources, **kwargs)

        monkeypatch.setattr(bayesian_gaussian, "niw_fit", spy_fit)
        weak = make_catalog_bayesian_gaussian(tiny_chain(), 2, num_iterations=3, init_num_iterations=1,
                                              progress=False)
        assert settings and all(s == {"kappa0": DEFAULT_KAPPA0, "nu0": None} for s in settings)
        settings.clear()
        strong = make_catalog_bayesian_gaussian(tiny_chain(), 2, num_iterations=3, init_num_iterations=1,
                                                kappa0=50.0, nu0=200, progress=False)
        assert settings and all(s == {"kappa0": 50.0, "nu0": 200} for s in settings)
        assert weak.cost_dict[2] != strong.cost_dict[2]


@pytest.mark.parametrize("keywords, expected", [
    pytest.param({"init_num_iterations": 4}, {"univariate": 4, "mv_normal": 4}, id="budget"),
    pytest.param({"init_with_mv_normal": False, "init_num_iterations": 3}, {"univariate": 3}, id="without mv normal"),
    pytest.param({"num_iterations": 3}, {"univariate": 200, "mv_normal": 200}, id="top-level budget"),
    pytest.param({"init_num_iterations": 0}, ValueError, id="init_num_iterations=0"),
    pytest.param({"initialization": None}, TypeError, id="initialization"),
    pytest.param({"n_phases": 2}, TypeError, id="n_phases"),
    pytest.param({"mv_normal_init": False}, TypeError, id="mv_normal_init"),
    pytest.param({"chain": None}, TypeError, id="chain"),
    pytest.param({"num_sources": 2}, TypeError, id="num_sources"),
])
def test_the_flat_init_keywords_reach_both_passes_and_removed_spellings_raise(monkeypatch, keywords, expected):
    """init_num_iterations bounds each pass and init_with_mv_normal gates the second; a removed spelling raises."""
    seen = spy_on_the_initialization_passes(monkeypatch)
    call = partial(make_catalog_bayesian_gaussian, tiny_chain(), 2,
                   **{"num_iterations": 1, "progress": False, **keywords})
    if isinstance(expected, dict):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            call()
        assert seen == expected
    else:
        with pytest.raises(expected):
            call()
        assert seen == {}


# The NIW relabeling loop


class _TaggedFit(list):
    """A NIW fit that remembers which call of the fit made it."""

    call_index: int


def test_the_bayesian_loop_reverts_checkpoints_and_resumes_only_with_its_settings(
        monkeypatch, tmp_path, scripted_costs):
    """The loop refits every iteration, keeps its cheapest labeling, and resumes only a NIW run with its prior."""
    with pytest.raises(ValueError, match="must be an integer >="):
        bayesian_relabel_loop(tiny_chain(), 2, num_iterations=0, progress=False)

    # No split pass runs: a budget that ends the search does not warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_catalog_bayesian_gaussian(tiny_chain(), 2, num_iterations=1, init_num_iterations=1, progress=False)
    assert not [w for w in caught if BUDGET_WARNING in str(w.message)]

    costs, fitted_on, scored_with = [], [], []
    real_one_iteration = relabel_module.relabel_samples_one_iteration
    real_fit, real_density = bayesian_gaussian.niw_fit, bayesian_gaussian.niw_aux_distribution

    def record_cost(*args, **kwargs):
        relabeled, cost = real_one_iteration(*args, **kwargs)
        costs.append(cost)
        return relabeled, cost

    def tagged_fit(chain, max_num_sources, **settings):
        fit = _TaggedFit(real_fit(chain, max_num_sources, **settings))
        fit.call_index = len(fitted_on)
        fitted_on.append(np.array(chain, copy=True))
        return fit

    def scripted_density(sample, aux_parameters, source_index):
        logpdf = np.array(real_density(sample, aux_parameters, source_index), dtype=float)
        tag = aux_parameters.call_index
        scored_with.append(tag)
        if tag == 1:
            return logpdf + 10.0
        if tag >= 2:
            other_source = np.atleast_1d(source_index)[:, None] != np.arange(sample.shape[0])[None, :]
            return logpdf - 1000.0 + 500.0 * other_source
        return logpdf

    with monkeypatch.context() as patch:
        patch.setattr(relabel_module, "relabel_samples_one_iteration", record_cost)
        # On real data the cost returned is the cheapest one seen.
        result = bayesian_relabel_loop(trans_dimensional_chain(), 4, num_iterations=15, progress=False)
        assert len(costs) > 1
        assert result.cost_dict[4] == min(costs)

        # Under fit 1 every log-density is 10 higher; under fit 2 every sample swaps its slots at a far higher cost.
        # A loop that kept the first fit would converge at iteration 2 instead of stopping on the increase.
        costs.clear()
        patch.setattr(bayesian_gaussian, "niw_fit", tagged_fit)
        patch.setattr(bayesian_gaussian, "niw_aux_distribution", scripted_density)
        result = bayesian_relabel_loop(tiny_chain(), 2, num_iterations=10, progress=False)
    assert len(costs) == 3
    assert scored_with == sorted(scored_with) and sorted(set(scored_with)) == [0, 1, 2]
    assert costs[1] == pytest.approx(costs[0] - 20.0)
    assert costs[2] > costs[1] + 100.0
    np.testing.assert_array_equal(result.get_chain(), tiny_chain().get_chain())
    assert result.cost_dict[2] == costs[1] == min(costs)
    np.testing.assert_array_equal(fitted_on[2], result.get_chain())

    # A run cut short by its budget resumes to the result of the uninterrupted run, and only with its own prior.
    samples = tiny_chain().get_chain().copy()
    samples[::2] = samples[::2, ::-1]
    scrambled = PosteriorChain(samples, 2, 1, trans_dimensional=True)
    uninterrupted = bayesian_relabel_loop(scrambled, 2, num_iterations=15, progress=False)
    bayesian_relabel_loop(scrambled, 2, num_iterations=2, checkpoint_dir=tmp_path, progress=False)
    checkpoint = load_checkpoint(tmp_path)
    assert checkpoint.completed_iterations == 2
    assert (checkpoint.state["method"], checkpoint.state["settings"]) == ("niw", {"kappa0": 0.01, "nu0": None})
    with pytest.raises(ValueError, match="was written with other settings"):
        bayesian_relabel_loop(scrambled, 2, num_iterations=15, kappa0=0.1, resume_from=tmp_path, progress=False)
    resumed = bayesian_relabel_loop(scrambled, 2, num_iterations=15, resume_from=tmp_path, progress=False)
    np.testing.assert_array_equal(resumed.get_chain(), uninterrupted.get_chain())
    assert resumed.cost_dict == uninterrupted.cost_dict

    # Scripted costs: the third iteration is dearer, so the second iteration's labeling and cost come back.
    calls = scripted_costs([-10.0, -20.0, -15.0])
    result = bayesian_relabel_loop(PosteriorChain(separated_chain(), 2, 1), 2, num_iterations=10, progress=False)
    assert len(calls) == 3
    np.testing.assert_array_equal(result.get_chain(), separated_chain(offset=2.0))
    assert result.cost_dict[2] == -20.0
