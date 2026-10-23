"""Tests for petra.fits: the per-slot normal fits, their log-densities, the factorization cache and the cost matrix."""

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from petra import fits
from petra.fits import (COVARIANCE_RIDGE, MIN_UNIVARIATE_SAMPLES, compute_cost_matrix, mv_normal_fit,
                        mv_normal_log_prob, regularize_covariance, uni_normal_fit, uni_normal_log_prob)
from petra.make_catalog import relabel_mv_normal
from petra.posterior_chain import PosteriorChain
from petra.relabel import relabel_samples_one_iteration
from petra.utils import find_prob_in_model
from tests.helpers import golden_chain

#: The replacement for a zero univariate std when the pooled std is 1.
UNIT_REPLACEMENT_STD = np.sqrt(COVARIANCE_RIDGE)


# Chains


def _interleaved_uniform_chain():
    """The old suite's 20,000 x 2 x 1 uniform chain: each slot holds a draw in even samples and NaN in odd ones."""
    rs = np.random.RandomState(0)
    chain = np.full((20_000, 2), np.nan)
    chain[::2, 0] = rs.uniform(0, 1, 10000)
    chain[::2, 1] = rs.uniform(0, 1, 10000)
    return chain.reshape(20_000, 2, 1)


def _thinned_uniform_chain():
    """The old suite's 10,000 x 2 x 1 uniform chain: slot 0 present in every 2nd sample, slot 1 in every 3rd."""
    rs = np.random.RandomState(0)
    first, second = rs.uniform(0, 1, 10000), rs.uniform(0, 1, 10000)
    chain = np.full((10_000, 2), np.nan)
    chain[::2, 0] = first[::2]
    chain[::3, 1] = second[::3]
    return chain.reshape(10_000, 2, 1)


def _scrambled_chain():
    """A (200, 6, 4) chain of six separated sources, labels scrambled, about 20% of rows absent."""
    rng = np.random.default_rng(28)
    n_samples, n_sources, n_params = 200, 6, 4
    centers = rng.normal(scale=4.0, size=(n_sources, n_params))
    widths = rng.uniform(0.5, 2.0, size=(n_sources, n_params))
    chain = centers + widths * rng.normal(size=(n_samples, n_sources, n_params))
    chain[rng.random((n_samples, n_sources)) < 0.2] = np.nan
    order = np.argsort(rng.random((n_samples, n_sources)), axis=1)
    return np.take_along_axis(chain, order[:, :, np.newaxis], axis=1)


def _stuck_chain(num_distinct, num_params=3, seed=1):
    """Sources A and C in every sample; B's 12 rows in slot 1 are `num_distinct` repeated states, 40 more in slot 3."""
    rng = np.random.default_rng(seed)
    chain = np.full((200, 4, num_params), np.nan)
    chain[:, 0, :] = rng.standard_normal((200, num_params))
    chain[:, 2, :] = rng.standard_normal((200, num_params)) + 12.0
    states = 6.0 + rng.standard_normal((num_distinct, num_params))
    chain[:12, 1, :] = states[np.arange(12) % num_distinct]
    chain[100:140, 3, :] = 6.0 + rng.standard_normal((40, num_params))
    return chain


def _uneven_repeats_chain():
    """Slot 1 holds three states four times each beside two single draws: five distinct rows, 3.9 effective."""
    chain = _stuck_chain(3)
    chain[100:102, 1, :] = 6.0 + np.random.default_rng(11).standard_normal((2, 3))
    return chain


def _forty_repeats_chain():
    """Slot 1 repeats one value 40 times, then holds 60 distinct draws."""
    rng = np.random.default_rng(2)
    chain = rng.normal(size=(100, 2, 1))
    chain[:40, 1, 0] = 0.25
    return chain


def _fixed_inside_chain(num_present=400, num_params=2, fixed=0.3, seed=0, num_samples=400):
    """Source A ~ N(0, I), source B held at `fixed` in its first `num_present` samples, and C far away."""
    rng = np.random.default_rng(seed)
    chain = np.empty((num_samples, 3, num_params))
    chain[:, 0] = rng.normal(size=(num_samples, num_params))
    chain[:, 1] = fixed
    chain[:, 2] = rng.normal(size=(num_samples, num_params)) * 0.5 + 6.0
    chain[num_present:, 1] = np.nan
    return chain


def _fixed_apart_chain():
    """Source A ~ N(0, I) beside two sources held at 5 and -5 in every sample."""
    rng = np.random.default_rng(0)
    chain = np.empty((300, 3, 2))
    chain[:, 0] = rng.normal(size=(300, 2))
    chain[:, 1] = 5.0
    chain[:, 2] = -5.0
    return chain


def _plane_chain():
    """Slot 1's hundreds of distinct rows lie on a plane."""
    chain = np.random.default_rng(4).normal(size=(300, 2, 3))
    chain[:, 1, 2] = chain[:, 1, 0] + chain[:, 1, 1]
    return chain


def _shared_dependence_chain():
    """Every slot's third column is twice its first."""
    chain = np.random.default_rng(5).normal(size=(300, 2, 3)) + np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 0.0]])
    chain[:, :, 2] = 2.0 * chain[:, :, 0]
    return chain


def _derived_column_chain(seed=0):
    """Four galactic-binary-like sources with columns [f, A, log10 f], shuffled per sample, and each entry's source."""
    rng = np.random.default_rng(seed)
    num_samples, num_sources = 400, 4
    f = np.array([1.0e-3, 2.0e-3, 2.0e-3 + 2e-8, 4.0e-3]) + 1.5e-9 * rng.normal(size=(num_samples, num_sources))
    amplitude = np.array([1e-22, 2e-22, 1.5e-22, 3e-22]) + 2e-23 * rng.normal(size=(num_samples, num_sources))
    chain = np.stack([f, amplitude, np.log10(f)], axis=2)
    order = np.argsort(rng.random((num_samples, num_sources)), axis=1)
    truth = np.take_along_axis(np.tile(np.arange(num_sources), (num_samples, 1)), order, axis=1)
    return np.take_along_axis(chain, order[:, :, None], axis=1), truth


def _derived_column_by_source():
    """The derived-column chain with every source in its own slot."""
    chain, truth = _derived_column_chain()
    return np.take_along_axis(chain, np.argsort(truth, axis=1)[:, :, None], axis=1)


def _frozen_column(value, num_values):
    """`num_values` copies of `value`, or for "round-off" six values 0.3 apart only in their last bits."""
    if value == "round-off":
        return 0.3 + np.arange(num_values) % 6 * np.spacing(0.3)
    return np.full(num_values, value)


# References that share no code with petra.fits


def _present_rows(chain, slot):
    rows = chain[:, slot, :]
    return rows[~np.isnan(rows).any(axis=1)]


def _pooled_reference(chain):
    """The pooled fallback: mean and ridged covariance of every present row of every slot."""
    rows = chain.reshape(-1, chain.shape[2])
    rows = rows[~np.isnan(rows).any(axis=1)]
    return rows.mean(axis=0), regularize_covariance(np.cov(rows, rowvar=False))


def _own_fit_reference(chain, slot):
    """A slot's own fit: its moments, an exactly constant column zeroed and ridged by its pooled variance."""
    rows = _present_rows(chain, slot)
    mean = rows.mean(axis=0)
    cov = np.atleast_2d(np.cov(rows, rowvar=False))
    constant = np.ptp(rows, axis=0) == 0
    if not constant.any():
        return mean, regularize_covariance(cov)
    cov[constant, :] = 0.0
    cov[:, constant] = 0.0
    pooled = chain.reshape(-1, chain.shape[2])
    pooled = pooled[~np.isnan(pooled).any(axis=1)]
    return np.where(constant, rows[0], mean), regularize_covariance(cov, np.var(pooled, axis=0, ddof=1))


def _fit_quietly(chain, max_num_sources):
    """Run mv_normal_fit; return its fit and the text of every UserWarning it gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        means, covs = mv_normal_fit(chain, max_num_sources)
    return means, covs, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _assert_pooled(means, covs, slot, chain):
    mean, cov = _pooled_reference(chain)
    np.testing.assert_allclose(means[slot], mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(covs[slot], cov, rtol=1e-12, atol=1e-14)


def _assert_own_fit(means, covs, slot, chain):
    mean, cov = _own_fit_reference(chain, slot)
    np.testing.assert_allclose(means[slot], mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(covs[slot], cov, rtol=1e-12, atol=1e-14)


def _mv_normal_entry(row, aux_parameters, label):
    """One row's log-density under one fit, from its own Cholesky factor and a solve."""
    means, covs = aux_parameters
    chol = np.linalg.cholesky(covs[label])
    solved = np.linalg.solve(chol, row - means[label])
    return -0.5 * (len(row) * np.log(2 * np.pi) + 2.0 * np.sum(np.log(np.diag(chol))) + solved @ solved)


def _uni_normal_entry(row, aux_parameters, label, param_index=1):
    """One row's univariate normal log-density of parameter `param_index` under one fit."""
    means, stds = aux_parameters
    z = (row[param_index] - means[label]) / stds[label]
    return -0.5 * np.log(2 * np.pi) - np.log(stds[label]) - 0.5 * z * z


def _reference_cost_matrix(sample, aux_parameters, prob_in_model, num_distributions, log_density=_mv_normal_entry):
    """The cost matrix one entry at a time: log p + log-density for a present row, log(1 - p) for an absent one."""
    cost = np.empty((num_distributions, len(sample)))
    for label in range(num_distributions):
        for slot, row in enumerate(sample):
            if np.isnan(row).any():
                cost[label, slot] = np.log1p(-prob_in_model[label])
            else:
                cost[label, slot] = np.log(prob_in_model[label]) + log_density(row, aux_parameters, label)
    return cost


@pytest.fixture
def cholesky_calls(monkeypatch):
    """Start from an empty factorization memo and record the shape of every np.linalg.cholesky call."""
    calls = []
    cholesky = np.linalg.cholesky

    def counting_cholesky(a, *args, **kwargs):
        calls.append(np.shape(a))
        return cholesky(a, *args, **kwargs)

    monkeypatch.setattr(fits, "_last_mv_normal_factors", None)
    monkeypatch.setattr(np.linalg, "cholesky", counting_cholesky)
    return calls


# The multivariate fit


def test_mv_normal_fit_of_healthy_slots_is_the_pandas_moments_plus_the_ridge():
    rng = np.random.default_rng(3)
    scales = np.array([1e-9, 1.0, 1e-22, 1e3])
    chain = rng.normal(size=(400, 2, 4)) * scales + np.array([1e-3, 0.0, 1e-21, 7.0])
    chain[rng.random(400) < 0.3, 1, :] = np.nan

    means, covs, messages = _fit_quietly(chain, 2)

    assert means.shape == (2, 4) and covs.shape == (2, 4, 4)
    assert messages == []
    for slot in range(2):
        rows = pd.DataFrame(_present_rows(chain, slot))
        expected_cov = np.array(rows.cov())
        expected_cov = expected_cov + COVARIANCE_RIDGE * np.diag(np.diag(expected_cov))
        np.testing.assert_array_equal(means[slot], np.array(rows.mean()))
        np.testing.assert_array_equal(covs[slot], expected_cov)

    uniform = _interleaved_uniform_chain()
    pinned_means = [0.49645889162008944, 0.495239621185434]
    pinned_variances = [0.08387139433093833, 0.08485848849960947]
    for max_num_sources in (2, 1):
        means, covs = mv_normal_fit(uniform, max_num_sources)
        assert means.shape == (max_num_sources, 1) and covs.shape == (max_num_sources, 1, 1)
        np.testing.assert_allclose(means[:, 0], pinned_means[:max_num_sources], rtol=1e-12)
        np.testing.assert_allclose(covs[:, 0, 0], np.multiply(pinned_variances[:max_num_sources], 1 + COVARIANCE_RIDGE),
                                   rtol=1e-12)

    fallback = regularize_covariance(np.zeros((2, 2)))
    np.testing.assert_allclose(fallback, COVARIANCE_RIDGE * np.eye(2))
    assert np.all(np.linalg.eigvalsh(fallback) > 0.0)


@pytest.mark.parametrize("num_params", [3, 8])
def test_a_slot_below_d_plus_two_rows_gets_exactly_the_pooled_fit(num_params):
    for extra_rows in (-2, -1, 0, 1, 2):
        rng = np.random.default_rng(num_params)
        chain = rng.normal(size=(200, 3, num_params)) + np.arange(3)[None, :, None] * 5.0
        chain[num_params + extra_rows:, 2, :] = np.nan

        means, covs, _ = _fit_quietly(chain, 3)

        if extra_rows < 2:
            _assert_pooled(means, covs, 2, chain)
            np.linalg.cholesky(covs[2])
        else:
            _assert_own_fit(means, covs, 2, chain)
        _assert_own_fit(means, covs, 0, chain)

    rng = np.random.default_rng(0)
    widened = rng.normal(loc=50.0, size=(300, 3, num_params))
    widened[:, 2, :] = np.nan
    widened[::3, 1, :] = np.nan
    means, covs, messages = _fit_quietly(widened, 3)
    _assert_pooled(means, covs, 2, widened)
    assert np.all(np.abs(means[2] - 50.0) < 0.2)
    assert messages == []

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        means, covs = mv_normal_fit(np.full((5, 2, num_params), np.nan), 2)
    assert means.shape == (2, num_params) and covs.shape == (2, num_params, num_params)
    assert np.isnan(means).all() and np.isnan(covs).all()


@pytest.mark.parametrize("build, effective, fresh_draw", [
    (partial(_stuck_chain, 1), "1.0", [6.3, 5.8, 6.1]),
    (partial(_stuck_chain, 3), "3.0", [6.3, 5.8, 6.1]),
    (_uneven_repeats_chain, "3.9", None),
    (partial(_fixed_inside_chain, 200), "1.0", None),
    (_forty_repeats_chain, None, None),
], ids=["one-state", "three-states", "uneven-repeats", "half-the-samples", "forty-repeats-sixty-distinct"])
def test_a_minority_slot_of_repeated_rows_borrows_the_pooled_covariance(build, effective, fresh_draw):
    chain = build()

    means, covs, messages = _fit_quietly(chain, chain.shape[1])

    if effective is None:
        assert messages == []
        _assert_own_fit(means, covs, 1, chain)
        return
    np.testing.assert_allclose(means[1], _present_rows(chain, 1).mean(axis=0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(covs[1], _pooled_reference(chain)[1], rtol=1e-12, atol=1e-14)
    assert len(messages) == 1
    assert messages[0].startswith("mv_normal_fit: source slot 1 ") and f" {effective} " in messages[0]
    if fresh_draw is not None:
        sample = np.full((4, 3), np.nan)
        sample[1] = fresh_draw
        assert mv_normal_log_prob(sample, (means, covs), 1)[1] > -20.0


@pytest.mark.parametrize("build", [
    partial(_fixed_inside_chain, 201), _fixed_inside_chain, _fixed_apart_chain, _plane_chain,
    _shared_dependence_chain, _derived_column_by_source,
], ids=["repeated-in-most-samples", "repeated-in-every-sample", "fixed-at-different-values", "rows-on-a-plane",
        "chain-wide-dependence", "derived-column"])
def test_repeated_states_in_most_samples_and_distinct_rows_keep_their_own_fit(build):
    chain = build()

    means, covs, messages = _fit_quietly(chain, chain.shape[1])

    assert messages == []
    for slot in range(chain.shape[1]):
        _assert_own_fit(means, covs, slot, chain)
    assert np.isfinite(mv_normal_log_prob(chain[0], (means, covs), np.arange(chain.shape[1]))).all()


@pytest.mark.parametrize("value", [0.0, 0.3, 1e22, "round-off"])
def test_a_constant_parameter_gets_its_value_and_the_pooled_variance_ridge(value):
    rng = np.random.default_rng(6)
    chain = rng.normal(size=(400, 2, 2)) + np.array([[0.0, 0.0], [5.0, 0.0]])
    chain[:, 1, 1] = _frozen_column(value, 400)

    means, covs, messages = _fit_quietly(chain, 2)

    assert messages == []
    if value != "round-off":
        assert means[1][1] == value
    assert covs[1][0, 1] == 0.0 and covs[1][1, 0] == 0.0
    assert covs[1][1, 1] == pytest.approx(COVARIANCE_RIDGE * np.var(chain[:, :, 1], ddof=1), rel=1e-12)

    base = np.random.default_rng(7).normal(size=(300, 2, 1)) + np.array([[0.0], [3.0]])
    everywhere = np.concatenate([base, _frozen_column(value, 600).reshape(300, 2, 1)], axis=2)
    means, covs, _ = _fit_quietly(everywhere, 2)
    _, zero_covs, _ = _fit_quietly(np.concatenate([base, np.zeros_like(base)], axis=2), 2)
    if value != "round-off":
        assert np.all(means[:, 1] == value)
    np.testing.assert_array_equal(covs, zero_covs)
    assert np.all(covs[:, 1, 1] == COVARIANCE_RIDGE)


def test_a_frozen_or_rescaled_parameter_does_not_change_the_catalog():
    def relabel(chain, width, **kwargs):
        posterior_chain = PosteriorChain(chain, width, chain.shape[2], trans_dimensional=True)
        return relabel_mv_normal(posterior_chain, width, progress=False, **kwargs).get_chain()

    rng = np.random.default_rng(0)
    centers, presence = [0.0, 1.0, 2.0], [1.0, 0.5, 0.3]
    base = np.full((600, 3, 1), np.nan)
    for i in range(600):
        sources = [k for k in range(3) if rng.random() < presence[k]]
        for k, slot in zip(sources, rng.permutation(3)[:len(sources)]):
            base[i, slot, 0] = rng.normal(centers[k], 0.1)
    reference = relabel(base, 3, num_iterations=60)[:, :, 0]
    frozen = np.concatenate([base, np.where(np.isnan(base), np.nan, 0.3)], axis=2)
    np.testing.assert_array_equal(relabel(frozen, 3, num_iterations=60)[:, :, 0], reference)

    chain, truth = _derived_column_chain()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        catalog = relabel(chain, 4, num_iterations=50)
    np.testing.assert_array_equal(catalog[:, :, :2], relabel(chain[:, :, :2], 4, num_iterations=50))
    for label in range(4):
        sources = [truth[i][chain[i, :, 0] == catalog[i, label, 0]][0] for i in range(len(chain))]
        assert np.unique(sources).size == 1

    rng = np.random.default_rng(5)
    chain = rng.normal(size=(200, 2, 2)) * [1.0, 0.05] + [[0.0, -2.0], [0.0, 2.0]]
    for sample in chain:
        rng.shuffle(sample)
    catalogs = []
    for units in (1.0, 1e-6, 1e-9):
        scaled = chain * [1.0, units]
        original = scaled.copy()
        restored = relabel(scaled, 2, num_iterations=5) / [1.0, units]
        ordered = restored[:, np.argsort(restored[:, :, 1].mean(axis=0)), :]
        assert np.all(ordered[:, 0, 1] < 0) and np.all(ordered[:, 1, 1] > 0)
        np.testing.assert_array_equal(scaled, original)
        catalogs.append(ordered)
    for ordered in catalogs[1:]:
        np.testing.assert_allclose(ordered, catalogs[0], rtol=1e-14, atol=0)

    rng = np.random.default_rng(5)
    chain = rng.normal(size=(200, 2, 2)) * [1.0, 0.0] + [[0.0, -1.0], [0.0, 1.0]]
    for sample in chain:
        rng.shuffle(sample)
    recovered = []
    for units in (1.0, 2.0 ** -40):
        restored = relabel(chain * [1.0, units], 2, num_iterations=5) / [1.0, units]
        ordered = restored[:, np.argsort(restored[:, :, 1].mean(axis=0)), :]
        np.testing.assert_array_equal(ordered[:, 0, 1], -1.0)
        np.testing.assert_array_equal(ordered[:, 1, 1], 1.0)
        recovered.append(ordered)
    np.testing.assert_array_equal(recovered[0], recovered[1])

    x = np.linspace(-1.0, 1.0, 30)
    mixed = np.stack([x, 1e-22 * x, np.full_like(x, 3.0)], axis=-1)[:, None, :]
    assert np.isfinite(mv_normal_log_prob(mixed[10], mv_normal_fit(mixed, 1), [0])).all()


# The univariate fit


def _healthy_uniform_case():
    chain = _interleaved_uniform_chain()
    valid = [chain[:, k, 0][~np.isnan(chain[:, k, 0])] for k in range(2)]
    return chain, 0, [np.mean(v) for v in valid], [np.std(v, ddof=1) for v in valid], None


def _narrow_slots_case():
    rng = np.random.default_rng(0)
    centres = np.array([1e-4, 5e-3, 5e-3 + 2e-7, 1e-2])
    widths = np.array([1e-9, 3e-9, 1e-8, 1e-8])
    chain = (centres + widths * rng.standard_normal((4000, 4)))[:, :, None]
    return (chain, 0, [np.mean(chain[:, k, 0]) for k in range(4)],
            [np.std(chain[:, k, 0], ddof=1) for k in range(4)], None)


def _constant_slot_case(constant):
    rng = np.random.default_rng(9 if constant != "round-off" else 10)
    num_samples = 400 if constant != "round-off" else 40
    chain = np.zeros((num_samples, 2, 1))
    chain[:, 0, 0] = rng.normal(size=num_samples)
    if constant == "round-off":
        chain[:, 1, 0] = 0.3 + np.arange(num_samples) % 9 * np.spacing(0.3)
    else:
        chain[:, 1, 0] = constant
    replacement = np.sqrt(COVARIANCE_RIDGE) * np.std(chain[:, :, 0], ddof=1)
    constant_mean = np.mean(chain[:, 1, 0]) if constant == "round-off" else constant
    return chain, 0, [np.mean(chain[:, 0, 0]), constant_mean], [np.std(chain[:, 0, 0], ddof=1), replacement], 1


def _frozen_everywhere_case(constant):
    return np.full((20, 2, 1), constant), 0, [constant, constant], [UNIT_REPLACEMENT_STD] * 2, None


def _sparse_slot_case():
    chain = np.full((40, 2, 2), np.nan)
    chain[:, 0, 0] = np.linspace(0.0, 1.0, 40)
    chain[:, 0, 1] = 100.0
    chain[:3, 1, :] = 0.5
    assert 3 < MIN_UNIVARIATE_SAMPLES
    pooled = chain[:, :, 0][~np.isnan(chain[:, :, 0])]
    return (chain, 0, [np.mean(chain[:, 0, 0]), np.mean(pooled)],
            [np.std(chain[:, 0, 0], ddof=1), np.std(pooled, ddof=1)], None)


def _all_absent_case():
    return np.full((20, 2, 1), np.nan), 0, [np.nan, np.nan], [UNIT_REPLACEMENT_STD] * 2, None


@pytest.mark.parametrize("build", [
    _healthy_uniform_case, _narrow_slots_case, partial(_constant_slot_case, 0.0), partial(_constant_slot_case, 0.3),
    partial(_constant_slot_case, "round-off"), partial(_frozen_everywhere_case, 0.0),
    partial(_frozen_everywhere_case, 0.3), _sparse_slot_case, _all_absent_case,
], ids=["healthy", "narrow-among-wide-spaced", "constant-0.0", "constant-0.3", "round-off", "frozen-everywhere-0.0",
        "frozen-everywhere-0.3", "sparse-slot-pools-its-parameter", "all-absent"])
def test_uni_normal_fit_keeps_healthy_stds_and_replaces_degenerate_ones(build):
    chain, param_index, expected_means, expected_stds, constant_slot = build()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        means, stds = uni_normal_fit(chain, chain.shape[1], param_index)

    np.testing.assert_array_equal(means, expected_means)
    np.testing.assert_array_equal(stds, expected_stds)
    if constant_slot is not None:
        cost = partial(compute_cost_matrix, log_prob=partial(uni_normal_log_prob, param_index=param_index))
        matrix = cost(chain[0], (means, stds), np.array([0.5, 0.5]), 2)
        assert np.isfinite(matrix).all()
        assert matrix[constant_slot, constant_slot] > matrix[constant_slot, 1 - constant_slot]


# The log-densities


def test_mv_normal_log_prob_matches_scipy_and_shifts_only_by_the_jacobian():
    sample = np.random.default_rng(2).normal(size=(3, 2))
    means = np.array([[0.5, -0.25], [1.0, 2.0]])
    covs = np.array([[[1.3, 0.4], [0.4, 0.8]], [[2.0, -0.3], [-0.3, 0.5]]])

    one = mv_normal_log_prob(sample, (means, covs), 1)
    both = mv_normal_log_prob(sample, (means, covs), [0, 1])

    assert one.shape == (3,) and both.shape == (2, 3)
    np.testing.assert_allclose(one, multivariate_normal(means[1], covs[1]).logpdf(sample), rtol=1e-12)
    for k in range(2):
        np.testing.assert_allclose(both[k], multivariate_normal(means[k], covs[k]).logpdf(sample), rtol=1e-12)

    chain = np.random.default_rng(7).normal(size=(30, 2, 2)) + [[0.0, -2.0], [0.0, 2.0]]
    units = np.array([1.0, 1e-22])
    ordinary = mv_normal_log_prob(chain[0], mv_normal_fit(chain, 2), [0, 1])
    rescaled = mv_normal_log_prob(chain[0] * units, mv_normal_fit(chain * units, 2), [0, 1])
    np.testing.assert_allclose(rescaled + np.log(units).sum(), ordinary, rtol=1e-12)


@pytest.mark.parametrize("log_prob, aux_parameters, error, anchor", [
    (mv_normal_log_prob, ([np.zeros(2), np.zeros(2)], [np.eye(2), np.zeros((2, 2))]), np.linalg.LinAlgError,
     "not positive definite"),
    (partial(uni_normal_log_prob, param_index=0), ([0.0, 1.0], [1.0, 0.0]), ValueError,
     "not strictly positive and finite"),
    (partial(uni_normal_log_prob, param_index=0), ([0.0, 1.0], [1.0, np.nan]), ValueError,
     "not strictly positive and finite"),
], ids=["singular-covariance", "zero-std", "nan-std"])
def test_log_probs_reject_an_undefined_scale(log_prob, aux_parameters, error, anchor):
    with pytest.raises(error, match=anchor) as excinfo:
        log_prob(np.zeros((2, 2)), aux_parameters, [0, 1])
    assert "[1]" in str(excinfo.value)


@pytest.mark.parametrize("second_fit, refactorized", [
    ("same-arrays", False), ("equal-copies", False), ("edited-in-place", True),
])
def test_the_factorization_is_cached_only_for_an_equal_fit(cholesky_calls, second_fit, refactorized):
    rng = np.random.default_rng(6)
    sample, other_sample = rng.normal(size=(2, 3, 2))
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    covs = np.array([np.eye(2), [[2.0, 0.3], [0.3, 1.0]]])
    mv_normal_log_prob(sample, (means, covs), [0, 1])

    if second_fit == "equal-copies":
        means, covs = means.copy(), covs.copy()
    elif second_fit == "edited-in-place":
        covs[1] = [[3.0, 0.5], [0.5, 2.0]]
    logpdf = mv_normal_log_prob(other_sample, (means, covs), np.array([0, 1]))

    assert cholesky_calls == [(2, 2, 2)] * (2 if refactorized else 1)
    for k in range(2):
        np.testing.assert_allclose(logpdf[k], multivariate_normal(means[k], covs[k]).logpdf(other_sample),
                                   rtol=1e-12)
    cached_means, cached_covs, _, factors = fits._last_mv_normal_factors
    assert not np.shares_memory(cached_means, means) and not np.shares_memory(cached_covs, covs)
    assert not any(array.flags.writeable for array in factors)
    assert means.flags.writeable and covs.flags.writeable and logpdf.flags.writeable
    expected = logpdf.copy()
    logpdf[:] = 0.0
    np.testing.assert_array_equal(mv_normal_log_prob(other_sample, (means, covs), [0, 1]), expected)

    chain = _scrambled_chain()
    cost = partial(compute_cost_matrix, log_prob=mv_normal_log_prob)
    prob_in_model = find_prob_in_model(chain, 6)
    for fit in (mv_normal_fit(chain, 6), mv_normal_fit(chain[::-1, ::-1], 6)):
        cholesky_calls.clear()
        relabel_samples_one_iteration(chain, fit, prob_in_model, 6, cost, progress=False)
        assert cholesky_calls == [(6, 4, 4)]

    good = ([np.zeros(2)], [np.eye(2)])
    expected = mv_normal_log_prob(sample, good, 0)
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        mv_normal_log_prob(sample, ([np.zeros(2)], [np.zeros((2, 2))]), 0)
    calls_after_failure = len(cholesky_calls)
    np.testing.assert_array_equal(mv_normal_log_prob(sample, good, 0), expected)
    assert len(cholesky_calls) == calls_after_failure


# The cost matrix


COST_METHODS = {
    "mv": (mv_normal_fit, mv_normal_log_prob, _mv_normal_entry),
    "uni": (partial(uni_normal_fit, param_index=1), partial(uni_normal_log_prob, param_index=1), _uni_normal_entry),
}


@pytest.mark.parametrize("method", ["mv", "uni"])
def test_cost_matrix_scores_present_rows_by_density_and_absent_rows_by_exclusion(method):
    fit, log_prob, log_density = COST_METHODS[method]
    cost = partial(compute_cost_matrix, log_prob=log_prob)
    reference = partial(_reference_cost_matrix, log_density=log_density)
    chain = _scrambled_chain()
    aux_parameters = fit(chain, 6)
    prob_in_model = find_prob_in_model(chain, 6)

    for sample in chain:
        np.testing.assert_allclose(cost(sample, aux_parameters, prob_in_model, 6),
                                   reference(sample, aux_parameters, prob_in_model, 6), rtol=1e-12, atol=1e-12)

    i, absent = np.argwhere(np.isnan(chain[:, :, 0]))[0]
    present = int(np.flatnonzero(~np.isnan(chain[i, :, 0]))[0])
    column = cost(chain[i], aux_parameters, prob_in_model, 6)[:, absent]
    np.testing.assert_array_equal(column, np.log1p(-prob_in_model))
    assert np.unique(column).size > 1

    narrow_fit = fit(chain, 3)
    narrow = cost(chain[i], narrow_fit, prob_in_model[:3], 3)
    assert narrow.shape == (3, 6)
    np.testing.assert_allclose(narrow, reference(chain[i], narrow_fit, prob_in_model[:3], 3), rtol=1e-12, atol=1e-12)
    wide = cost(chain[i, :4], aux_parameters, prob_in_model, 6)
    assert wide.shape == (6, 4)
    np.testing.assert_allclose(wide, reference(chain[i, :4], aux_parameters, prob_in_model, 6), rtol=1e-12,
                               atol=1e-12)

    unclipped = np.concatenate([[1.0, 0.0], prob_in_model[2:]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matrix = cost(chain[i], aux_parameters, unclipped, 6)
    assert matrix[0, absent] == -np.inf and matrix[1, present] == -np.inf

    if method == "mv":
        golden = golden_chain().get_chain()
        matrix = cost(golden[0], mv_normal_fit(golden, 2), find_prob_in_model(golden, 2, eps=0), 2)
        np.testing.assert_allclose(matrix, np.array([[0.30395628, 0.29753514], [-0.05770942, -0.06278939]]).T,
                                   rtol=1e-8, atol=1e-8)
        thinned = _thinned_uniform_chain()
        matrix = cost(thinned[2], mv_normal_fit(thinned, 2), find_prob_in_model(thinned, 2, eps=0), 2)
        np.testing.assert_allclose(matrix, [[-0.44169619, -0.69314718], [-0.8618358, -0.40556511]], rtol=1e-7,
                                   atol=1e-8)
    else:
        uniform = _interleaved_uniform_chain()
        uniform_fit = uni_normal_fit(uniform, 2, 0)
        prob = find_prob_in_model(uniform, 2, eps=0)
        np.testing.assert_allclose(prob, 0.5)
        matrix = compute_cost_matrix(uniform[0], uniform_fit, prob, 2,
                                     log_prob=partial(uni_normal_log_prob, param_index=0))
        expected = _reference_cost_matrix(uniform[0], uniform_fit, prob, 2,
                                          log_density=partial(_uni_normal_entry, param_index=0))
        np.testing.assert_allclose(matrix, expected, rtol=1e-12)
        both = uni_normal_log_prob(uniform[0], uniform_fit, [0, 1], param_index=0)
        np.testing.assert_array_equal(uni_normal_log_prob(uniform[0], uniform_fit, 1, param_index=0), both[1])


def test_an_assignment_matches_a_straightforward_reference_labeling():
    chain = _scrambled_chain()
    aux_parameters = mv_normal_fit(chain, 6)
    prob_in_model = find_prob_in_model(chain, 6)

    relabeled, cost = relabel_samples_one_iteration(chain, aux_parameters, prob_in_model, 6,
                                                    partial(compute_cost_matrix, log_prob=mv_normal_log_prob),
                                                    progress=False)
    expected_relabeled, expected_cost = relabel_samples_one_iteration(chain, aux_parameters, prob_in_model, 6,
                                                                      _reference_cost_matrix, progress=False)

    np.testing.assert_array_equal(relabeled, expected_relabeled)
    assert cost == pytest.approx(expected_cost, rel=1e-12, abs=1e-12)
