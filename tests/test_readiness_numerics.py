"""Regressions for numerical failures reproduced during the readiness audit."""

import numpy as np
import pytest

from petra import PosteriorChain, make_catalog_bayesian_gaussian, make_catalog_mv_normal
from petra.aux_distributions import mv_normal_aux_distribution
from petra.bayesian_gaussian import compute_niw_prior, niw_aux_distribution, niw_fit, precompute_t_params
from petra.cost_matrix import create_compute_cost_matrix
from petra.parametric_fits import mv_normal_fit
from petra.relabel import relabel_samples_one_iteration


@pytest.mark.parametrize("make_catalog", [make_catalog_mv_normal, make_catalog_bayesian_gaussian])
def test_gaussian_catalog_recovery_survives_changing_one_parameters_units(make_catalog):
    rng = np.random.default_rng(5)
    chain = rng.normal(size=(200, 2, 2)) * [1.0, 0.05] + [[0.0, -2.0], [0.0, 2.0]]
    for sample in chain:
        rng.shuffle(sample)

    catalogs = []
    for units in (1.0, 1e-9):
        scaled_chain = chain * [1.0, units]
        original = scaled_chain.copy()
        catalog = make_catalog(
            PosteriorChain(scaled_chain, 2, 2),
            2,
            num_iterations=5,
            initialization_param_index=1,
            init_num_iterations=5,
            progress=False,
        )
        restored = catalog.chain / [1.0, units]
        ordered = restored[:, np.argsort(restored[:, :, 1].mean(axis=0)), :]
        assert np.all(ordered[:, 0, 1] < 0)
        assert np.all(ordered[:, 1, 1] > 0)
        np.testing.assert_array_equal(scaled_chain, original)
        catalogs.append(ordered)

    np.testing.assert_allclose(catalogs[0], catalogs[1], rtol=1e-14, atol=0)


def test_constant_source_parameters_keep_their_separation_after_changing_units():
    rng = np.random.default_rng(5)
    chain = rng.normal(size=(200, 2, 2)) * [1.0, 0.0] + [[0.0, -1.0], [0.0, 1.0]]
    for sample in chain:
        rng.shuffle(sample)

    recovered = []
    for units in (1.0, 2.0 ** -40):
        catalog = make_catalog_mv_normal(
            PosteriorChain(chain * [1.0, units], 2, 2), 2,
            num_iterations=5, initialization_param_index=1,
            init_num_iterations=5, progress=False,
        )
        restored = catalog.chain / [1.0, units]
        ordered = restored[:, np.argsort(restored[:, :, 1].mean(axis=0)), :]
        np.testing.assert_array_equal(ordered[:, 0, 1], -1.0)
        np.testing.assert_array_equal(ordered[:, 1, 1], 1.0)
        recovered.append(ordered)

    np.testing.assert_array_equal(recovered[0], recovered[1])


@pytest.mark.parametrize("fit,evaluate", [
    (mv_normal_fit, mv_normal_aux_distribution),
    (niw_fit, niw_aux_distribution),
])
def test_gaussian_density_change_of_units_has_only_the_jacobian_shift(fit, evaluate):
    rng = np.random.default_rng(7)
    chain = rng.normal(size=(30, 2, 2)) + [[0.0, -2.0], [0.0, 2.0]]
    units = np.array([1.0, 1e-22])
    ordinary = evaluate(chain[0], fit(chain, 2), [0, 1])
    rescaled = evaluate(chain[0] * units, fit(chain * units, 2), [0, 1])
    np.testing.assert_allclose(rescaled + np.log(units).sum(), ordinary, rtol=1e-12)


@pytest.mark.parametrize("fit,evaluate", [
    (mv_normal_fit, mv_normal_aux_distribution),
    (niw_fit, niw_aux_distribution),
])
def test_mixed_scale_rank_deficient_fits_with_constant_parameters_stay_finite(fit, evaluate):
    x = np.linspace(-1.0, 1.0, 30)
    chain = np.stack([x, 1e-22 * x, np.full_like(x, 3.0)], axis=-1)[:, None, :]
    parameters = fit(chain, 1)
    log_prob = evaluate(chain[10], parameters, [0])
    assert np.isfinite(log_prob).all()


@pytest.mark.parametrize("parameter", ["kappa0", "nu0"])
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_niw_prior_hyperparameters_must_be_finite(parameter, value):
    chain = np.arange(12, dtype=float).reshape(3, 2, 2)
    with pytest.raises(ValueError, match=rf"{parameter}.*finite"):
        compute_niw_prior(chain, 2, **{parameter: value})


@pytest.mark.parametrize("invalid_parameters", [{"Psi_n": np.zeros((1, 1))}, {"nu_n": 0}])
def test_one_degenerate_niw_source_uses_inclusion_probabilities(invalid_parameters):
    valid = {"m_n": np.zeros(1), "kappa_n": 1.0, "nu_n": 3, "Psi_n": np.eye(1)}
    parameters = precompute_t_params([{**valid, **invalid_parameters}, valid], 1)
    sample = np.array([[0.0], [np.nan]])
    probabilities = np.array([0.8, 0.2])

    log_density = niw_aux_distribution(sample, parameters, [0, 1])
    assert log_density[0, 0] == 0.0
    assert np.isnan(log_density[:, 1]).all()
    compute_cost = create_compute_cost_matrix(niw_aux_distribution)
    cost = compute_cost(sample, parameters, probabilities, 2)
    np.testing.assert_allclose(cost[0], [np.log(0.8), np.log(0.2)])
    assert np.isfinite(cost).all()

    relabeled, total_cost = relabel_samples_one_iteration(
        sample[None, :, :], parameters, probabilities, 2, compute_cost, progress=False,
    )
    np.testing.assert_array_equal(relabeled[0], sample)
    assert np.isfinite(total_cost)
