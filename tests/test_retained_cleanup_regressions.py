"""Regression coverage for repaired paths retained by the package cleanup."""

from collections.abc import Callable

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import petra.relabel as relabel_module
from petra.bayesian_gaussian import compute_niw_prior, make_catalog_bayesian_gaussian
from petra.copula_flows import copula_flows_aux_distribution
from petra.cost_matrix import create_compute_cost_matrix
from petra.flow_utils import DEFAULT_LOG_PROB_FLOOR
from petra.initialization import label_chain_by_histogram
from petra.posterior_chain import PosteriorChain
from petra.relabel import load_checkpoint, run_relabeling_loop
from petra.samples_io import (load_samples, load_samples_fixed_num_sources,
                              load_samples_product_space, load_samples_ucbmcmc)
from petra.utils import UniformPrior


@pytest.mark.parametrize(
    "aux_distribution",
    [copula_flows_aux_distribution],
    ids=["copula"],
)
def test_flow_cost_matrix_uses_exclusion_probability_for_an_absent_slot(
        aux_distribution: Callable) -> None:
    """A floored absent density must not hide the spike half of the model."""
    sample = np.array([[0.5], [np.nan]])
    flows = [UniformPrior([0.0], [np.exp(3.0)]), UniformPrior([0.0], [1.0])]
    prob_in_model = np.array([0.9, 0.1])

    compute_cost = create_compute_cost_matrix(aux_distribution)
    cost = compute_cost(sample, flows, prob_in_model, num_distributions=2)

    expected = np.array([
        [-3.1053605156578263, -2.302585092994046],
        [-2.3025850929940455, -0.10536051565782631],
    ])
    assert np.allclose(cost, expected, atol=1e-6)

    rows, columns = linear_sum_assignment(cost, maximize=True)
    assert np.array_equal(rows, np.array([0, 1]))
    assert np.array_equal(columns, np.array([0, 1]))


@pytest.mark.parametrize(
    "aux_distribution",
    [copula_flows_aux_distribution],
    ids=["copula"],
)
def test_flow_cost_matrix_keeps_the_density_floor_for_a_present_invalid_point(
        aux_distribution: Callable) -> None:
    sample = np.array([[2.0]])
    prob_in_model = np.array([0.25])
    compute_cost = create_compute_cost_matrix(aux_distribution)

    cost = compute_cost(
        sample,
        [UniformPrior([0.0], [1.0])],
        prob_in_model,
        num_distributions=1,
    )

    assert np.isfinite(cost[0, 0])
    assert cost[0, 0] == pytest.approx(DEFAULT_LOG_PROB_FLOOR + np.log(0.25))


def test_every_checkpoint_contains_the_best_state_available_at_that_iteration(
        tmp_path, monkeypatch) -> None:
    """The latest checkpoint must be safe to resume after a worse iteration."""
    initial = np.array([[[0.0], [10.0]], [[1.0], [11.0]]])
    produced = [initial + 1.0, initial + 2.0, initial + 3.0]
    scripted_costs = [1.0, 2.0, 3.0]
    call_count = 0

    def fake_one_iteration(posterior_chain, aux_parameters, prob_in_model,
                           max_num_sources, compute_cost_matrix, progress=True):
        nonlocal call_count
        result = PosteriorChain(
            produced[call_count],
            num_sources=2,
            num_params_per_source=1,
            prob_in_model=prob_in_model,
            cost_dict={max_num_sources: scripted_costs[call_count]},
        )
        call_count += 1
        return result

    monkeypatch.setattr(
        relabel_module, "relabel_posterior_chain_one_iteration", fake_one_iteration
    )

    posterior_chain = PosteriorChain(initial, 2, 1)

    def param_fit(chain, max_num_sources):
        return None

    def compute_cost_matrix(*args):
        return np.zeros((2, 2))

    result = run_relabeling_loop(
        posterior_chain,
        param_fit,
        compute_cost_matrix,
        num_iterations=2,
        checkpoint_dir=str(tmp_path),
        progress=False,
    )
    latest, completed = load_checkpoint(str(tmp_path))

    assert completed == 2
    assert result.cost_dict[2] == 1.0
    assert latest.cost_dict[2] == 1.0
    assert np.array_equal(latest.chain, result.chain)

    resumed = run_relabeling_loop(
        posterior_chain,
        param_fit,
        compute_cost_matrix,
        num_iterations=3,
        checkpoint_dir=str(tmp_path),
        resume_from=str(tmp_path),
        progress=False,
    )
    latest_after_resume, completed = load_checkpoint(str(tmp_path))

    assert completed == 3
    assert resumed.cost_dict[2] == 1.0
    assert latest_after_resume.cost_dict[2] == 1.0
    assert np.array_equal(latest_after_resume.chain, result.chain)


def test_one_parameter_text_file_requires_an_explicit_layout(tmp_path) -> None:
    """A one-parameter fixed chain must not be silently read as product-space."""
    path = tmp_path / "ambiguous.dat"
    np.savetxt(path, np.array([[1, 2, 0, 0, 0, 0], [3, 4, 0, 0, 0, 0]]))

    explicit = load_samples_fixed_num_sources(str(path), num_params_per_source=1)
    assert explicit.chain.shape == (2, 2, 1)
    assert np.array_equal(explicit.chain[:, :, 0], np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError, match="source counts"):
        load_samples_product_space(str(path), num_params_per_source=1)

    product_path = tmp_path / "product.dat"
    np.savetxt(product_path, [[1, 2, 1, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0]])
    explicit_product = load_samples_product_space(str(product_path), num_params_per_source=1)
    np.testing.assert_array_equal(explicit_product.chain[:, :, 0], [[1, 2], [3, np.nan]])

    with pytest.raises(ValueError, match="ambiguous.*load_samples_fixed_num_sources.*load_samples_product_space"):
        load_samples(str(path), num_params_per_source=1)


@pytest.mark.parametrize("count", [-2, 0.5, np.nan, np.inf, 2])
def test_product_space_rejects_invalid_source_counts(tmp_path, count) -> None:
    path = tmp_path / "invalid-product.dat"
    # Two data columns followed by the zero-based source count and four metadata columns.
    np.savetxt(path, [[1, 2, count, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="source counts"):
        load_samples_product_space(str(path), num_params_per_source=1)


@pytest.mark.parametrize("row", [[1, 2, 3], [1, 2, 3, 0, 0, 0, 0, 0]])
def test_product_space_rejects_invalid_column_layout(tmp_path, row) -> None:
    path = tmp_path / "invalid-layout.dat"
    np.savetxt(path, [row])
    with pytest.raises(ValueError, match="data columns"):
        load_samples_product_space(str(path), num_params_per_source=2)


def test_product_space_preserves_a_sample_with_no_sources(tmp_path) -> None:
    path = tmp_path / "empty-sources.dat"
    np.savetxt(path, [[0, -1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])
    restored = load_samples_product_space(str(path), num_params_per_source=1)
    np.testing.assert_array_equal(restored.chain[:, :, 0], [[np.nan], [5]])


@pytest.mark.parametrize(
    "loader,args",
    [
        (load_samples, ("missing.dat",)),
        (load_samples_fixed_num_sources, ("missing.dat",)),
        (load_samples_product_space, ("missing.dat",)),
        (load_samples_ucbmcmc, ("missing",)),
    ],
)
def test_sample_loaders_reject_an_invalid_parameter_count_before_reading(
        loader: Callable, args: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="num_params_per_source must be a positive integer"):
        loader(*args, num_params_per_source=0)


@pytest.mark.parametrize(
    "chain",
    [
        np.full((2, 2, 2), np.nan),
        np.array([[[1.0, 2.0], [np.nan, np.nan]]]),
        np.array([[[np.inf, 2.0]], [[1.0, 2.0]]]),
    ],
    ids=["no-complete-vectors", "one-complete-vector", "one-finite-vector"],
)
def test_niw_prior_requires_two_complete_vectors(chain: np.ndarray) -> None:
    with pytest.raises(ValueError, match="at least two complete, finite source vectors"):
        compute_niw_prior(chain, max_num_sources=2)


@pytest.mark.parametrize("num_valid", [0, 1])
@pytest.mark.filterwarnings("error")
def test_public_bayesian_entry_rejects_sparse_data_before_initialization(num_valid) -> None:
    values = np.full((2, 2, 2), np.nan)
    if num_valid:
        values[0, 0] = [1, 2]
    posterior_chain = PosteriorChain(values, 2, 2, trans_dimensional=True)
    with pytest.raises(ValueError, match="at least two complete, finite source vectors"):
        make_catalog_bayesian_gaussian(posterior_chain, 2, progress=False)


def test_niw_prior_is_finite_with_two_complete_vectors() -> None:
    chain = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    prior = compute_niw_prior(chain, max_num_sources=1)

    assert np.array_equal(prior["m0"], np.array([2.0, 3.0]))
    assert np.all(np.isfinite(prior["Psi0"]))
    np.linalg.cholesky(prior["Psi0"])


def test_zero_surrounding_bins_still_claims_the_center_histogram_bin() -> None:
    values = np.full((2, 1, 1), 0.001000001)
    chain = PosteriorChain(values, 1, 1)

    labeled = label_chain_by_histogram(
        chain,
        num_surrounding_bins=0,
        num_extra_entries=1,
        low_num_samples=1,
    )

    assert labeled.num_sources == 1
    assert np.array_equal(labeled.chain, values)
