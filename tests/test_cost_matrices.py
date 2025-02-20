import pytest
import numpy as np

from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import create_parametric_fit, mv_normal_fit
from petra.aux_distributions import mv_normal_aux_distribution
from petra.cost_matrix import create_compute_cost_matrix
from petra.utils import find_prob_in_model


@pytest.fixture
def posterior_chain_no_nans_fixture():
    np.random.seed(0)
    chain = np.zeros((10_000, 2))
    chain_signal_1 = np.random.uniform(0, 1, 10000)
    chain_signal_2 = np.random.uniform(0, 1, 10000)
    chain[:, 0] = chain_signal_1
    chain[:, 1] = chain_signal_2
    return PosteriorChain(chain, 2, 1)


@pytest.fixture
def posterior_chain_nans_fixture():
    np.random.seed(0)
    chain = np.zeros((10_000, 2))
    chain_signal_1 = np.random.uniform(0, 1, 10000)
    chain_signal_2 = np.random.uniform(0, 1, 10000)
    chain[::2, 0] = chain_signal_1[::2]
    chain[1::2, 0] = np.nan
    chain[::3, 1] = chain_signal_2[::3]
    chain[1::3, 1] = np.nan
    chain[2::3, 1] = np.nan
    return PosteriorChain(chain, 2, 1)


def test_compute_mv_gaussian_cost_matrix_no_nans(posterior_chain_no_nans_fixture):
    max_num_sources = 2
    chain = posterior_chain_no_nans_fixture.get_chain()
    fit_mv_gaussian = create_parametric_fit(mv_normal_fit)

    mv_gaussian_params = fit_mv_gaussian(chain, max_num_sources=max_num_sources)
    compute_mv_gaussian_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)
    prob_in_model = find_prob_in_model(chain, max_num_sources=max_num_sources, eps=0)

    cost_matrix = compute_mv_gaussian_cost_matrix(chain[0], mv_gaussian_params, prob_in_model, num_distributions=max_num_sources)
    assert cost_matrix.shape == (2, 2)
    cost_matrix_expected = np.array([[ 0.30395628,  0.29753514],
                                      [-0.05770942, -0.06278939]]).T
    for i in range(2):
        for j in range(2):
            assert np.isclose(cost_matrix[i, j], cost_matrix_expected[i, j], rtol=1e-8)


def test_compute_mv_gaussian_cost_matrix_with_nans(posterior_chain_nans_fixture):
    max_num_sources = 2
    chain = posterior_chain_nans_fixture.get_chain()
    fit_mv_gaussian = create_parametric_fit(mv_normal_fit)

    mv_gaussian_params = fit_mv_gaussian(chain, max_num_sources=max_num_sources)
    compute_mv_gaussian_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)
    prob_in_model = find_prob_in_model(chain, max_num_sources=max_num_sources, eps=0)
    cost_matrix = compute_mv_gaussian_cost_matrix(chain[2], mv_gaussian_params, prob_in_model, num_distributions=max_num_sources)
    cost_matrix_expected = np.array([[ -0.44169619,  -0.40556511],
                                      [-0.8618358, -0.40556511]])
    for i in range(2):
        for j in range(2):
            assert np.isclose(cost_matrix[i, j], cost_matrix_expected[i, j], rtol=1e-7)


# def test_jit_compute_mv_gaussian_cost_matrix_with_nans(posterior_chain_nans_fixture):
#     max_num_sources = 2
#     chain = posterior_chain_nans_fixture.get_chain()
#     fit_mv_gaussian = create_parametric_fit(mv_normal_fit)

#     mv_gaussian_params = fit_mv_gaussian(chain, max_num_sources=max_num_sources)
#     jitted_mv_gaussian_cost_matrix = create_compute_cost_matrix_numba(mv_normal_aux_distribution)
#     prob_in_model = find_prob_in_model(chain, max_num_sources=max_num_sources, eps=0)
#     cost_matrix = jitted_mv_gaussian_cost_matrix(chain[2], mv_gaussian_params, prob_in_model, num_distributions=max_num_sources)
#     cost_matrix_expected = np.array([[ -0.44169619,  -0.40556511],
#                                       [-0.8618358, -0.40556511]])
#     for i in range(2):
#         for j in range(2):
#             assert np.isclose(cost_matrix[i, j], cost_matrix_expected[i, j], rtol=1e-7)
