import pytest
import numpy as np

from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import uni_normal_fit_single_parameter, create_parametric_fit
from petra.aux_distributions import uni_normal_aux_distribution_single_parameter
from petra.cost_matrix import create_compute_cost_matrix


@pytest.fixture
def posterior_chain_nans_fixture():
    np.random.seed(0)
    chain = np.zeros((20_000, 2))
    chain_signal_1 = np.random.uniform(0, 1, 10000)
    chain_signal_2 = np.random.uniform(0, 1, 10000)
    chain[::2, 0] = chain_signal_1
    chain[1::2, 0] = np.nan
    chain[::2, 1] = chain_signal_2
    chain[1::2, 1] = np.nan
    return PosteriorChain(chain, 2, 1, trans_dimensional=True, )


def test_fit_uni_gaussian_single_param(posterior_chain_nans_fixture):
    chain = posterior_chain_nans_fixture.get_chain()
    uni_normal_single_param_fit = create_parametric_fit(uni_normal_fit_single_parameter, single_parameter=0)
    means, std_devs = uni_normal_single_param_fit(chain, max_num_sources=2)
    assert len(means) == 2
    assert len(std_devs) == 2
    assert np.isclose(means[0], 0.49645889)
    assert np.isclose(means[1], 0.49523962)
    assert np.isclose(std_devs[0], 0.2895911)
    assert np.isclose(std_devs[1], 0.29129024)


def test_uni_gaussian_cost_matrix(posterior_chain_nans_fixture):
    chain = posterior_chain_nans_fixture.get_chain()
    uni_normal_single_param_fit = create_parametric_fit(uni_normal_fit_single_parameter, single_parameter=0)
    uni_fit = uni_normal_single_param_fit(chain, max_num_sources=2)
    compute_cost_matrix = create_compute_cost_matrix(uni_normal_aux_distribution_single_parameter, single_parameter=0)
    # prob_in_model = find_prob_in_model(chain, max_num_sources=2)
    prob_in_model = np.array([1, 1])  # to check with the old code; this is actually incorrect
    cost_matrix = compute_cost_matrix(chain[0], uni_fit, prob_in_model, num_distributions=2)
    print(cost_matrix)
    assert cost_matrix.shape == (2, 2)
    assert np.isclose(cost_matrix[0, 0], 0.30400465)
    assert np.isclose(cost_matrix[0, 1], -0.05769722)
    assert np.isclose(cost_matrix[1, 0], 0.29758345)
    assert np.isclose(cost_matrix[1, 1], -0.06277712)









