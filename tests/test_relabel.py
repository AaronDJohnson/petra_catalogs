import pytest
import numpy as np

from petra.posterior_chain import PosteriorChain
from petra.relabel import relabel_samples_one_iteration as relabel_samples
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


def test_mv_gaussian_relabel_no_nans(posterior_chain_no_nans_fixture):
    max_num_sources = 2
    chain = posterior_chain_no_nans_fixture.get_chain()
    fit_mv_gaussian = create_parametric_fit(mv_normal_fit)
    mv_gaussian_params = fit_mv_gaussian(chain, max_num_sources=max_num_sources)

    compute_mv_gaussian_cost_matrix = create_compute_cost_matrix(mv_normal_aux_distribution)
    prob_in_model = find_prob_in_model(chain, max_num_sources=max_num_sources, eps=0)

    relabeled_chain, total_cost = relabel_samples(chain, mv_gaussian_params, prob_in_model, max_num_sources, compute_mv_gaussian_cost_matrix)

    assert len(relabeled_chain) == len(chain)
    first_three_expected = np.array([[[0.5488135 ],
         [0.74826798]],
        [[0.71518937],
         [0.18020271]],
        [[0.60276338],
         [0.38902314]]])
    for i in range(3):
        assert np.allclose(relabeled_chain[i], first_three_expected[i])

    last_three_expected = np.array([
        [[0.44645576],
         [0.75842952]],
        [[0.36012661],
         [0.02378743]],
        [[0.62588665],
         [0.81357508]]])
    for i in range(3):
        assert np.allclose(relabeled_chain[-3 + i], last_three_expected[i])

    total_cost_expected = 0.3616456656513372

    assert np.isclose(total_cost, total_cost_expected)
