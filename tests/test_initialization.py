import pytest
import numpy as np

from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import uni_normal_fit_single_parameter, create_parametric_fit
from petra.aux_distributions import uni_normal_aux_distribution_single_parameter
from petra.cost_matrix import create_compute_cost_matrix
from petra.utils import find_prob_in_model


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
    """The fit is the sample mean/std (ddof=1) of the non-NaN entries of that slot."""
    chain = posterior_chain_nans_fixture.get_chain()
    uni_normal_single_param_fit = create_parametric_fit(uni_normal_fit_single_parameter, single_parameter=0)
    means, std_devs = uni_normal_single_param_fit(chain, max_num_sources=2)
    assert len(means) == 2
    assert len(std_devs) == 2

    # Reference computed directly from the fixture rather than pinned as a magic number,
    # so the test states what the fit is supposed to be. ddof=1 matches the sample
    # covariance used by mv_normal_fit.
    for source in range(2):
        column = chain[:, source, 0]
        valid = column[~np.isnan(column)]
        assert np.isclose(means[source], np.mean(valid))
        assert np.isclose(std_devs[source], np.std(valid, ddof=1))

    # Both slots are draws from Uniform(0, 1): mean 1/2, std 1/sqrt(12) ~ 0.2887.
    assert np.allclose(means, 0.5, atol=0.01)
    assert np.allclose(std_devs, 1 / np.sqrt(12), atol=0.01)


def test_uni_gaussian_cost_matrix(posterior_chain_nans_fixture):
    """cost[i, j] == log(prob_in_model[i]) + logpdf_i(sample[j]) when nothing is absent."""
    chain = posterior_chain_nans_fixture.get_chain()
    uni_normal_single_param_fit = create_parametric_fit(uni_normal_fit_single_parameter, single_parameter=0)
    means, std_devs = uni_normal_single_param_fit(chain, max_num_sources=2)
    compute_cost_matrix = create_compute_cost_matrix(uni_normal_aux_distribution_single_parameter, single_parameter=0)

    # Each slot is present in half the samples, so the true inclusion probability is 0.5.
    # (The previous version passed [1, 1] with a comment saying it was incorrect.)
    prob_in_model = find_prob_in_model(chain, max_num_sources=2, eps=0)
    assert np.allclose(prob_in_model, 0.5)

    sample = chain[0]
    assert not np.isnan(sample).any(), "sample 0 has both slots occupied, so no NaN fill applies"
    cost_matrix = compute_cost_matrix(sample, (means, std_devs), prob_in_model, num_distributions=2)
    assert cost_matrix.shape == (2, 2)

    # Independent reference: a plain Gaussian log-density plus the log inclusion prior.
    values = sample[:, 0]
    expected = np.log(prob_in_model)[:, None] + (
        -0.5 * np.log(2 * np.pi)
        - np.log(std_devs)[:, None]
        - 0.5 * ((values[None, :] - means[:, None]) / std_devs[:, None]) ** 2
    )
    assert np.allclose(cost_matrix, expected)
