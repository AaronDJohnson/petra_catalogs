import pytest
import numpy as np
from petra.posterior_chain import PosteriorChain
from petra.parametric_fits import mv_normal_fit, create_parametric_fit

@pytest.fixture
def posterior_chain_fixture():
    np.random.seed(0)
    chain = np.zeros((20_000, 2))
    chain_signal_1 = np.random.uniform(0, 1, 10000)
    chain_signal_2 = np.random.uniform(0, 1, 10000)
    chain[::2, 0] = chain_signal_1
    chain[1::2, 0] = np.nan
    chain[::2, 1] = chain_signal_2
    chain[1::2, 1] = np.nan
    return PosteriorChain(chain, 2, 1)

def test_fit_mv_gaussian(posterior_chain_fixture):
    chain = posterior_chain_fixture.get_chain()
    mv_normal_param_fit = create_parametric_fit(mv_normal_fit)
    means, cov_matrices = mv_normal_param_fit(chain, max_num_sources=2)
    assert len(means) == 2
    assert len(cov_matrices) == 2
    assert np.isclose(means[0][0], 0.49645889162008944, rtol=1e-8)
    assert np.isclose(means[1][0], 0.495239621185434, rtol=1e-8)
    assert np.isclose(cov_matrices[0][0, 0], 0.08387139433093833, rtol=1e-8)
    assert np.isclose(cov_matrices[1][0, 0], 0.08485848849960947, rtol=1e-8)

def test_fit_mv_gaussian_max_sources(posterior_chain_fixture):
    chain = posterior_chain_fixture.get_chain()
    mv_normal_param_fit = create_parametric_fit(mv_normal_fit)
    means, cov_matrices = mv_normal_param_fit(chain, max_num_sources=1)
    assert len(means) == 1
    assert len(cov_matrices) == 1
    assert np.isclose(means[0][0], 0.49645889162008944, rtol=1e-8)
    assert np.isclose(cov_matrices[0][0, 0], 0.08387139433093833, rtol=1e-8)
