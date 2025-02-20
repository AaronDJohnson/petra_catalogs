import pytest
import numpy as np
from petra.utils import find_prob_in_model
from petra.posterior_chain import PosteriorChain

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

def test_find_pastro(posterior_chain_fixture):
    chain = posterior_chain_fixture.get_chain()
    prob_in_model = find_prob_in_model(chain, max_num_sources=2)
    assert prob_in_model.shape[0] == 2
    assert np.isclose(prob_in_model[0], 0.5, rtol=1e-8)
    assert np.isclose(prob_in_model[1], 0.5, rtol=1e-8)

def test_find_pastro_max_sources(posterior_chain_fixture):
    chain = posterior_chain_fixture.get_chain()
    prob_in_model = find_prob_in_model(chain, max_num_sources=1)
    assert prob_in_model.shape[0] == 1
    assert np.isclose(prob_in_model[0], 0.5, rtol=1e-8)
