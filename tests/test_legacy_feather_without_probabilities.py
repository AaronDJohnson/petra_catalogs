"""Cover legacy metadata without probabilities, as in the bundled LISA example."""

import numpy as np
import pandas as pd

from petra.posterior_chain import PosteriorChain


def test_legacy_feather_can_omit_uncomputed_probabilities(tmp_path):
    chain = np.array([[[1.0, 2.0], [np.nan, np.nan]],
                      [[3.0, 4.0], [5.0, 6.0]]])
    frame = pd.DataFrame(chain.reshape(2, 4), columns=[str(i) for i in range(4)])
    frame['num_sources'] = 2
    frame['num_params_per_source'] = 2
    frame['transdimensional'] = True
    path = tmp_path / 'legacy.feather'
    frame.to_feather(path)

    restored = PosteriorChain.read_feather(str(path))

    np.testing.assert_array_equal(restored.chain, chain)
    assert restored.trans_dimensional
    assert restored.prob_in_model is None
    assert restored.cost_dict == {}
