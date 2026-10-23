"""Shared fixtures: scripted relabeling costs, the two-cluster checkpoint chain and the thinned LISA example.

matplotlib is optional and no test needs it; when it is installed, the non-interactive Agg backend is selected
before anything can create a figure.
"""

import pytest

from petra import relabel as relabel_module
from petra.posterior_chain import PosteriorChain
from tests import helpers
from tests.helpers import LISA_EXAMPLE, separated_chain

try:
    import matplotlib
except ModuleNotFoundError:
    # The [plots] extra is not installed, so there is no backend to select.
    pass
else:
    matplotlib.use("Agg")


@pytest.fixture
def scripted_costs(monkeypatch):
    """Return install(costs), which makes iteration k of the loop return separated_chain(k + 1.0) at costs[k]."""
    def install(costs):
        calls = []

        def fake(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix, progress=True):
            k = len(calls)
            calls.append({"chain": chain, "aux_parameters": aux_parameters, "prob_in_model": prob_in_model})
            return separated_chain(offset=k + 1.0), costs[k]

        monkeypatch.setattr(relabel_module, "relabel_samples_one_iteration", fake)
        return calls

    return install


@pytest.fixture
def two_cluster_chain():
    """The shuffled two-source chain of helpers.two_cluster_chain."""
    return helpers.two_cluster_chain()


@pytest.fixture(scope="session")
def lisa_sub():
    """Every tenth sample of the shipped LISA example, 200 x 20 x 8; skipped when the data file is missing."""
    if not LISA_EXAMPLE.is_file():
        pytest.skip(f"the LISA example {LISA_EXAMPLE} is missing")
    full = PosteriorChain.read_feather(str(LISA_EXAMPLE))
    return PosteriorChain(full.chain[::10], full.num_sources, full.num_params_per_source)
